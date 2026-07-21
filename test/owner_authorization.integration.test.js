const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const path = require("node:path");
const test = require("node:test");
const { Client } = require("pg");

const shouldRun = process.env.RUN_DB_TESTS === "1";
const port = Number(process.env.TEST_OWNER_API_PORT || 5005);
const baseUrl = `http://127.0.0.1:${port}`;

async function request(pathname, options) {
  const response = await fetch(`${baseUrl}${pathname}`, options);
  return { response, body: await response.json() };
}

test("owner data is isolated and cancellation releases capacity once", { skip: !shouldRun }, async (testContext) => {
  const server = spawn(process.execPath, ["index.js"], {
    cwd: path.join(__dirname, ".."),
    env: { ...process.env, PORT: String(port) },
    stdio: "ignore",
  });
  const client = new Client({
    host: process.env.DB_HOST, port: Number(process.env.DB_PORT), user: process.env.DB_USER,
    password: process.env.DB_PASSWORD, database: process.env.DB_NAME,
  });
  testContext.after(async () => { server.kill(); await once(server, "exit"); await client.end(); });
  await client.connect();
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try { if ((await fetch(`${baseUrl}/api/db-test`)).ok) break; } catch (_) { /* retry */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  async function register(name, email, role = "student") {
    const registration = await request("/api/auth/register", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password: "integration-password" }),
    });
    assert.equal(registration.response.status, 200);
    if (role !== "student") {
      await client.query("UPDATE users SET role=$1 WHERE id=$2", [role, registration.body.user.id]);
      const login = await request("/api/auth/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password: "integration-password" }),
      });
      assert.equal(login.response.status, 200);
      registration.body.token = login.body.token;
      registration.body.user = login.body.user;
    }
    return {
      user: registration.body.user,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${registration.body.token}` },
    };
  }
  async function createHostel(owner, name) {
    const created = await request("/api/owner/hostels", {
      method: "POST", headers: owner.headers,
      body: JSON.stringify({ name, address: `${name} Road`, city: "Lahore", total_capacity: 2 }),
    });
    assert.equal(created.response.status, 201);
    return created.body.hostel;
  }

  const ownerA = await register("Owner A", "owner.a@example.test", "owner");
  const ownerB = await register("Owner B", "owner.b@example.test", "owner");
  const hostelA = await createHostel(ownerA, "Owner A Hostel");
  const hostelB = await createHostel(ownerB, "Owner B Hostel");
  const ownerAHostels = await request("/api/owner/hostels", { headers: ownerA.headers });
  assert.equal(ownerAHostels.response.status, 200);
  assert.deepEqual(ownerAHostels.body.map((hostel) => hostel.id), [hostelA.id]);

  const profileUpdate = await request(`/api/owner/hostels/${hostelB.id}`, {
    method: "PATCH", headers: ownerA.headers, body: JSON.stringify({ name: "Unauthorized" }),
  });
  assert.equal(profileUpdate.response.status, 404);

  const student = await register("Queue Student", "owner.queue.student@example.test", "student");
  const booking = await request("/api/bookings", {
    method: "POST", headers: student.headers,
    body: JSON.stringify({ hostel_id: hostelB.id, check_in: "2032-01-10", check_out: "2032-01-20", status: "pending" }),
  });
  assert.equal(booking.response.status, 201);
  const complaint = await request("/api/complaints", {
    method: "POST", headers: student.headers,
    body: JSON.stringify({ hostel_id: hostelB.id, category: "Other", severity: "low", description: "Owner isolation regression" }),
  });
  assert.equal(complaint.response.status, 201);

  const bookingsAsA = await request("/api/owner/bookings", { headers: ownerA.headers });
  assert.equal(bookingsAsA.response.status, 200);
  assert.ok(!bookingsAsA.body.some((item) => item.id === booking.body.booking.id));
  const complaintsAsA = await request("/api/owner/complaints", { headers: ownerA.headers });
  assert.equal(complaintsAsA.response.status, 200);
  assert.ok(!complaintsAsA.body.some((item) => item.id === complaint.body.complaint.id));

  const bookingUpdate = await request(`/api/owner/bookings/${booking.body.booking.id}`, {
    method: "PATCH", headers: ownerA.headers, body: JSON.stringify({ status: "cancelled" }),
  });
  assert.equal(bookingUpdate.response.status, 404);
  const complaintUpdate = await request(`/api/owner/complaints/${complaint.body.complaint.id}`, {
    method: "PATCH", headers: ownerA.headers, body: JSON.stringify({ status: "resolved" }),
  });
  assert.equal(complaintUpdate.response.status, 404);

  const beforeReject = await client.query("SELECT available_capacity FROM hostels WHERE id=$1", [hostelB.id]);
  assert.equal(Number(beforeReject.rows[0].available_capacity), 1);
  const rejection = await request(`/api/owner/bookings/${booking.body.booking.id}`, {
    method: "PATCH", headers: ownerB.headers, body: JSON.stringify({ status: "cancelled" }),
  });
  assert.equal(rejection.response.status, 200);
  const afterReject = await client.query("SELECT available_capacity FROM hostels WHERE id=$1", [hostelB.id]);
  assert.equal(Number(afterReject.rows[0].available_capacity), 2);
  const repeatedRejection = await request(`/api/owner/bookings/${booking.body.booking.id}`, {
    method: "PATCH", headers: ownerB.headers, body: JSON.stringify({ status: "cancelled" }),
  });
  assert.equal(repeatedRejection.response.status, 200);
  const afterRepeatedReject = await client.query("SELECT available_capacity FROM hostels WHERE id=$1", [hostelB.id]);
  assert.equal(Number(afterRepeatedReject.rows[0].available_capacity), 2);
});