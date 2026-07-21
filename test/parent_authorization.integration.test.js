const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const path = require("node:path");
const test = require("node:test");
const { Client } = require("pg");

const shouldRun = process.env.RUN_DB_TESTS === "1";
const port = Number(process.env.TEST_PARENT_API_PORT || 5007);
const baseUrl = `http://127.0.0.1:${port}`;

async function request(pathname, options) {
  const response = await fetch(`${baseUrl}${pathname}`, options);
  return { response, body: await response.json() };
}

test("parent links are consented, isolated, and immediately revocable", { skip: !shouldRun }, async (testContext) => {
  const server = spawn(process.execPath, ["index.js"], {
    cwd: path.join(__dirname, ".."), env: { ...process.env, PORT: String(port) }, stdio: "ignore",
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
    }
    return { user: registration.body.user, headers: { "Content-Type": "application/json", Authorization: `Bearer ${registration.body.token}` } };
  }
  async function createHostel(owner, name) {
    const created = await request("/api/owner/hostels", {
      method: "POST", headers: owner.headers,
      body: JSON.stringify({ name, address: `${name} Road`, city: "Lahore", total_capacity: 2 }),
    });
    assert.equal(created.response.status, 201);
    return created.body.hostel;
  }

  const owner = await register("Parent Hostel Owner", "parent.owner@example.test", "owner");
  const hostel = await createHostel(owner, "Parent Access Hostel");
  const studentA = await register("Linked Student", "linked.student@example.test");
  const studentB = await register("Unlinked Student", "unlinked.student@example.test");
  const parentA = await register("Parent A", "parent.a@example.test", "parent");
  const parentB = await register("Parent B", "parent.b@example.test", "parent");
  const booking = await request("/api/bookings", {
    method: "POST", headers: studentA.headers,
    body: JSON.stringify({ hostel_id: hostel.id, check_in: "2034-01-10", check_out: "2034-01-20", status: "pending" }),
  });
  assert.equal(booking.response.status, 201);

  const pendingBeforeAccept = await request("/api/parent/children", { headers: parentA.headers });
  assert.equal(pendingBeforeAccept.response.status, 200);
  assert.equal(pendingBeforeAccept.body.children.length, 0);
  const invitation = await request("/api/student/parents/invitations", {
    method: "POST", headers: studentA.headers, body: JSON.stringify({ email: parentA.user.email }),
  });
  assert.equal(invitation.response.status, 201);
  assert.ok(invitation.body.test_invitation_token);
  const accept = await request("/api/parent/invitations/accept", {
    method: "POST", headers: parentA.headers, body: JSON.stringify({ token: invitation.body.test_invitation_token }),
  });
  assert.equal(accept.response.status, 200);
  const replay = await request("/api/parent/invitations/accept", {
    method: "POST", headers: parentA.headers, body: JSON.stringify({ token: invitation.body.test_invitation_token }),
  });
  assert.equal(replay.response.status, 400);

  const children = await request("/api/parent/children", { headers: parentA.headers });
  assert.deepEqual(children.body.children.map((child) => child.id), [studentA.user.id]);
  const overview = await request(`/api/parent/children/${studentA.user.id}/overview`, { headers: parentA.headers });
  assert.equal(overview.response.status, 200);
  assert.equal(overview.body.bookings[0].id, booking.body.booking.id);
  assert.equal(overview.body.bookings[0].hostel_name, hostel.name);

  const otherStudent = await request(`/api/parent/children/${studentB.user.id}/overview`, { headers: parentA.headers });
  assert.equal(otherStudent.response.status, 404);
  const unrelatedParent = await request(`/api/parent/children/${studentA.user.id}/overview`, { headers: parentB.headers });
  assert.equal(unrelatedParent.response.status, 404);
  const revoke = await request(`/api/student/parents/${parentA.user.id}`, { method: "DELETE", headers: studentA.headers });
  assert.equal(revoke.response.status, 200);
  const revokedOverview = await request(`/api/parent/children/${studentA.user.id}/overview`, { headers: parentA.headers });
  assert.equal(revokedOverview.response.status, 404);
});
