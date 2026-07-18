const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const path = require("node:path");
const test = require("node:test");
const { Client } = require("pg");

const shouldRun = process.env.RUN_DB_TESTS === "1";
const port = Number(process.env.TEST_WARDEN_API_PORT || 5006);
const baseUrl = `http://127.0.0.1:${port}`;

async function request(pathname, options) {
  const response = await fetch(`${baseUrl}${pathname}`, options);
  return { response, body: await response.json() };
}

test("warden assignments and notification preferences scope operational delivery", { skip: !shouldRun }, async (testContext) => {
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
    const result = await request("/api/owner/hostels", {
      method: "POST", headers: owner.headers,
      body: JSON.stringify({ name, address: `${name} Road`, city: "Lahore", total_capacity: 2 }),
    });
    assert.equal(result.response.status, 201);
    return result.body.hostel;
  }

  const ownerA = await register("Warden Owner A", "warden.owner.a@example.test", "owner");
  const ownerB = await register("Warden Owner B", "warden.owner.b@example.test", "owner");
  const hostelA = await createHostel(ownerA, "Warden Hostel A");
  const hostelB = await createHostel(ownerB, "Warden Hostel B");
  const wardenA = await register("Warden A", "warden.a@example.test", "warden");
  const wardenB = await register("Warden B", "warden.b@example.test", "warden");
  for (const assignment of [[ownerA, wardenA, hostelA], [ownerB, wardenB, hostelB]]) {
    const result = await request("/api/owner/wardens", {
      method: "POST", headers: assignment[0].headers,
      body: JSON.stringify({ email: assignment[1].user.email, hostel_id: assignment[2].id }),
    });
    assert.equal(result.response.status, 201);
  }

  const student = await register("Notification Student", "notification.student@example.test");
  const booking = await request("/api/bookings", {
    method: "POST", headers: student.headers,
    body: JSON.stringify({ hostel_id: hostelA.id, check_in: "2033-01-10", check_out: "2033-01-20", status: "pending" }),
  });
  assert.equal(booking.response.status, 201);
  const complaint = await request("/api/complaints", {
    method: "POST", headers: student.headers,
    body: JSON.stringify({ hostel_id: hostelA.id, category: "Other", severity: "low", description: "Warden isolation regression" }),
  });
  assert.equal(complaint.response.status, 201);

  const wardenAQueue = await request("/api/warden/bookings", { headers: wardenA.headers });
  assert.equal(wardenAQueue.response.status, 200);
  assert.ok(wardenAQueue.body.some((item) => item.id === booking.body.booking.id));
  const forbiddenBooking = await request(`/api/warden/bookings/${booking.body.booking.id}`, {
    method: "PATCH", headers: wardenB.headers, body: JSON.stringify({ status: "confirmed" }),
  });
  assert.equal(forbiddenBooking.response.status, 404);
  const forbiddenComplaint = await request(`/api/warden/complaints/${complaint.body.complaint.id}`, {
    method: "PATCH", headers: wardenB.headers, body: JSON.stringify({ status: "resolved" }),
  });
  assert.equal(forbiddenComplaint.response.status, 404);

  const bookingUpdate = await request(`/api/warden/bookings/${booking.body.booking.id}`, {
    method: "PATCH", headers: wardenA.headers, body: JSON.stringify({ status: "confirmed" }),
  });
  assert.equal(bookingUpdate.response.status, 200);
  const room = await request("/api/rooms", {
    method: "POST", headers: ownerA.headers,
    body: JSON.stringify({ hostel_id: hostelA.id, room_number: "A-101", capacity: 1, room_type: "single" }),
  });
  assert.equal(room.response.status, 201);
  const forbiddenRoomAssignment = await request(`/api/bookings/${booking.body.booking.id}/assign-room`, {
    method: "POST", headers: wardenB.headers, body: JSON.stringify({ room_id: room.body.room.id }),
  });
  assert.equal(forbiddenRoomAssignment.response.status, 404);
  const roomAssignment = await request(`/api/bookings/${booking.body.booking.id}/assign-room`, {
    method: "POST", headers: wardenA.headers, body: JSON.stringify({ room_id: room.body.room.id }),
  });
  assert.equal(roomAssignment.response.status, 200);
  const checkIn = await request(`/api/warden/bookings/${booking.body.booking.id}/check-in`, {
    method: "POST", headers: wardenA.headers,
  });
  assert.equal(checkIn.response.status, 200);
  const checkOut = await request(`/api/warden/bookings/${booking.body.booking.id}/check-out`, {
    method: "POST", headers: wardenA.headers,
  });
  assert.equal(checkOut.response.status, 200);
  const hostelAfterCheckout = await client.query("SELECT available_capacity FROM hostels WHERE id=$1", [hostelA.id]);
  const roomAfterCheckout = await client.query("SELECT available_capacity FROM rooms WHERE id=$1", [room.body.room.id]);
  assert.equal(Number(hostelAfterCheckout.rows[0].available_capacity), 2);
  assert.equal(Number(roomAfterCheckout.rows[0].available_capacity), 1, JSON.stringify(roomAfterCheckout.rows[0]));
  const repeatedCheckout = await request(`/api/warden/bookings/${booking.body.booking.id}/check-out`, {
    method: "POST", headers: wardenA.headers,
  });
  assert.equal(repeatedCheckout.response.status, 409);
  const complaintUpdate = await request(`/api/warden/complaints/${complaint.body.complaint.id}`, {
    method: "PATCH", headers: wardenA.headers, body: JSON.stringify({ status: "in_progress" }),
  });
  assert.equal(complaintUpdate.response.status, 200, JSON.stringify(complaintUpdate.body));
  const statusNotifications = await request("/api/notifications", { headers: student.headers });
  assert.equal(statusNotifications.response.status, 200);
  assert.equal(statusNotifications.body.filter((item) => item.type === "booking_status" || item.type === "complaint_status").length, 3);

  const favorite = await request("/api/favorites", {
    method: "POST", headers: student.headers, body: JSON.stringify({ hostel_id: hostelA.id }),
  });
  assert.equal(favorite.response.status, 201);

  const disableAnnouncements = await request("/api/notification-preferences", {
    method: "PUT", headers: student.headers, body: JSON.stringify({ announcements: false }),
  });
  assert.equal(disableAnnouncements.response.status, 200);
  const mutedAnnouncement = await request("/api/announcements", {
    method: "POST", headers: ownerA.headers,
    body: JSON.stringify({ hostel_id: hostelA.id, title: "Muted", body: "No delivery expected", audience: "favorited" }),
  });
  assert.equal(mutedAnnouncement.response.status, 201);
  const mutedNotifications = await request("/api/notifications", { headers: student.headers });
  assert.equal(mutedNotifications.body.filter((item) => item.type === "announcement").length, 0);

  await request("/api/notification-preferences", { method: "PUT", headers: student.headers, body: JSON.stringify({ announcements: true }) });
  const deliveredAnnouncement = await request("/api/announcements", {
    method: "POST", headers: wardenA.headers,
    body: JSON.stringify({ hostel_id: hostelA.id, title: "Delivered", body: "Delivery expected", audience: "favorited" }),
  });
  assert.equal(deliveredAnnouncement.response.status, 201);
  const deliveredNotifications = await request("/api/notifications", { headers: student.headers });
  const announcementNotification = deliveredNotifications.body.find((item) => item.type === "announcement");
  assert.ok(announcementNotification);
  const read = await request(`/api/notifications/${announcementNotification.id}/read`, { method: "PATCH", headers: student.headers });
  assert.equal(read.response.status, 200);
  assert.ok(read.body.notification.read_at);
});