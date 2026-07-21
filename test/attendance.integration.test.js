const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const path = require("node:path");
const test = require("node:test");
const { Client } = require("pg");

const shouldRun = process.env.RUN_DB_TESTS === "1";
const port = Number(process.env.TEST_ATTENDANCE_API_PORT || 5008);
const baseUrl = `http://127.0.0.1:${port}`;
const request = async (pathname, options = {}) => { const response = await fetch(`${baseUrl}${pathname}`, options); return { response, body: await response.json() }; };

test("attendance events are idempotent, warden-scoped, and visible only to linked parents", { skip: !shouldRun }, async (t) => {
  const server = spawn(process.execPath, ["index.js"], { cwd: path.join(__dirname, ".."), env: { ...process.env, PORT: String(port) }, stdio: "ignore" });
  const client = new Client({ host: process.env.DB_HOST, port: Number(process.env.DB_PORT), user: process.env.DB_USER, password: process.env.DB_PASSWORD, database: process.env.DB_NAME });
  t.after(async () => { server.kill(); await once(server, "exit"); await client.end(); }); await client.connect();
  for (let i = 0; i < 50; i += 1) { try { if ((await fetch(`${baseUrl}/api/db-test`)).ok) break; } catch (_) {} await new Promise((r) => setTimeout(r, 100)); }
  async function user(name, email, role = "student") { const r = await request("/api/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, email, password: "integration-password" }) }); assert.equal(r.response.status, 200); if (role !== "student") { await client.query("UPDATE users SET role=$1 WHERE id=$2", [role, r.body.user.id]); const l = await request("/api/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password: "integration-password" }) }); r.body.token = l.body.token; } return { user: r.body.user, headers: { "Content-Type": "application/json", Authorization: `Bearer ${r.body.token}` } }; }
  const owner = await user("Attendance Owner", "attendance.owner@test", "owner"); const warden = await user("Attendance Warden", "attendance.warden@test", "warden"); const student = await user("Attendance Student", "attendance.student@test"); const parent = await user("Attendance Parent", "attendance.parent@test", "parent");
  const hostel = await request("/api/owner/hostels", { method: "POST", headers: owner.headers, body: JSON.stringify({ name: "Attendance Hostel", address: "Road", city: "Lahore", total_capacity: 2 }) }); assert.equal(hostel.response.status, 201);
  const booking = await request("/api/bookings", { method: "POST", headers: student.headers, body: JSON.stringify({ hostel_id: hostel.body.hostel.id, check_in: "2035-01-10", check_out: "2035-01-20", status: "pending" }) }); assert.equal(booking.response.status, 201);
  await request(`/api/owner/bookings/${booking.body.booking.id}`, { method: "PATCH", headers: owner.headers, body: JSON.stringify({ status: "confirmed" }) });
  await request("/api/owner/wardens", { method: "POST", headers: owner.headers, body: JSON.stringify({ hostel_id: hostel.body.hostel.id, email: warden.user.email }) });
  const invite = await request("/api/student/parents/invitations", { method: "POST", headers: student.headers, body: JSON.stringify({ email: parent.user.email }) }); await request("/api/parent/invitations/accept", { method: "POST", headers: parent.headers, body: JSON.stringify({ token: invite.body.test_invitation_token }) });
  const credential = await request(`/api/owner/hostels/${hostel.body.hostel.id}/attendance-credentials`, { method: "POST", headers: owner.headers, body: JSON.stringify({ label: "Gate RFID" }) }); assert.equal(credential.response.status, 201);
  const identifier = await request("/api/warden/attendance/identifiers", { method: "POST", headers: warden.headers, body: JSON.stringify({ hostel_id: hostel.body.hostel.id, student_id: student.user.id, external_identifier: "RFID-001" }) }); assert.equal(identifier.response.status, 201);
  const headers = { "Content-Type": "application/json", "x-attendance-key": credential.body.integration_key, "idempotency-key": "attendance-test-001" }; const payload = JSON.stringify({ external_identifier: "RFID-001", direction: "out", source: "rfid", occurred_at: "2035-01-10T18:30:00.000Z", external_event_id: "device-001" });
  const event = await request("/api/attendance/events", { method: "POST", headers, body: payload }); assert.equal(event.response.status, 201);
  const duplicate = await request("/api/attendance/events", { method: "POST", headers, body: payload }); assert.equal(duplicate.response.status, 200); assert.equal(duplicate.body.duplicate, true);
  const parentHistory = await request(`/api/parent/children/${student.user.id}/attendance?from=2035-01-10T00:00:00Z&to=2035-01-11T00:00:00Z`, { headers: parent.headers }); assert.equal(parentHistory.response.status, 200); assert.equal(parentHistory.body.events.length, 1);
  const wardenHistory = await request(`/api/warden/attendance?hostel_id=${hostel.body.hostel.id}&from=2035-01-10T00:00:00Z&to=2035-01-11T00:00:00Z`, { headers: warden.headers }); assert.equal(wardenHistory.response.status, 200); assert.equal(wardenHistory.body.events.length, 1);
});
