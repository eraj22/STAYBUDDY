const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const path = require("node:path");
const test = require("node:test");
const { Client } = require("pg");

const shouldRun = process.env.RUN_DB_TESTS === "1";
const port = Number(process.env.TEST_API_PORT || 5002);
const baseUrl = `http://127.0.0.1:${port}`;

async function waitForApi() {
  let lastError;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const response = await fetch(`${baseUrl}/api/db-test`);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw lastError || new Error("Timed out waiting for the test API");
}

async function request(pathname, options) {
  const response = await fetch(`${baseUrl}${pathname}`, options);
  const body = await response.json();
  return { response, body };
}

test("booking lifecycle maintains catalog availability", { skip: !shouldRun }, async (testContext) => {
  const root = path.join(__dirname, "..");
  const server = spawn(process.execPath, ["index.js"], {
    cwd: root,
    env: { ...process.env, PORT: String(port) },
    stdio: "ignore",
  });
  const client = new Client({
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  });

  testContext.after(async () => {
    server.kill();
    await once(server, "exit");
    await client.end();
  });

  await client.connect();
  await waitForApi();

  const userResult = await client.query("SELECT id, email FROM users ORDER BY id LIMIT 1");
  const hostelResult = await client.query(
    `SELECT id, available_capacity, source_available_capacity
     FROM hostels
     WHERE source='ml_catalog' AND available_capacity > 0
     ORDER BY external_id
     LIMIT 1`
  );
  assert.equal(userResult.rowCount, 1, "test database must contain one user");
  assert.equal(hostelResult.rowCount, 1, "test database must contain an available catalog hostel");

  const userId = userResult.rows[0].id;
  const hostel = hostelResult.rows[0];
  const before = Number(hostel.available_capacity);

  const login = await request("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: userResult.rows[0].email, password: "anything" }),
  });
  assert.equal(login.response.status, 200);
  const authHeaders = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${login.body.token}`,
  };

  const firstBooking = await request("/api/bookings", {
    method: "POST",
    headers: authHeaders,
    body: JSON.stringify({
      user_id: userId,
      hostel_id: hostel.id,
      check_in: "2030-01-10",
      check_out: "2030-01-20",
      status: "pending",
    }),
  });
  assert.equal(firstBooking.response.status, 201);

  const duringFirst = await client.query("SELECT available_capacity FROM hostels WHERE id=$1", [hostel.id]);
  assert.equal(Number(duringFirst.rows[0].available_capacity), before - 1);

  const cancellation = await request(`/api/bookings/${firstBooking.body.booking.id}/cancel`, { method: "POST", headers: authHeaders });
  assert.equal(cancellation.response.status, 200);

  const afterCancel = await client.query("SELECT available_capacity FROM hostels WHERE id=$1", [hostel.id]);
  assert.equal(Number(afterCancel.rows[0].available_capacity), before);

  const secondBooking = await request("/api/bookings", {
    method: "POST",
    headers: authHeaders,
    body: JSON.stringify({
      user_id: userId,
      hostel_id: hostel.id,
      check_in: "2030-02-10",
      check_out: "2030-02-20",
      status: "pending",
    }),
  });
  assert.equal(secondBooking.response.status, 201);

  const statusUpdate = await request(`/api/bookings/${secondBooking.body.booking.id}`, {
    method: "PUT",
    headers: authHeaders,
    body: JSON.stringify({ status: "cancelled" }),
  });
  assert.equal(statusUpdate.response.status, 200);

  const afterStatusUpdate = await client.query("SELECT available_capacity FROM hostels WHERE id=$1", [hostel.id]);
  assert.equal(Number(afterStatusUpdate.rows[0].available_capacity), before);
  assert.ok(before <= Number(hostel.source_available_capacity));
});

test("a student cannot read, cancel, or forge another student's data", { skip: !shouldRun }, async (testContext) => {
  const authPort = port + 1;
  const authBaseUrl = `http://127.0.0.1:${authPort}`;
  const root = path.join(__dirname, "..");
  const server = spawn(process.execPath, ["index.js"], {
    cwd: root,
    env: { ...process.env, PORT: String(authPort) },
    stdio: "ignore",
  });
  const client = new Client({
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  });

  testContext.after(async () => {
    server.kill();
    await once(server, "exit");
    await client.end();
  });

  await client.connect();

  async function waitForAuthApi() {
    let lastError;
    for (let attempt = 0; attempt < 50; attempt += 1) {
      try {
        const response = await fetch(`${authBaseUrl}/api/db-test`);
        if (response.ok) return;
      } catch (error) {
        lastError = error;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw lastError || new Error("Timed out waiting for the test API");
  }
  async function authRequest(pathname, options) {
    const response = await fetch(`${authBaseUrl}${pathname}`, options);
    const body = await response.json();
    return { response, body };
  }
  async function loginAs(email) {
    const login = await authRequest("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password: "anything" }),
    });
    assert.equal(login.response.status, 200);
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${login.body.token}`,
    };
  }

  await waitForAuthApi();

  const usersResult = await client.query("SELECT id, email FROM users WHERE role='student' ORDER BY id LIMIT 1");
  assert.equal(usersResult.rowCount, 1, "test database must contain a student for isolation checks");
  const userA = usersResult.rows[0];

  const headersA = await loginAs(userA.email);
  const secondStudent = await authRequest("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: "Isolation Student B",
      email: "isolation.student.b@example.test",
      password: "test-password",
      role: "student",
    }),
  });
  assert.equal(secondStudent.response.status, 200);
  const userB = secondStudent.body.user;
  const headersB = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${secondStudent.body.token}`,
  };

  // A forged/mismatched Authorization header must be rejected outright.
  const noAuth = await authRequest("/api/bookings", { method: "GET" });
  assert.equal(noAuth.response.status, 401);
  const forgedAuth = await authRequest("/api/bookings", {
    method: "GET",
    headers: { Authorization: "Bearer not-a-real-token" },
  });
  assert.equal(forgedAuth.response.status, 401);

  // Create a booking as user A using their own token.
  const hostelResult = await client.query(
    `SELECT id FROM hostels WHERE source='ml_catalog' AND available_capacity > 0 ORDER BY external_id LIMIT 1`
  );
  assert.equal(hostelResult.rowCount, 1, "test database must contain an available catalog hostel");
  const hostelId = hostelResult.rows[0].id;

  const bookingAsA = await authRequest("/api/bookings", {
    method: "POST",
    headers: headersA,
    body: JSON.stringify({
      user_id: userA.id,
      hostel_id: hostelId,
      check_in: "2031-01-10",
      check_out: "2031-01-20",
      status: "pending",
    }),
  });
  assert.equal(bookingAsA.response.status, 201);
  const bookingId = bookingAsA.body.booking.id;

  const favoriteAsA = await authRequest("/api/favorites", {
    method: "POST",
    headers: headersA,
    body: JSON.stringify({ hostel_id: hostelId }),
  });
  assert.equal(favoriteAsA.response.status, 201);
  assert.equal(favoriteAsA.body.favorite.user_id, userA.id);

  // A spoofed body user_id cannot create a review under another identity.
  const reviewAsA = await authRequest("/api/reviews", {
    method: "POST",
    headers: headersA,
    body: JSON.stringify({
      user_id: userB.id,
      hostel_id: hostelId,
      overall_rating: 4,
      text_review: "Disposable authorization regression review.",
    }),
  });
  assert.equal(reviewAsA.response.status, 201);
  assert.equal(reviewAsA.body.review.user_id, userA.id);

  const hostelDetail = await authRequest(`/api/hostels/${hostelId}`, {
    method: "GET",
    headers: headersA,
  });
  assert.equal(hostelDetail.response.status, 200);
  assert.ok(hostelDetail.body.reviews.some((review) => review.id === reviewAsA.body.review.id));

  // A spoofed body user_id cannot create a complaint under another identity.
  const complaintAsA = await authRequest("/api/complaints", {
    method: "POST",
    headers: headersA,
    body: JSON.stringify({
      user_id: userB.id,
      hostel_id: hostelId,
      category: "Maintenance",
      severity: "low",
      description: "Disposable authorization regression complaint.",
    }),
  });
  assert.equal(complaintAsA.response.status, 201);
  assert.equal(complaintAsA.body.complaint.user_id, userA.id);

  // Even if user B's request body claims to be user A, the token identity wins:
  // the booking is created under user B, not user A.
  const bookingBodySpoofed = await authRequest("/api/bookings", {
    method: "POST",
    headers: headersB,
    body: JSON.stringify({
      user_id: userA.id,
      hostel_id: hostelId,
      check_in: "2031-02-10",
      check_out: "2031-02-20",
      status: "pending",
    }),
  });
  assert.equal(bookingBodySpoofed.response.status, 201);
  assert.equal(bookingBodySpoofed.body.booking.user_id, userB.id);

  // User B cannot see user A's bookings in their own list.
  const bookingsAsB = await authRequest("/api/bookings", { method: "GET", headers: headersB });
  assert.equal(bookingsAsB.response.status, 200);
  assert.ok(!bookingsAsB.body.some((b) => b.id === bookingId));

  // User B cannot cancel user A's booking.
  const cancelAttempt = await authRequest(`/api/bookings/${bookingId}/cancel`, {
    method: "POST",
    headers: headersB,
  });
  assert.equal(cancelAttempt.response.status, 403);

  const updateAttempt = await authRequest(`/api/bookings/${bookingId}`, {
    method: "PUT",
    headers: headersB,
    body: JSON.stringify({ status: "cancelled" }),
  });
  assert.equal(updateAttempt.response.status, 403);

  // User B cannot read user A's favorites list.
  const favReadAttempt = await authRequest(`/api/favorites/${userA.id}`, {
    method: "GET",
    headers: headersB,
  });
  assert.equal(favReadAttempt.response.status, 403);

  // User B's list endpoints expose none of student A's data.
  const favoritesAsB = await authRequest(`/api/favorites/${userB.id}`, {
    method: "GET",
    headers: headersB,
  });
  assert.equal(favoritesAsB.response.status, 200);
  assert.ok(!favoritesAsB.body.some((favorite) => favorite.favorite_id === favoriteAsA.body.favorite.id));

  const complaintsAsB = await authRequest("/api/complaints", {
    method: "GET",
    headers: headersB,
  });
  assert.equal(complaintsAsB.response.status, 200);
  assert.ok(!complaintsAsB.body.some((complaint) => complaint.id === complaintAsA.body.complaint.id));

  // Cleanup: cancel the bookings we created so repeated runs stay clean.
  await authRequest(`/api/bookings/${bookingId}/cancel`, { method: "POST", headers: headersA });
  await authRequest(`/api/bookings/${bookingBodySpoofed.body.booking.id}/cancel`, { method: "POST", headers: headersB });
});