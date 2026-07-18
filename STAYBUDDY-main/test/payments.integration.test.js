const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const http = require("node:http");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const path = require("node:path");
const test = require("node:test");
const { Client } = require("pg");

const shouldRun = process.env.RUN_DB_TESTS === "1";
const port = Number(process.env.TEST_PAYMENTS_API_PORT || 5007);
const stripeMockPort = Number(process.env.TEST_STRIPE_MOCK_PORT || 5008);
const baseUrl = `http://127.0.0.1:${port}`;
const stripeWebhookSecret = "whsec_test_payments_integration";
const stripeSecretKey = "sk_test_payments_integration";

async function request(pathname, options) {
  const response = await fetch(`${baseUrl}${pathname}`, options);
  return { response, body: await response.json() };
}

function signStripePayload(payload, secret) {
  const timestamp = Math.floor(Date.now() / 1000);
  const signedPayload = `${timestamp}.${payload}`;
  const signature = crypto.createHmac("sha256", secret).update(signedPayload).digest("hex");
  return `t=${timestamp},v1=${signature}`;
}

test("Stripe payment intents are created server-side and only the verified webhook can settle them", { skip: !shouldRun }, async (testContext) => {
  // Minimal mock of Stripe's REST API so the payment-intent route can be exercised without real credentials.
  let lastIntentId = null;
  const stripeMock = http.createServer((req, res) => {
    if (req.method === "POST" && req.url === "/v1/payment_intents") {
      lastIntentId = `pi_test_${crypto.randomBytes(8).toString("hex")}`;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ id: lastIntentId, client_secret: `${lastIntentId}_secret` }));
      return;
    }
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "not found" }));
  });
  await new Promise((resolve) => stripeMock.listen(stripeMockPort, resolve));

  const server = spawn(process.execPath, ["index.js"], {
    cwd: path.join(__dirname, ".."),
    env: {
      ...process.env,
      PORT: String(port),
      STRIPE_SECRET_KEY: stripeSecretKey,
      STRIPE_WEBHOOK_SECRET: stripeWebhookSecret,
      STRIPE_API_BASE: `http://127.0.0.1:${stripeMockPort}`,
    },
    stdio: "ignore",
  });
  const client = new Client({
    host: process.env.DB_HOST, port: Number(process.env.DB_PORT), user: process.env.DB_USER,
    password: process.env.DB_PASSWORD, database: process.env.DB_NAME,
  });
  testContext.after(async () => {
    server.kill(); await once(server, "exit");
    await client.end();
    stripeMock.close(); await once(stripeMock, "close");
  });
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

  const owner = await register("Payments Owner", "payments.owner@example.test", "owner");
  const hostel = await request("/api/owner/hostels", {
    method: "POST", headers: owner.headers,
    body: JSON.stringify({ name: "Payments Hostel", address: "Payments Road", city: "Lahore", total_capacity: 2 }),
  });
  assert.equal(hostel.response.status, 201);
  const priced = await request(`/api/owner/hostels/${hostel.body.hostel.id}`, {
    method: "PATCH", headers: owner.headers, body: JSON.stringify({ single_room_price: 15000 }),
  });
  assert.equal(priced.response.status, 200);

  const student = await register("Payments Student", "payments.student@example.test");
  const booking = await request("/api/bookings", {
    method: "POST", headers: student.headers,
    body: JSON.stringify({ hostel_id: hostel.body.hostel.id, check_in: "2033-02-01", check_out: "2033-02-20", status: "pending" }),
  });
  assert.equal(booking.response.status, 201);

  const pendingIntent = await request("/api/payments/intents", {
    method: "POST", headers: student.headers, body: JSON.stringify({ booking_id: booking.body.booking.id }),
  });
  assert.equal(pendingIntent.response.status, 409, "only confirmed bookings should be payable");

  const confirmed = await request(`/api/owner/bookings/${booking.body.booking.id}`, {
    method: "PATCH", headers: owner.headers, body: JSON.stringify({ status: "confirmed" }),
  });
  assert.equal(confirmed.response.status, 200);

  const intent = await request("/api/payments/intents", {
    method: "POST", headers: student.headers, body: JSON.stringify({ booking_id: booking.body.booking.id }),
  });
  assert.equal(intent.response.status, 201, JSON.stringify(intent.body));
  assert.equal(intent.body.payment.status, "pending");
  assert.ok(intent.body.client_secret);
  assert.equal(intent.body.payment.provider_payment_id, lastIntentId);

  const otherStudent = await register("Other Payments Student", "other.payments.student@example.test");
  const forbiddenPayments = await request("/api/payments", { headers: otherStudent.headers });
  assert.equal(forbiddenPayments.response.status, 200);
  assert.equal(forbiddenPayments.body.length, 0);

  const succeededEvent = JSON.stringify({
    type: "payment_intent.succeeded",
    data: { object: { id: lastIntentId, charges: { data: [{ receipt_url: "https://example.test/receipt/1" }] } } },
  });
  const forgedWebhook = await fetch(`${baseUrl}/api/payments/webhook/stripe`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "stripe-signature": "t=1,v1=deadbeef" },
    body: succeededEvent,
  });
  assert.equal(forgedWebhook.status, 400, "an invalid signature must never settle a payment");

  const stillPending = await request("/api/payments", { headers: student.headers });
  assert.equal(stillPending.body[0].status, "pending");

  const validWebhook = await fetch(`${baseUrl}/api/payments/webhook/stripe`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "stripe-signature": signStripePayload(succeededEvent, stripeWebhookSecret) },
    body: succeededEvent,
  });
  assert.equal(validWebhook.status, 200);

  const settledPayments = await request("/api/payments", { headers: student.headers });
  assert.equal(settledPayments.response.status, 200);
  assert.equal(settledPayments.body[0].status, "succeeded");
  assert.equal(settledPayments.body[0].receipt_url, "https://example.test/receipt/1");
});
