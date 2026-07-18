require("dotenv").config();

console.log("ENV CHECK:", {
  DB_HOST: process.env.DB_HOST,
  DB_PORT: process.env.DB_PORT,
  DB_USER: process.env.DB_USER,
  DB_NAME: process.env.DB_NAME,
});

const express = require("express");
const cors = require("cors");
const crypto = require("crypto");
const { Pool } = require("pg");
const nodemailer = require("nodemailer");
const { hashPassword, isPassword, verifyPassword } = require("./passwords");

const {
  buildUserPreferenceFromHistory,
  softNormalize,
  validateAndFallback,
  explain,
  actionWeight,
  clamp01
} = require("./recommender");

const app = express();
app.use(cors());
app.post("/api/payments/webhook/stripe", express.raw({ type: "application/json" }), async (req, res) => {
  try {
    const secret = process.env.STRIPE_WEBHOOK_SECRET;
    const signature = req.headers["stripe-signature"];
    if (!secret || !signature || !Buffer.isBuffer(req.body)) return res.status(400).json({ error: "Invalid payment webhook" });
    const parts = Object.fromEntries(signature.split(",").map((part) => part.split("=")));
    const timestamp = Number(parts.t);
    if (!Number.isFinite(timestamp) || Math.abs(Date.now() / 1000 - timestamp) > 300 || !parts.v1) return res.status(400).json({ error: "Expired payment webhook" });
    const expected = crypto.createHmac("sha256", secret).update(`${timestamp}.${req.body.toString("utf8")}`).digest("hex");
    const received = Buffer.from(parts.v1, "hex");
    const expectedBuffer = Buffer.from(expected, "hex");
    if (received.length !== expectedBuffer.length || !crypto.timingSafeEqual(received, expectedBuffer)) return res.status(400).json({ error: "Invalid payment webhook signature" });
    const event = JSON.parse(req.body.toString("utf8"));
    const intent = event?.data?.object;
    if (!intent?.id) return res.status(200).json({ received: true });
    if (event.type === "payment_intent.succeeded" || event.type === "payment_intent.payment_failed") {
      const status = event.type === "payment_intent.succeeded" ? "succeeded" : "failed";
      const receiptUrl = intent?.charges?.data?.[0]?.receipt_url || null;
      await pool.query("UPDATE payments SET status=$1, receipt_url=COALESCE($2,receipt_url), updated_at=NOW() WHERE provider='stripe' AND provider_payment_id=$3", [status, receiptUrl, intent.id]);
    }
    res.status(200).json({ received: true });
  } catch (error) {
    console.error("Stripe webhook failed:", error.message);
    res.status(400).json({ error: "Invalid payment webhook" });
  }
});
app.use(express.json());

// Connect PostgreSQL
const pool = new Pool({
  host:     process.env.DB_HOST,
  user:     process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port:     Number(process.env.DB_PORT),
});

// Set search_path on every new connection
pool.on("connect", (client) => {
  client.query("SET search_path TO public");
});

// Test DB connection on startup
pool.connect((err, client, release) => {
  if (err) {
    console.error("❌ DB connection failed:", err.message);
  } else {
    console.log("✅ DB connected successfully");
    release();
  }
});

// ── Email transporter (Gmail App Password) ────────────────────────
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});

// In-memory OTP store: email -> { otp, expiresAt }
const otpStore = new Map();

const PASSWORD_RESET_TTL_MINUTES = 15;
let passwordResetTableReady;

async function ensurePasswordResetTable() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS password_reset_tokens (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      token_hash CHAR(64) NOT NULL UNIQUE,
      expires_at TIMESTAMPTZ NOT NULL,
      used_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE INDEX IF NOT EXISTS password_reset_tokens_active_user_idx
      ON password_reset_tokens (user_id, expires_at)
      WHERE used_at IS NULL
  `);
  console.log("✅ password reset tokens table ready");
}
passwordResetTableReady = ensurePasswordResetTable().catch((error) => {
  console.error("❌ password reset tokens table setup failed:", error.message);
  throw error;
});

let operationalTablesReady;
async function ensureOperationalTables() {
  await pool.query("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check");
  await pool.query("ALTER TABLE users ADD CONSTRAINT users_role_check CHECK (role IN ('student', 'owner', 'warden', 'admin'))");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS student_profiles (
      user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      university VARCHAR(150), department VARCHAR(150), gender VARCHAR(30),
      budget_max NUMERIC(12, 2) CHECK (budget_max IS NULL OR budget_max >= 0),
      max_distance_km NUMERIC(6, 2) CHECK (max_distance_km IS NULL OR max_distance_km >= 0),
      study_preference NUMERIC(3, 2) CHECK (study_preference IS NULL OR (study_preference >= 0 AND study_preference <= 1)),
      food_preference VARCHAR(30), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS notification_preferences (
      user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      booking_updates BOOLEAN NOT NULL DEFAULT true, complaint_updates BOOLEAN NOT NULL DEFAULT true,
      announcements BOOLEAN NOT NULL DEFAULT true, email_enabled BOOLEAN NOT NULL DEFAULT true,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS hostel_owners (
      owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
      claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (owner_user_id, hostel_id), UNIQUE (hostel_id)
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS warden_assignments (
      warden_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
      assigned_by_user_id INTEGER NOT NULL REFERENCES users(id),
      assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (warden_user_id, hostel_id)
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS complaint_status_history (
      id SERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
      previous_status VARCHAR(30), next_status VARCHAR(30) NOT NULL,
      changed_by_user_id INTEGER NOT NULL REFERENCES users(id), changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS announcements (
      id SERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
      author_user_id INTEGER NOT NULL REFERENCES users(id), title VARCHAR(180) NOT NULL, body TEXT NOT NULL,
      audience VARCHAR(20) NOT NULL DEFAULT 'residents' CHECK (audience IN ('residents', 'booked', 'favorited')),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS notifications (
      id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      announcement_id INTEGER REFERENCES announcements(id) ON DELETE CASCADE, type VARCHAR(40) NOT NULL,
      title VARCHAR(180) NOT NULL, body TEXT NOT NULL, data JSONB NOT NULL DEFAULT '{}'::jsonb,
      read_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS room_assignment_history (
      id SERIAL PRIMARY KEY, booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
      previous_room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL, next_room_id INTEGER NOT NULL REFERENCES rooms(id),
      assigned_by_user_id INTEGER NOT NULL REFERENCES users(id), assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS booking_stays (
      booking_id INTEGER PRIMARY KEY REFERENCES bookings(id) ON DELETE CASCADE,
      checked_in_at TIMESTAMPTZ, checked_in_by_user_id INTEGER REFERENCES users(id),
      checked_out_at TIMESTAMPTZ, checked_out_by_user_id INTEGER REFERENCES users(id),
      CHECK (checked_out_at IS NULL OR checked_in_at IS NOT NULL),
      CHECK (checked_out_at IS NULL OR checked_out_at >= checked_in_at)
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS payments (
      id SERIAL PRIMARY KEY, booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
      user_id INTEGER NOT NULL REFERENCES users(id), provider VARCHAR(30) NOT NULL,
      provider_payment_id VARCHAR(255) UNIQUE, amount NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
      currency CHAR(3) NOT NULL DEFAULT 'PKR', status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'succeeded', 'failed', 'cancelled', 'refunded')),
      receipt_url TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query("CREATE INDEX IF NOT EXISTS hostel_owners_owner_idx ON hostel_owners (owner_user_id)");
  await pool.query("CREATE INDEX IF NOT EXISTS warden_assignments_hostel_idx ON warden_assignments (hostel_id)");
  await pool.query("CREATE INDEX IF NOT EXISTS complaint_status_history_complaint_idx ON complaint_status_history (complaint_id, changed_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS announcements_hostel_created_idx ON announcements (hostel_id, created_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS notifications_user_created_idx ON notifications (user_id, created_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS room_assignment_history_booking_idx ON room_assignment_history (booking_id, assigned_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS payments_user_created_idx ON payments (user_id, created_at DESC)");
  console.log("✅ profiles and staff ownership tables ready");
}
operationalTablesReady = ensureOperationalTables().catch((error) => {
  console.error("❌ profiles and staff ownership table setup failed:", error.message);
  throw error;
});

// ── Auth token signing (HMAC-SHA256) ───────────────────────────────
// Tokens are opaque and forgeable unless they carry a signature that only
// the server can produce. AUTH_SECRET must be set in .env; a fallback is
// provided only so local dev never silently 500s, but it is NOT safe to
// share/deploy with the fallback value.
const AUTH_SECRET = process.env.AUTH_SECRET;
if (!AUTH_SECRET) {
  console.warn(
    "⚠️  AUTH_SECRET is not set in .env — using an insecure development-only fallback. " +
    "Set AUTH_SECRET before deploying or sharing this server."
  );
}
const EFFECTIVE_AUTH_SECRET = AUTH_SECRET || "insecure-dev-only-fallback-secret";

function base64UrlEncode(input) {
  return Buffer.from(input).toString("base64url");
}
function signAuthToken(user) {
  const payload = { id: user.id, role: user.role, iat: Date.now() };
  const encodedPayload = base64UrlEncode(JSON.stringify(payload));
  const signature = crypto
    .createHmac("sha256", EFFECTIVE_AUTH_SECRET)
    .update(encodedPayload)
    .digest("base64url");
  return `${encodedPayload}.${signature}`;
}
function verifyAuthToken(token) {
  if (!token || typeof token !== "string" || !token.includes(".")) return null;
  const [encodedPayload, signature] = token.split(".");
  if (!encodedPayload || !signature) return null;
  const expectedSignature = crypto
    .createHmac("sha256", EFFECTIVE_AUTH_SECRET)
    .update(encodedPayload)
    .digest("base64url");
  const providedBuffer = Buffer.from(signature);
  const expectedBuffer = Buffer.from(expectedSignature);
  if (
    providedBuffer.length !== expectedBuffer.length ||
    !crypto.timingSafeEqual(providedBuffer, expectedBuffer)
  ) {
    return null;
  }
  try {
    const payload = JSON.parse(Buffer.from(encodedPayload, "base64url").toString("utf8"));
    if (!isValidInt(Number(payload.id)) || !payload.role) return null;
    return { id: Number(payload.id), role: String(payload.role) };
  } catch {
    return null;
  }
}
function hashPasswordResetToken(token) {
  return crypto.createHash("sha256").update(token).digest("hex");
}
function passwordResetUrl(token) {
  const baseUrl = process.env.PASSWORD_RESET_URL || "staybuddy://reset-password";
  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}token=${encodeURIComponent(token)}`;
}
async function sendPasswordResetEmail(user, token) {
  await transporter.sendMail({
    from: `"StayBuddy" <${process.env.EMAIL_USER}>`,
    to: user.email,
    subject: "Reset your StayBuddy password",
    text: `Use this link within ${PASSWORD_RESET_TTL_MINUTES} minutes to reset your password: ${passwordResetUrl(token)}`,
  });
}
// Express middleware: requires a valid signed token and attaches req.user = { id, role }.
function requireAuth(req, res, next) {
  const auth = req.headers.authorization || "";
  const token = auth.replace("Bearer ", "").trim();
  const user = verifyAuthToken(token);
  if (!user) return res.status(401).json({ error: "Missing or invalid auth token" });
  req.user = user;
  next();
}
function requireRole(...roles) {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: "You do not have permission for this action" });
    }
    next();
  };
}
async function ownerHasHostel(client, ownerUserId, hostelId) {
  const result = await client.query(
    "SELECT 1 FROM hostel_owners WHERE owner_user_id=$1 AND hostel_id=$2",
    [ownerUserId, hostelId]
  );
  return result.rowCount === 1;
}
async function wardenHasHostel(client, wardenUserId, hostelId) {
  const result = await client.query(
    "SELECT 1 FROM warden_assignments WHERE warden_user_id=$1 AND hostel_id=$2",
    [wardenUserId, hostelId]
  );
  return result.rowCount === 1;
}
async function createNotification(client, userId, type, title, body, data, preferenceColumn) {
  const preferenceColumns = new Set(["booking_updates", "complaint_updates", "announcements"]);
  if (!preferenceColumns.has(preferenceColumn)) throw new Error("Invalid notification preference");
  await client.query(
    `INSERT INTO notifications (user_id, type, title, body, data)
     SELECT $1,$2,$3,$4,$5::jsonb
     WHERE COALESCE((SELECT ${preferenceColumn} FROM notification_preferences WHERE user_id=$1), true)`,
    [userId, type, title, body, JSON.stringify(data)]
  );
}

// ---------- helpers ----------
function isValidInt(n) {
  return Number.isInteger(n) && n > 0;
}
function generateOtp() {
  return Math.floor(1000 + Math.random() * 9000).toString();
}
function isValidDateInput(value) {
  return !Number.isNaN(Date.parse(value));
}
function normalizeBookingStatus(status) {
  return String(status || "pending").trim().toLowerCase();
}
function isBookingStatus(status) {
  return ["pending", "confirmed", "cancelled", "completed"].includes(status);
}
function isCapacityHoldingStatus(status) {
  return status === "pending" || status === "confirmed";
}
async function releaseHostelCapacity(client, hostelId) {
  const result = await client.query(
    `UPDATE hostels
     SET available_capacity = available_capacity + 1
     WHERE id=$1
       AND available_capacity < COALESCE(source_available_capacity, total_capacity)
     RETURNING id`,
    [hostelId]
  );
  return result.rowCount === 1;
}
function isValidComplaintStatus(status) {
  return ["open", "in_progress", "resolved", "closed"].includes(status);
}
function priorityToSeverity(priority) {
  const map = { "High": "high", "Medium": "medium", "Low": "low" };
  return map[priority] || "medium";
}
const hostelResponseColumns = [
  "id", "external_id", "name", "address", "city", "university", "hostel_type", "area",
  "verified", "latitude", "longitude", "distance_from_fast_km", "price_tier",
  "single_room_price", "double_room_price", "dorm_room_price", "meal_included", "food_type",
  "food_rating", "room_types_available", "total_rooms", "total_capacity", "available_capacity",
  "amenities", "internet_speed_mbps", "noise_level", "curfew_hour", "study_environment_score",
  "description"
].join(", ");
const hostelResponseColumnsWithAlias = hostelResponseColumns
  .split(", ")
  .map((column) => `h.${column}`)
  .join(", ");
async function callAiComplaintCategorizer(description) {
  try {
    const response = await fetch("http://127.0.0.1:8001/categorize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: description })
    });
    if (!response.ok) {
      console.warn("AI categorizer returned", response.status);
      return null;
    }
    return await response.json();
  } catch (err) {
    console.warn("AI categorizer unavailable:", err.message);
    return null;
  }
}

// ✅ Test API
app.get("/", (req, res) => {
  res.json({ message: "StayBuddy API running ✅" });
});

// ✅ Test DB Connection
app.get("/api/db-test", async (req, res) => {
  try {
    const result = await pool.query("SELECT NOW() as now");
    res.json({ db_time: result.rows[0].now });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Database connection failed" });
  }
});

// ══════════════════════════════════════════════════════════════════
//  OWNER ROUTES
// ══════════════════════════════════════════════════════════════════

// Create owners table on startup
async function ensureOwnerTable() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS owners (
      id          SERIAL PRIMARY KEY,
      full_name   VARCHAR(120) NOT NULL,
      email       VARCHAR(120) NOT NULL UNIQUE,
      cnic        VARCHAR(20)  NOT NULL,
      city        VARCHAR(80)  NOT NULL,
      phone       VARCHAR(20)  NOT NULL,
      is_verified BOOLEAN      NOT NULL DEFAULT false,
      created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    )
  `);
  console.log("✅ owners table ready");
}
ensureOwnerTable().catch(console.error);

// POST /api/owner/send-otp
app.post("/api/owner/send-otp", async (req, res) => {
  try {
    const { full_name, email, cnic, city, phone } = req.body;

    if (!full_name || !email || !cnic || !city || !phone) {
      return res.status(400).json({ error: "All fields are required" });
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ error: "Invalid email address" });
    }

    // Check if already verified
    const existing = await pool.query(
      "SELECT id, is_verified FROM owners WHERE email = $1",
      [email]
    );

    if (existing.rows.length > 0 && existing.rows[0].is_verified) {
      return res.status(409).json({
        error: "This email is already registered. Please login instead."
      });
    }

    // Insert or update owner (unverified)
    if (existing.rows.length > 0) {
      await pool.query(
        "UPDATE owners SET full_name=$1, cnic=$2, city=$3, phone=$4 WHERE email=$5",
        [full_name, cnic, city, phone, email]
      );
    } else {
      await pool.query(
        "INSERT INTO owners (full_name, email, cnic, city, phone, is_verified) VALUES ($1,$2,$3,$4,$5,false)",
        [full_name, email, cnic, city, phone]
      );
    }

    // Generate OTP with 10 min expiry
    const otp = generateOtp();
    otpStore.set(email, { otp, expiresAt: Date.now() + 10 * 60 * 1000 });

    console.log(`📧 OTP for ${email}: ${otp}`); // debug log

    // Send email
    await transporter.sendMail({
      from: `"StayBuddy" <${process.env.EMAIL_USER}>`,
      to: email,
      subject: "Your StayBuddy OTP Code",
      html: `
        <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:32px;background:#f4f4f4;border-radius:12px;">
          <h2 style="color:#0B7C80;">StayBuddy Owner Verification</h2>
          <p style="color:#555;">Hello <strong>${full_name}</strong>,</p>
          <p style="color:#555;">Your one-time verification code is:</p>
          <div style="background:#0B7C80;color:white;font-size:40px;font-weight:bold;letter-spacing:16px;text-align:center;padding:24px;border-radius:10px;margin:24px 0;">
            ${otp}
          </div>
          <p style="color:#888;font-size:13px;">This code expires in <strong>10 minutes</strong>. Do not share it.</p>
          <p style="color:#bbb;font-size:12px;">StayBuddy — Your Intelligent Hostel Companion</p>
        </div>
      `,
    });

    res.json({ success: true, message: "OTP sent to your email" });
  } catch (err) {
    console.error("Error sending OTP:", err);
    if (err.code === "EAUTH") {
      return res.status(500).json({
        error: "Email authentication failed. Check EMAIL_USER and EMAIL_PASS in .env",
        hint: "Use Gmail App Password, not your normal Gmail password."
      });
    }
    res.status(500).json({ error: "Failed to send OTP: " + err.message });
  }
});

// POST /api/owner/verify-otp
app.post("/api/owner/verify-otp", async (req, res) => {
  try {
    const { email, otp } = req.body;

    if (!email || !otp) {
      return res.status(400).json({ error: "email and otp are required" });
    }

    const record = otpStore.get(email);

    if (!record) {
      return res.status(400).json({ error: "No OTP found. Please request a new one." });
    }

    if (Date.now() > record.expiresAt) {
      otpStore.delete(email);
      return res.status(400).json({ error: "OTP expired. Please request a new one." });
    }

    if (record.otp !== otp.toString().trim()) {
      return res.status(400).json({ error: "Incorrect OTP. Please try again." });
    }

    // Mark as verified
    otpStore.delete(email);
    const result = await pool.query(
      "UPDATE owners SET is_verified=true WHERE email=$1 RETURNING id, full_name, email, city, phone, created_at",
      [email]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "Owner not found." });
    }

    res.json({
      success: true,
      message: "Owner verified and registered ✅",
      owner: result.rows[0],
    });
  } catch (err) {
    console.error("Error verifying OTP:", err);
    res.status(500).json({ error: "Failed to verify OTP" });
  }
});

// POST /api/owner/login
app.post("/api/owner/login", async (req, res) => {
  try {
    const { email } = req.body;
    if (!email) return res.status(400).json({ error: "email is required" });

    const result = await pool.query(
      "SELECT id, full_name, email, city, is_verified FROM owners WHERE email=$1",
      [email]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "No owner account found with this email." });
    }
    if (!result.rows[0].is_verified) {
      return res.status(403).json({ error: "Account not verified. Please complete OTP verification." });
    }

    res.json({ success: true, owner: result.rows[0] });
  } catch (err) {
    res.status(500).json({ error: "Login failed" });
  }
});

// GET /api/owners (view all owners)
app.get("/api/owners", async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT id, full_name, email, cnic, city, phone, is_verified, created_at FROM owners ORDER BY created_at DESC"
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch owners" });
  }
});

// ══════════════════════════════════════════════════════════════════
//  EXISTING ROUTES
// ══════════════════════════════════════════════════════════════════

app.get("/api/hostels", async (req, res) => {
  try {
    const result = await pool.query(`SELECT ${hostelResponseColumns} FROM hostels ORDER BY id ASC`);
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch hostels" }); }
});

app.get("/api/hostels/:id", async (req, res, next) => {
  try {
    if (["search", "compare"].includes(req.params.id)) return next();
    const hostelId = parseInt(req.params.id, 10);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    const hostelQ = await pool.query(`SELECT ${hostelResponseColumns} FROM hostels WHERE id=$1`, [hostelId]);
    if (hostelQ.rows.length === 0) return res.status(404).json({ error: "Hostel not found" });
    const reviewsQ = await pool.query("SELECT id, user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review, created_at FROM reviews WHERE hostel_id=$1 ORDER BY created_at DESC", [hostelId]);
    res.json({ ...hostelQ.rows[0], reviews: reviewsQ.rows });
  } catch (err) { res.status(500).json({ error: "Failed to fetch hostel details" }); }
});

// GET /api/hostels/search/by-city
app.get("/api/hostels/search/by-city", async (req, res) => {
  try {
    const { city, min_price, max_price } = req.query;
    if (!city) return res.status(400).json({ error: "city query parameter is required" });

    const params = [`%${city}%`];
    let query = `SELECT ${hostelResponseColumns} FROM hostels WHERE city ILIKE $1`;
    for (const [price, operator] of [[min_price, ">="], [max_price, "<="]]) {
      if (price == null) continue;
      const parsed = Number(price);
      if (!Number.isFinite(parsed) || parsed < 0) {
        return res.status(400).json({ error: "min_price and max_price must be non-negative numbers" });
      }
      params.push(parsed);
      query += ` AND single_room_price ${operator} $${params.length}`;
    }
    query += " ORDER BY name ASC";
    const result = await pool.query(query, params);
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to search hostels by city: " + err.message }); }
});

// GET /api/hostels/search/by-university
app.get("/api/hostels/search/by-university", async (req, res) => {
  try {
    const { university } = req.query;
    if (!university) return res.status(400).json({ error: "university query parameter is required" });
    
    const result = await pool.query(
      `SELECT ${hostelResponseColumns} FROM hostels WHERE university ILIKE $1 ORDER BY name ASC`,
      [`%${university}%`]
    );
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to search hostels by university: " + err.message }); }
});

// GET /api/hostels/search/nearby (GPS proximity search)
app.get("/api/hostels/search/nearby", async (req, res) => {
  try {
    const { latitude, longitude, radius_km } = req.query;
    
    if (!latitude || !longitude) {
      return res.status(400).json({ error: "latitude and longitude query parameters are required" });
    }
    
    const lat = parseFloat(latitude);
    const lon = parseFloat(longitude);
    const radiusKm = parseFloat(radius_km) || 5;
    
    if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      return res.status(400).json({ error: "Invalid latitude or longitude values" });
    }
    
    const result = await pool.query(`
      SELECT * FROM (
        SELECT
          ${hostelResponseColumns},
          (6371 * acos(cos(radians($1)) * cos(radians(latitude)) *
          cos(radians(longitude) - radians($2)) +
          sin(radians($1)) * sin(radians(latitude)))) AS distance_km
        FROM hostels
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
      ) AS nearby_hostels
      WHERE distance_km <= $3
      ORDER BY distance_km ASC
    `, [lat, lon, radiusKm]);
    
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to search nearby hostels: " + err.message }); }
});

// GET /api/hostels/search/filter
app.get("/api/hostels/search/filter", async (req, res) => {
  try {
    const { city, university, hostel_type, verified, min_capacity, max_capacity, min_price, max_price, amenities, search_text } = req.query;

    let query = `SELECT ${hostelResponseColumns} FROM hostels WHERE 1=1`;
    let params = [];
    
    if (city) {
      query += ` AND city ILIKE $${params.length + 1}`;
      params.push(`%${city}%`);
    }
    if (university) {
      query += ` AND university ILIKE $${params.length + 1}`;
      params.push(`%${university}%`);
    }
    if (hostel_type) {
      query += ` AND hostel_type ILIKE $${params.length + 1}`;
      params.push(`%${hostel_type}%`);
    }
    if (verified != null) {
      if (!["true", "false"].includes(String(verified).toLowerCase())) {
        return res.status(400).json({ error: "verified must be true or false" });
      }
      query += ` AND verified=$${params.length + 1}`;
      params.push(String(verified).toLowerCase() === "true");
    }
    if (min_capacity) {
      const minCap = parseInt(min_capacity, 10);
      if (isValidInt(minCap)) {
        query += ` AND available_capacity >= $${params.length + 1}`;
        params.push(minCap);
      }
    }
    if (max_capacity) {
      const maxCap = parseInt(max_capacity, 10);
      if (isValidInt(maxCap)) {
        query += ` AND available_capacity <= $${params.length + 1}`;
        params.push(maxCap);
      }
    }
    for (const [price, operator] of [[min_price, ">="], [max_price, "<="]]) {
      if (price == null) continue;
      const parsed = Number(price);
      if (!Number.isFinite(parsed) || parsed < 0) {
        return res.status(400).json({ error: "min_price and max_price must be non-negative numbers" });
      }
      query += ` AND single_room_price ${operator} $${params.length + 1}`;
      params.push(parsed);
    }
    if (amenities) {
      const amenityList = String(amenities)
        .split(",")
        .map((amenity) => amenity.trim())
        .filter(Boolean);
      if (amenityList.length === 0) return res.status(400).json({ error: "amenities must contain one or more values" });
      query += ` AND amenities @> $${params.length + 1}::jsonb`;
      params.push(JSON.stringify(amenityList));
    }
    if (search_text) {
      query += ` AND (name ILIKE $${params.length + 1} OR address ILIKE $${params.length + 1} OR area ILIKE $${params.length + 1} OR description ILIKE $${params.length + 1})`;
      const searchPattern = `%${search_text}%`;
      params.push(searchPattern, searchPattern, searchPattern, searchPattern);
    }
    
    query += " ORDER BY name ASC";
    const result = await pool.query(query, params);
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to filter hostels: " + err.message }); }
});

// GET /api/hostels/compare?ids=1,2
app.get("/api/hostels/compare", async (req, res) => {
  try {
    const ids = String(req.query.ids || "")
      .split(",")
      .map((value) => parseInt(value.trim(), 10));

    if (ids.length < 2 || ids.some((id) => !isValidInt(id)) || new Set(ids).size !== ids.length) {
      return res.status(400).json({ error: "ids must contain at least two unique positive hostel IDs" });
    }

    const result = await pool.query(`
      SELECT ${hostelResponseColumnsWithAlias},
             ROUND(AVG(r.overall_rating)::numeric, 2) AS average_rating,
             COUNT(r.id)::integer AS review_count
      FROM hostels h
      LEFT JOIN reviews r ON r.hostel_id = h.id
      WHERE h.id = ANY($1::int[])
      GROUP BY h.id
      ORDER BY array_position($1::int[], h.id)
    `, [ids]);

    if (result.rows.length !== ids.length) {
      return res.status(404).json({ error: "One or more hostels were not found" });
    }

    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to compare hostels: " + err.message }); }
});

// ══════════════════════════════════════════════════════════════════
//  FAVORITES ROUTES
// ══════════════════════════════════════════════════════════════════

// POST /api/favorites (add to favorites)
app.post("/api/favorites", requireAuth, async (req, res) => {
  try {
    const { hostel_id } = req.body;
    const uid = req.user.id;
    const hid = parseInt(hostel_id, 10);
    
    if (!isValidInt(hid)) {
      return res.status(400).json({ error: "hostel_id must be a positive integer" });
    }
    
    const userResult = await pool.query("SELECT id FROM users WHERE id=$1", [uid]);
    if (userResult.rows.length === 0) return res.status(404).json({ error: "User not found" });
    
    const hostelResult = await pool.query("SELECT id FROM hostels WHERE id=$1", [hid]);
    if (hostelResult.rows.length === 0) return res.status(404).json({ error: "Hostel not found" });
    
    const result = await pool.query(
      "INSERT INTO favorites (user_id, hostel_id) VALUES ($1, $2) ON CONFLICT DO NOTHING RETURNING *",
      [uid, hid]
    );
    
    if (result.rows.length === 0) {
      return res.status(409).json({ error: "Hostel is already in favorites" });
    }
    
    res.status(201).json({ message: "Hostel added to favorites ✅", favorite: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to add favorite: " + err.message }); }
});

// DELETE /api/favorites/:user_id/:hostel_id (remove from favorites)
app.delete("/api/favorites/:user_id/:hostel_id", requireAuth, async (req, res) => {
  try {
    const uid = parseInt(req.params.user_id, 10);
    const hid = parseInt(req.params.hostel_id, 10);
    
    if (!isValidInt(uid) || !isValidInt(hid)) {
      return res.status(400).json({ error: "user_id and hostel_id must be positive integers" });
    }
    if (uid !== req.user.id) {
      return res.status(403).json({ error: "Cannot modify another user's favorites" });
    }
    
    const result = await pool.query(
      "DELETE FROM favorites WHERE user_id=$1 AND hostel_id=$2 RETURNING *",
      [uid, hid]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: "Favorite not found" });
    }
    
    res.json({ message: "Hostel removed from favorites ✅", favorite: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to remove favorite: " + err.message }); }
});

// GET /api/favorites/:user_id (get user's favorites)
app.get("/api/favorites/:user_id", requireAuth, async (req, res) => {
  try {
    const uid = parseInt(req.params.user_id, 10);
    if (!isValidInt(uid)) return res.status(400).json({ error: "Invalid user id" });
    if (uid !== req.user.id) {
      return res.status(403).json({ error: "Cannot view another user's favorites" });
    }
    
    const userResult = await pool.query("SELECT id FROM users WHERE id=$1", [uid]);
    if (userResult.rows.length === 0) return res.status(404).json({ error: "User not found" });
    
    const result = await pool.query(`
            SELECT f.id AS favorite_id, f.user_id, f.created_at,
              h.id AS hostel_id, ${hostelResponseColumnsWithAlias}
      FROM favorites f
      JOIN hostels h ON f.hostel_id = h.id
      WHERE f.user_id=$1
      ORDER BY f.created_at DESC
    `, [uid]);
    
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch favorites: " + err.message }); }
});

app.get("/api/users/:id", async (req, res) => {
  try {
    const userId = parseInt(req.params.id, 10);
    if (!isValidInt(userId)) return res.status(400).json({ error: "Invalid user id" });
    const userQ = await pool.query("SELECT id, name, email, role, created_at FROM users WHERE id=$1", [userId]);
    if (userQ.rows.length === 0) return res.status(404).json({ error: "User not found" });
    res.json(userQ.rows[0]);
  } catch (err) { res.status(500).json({ error: "Failed to fetch user profile" }); }
});

app.get("/api/interactions", async (req, res) => {
  try {
    const result = await pool.query("SELECT id, user_id, hostel_id, action_type, created_at FROM interactions ORDER BY created_at DESC");
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch interactions" }); }
});

app.post("/api/interactions", async (req, res) => {
  try {
    const { user_id, hostel_id, action_type } = req.body;
    const uid = parseInt(user_id, 10);
    const hid = hostel_id == null ? null : parseInt(hostel_id, 10);
    if (!isValidInt(uid)) return res.status(400).json({ error: "user_id must be a positive integer" });
    if (hid !== null && !isValidInt(hid)) return res.status(400).json({ error: "hostel_id must be a positive integer or null" });
    if (!action_type) return res.status(400).json({ error: "action_type is required" });
    const result = await pool.query("INSERT INTO interactions (user_id, hostel_id, action_type) VALUES ($1,$2,$3) RETURNING *", [uid, hid, action_type]);
    res.status(201).json({ message: "Interaction saved ✅", interaction: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to save interaction" }); }
});

app.get("/api/bookings", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;
    const result = await pool.query(
      `SELECT b.id, b.user_id, b.hostel_id, b.room_id, b.check_in, b.check_out, b.status, b.created_at,
              h.name AS hostel_name, h.city AS hostel_city, h.area AS hostel_area, h.single_room_price
       FROM bookings b
       JOIN hostels h ON h.id = b.hostel_id
       WHERE b.user_id=$1
       ORDER BY b.created_at DESC, b.id DESC`,
      [userId]
    );
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch bookings" }); }
});

app.post("/api/bookings", requireAuth, async (req, res) => {
  const client = await pool.connect();
  try {
    const { hostel_id, check_in, check_out, status } = req.body;
    const userId = req.user.id;
    const hostelId = parseInt(hostel_id, 10);
    const bookingStatus = normalizeBookingStatus(status);

    if (!isValidInt(hostelId)) {
      return res.status(400).json({ error: "hostel_id must be a positive integer" });
    }
    if (!check_in || !check_out || !isValidDateInput(check_in) || !isValidDateInput(check_out)) {
      return res.status(400).json({ error: "check_in and check_out must be valid dates" });
    }
    if (new Date(check_out) <= new Date(check_in)) {
      return res.status(400).json({ error: "check_out must be after check_in" });
    }
    if (!isBookingStatus(bookingStatus)) {
      return res.status(400).json({ error: "status must be one of pending, confirmed, cancelled, completed" });
    }

    await client.query("BEGIN");

    const hostelResult = await client.query(
      "SELECT id, name, available_capacity FROM hostels WHERE id=$1 FOR UPDATE",
      [hostelId]
    );
    if (hostelResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Hostel not found" });
    }

    const userResult = await client.query(
      "SELECT id FROM users WHERE id=$1",
      [userId]
    );
    if (userResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "User not found" });
    }

    const hostel = hostelResult.rows[0];
    if (isCapacityHoldingStatus(bookingStatus) && Number(hostel.available_capacity) <= 0) {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "No beds available for this hostel" });
    }

    const bookingResult = await client.query(
      "INSERT INTO bookings (user_id, hostel_id, check_in, check_out, status) VALUES ($1,$2,$3,$4,$5) RETURNING *",
      [userId, hostelId, check_in, check_out, bookingStatus]
    );

    if (isCapacityHoldingStatus(bookingStatus)) {
      await client.query(
        "UPDATE hostels SET available_capacity = available_capacity - 1 WHERE id=$1",
        [hostelId]
      );
    }

    await client.query("COMMIT");
    res.status(201).json({
      message: "Booking created ✅",
      booking: bookingResult.rows[0],
    });
  } catch (err) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to create booking: " + err.message });
  } finally {
    client.release();
  }
});

app.put("/api/bookings/:id", requireAuth, async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = parseInt(req.params.id, 10);
    if (!isValidInt(bookingId)) {
      return res.status(400).json({ error: "Invalid booking id" });
    }

    const { check_in, check_out, status } = req.body;
    const nextStatus = status == null ? null : normalizeBookingStatus(status);

    if (check_in != null && !isValidDateInput(check_in)) {
      return res.status(400).json({ error: "check_in must be a valid date" });
    }
    if (check_out != null && !isValidDateInput(check_out)) {
      return res.status(400).json({ error: "check_out must be a valid date" });
    }
    if (nextStatus != null && !isBookingStatus(nextStatus)) {
      return res.status(400).json({ error: "status must be one of pending, confirmed, cancelled, completed" });
    }

    await client.query("BEGIN");

    const bookingResult = await client.query(
      "SELECT id, user_id, hostel_id, room_id, check_in, check_out, status FROM bookings WHERE id=$1 FOR UPDATE",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Booking not found" });
    }

    const booking = bookingResult.rows[0];
    if (booking.user_id !== req.user.id) {
      await client.query("ROLLBACK");
      return res.status(403).json({ error: "Cannot update another user's booking" });
    }
    const updatedCheckIn = check_in || booking.check_in;
    const updatedCheckOut = check_out || booking.check_out;
    const updatedStatus = nextStatus || normalizeBookingStatus(booking.status);

    if (new Date(updatedCheckOut) <= new Date(updatedCheckIn)) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "check_out must be after check_in" });
    }

    const currentHoldsCapacity = isCapacityHoldingStatus(normalizeBookingStatus(booking.status));
    const nextHoldsCapacity = isCapacityHoldingStatus(updatedStatus);

    if (!currentHoldsCapacity && nextHoldsCapacity) {
      const hostelResult = await client.query(
        "SELECT id, available_capacity FROM hostels WHERE id=$1 FOR UPDATE",
        [booking.hostel_id]
      );
      if (hostelResult.rows.length === 0) {
        await client.query("ROLLBACK");
        return res.status(404).json({ error: "Hostel not found" });
      }
      if (Number(hostelResult.rows[0].available_capacity) <= 0) {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "No beds available for this hostel" });
      }
      await client.query(
        "UPDATE hostels SET available_capacity = available_capacity - 1 WHERE id=$1",
        [booking.hostel_id]
      );
      if (booking.room_id != null) {
        const roomResult = await client.query(
          "SELECT id, available_capacity FROM rooms WHERE id=$1 FOR UPDATE",
          [booking.room_id]
        );
        if (roomResult.rows.length === 0 || Number(roomResult.rows[0].available_capacity) <= 0) {
          await client.query("ROLLBACK");
          return res.status(409).json({ error: "Assigned room has no available capacity" });
        }
        await client.query(
          "UPDATE rooms SET available_capacity = available_capacity - 1 WHERE id=$1",
          [booking.room_id]
        );
      }
    } else if (currentHoldsCapacity && !nextHoldsCapacity) {
      if (!await releaseHostelCapacity(client, booking.hostel_id)) {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "Hostel availability is already fully restored" });
      }
      if (booking.room_id != null) {
        await client.query(
          "UPDATE rooms SET available_capacity = available_capacity + 1 WHERE id=$1",
          [booking.room_id]
        );
      }
    }

    const updatedResult = await client.query(
      "UPDATE bookings SET check_in=$1, check_out=$2, status=$3 WHERE id=$4 RETURNING *",
      [updatedCheckIn, updatedCheckOut, updatedStatus, bookingId]
    );

    await client.query("COMMIT");
    res.json({ message: "Booking updated ✅", booking: updatedResult.rows[0] });
  } catch (err) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to update booking: " + err.message });
  } finally {
    client.release();
  }
});

app.post("/api/bookings/:id/cancel", requireAuth, async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = parseInt(req.params.id, 10);
    if (!isValidInt(bookingId)) {
      return res.status(400).json({ error: "Invalid booking id" });
    }

    await client.query("BEGIN");

    const bookingResult = await client.query(
      "SELECT id, user_id, hostel_id, room_id, status FROM bookings WHERE id=$1 FOR UPDATE",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Booking not found" });
    }

    const booking = bookingResult.rows[0];
    if (booking.user_id !== req.user.id) {
      await client.query("ROLLBACK");
      return res.status(403).json({ error: "Cannot cancel another user's booking" });
    }
    const currentStatus = normalizeBookingStatus(booking.status);
    if (currentStatus === "cancelled") {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "Booking is already cancelled" });
    }

    if (isCapacityHoldingStatus(currentStatus)) {
      if (!await releaseHostelCapacity(client, booking.hostel_id)) {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "Hostel availability is already fully restored" });
      }
      if (booking.room_id != null) {
        await client.query(
          "UPDATE rooms SET available_capacity = available_capacity + 1 WHERE id=$1",
          [booking.room_id]
        );
      }
    }

    const updatedResult = await client.query(
      "UPDATE bookings SET status='cancelled' WHERE id=$1 RETURNING *",
      [bookingId]
    );

    await client.query("COMMIT");
    res.json({ message: "Booking cancelled ✅", booking: updatedResult.rows[0] });
  } catch (err) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to cancel booking: " + err.message });
  } finally {
    client.release();
  }
});

app.get("/api/reviews", async (req, res) => {
  try {
    const result = await pool.query("SELECT id, user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review, created_at FROM reviews ORDER BY id ASC");
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch reviews" }); }
});

app.post("/api/reviews", requireAuth, async (req, res) => {
  try {
    const { hostel_id, overall_rating, cleanliness, facilities, management, text_review } = req.body;
    const uid = req.user.id; const hid = parseInt(hostel_id, 10);
    if (!isValidInt(hid)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    const result = await pool.query("INSERT INTO reviews (user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *", [uid, hid, overall_rating, cleanliness, facilities, management, text_review || null]);
    res.status(201).json({ message: "Review saved ✅", review: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to save review" }); }
});

app.get("/api/complaints", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;
    const result = await pool.query(
      `SELECT id, user_id, hostel_id, category, severity, status, assigned_to, description, created_at, resolved_at
       FROM complaints
       WHERE user_id=$1
       ORDER BY id ASC`,
      [userId]
    );
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch complaints" }); }
});

app.post("/api/complaints", requireAuth, async (req, res) => {
  try {
    const { hostel_id, category, severity, status, assigned_to, description } = req.body;
    const uid = req.user.id; const hid = parseInt(hostel_id, 10);
    const assigned = assigned_to == null ? null : parseInt(assigned_to, 10);
    if (!isValidInt(hid)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    if (!description) return res.status(400).json({ error: "description is required" });
    
    let finalCategory = category;
    let finalSeverity = severity;
    let aiResult = null;
    
    if (!category || !severity) {
      aiResult = await callAiComplaintCategorizer(description);
      if (aiResult && aiResult.success) {
        finalCategory = finalCategory || aiResult.category || "Other";
        finalSeverity = finalSeverity || priorityToSeverity(aiResult.priority || "Medium");
      } else {
        finalCategory = finalCategory || "Other";
        finalSeverity = finalSeverity || "medium";
      }
    }
    
    const result = await pool.query(
      "INSERT INTO complaints (user_id, hostel_id, category, severity, status, assigned_to, description) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *",
      [uid, hid, finalCategory, finalSeverity, status || "open", assigned, description]
    );
    
    res.status(201).json({
      message: "Complaint saved ✅",
      complaint: result.rows[0],
      ai_analysis: aiResult ? {
        category: aiResult.category,
        priority: aiResult.priority,
        confidence: aiResult.confidence,
        suggestion: aiResult.suggestion
      } : null
    });
  } catch (err) { res.status(500).json({ error: "Failed to save complaint: " + err.message }); }
});

app.patch("/api/complaints/:id", async (req, res) => {
  try {
    const complaintId = parseInt(req.params.id, 10);
    if (!isValidInt(complaintId)) return res.status(400).json({ error: "Invalid complaint id" });
    
    const { status, assigned_to, resolved_at } = req.body;
    
    if (status != null && !isValidComplaintStatus(status)) {
      return res.status(400).json({ error: "status must be one of open, in_progress, resolved, closed" });
    }
    
    const complaintResult = await pool.query(
      "SELECT id, status FROM complaints WHERE id=$1",
      [complaintId]
    );
    if (complaintResult.rows.length === 0) {
      return res.status(404).json({ error: "Complaint not found" });
    }
    
    let updateQuery = "UPDATE complaints SET ";
    const params = [];
    const updates = [];
    let paramCount = 1;
    
    if (status != null) {
      updates.push(`status=$${paramCount}`);
      params.push(status);
      paramCount++;
    }
    if (assigned_to != null) {
      const assignedId = parseInt(assigned_to, 10);
      if (!isValidInt(assignedId)) return res.status(400).json({ error: "assigned_to must be a positive integer" });
      updates.push(`assigned_to=$${paramCount}`);
      params.push(assignedId);
      paramCount++;
    }
    if (resolved_at != null) {
      updates.push(`resolved_at=$${paramCount}`);
      params.push(resolved_at);
      paramCount++;
    }
    
    if (updates.length === 0) {
      return res.status(400).json({ error: "No fields to update" });
    }
    
    updateQuery += updates.join(", ") + ` WHERE id=$${paramCount} RETURNING *`;
    params.push(complaintId);
    
    const result = await pool.query(updateQuery, params);
    res.json({ message: "Complaint updated ✅", complaint: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to update complaint: " + err.message }); }
});

app.post("/api/recommendations", async (req, res) => {
  try {
    const { student_id, top_k } = req.body;
    const sid = parseInt(student_id, 10);
    if (!isValidInt(sid)) return res.status(400).json({ error: "student_id must be a positive integer" });
    const k = Number.isInteger(top_k) && top_k > 0 ? top_k : 5;
    const hostelsQ = await pool.query("SELECT id, name, city, available_capacity, total_capacity, description FROM hostels ORDER BY id ASC");
    const ratingsQ = await pool.query("SELECT hostel_id, AVG(overall_rating)::float AS avg_rating, COUNT(*)::int AS n_reviews FROM reviews GROUP BY hostel_id");
    const interQ = await pool.query("SELECT user_id, hostel_id, action_type, created_at FROM interactions WHERE user_id=$1 ORDER BY created_at DESC", [sid]);
    const hostels = hostelsQ.rows;
    const hostelMap = new Map(hostels.map(h => [h.id, h]));
    const ratingMap = new Map(); for (const r of ratingsQ.rows) ratingMap.set(r.hostel_id, r);
    const userInteractions = interQ.rows;
    const userPref = buildUserPreferenceFromHistory(userInteractions, hostelMap);
    const ratingNormArr = softNormalize(hostels.map(h => ratingMap.get(h.id)?.avg_rating ?? 0));
    const availNormArr = softNormalize(hostels.map(h => h.available_capacity ?? 0));
    const affinityByHostel = new Map();
    for (const it of userInteractions) { const w = actionWeight(it.action_type); affinityByHostel.set(it.hostel_id, (affinityByHostel.get(it.hostel_id) || 0) + w); }
    const affinityNormArr = softNormalize(hostels.map(h => affinityByHostel.get(h.id) || 0));
    const W = { userAffinity: 0.45, rating: 0.30, availability: 0.15, city: 0.10 };
    const scored = hostels.map((h, idx) => {
      const cityMatch = userPref.topCity && h.city === userPref.topCity ? clamp01(userPref.cityStrength) : 0;
      const score = W.userAffinity * affinityNormArr[idx] + W.rating * ratingNormArr[idx] + W.availability * availNormArr[idx] + W.city * cityMatch;
      return { hostel: { id: h.id, name: h.name, city: h.city, available_capacity: h.available_capacity, avg_rating: ratingMap.get(h.id)?.avg_rating ?? null, n_reviews: ratingMap.get(h.id)?.n_reviews ?? 0 }, score: Number(score.toFixed(4)), signals: { userAffinity: Number(affinityNormArr[idx].toFixed(4)), ratingNorm: Number(ratingNormArr[idx].toFixed(4)), availNorm: Number(availNormArr[idx].toFixed(4)), cityMatch: Number(cityMatch.toFixed(4)) } };
    });
    scored.sort((a, b) => b.score - a.score);
    const validation = validateAndFallback(scored, { hasEnoughSignals: userInteractions.length >= 3 });
    let finalList = scored;
    if (validation.usedFallback) { finalList = scored.slice().sort((a, b) => (b.hostel.avg_rating ?? 0) - (a.hostel.avg_rating ?? 0) || (b.hostel.available_capacity ?? 0) - (a.hostel.available_capacity ?? 0)); }
    res.json({ student_id: sid, model: validation.usedFallback ? "hybrid_v1_fallback" : "hybrid_v1", personalization: { inferred_top_city: userPref.topCity, confidence_interactions: userInteractions.length }, validation, recommendations: finalList.slice(0, k).map((rec, i) => ({ rank: i + 1, ...rec.hostel, score: rec.score, why: explain(rec, userPref), debug_signals: rec.signals })) });
  } catch (err) { res.status(500).json({ error: "Failed to generate recommendations" }); }
});

app.post("/api/recommendations/ml", async (req, res) => {
  try {
    const requestedTopK = req.body?.top_k == null ? 5 : req.body.top_k;
    if (!Number.isInteger(requestedTopK) || requestedTopK < 1 || requestedTopK > 10) {
      return res.status(400).json({ error: "top_k must be an integer between 1 and 10" });
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    let aiResponse;
    try {
      aiResponse = await fetch(process.env.AI_RECOMMENDER_URL || "http://127.0.0.1:8000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...req.body, top_k: Math.min(requestedTopK * 3, 20) }),
        signal: controller.signal
      });
    } finally {
      clearTimeout(timeout);
    }

    if (!aiResponse.ok) {
      return res.status(502).json({ error: "Recommendation service returned an error" });
    }

    const aiResult = await aiResponse.json();
    if (!aiResult.success || !Array.isArray(aiResult.recommendations)) {
      return res.status(502).json({ error: "Recommendation service returned an invalid response" });
    }

    const externalIds = aiResult.recommendations
      .map((recommendation) => recommendation.hostel_id)
      .filter((externalId) => typeof externalId === "string" && externalId.length > 0);

    if (externalIds.length === 0) {
      return res.json({
        model: "ml_hybrid_catalog_v1",
        student_type: aiResult.student_type,
        alpha_used: aiResult.alpha_used,
        count: 0,
        unavailable_count: 0,
        recommendations: []
      });
    }

    const catalogResult = await pool.query(
      `SELECT ${hostelResponseColumnsWithAlias}
       FROM hostels h
       WHERE h.external_id = ANY($1::varchar[])
         AND h.available_capacity > 0
       ORDER BY array_position($1::varchar[], h.external_id)`,
      [externalIds]
    );
    const catalogByExternalId = new Map(catalogResult.rows.map((hostel) => [hostel.external_id, hostel]));
    const recommendations = aiResult.recommendations
      .filter((recommendation) => catalogByExternalId.has(recommendation.hostel_id))
      .slice(0, requestedTopK)
      .map((recommendation, index) => ({
        rank: index + 1,
        ...catalogByExternalId.get(recommendation.hostel_id),
        model_score: recommendation.hybrid_score,
        content_score: recommendation.cb_score,
        collaborative_score: recommendation.cf_score,
        student_type: recommendation.student_type,
        alpha_used: recommendation.alpha_used
      }));

    res.json({
      model: "ml_hybrid_catalog_v1",
      student_type: aiResult.student_type,
      alpha_used: aiResult.alpha_used,
      count: recommendations.length,
      unavailable_count: externalIds.length - catalogResult.rows.length,
      recommendations
    });
  } catch (err) {
    const error = err.name === "AbortError" ? "Recommendation service timed out" : "Recommendation service is unavailable";
    res.status(503).json({ error });
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    const { message, student_id } = req.body;
    if (!message) return res.status(400).json({ error: "message is required" });
    const sid = student_id ? parseInt(student_id, 10) : null;
    const hostelsQ = await pool.query("SELECT id, name, city, available_capacity FROM hostels ORDER BY available_capacity DESC LIMIT 3");
    res.json({ student_id: isValidInt(sid) ? sid : null, reply: "I can help you with hostel availability, reviews, and complaints. Here are some hostels with availability right now:", suggestions: hostelsQ.rows });
  } catch (err) { res.status(500).json({ error: "Chat service failed" }); }
});


// ══════════════════════════════════════════════════════════════════
//  AUTH ROUTES  (warden / student login)
// ══════════════════════════════════════════════════════════════════

// POST /api/auth/login
// Body: { email, password }
// Works for any role in users table (student, warden, admin)
app.post("/api/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ error: "email and password are required" });

    const result = await pool.query(
      "SELECT id, name, email, role, phone, password_hash FROM public.users WHERE email=$1",
      [email]
    );

    if (result.rows.length === 0)
      return res.status(401).json({ error: "Invalid email or password" });

    const user = result.rows[0];
    if (!await verifyPassword(password, user.password_hash)) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    const token = signAuthToken(user);

    res.json({
      success: true,
      token,
      user: {
        id:    user.id,
        name:  user.name,
        email: user.email,
        role:  user.role,
        phone: user.phone,
      }
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Login failed: " + err.message });
  }
});

// POST /api/auth/register
app.post("/api/auth/register", async (req, res) => {
  try {
    const { name, email, password, role = "student" } = req.body;
    if (!name || !email || !password)
      return res.status(400).json({ error: "name, email and password are required" });
    if (role !== "student") {
      return res.status(403).json({ error: "Public registration is available for students only" });
    }
    if (!isPassword(password)) {
      return res.status(400).json({ error: "password must be at least 8 characters long" });
    }

    const exists = await pool.query("SELECT id FROM public.users WHERE email=$1", [email]);
    if (exists.rows.length > 0)
      return res.status(409).json({ error: "Email already registered." });

    const passwordHash = await hashPassword(password);
    const result = await pool.query(
      "INSERT INTO public.users (name, email, password_hash, role) VALUES ($1,$2,$3,$4) RETURNING id, name, email, role",
      [name, email, passwordHash, role]
    );
    const user = result.rows[0];
    const token = signAuthToken(user);
    res.json({ success: true, token, user });
  } catch (err) {
    res.status(500).json({ error: "Registration failed: " + err.message });
  }
});

// POST /api/auth/password-reset/request
// Body: { email }. Always returns the same message to avoid account enumeration.
app.post("/api/auth/password-reset/request", async (req, res) => {
  try {
    const email = String(req.body.email || "").trim().toLowerCase();
    const genericResponse = { success: true, message: "If an account exists, a password reset link has been sent." };
    if (!email) return res.status(400).json({ error: "email is required" });

    await passwordResetTableReady;
    const userResult = await pool.query(
      "SELECT id, name, email FROM users WHERE LOWER(email)=LOWER($1)",
      [email]
    );
    if (userResult.rowCount === 0) return res.json(genericResponse);

    const user = userResult.rows[0];
    const token = crypto.randomBytes(32).toString("base64url");
    const tokenHash = hashPasswordResetToken(token);
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      await client.query(
        "UPDATE password_reset_tokens SET used_at=NOW() WHERE user_id=$1 AND used_at IS NULL",
        [user.id]
      );
      await client.query(
        "INSERT INTO password_reset_tokens (user_id, token_hash, expires_at) VALUES ($1,$2,NOW() + ($3 * INTERVAL '1 minute'))",
        [user.id, tokenHash, PASSWORD_RESET_TTL_MINUTES]
      );
      await client.query("COMMIT");
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }

    if (process.env.TEST_PASSWORD_RESET_TOKENS === "1") {
      return res.json({ ...genericResponse, test_reset_token: token });
    }

    try {
      await sendPasswordResetEmail(user, token);
      return res.json(genericResponse);
    } catch (error) {
      await pool.query("DELETE FROM password_reset_tokens WHERE token_hash=$1", [tokenHash]);
      console.error("Password reset email failed:", error.message);
      return res.status(500).json({ error: "Unable to send password reset email. Please try again later." });
    }
  } catch (error) {
    console.error("Password reset request failed:", error);
    res.status(500).json({ error: "Unable to request a password reset" });
  }
});

// POST /api/auth/password-reset/confirm
// Body: { token, password }
app.post("/api/auth/password-reset/confirm", async (req, res) => {
  const { token, password } = req.body;
  if (!token || !isPassword(password)) {
    return res.status(400).json({ error: "A valid reset token and password of at least 8 characters are required" });
  }

  const client = await pool.connect();
  try {
    await passwordResetTableReady;
    const tokenHash = hashPasswordResetToken(token);
    await client.query("BEGIN");
    const tokenResult = await client.query(
      `SELECT id, user_id FROM password_reset_tokens
       WHERE token_hash=$1 AND used_at IS NULL AND expires_at > NOW()
       FOR UPDATE`,
      [tokenHash]
    );
    if (tokenResult.rowCount === 0) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "This password reset token is invalid or expired" });
    }

    const passwordHash = await hashPassword(password);
    const userId = tokenResult.rows[0].user_id;
    await client.query("UPDATE users SET password_hash=$1 WHERE id=$2", [passwordHash, userId]);
    await client.query("UPDATE password_reset_tokens SET used_at=NOW() WHERE user_id=$1 AND used_at IS NULL", [userId]);
    await client.query("COMMIT");
    res.json({ success: true, message: "Password reset successfully. Please sign in with your new password." });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Password reset confirmation failed:", error);
    res.status(500).json({ error: "Unable to reset password" });
  } finally {
    client.release();
  }
});

// GET /api/auth/me  (requires token in Authorization header)
app.get("/api/auth/me", requireAuth, async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT id, name, email, role, phone FROM public.users WHERE id=$1", [req.user.id]);
    if (result.rows.length === 0)
      return res.status(404).json({ error: "User not found" });

    res.json({ success: true, user: result.rows[0] });
  } catch (err) {
    res.status(500).json({ error: "Auth check failed" });
  }
});

app.get("/api/student/profile", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT u.id, u.name, u.email, u.phone, p.university, p.department, p.gender,
              p.budget_max, p.max_distance_km, p.study_preference, p.food_preference
       FROM users u LEFT JOIN student_profiles p ON p.user_id=u.id WHERE u.id=$1`,
      [req.user.id]
    );
    res.json(result.rows[0]);
  } catch (error) { res.status(500).json({ error: "Failed to fetch student profile" }); }
});

app.put("/api/student/profile", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    const { name, phone, university, department, gender, budget_max, max_distance_km, study_preference, food_preference } = req.body;
    for (const value of [budget_max, max_distance_km, study_preference]) {
      if (value != null && (!Number.isFinite(Number(value)) || Number(value) < 0)) {
        return res.status(400).json({ error: "Numeric profile preferences must be non-negative" });
      }
    }
    if (study_preference != null && Number(study_preference) > 1) {
      return res.status(400).json({ error: "study_preference must be between 0 and 1" });
    }
    await pool.query("UPDATE users SET name=COALESCE($1,name), phone=COALESCE($2,phone) WHERE id=$3", [name || null, phone || null, req.user.id]);
    const result = await pool.query(
      `INSERT INTO student_profiles (user_id, university, department, gender, budget_max, max_distance_km, study_preference, food_preference)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
       ON CONFLICT (user_id) DO UPDATE SET university=COALESCE(EXCLUDED.university, student_profiles.university),
         department=COALESCE(EXCLUDED.department, student_profiles.department), gender=COALESCE(EXCLUDED.gender, student_profiles.gender),
         budget_max=COALESCE(EXCLUDED.budget_max, student_profiles.budget_max), max_distance_km=COALESCE(EXCLUDED.max_distance_km, student_profiles.max_distance_km),
         study_preference=COALESCE(EXCLUDED.study_preference, student_profiles.study_preference), food_preference=COALESCE(EXCLUDED.food_preference, student_profiles.food_preference), updated_at=NOW()
       RETURNING *`,
      [req.user.id, university || null, department || null, gender || null, budget_max ?? null, max_distance_km ?? null, study_preference ?? null, food_preference || null]
    );
    res.json({ success: true, profile: result.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to update student profile" }); }
});

app.get("/api/notification-preferences", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      "INSERT INTO notification_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO UPDATE SET user_id=EXCLUDED.user_id RETURNING *", [req.user.id]
    );
    res.json(result.rows[0]);
  } catch (error) { res.status(500).json({ error: "Failed to fetch notification preferences" }); }
});

app.put("/api/notification-preferences", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const { booking_updates, complaint_updates, announcements, email_enabled } = req.body;
    if ([booking_updates, complaint_updates, announcements, email_enabled].some((value) => value != null && typeof value !== "boolean")) {
      return res.status(400).json({ error: "Notification preferences must be boolean values" });
    }
    const result = await pool.query(
      `INSERT INTO notification_preferences (user_id, booking_updates, complaint_updates, announcements, email_enabled)
       VALUES ($1,COALESCE($2,true),COALESCE($3,true),COALESCE($4,true),COALESCE($5,true))
       ON CONFLICT (user_id) DO UPDATE SET booking_updates=COALESCE($2,notification_preferences.booking_updates), complaint_updates=COALESCE($3,notification_preferences.complaint_updates), announcements=COALESCE($4,notification_preferences.announcements), email_enabled=COALESCE($5,notification_preferences.email_enabled), updated_at=NOW()
       RETURNING *`, [req.user.id, booking_updates ?? null, complaint_updates ?? null, announcements ?? null, email_enabled ?? null]
    );
    res.json(result.rows[0]);
  } catch (error) { res.status(500).json({ error: "Failed to update notification preferences" }); }
});

app.get("/api/owner/hostels", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(`SELECT ${hostelResponseColumns} FROM hostels h JOIN hostel_owners ho ON ho.hostel_id=h.id WHERE ho.owner_user_id=$1 ORDER BY h.id`, [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch owner hostels" }); }
});

app.post("/api/owner/hostels", requireAuth, requireRole("owner"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const { name, address, city, university, total_capacity, description } = req.body;
    const totalCapacity = Number(total_capacity);
    if (!name || !address || !city || !Number.isInteger(totalCapacity) || totalCapacity <= 0) return res.status(400).json({ error: "name, address, city, and a positive total_capacity are required" });
    await client.query("BEGIN");
    const hostel = await client.query(
      "INSERT INTO hostels (name,address,city,university,total_capacity,available_capacity,source_available_capacity,description,source) VALUES ($1,$2,$3,$4,$5,$5,$5,$6,'owner') RETURNING *",
      [name, address, city, university || null, totalCapacity, description || null]
    );
    await client.query("INSERT INTO hostel_owners (owner_user_id,hostel_id) VALUES ($1,$2)", [req.user.id, hostel.rows[0].id]);
    await client.query("COMMIT");
    res.status(201).json({ hostel: hostel.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to register hostel" }); } finally { client.release(); }
});

app.get("/api/owner/bookings", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(`SELECT b.*, u.name AS student_name, u.email AS student_email, h.name AS hostel_name FROM bookings b JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id JOIN users u ON u.id=b.user_id JOIN hostels h ON h.id=b.hostel_id WHERE ho.owner_user_id=$1 ORDER BY b.created_at DESC`, [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch owner bookings" }); }
});

app.patch("/api/owner/bookings/:id", requireAuth, requireRole("owner"), async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = Number(req.params.id); const nextStatus = normalizeBookingStatus(req.body.status);
    if (!isValidInt(bookingId) || !isBookingStatus(nextStatus)) return res.status(400).json({ error: "A valid booking id and status are required" });
    await client.query("BEGIN");
    const booking = await client.query("SELECT b.* FROM bookings b JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id WHERE b.id=$1 AND ho.owner_user_id=$2 FOR UPDATE", [bookingId, req.user.id]);
    if (booking.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    const current = normalizeBookingStatus(booking.rows[0].status);
    if (isCapacityHoldingStatus(current) && !isCapacityHoldingStatus(nextStatus)) await releaseHostelCapacity(client, booking.rows[0].hostel_id);
    if (!isCapacityHoldingStatus(current) && isCapacityHoldingStatus(nextStatus)) { const hostel = await client.query("SELECT available_capacity FROM hostels WHERE id=$1 FOR UPDATE", [booking.rows[0].hostel_id]); if (Number(hostel.rows[0].available_capacity) <= 0) { await client.query("ROLLBACK"); return res.status(409).json({ error: "No beds available" }); } await client.query("UPDATE hostels SET available_capacity=available_capacity-1 WHERE id=$1", [booking.rows[0].hostel_id]); }
    const updated = await client.query("UPDATE bookings SET status=$1 WHERE id=$2 RETURNING *", [nextStatus, bookingId]);
    await client.query("COMMIT"); res.json({ booking: updated.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); console.error("Warden booking update failed:", error.message); res.status(500).json({ error: "Failed to update booking" }); } finally { client.release(); }
});

app.patch("/api/owner/hostels/:id", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const hostelId = Number(req.params.id);
    if (!isValidInt(hostelId) || !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const allowed = ["name", "address", "city", "university", "description", "area", "hostel_type", "single_room_price", "double_room_price", "dorm_room_price", "amenities"];
    const entries = Object.entries(req.body).filter(([key, value]) => allowed.includes(key) && value != null);
    if (entries.length === 0) return res.status(400).json({ error: "No editable hostel fields supplied" });
    if (entries.some(([key, value]) => key.endsWith("_price") && (!Number.isFinite(Number(value)) || Number(value) < 0))) return res.status(400).json({ error: "Prices must be non-negative" });
    const values = entries.map(([, value]) => value);
    const columns = entries.map(([key], index) => `${key}=$${index + 1}`);
    const result = await pool.query(`UPDATE hostels SET ${columns.join(", ")} WHERE id=$${values.length + 1} RETURNING *`, [...values, hostelId]);
    res.json({ hostel: result.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to update hostel profile" }); }
});

app.get("/api/owner/complaints", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(`SELECT c.*, u.name AS student_name, h.name AS hostel_name FROM complaints c JOIN hostel_owners ho ON ho.hostel_id=c.hostel_id JOIN users u ON u.id=c.user_id JOIN hostels h ON h.id=c.hostel_id WHERE ho.owner_user_id=$1 ORDER BY c.created_at DESC`, [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch owner complaints" }); }
});

app.patch("/api/owner/complaints/:id", requireAuth, requireRole("owner"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const complaintId = Number(req.params.id); const status = req.body.status;
    if (!isValidInt(complaintId) || !isValidComplaintStatus(status)) return res.status(400).json({ error: "A valid complaint id and status are required" });
    await client.query("BEGIN");
    const complaint = await client.query("SELECT c.* FROM complaints c JOIN hostel_owners ho ON ho.hostel_id=c.hostel_id WHERE c.id=$1 AND ho.owner_user_id=$2 FOR UPDATE", [complaintId, req.user.id]);
    if (complaint.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Complaint not found" }); }
    const updated = await client.query("UPDATE complaints SET status=$1, resolved_at=CASE WHEN $1 IN ('resolved','closed') THEN NOW() ELSE NULL END WHERE id=$2 RETURNING *", [status, complaintId]);
    await client.query("INSERT INTO complaint_status_history (complaint_id, previous_status, next_status, changed_by_user_id) VALUES ($1,$2,$3,$4)", [complaintId, complaint.rows[0].status, status, req.user.id]);
    await client.query("COMMIT"); res.json({ complaint: updated.rows[0] });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Warden complaint update failed:", error.message);
    res.status(500).json({
      error: "Failed to update complaint",
      ...(process.env.RUN_DB_TESTS === "1" ? { detail: error.message } : {}),
    });
  } finally { client.release(); }
});

app.get("/api/owner/dashboard", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(`SELECT h.id, h.name, h.total_capacity, h.available_capacity,
      COUNT(b.id) FILTER (WHERE b.status='pending')::int AS pending_bookings,
      COUNT(b.id) FILTER (WHERE b.status='confirmed')::int AS confirmed_bookings,
      COUNT(b.id) FILTER (WHERE b.status='cancelled')::int AS cancelled_bookings
      FROM hostels h JOIN hostel_owners ho ON ho.hostel_id=h.id LEFT JOIN bookings b ON b.hostel_id=h.id
      WHERE ho.owner_user_id=$1 GROUP BY h.id ORDER BY h.id`, [req.user.id]);
    res.json({ hostels: result.rows });
  } catch (error) { res.status(500).json({ error: "Failed to fetch owner dashboard" }); }
});

app.get("/api/owner/wardens", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(`SELECT wa.hostel_id, wa.assigned_at, u.id, u.name, u.email, h.name AS hostel_name FROM warden_assignments wa JOIN hostel_owners ho ON ho.hostel_id=wa.hostel_id JOIN users u ON u.id=wa.warden_user_id JOIN hostels h ON h.id=wa.hostel_id WHERE ho.owner_user_id=$1 ORDER BY wa.assigned_at DESC`, [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch wardens" }); }
});

app.post("/api/owner/wardens", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const hostelId = Number(req.body.hostel_id); const email = String(req.body.email || "").trim().toLowerCase();
    if (!isValidInt(hostelId) || !email) return res.status(400).json({ error: "hostel_id and warden email are required" });
    if (!await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const warden = await pool.query("SELECT id, name, email FROM users WHERE LOWER(email)=LOWER($1) AND role='warden'", [email]);
    if (warden.rowCount === 0) return res.status(404).json({ error: "A warden account with this email was not found" });
    const assignment = await pool.query("INSERT INTO warden_assignments (warden_user_id,hostel_id,assigned_by_user_id) VALUES ($1,$2,$3) ON CONFLICT (warden_user_id,hostel_id) DO UPDATE SET assigned_by_user_id=EXCLUDED.assigned_by_user_id, assigned_at=NOW() RETURNING *", [warden.rows[0].id, hostelId, req.user.id]);
    res.status(201).json({ assignment: assignment.rows[0], warden: warden.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to assign warden" }); }
});

// GET /api/warden/dashboard
app.get("/api/warden/dashboard", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT h.id, h.name, h.city, h.total_capacity, h.available_capacity,
        COUNT(DISTINCT b.id)::int AS total_bookings,
        COUNT(DISTINCT c.id) FILTER (WHERE c.status NOT IN ('resolved','closed'))::int AS pending_complaints
       FROM warden_assignments wa JOIN hostels h ON h.id=wa.hostel_id
       LEFT JOIN bookings b ON b.hostel_id=h.id LEFT JOIN complaints c ON c.hostel_id=h.id
       WHERE wa.warden_user_id=$1 GROUP BY h.id ORDER BY h.id`, [req.user.id]
    );
    res.json({ success: true, hostels: result.rows });
  } catch (err) {
    res.status(500).json({ error: "Dashboard fetch failed: " + err.message });
  }
});

app.get("/api/warden/bookings", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT b.*, u.name AS student_name, u.email AS student_email, h.name AS hostel_name
       FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id
       JOIN users u ON u.id=b.user_id JOIN hostels h ON h.id=b.hostel_id
       WHERE wa.warden_user_id=$1 ORDER BY b.created_at DESC`, [req.user.id]
    );
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch warden bookings" }); }
});

app.patch("/api/warden/bookings/:id", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const bookingId = Number(req.params.id); const nextStatus = normalizeBookingStatus(req.body.status);
    if (!isValidInt(bookingId) || !isBookingStatus(nextStatus)) return res.status(400).json({ error: "A valid booking id and status are required" });
    await client.query("BEGIN");
    const booking = await client.query(
      "SELECT b.* FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id WHERE b.id=$1 AND wa.warden_user_id=$2 FOR UPDATE",
      [bookingId, req.user.id]
    );
    if (booking.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    const currentStatus = normalizeBookingStatus(booking.rows[0].status);
    if (isCapacityHoldingStatus(currentStatus) && !isCapacityHoldingStatus(nextStatus)) await releaseHostelCapacity(client, booking.rows[0].hostel_id);
    if (!isCapacityHoldingStatus(currentStatus) && isCapacityHoldingStatus(nextStatus)) {
      const hostel = await client.query("SELECT available_capacity FROM hostels WHERE id=$1 FOR UPDATE", [booking.rows[0].hostel_id]);
      if (Number(hostel.rows[0].available_capacity) <= 0) { await client.query("ROLLBACK"); return res.status(409).json({ error: "No beds available" }); }
      await client.query("UPDATE hostels SET available_capacity=available_capacity-1 WHERE id=$1", [booking.rows[0].hostel_id]);
    }
    const updated = await client.query("UPDATE bookings SET status=$1 WHERE id=$2 RETURNING *", [nextStatus, bookingId]);
    await createNotification(client, booking.rows[0].user_id, "booking_status", "Booking status updated", `Your booking is now ${nextStatus}.`, { booking_id: bookingId, status: nextStatus }, "booking_updates");
    await client.query("COMMIT");
    res.json({ booking: updated.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to update booking" }); } finally { client.release(); }
});

app.get("/api/warden/complaints", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT c.*, u.name AS student_name, u.email AS student_email, h.name AS hostel_name
       FROM complaints c JOIN warden_assignments wa ON wa.hostel_id=c.hostel_id
       JOIN users u ON u.id=c.user_id JOIN hostels h ON h.id=c.hostel_id
       WHERE wa.warden_user_id=$1 ORDER BY c.created_at DESC`, [req.user.id]
    );
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch warden complaints" }); }
});

app.patch("/api/warden/complaints/:id", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const complaintId = Number(req.params.id); const status = req.body.status;
    if (!isValidInt(complaintId) || !isValidComplaintStatus(status)) return res.status(400).json({ error: "A valid complaint id and status are required" });
    await client.query("BEGIN");
    const complaint = await client.query(
      "SELECT c.* FROM complaints c JOIN warden_assignments wa ON wa.hostel_id=c.hostel_id WHERE c.id=$1 AND wa.warden_user_id=$2 FOR UPDATE",
      [complaintId, req.user.id]
    );
    if (complaint.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Complaint not found" }); }
    const updated = await client.query("UPDATE complaints SET status=$1::varchar, assigned_to=$2, resolved_at=CASE WHEN $1::varchar IN ('resolved','closed') THEN NOW() ELSE NULL END WHERE id=$3 RETURNING *", [status, req.user.id, complaintId]);
    await client.query("INSERT INTO complaint_status_history (complaint_id, previous_status, next_status, changed_by_user_id) VALUES ($1,$2,$3,$4)", [complaintId, complaint.rows[0].status, status, req.user.id]);
    await createNotification(client, complaint.rows[0].user_id, "complaint_status", "Complaint status updated", `Your complaint is now ${status}.`, { complaint_id: complaintId, status }, "complaint_updates");
    await client.query("COMMIT");
    res.json({ complaint: updated.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); console.error("Warden complaint update failed:", error.message); res.status(500).json({ error: "Failed to update complaint" }); } finally { client.release(); }
});

async function staffCanManageHostel(client, user, hostelId) {
  if (user.role === "owner") return ownerHasHostel(client, user.id, hostelId);
  if (user.role === "warden") return wardenHasHostel(client, user.id, hostelId);
  return false;
}

app.post("/api/announcements", requireAuth, requireRole("owner", "warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const hostelId = Number(req.body.hostel_id); const title = String(req.body.title || "").trim();
    const body = String(req.body.body || "").trim(); const audience = req.body.audience || "residents";
    if (!isValidInt(hostelId) || !title || !body || !["residents", "booked", "favorited"].includes(audience)) return res.status(400).json({ error: "hostel_id, title, body, and a valid audience are required" });
    if (!await staffCanManageHostel(client, req.user, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    await client.query("BEGIN");
    const announcement = await client.query("INSERT INTO announcements (hostel_id,author_user_id,title,body,audience) VALUES ($1,$2,$3,$4,$5) RETURNING *", [hostelId, req.user.id, title, body, audience]);
    const includeBookings = audience !== "favorited";
    const includeFavorites = audience !== "booked";
    await client.query(
      `WITH recipients AS (
        SELECT DISTINCT user_id FROM bookings WHERE hostel_id=$1 AND status IN ('pending','confirmed') AND $2
        UNION
        SELECT DISTINCT user_id FROM favorites WHERE hostel_id=$1 AND $3
      )
      INSERT INTO notifications (user_id,announcement_id,type,title,body,data)
      SELECT r.user_id,$4,'announcement',$5,$6,jsonb_build_object('hostel_id',$1)
      FROM recipients r LEFT JOIN notification_preferences np ON np.user_id=r.user_id
      WHERE COALESCE(np.announcements,true)`,
      [hostelId, includeBookings, includeFavorites, announcement.rows[0].id, title, body]
    );
    await client.query("COMMIT");
    res.status(201).json({ announcement: announcement.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to post announcement" }); } finally { client.release(); }
});

app.get("/api/announcements", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    if (req.user.role === "student") {
      const result = await pool.query(
        `SELECT DISTINCT a.* FROM announcements a
         LEFT JOIN bookings b ON b.hostel_id=a.hostel_id AND b.user_id=$1 AND b.status IN ('pending','confirmed')
         LEFT JOIN favorites f ON f.hostel_id=a.hostel_id AND f.user_id=$1
         WHERE (a.audience IN ('residents','booked') AND b.user_id IS NOT NULL) OR (a.audience IN ('residents','favorited') AND f.user_id IS NOT NULL)
         ORDER BY a.created_at DESC`, [req.user.id]
      );
      return res.json(result.rows);
    }
    const result = await pool.query(
      `SELECT DISTINCT a.* FROM announcements a LEFT JOIN hostel_owners ho ON ho.hostel_id=a.hostel_id LEFT JOIN warden_assignments wa ON wa.hostel_id=a.hostel_id
       WHERE (ho.owner_user_id=$1 OR wa.warden_user_id=$1) ORDER BY a.created_at DESC`, [req.user.id]
    );
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch announcements" }); }
});

app.get("/api/notifications", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query("SELECT * FROM notifications WHERE user_id=$1 ORDER BY created_at DESC", [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch notifications" }); }
});

app.patch("/api/notifications/:id/read", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const notificationId = Number(req.params.id);
    if (!isValidInt(notificationId)) return res.status(400).json({ error: "Invalid notification id" });
    const result = await pool.query("UPDATE notifications SET read_at=COALESCE(read_at,NOW()) WHERE id=$1 AND user_id=$2 RETURNING *", [notificationId, req.user.id]);
    if (result.rowCount === 0) return res.status(404).json({ error: "Notification not found" });
    res.json({ notification: result.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to mark notification as read" }); }
});

// ══════════════════════════════════════════════════════════════════
//  ROOM MANAGEMENT ROUTES
// ══════════════════════════════════════════════════════════════════

// GET /api/hostels/:hostel_id/rooms
app.get("/api/hostels/:hostel_id/rooms", requireAuth, requireRole("owner", "warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const hostelId = parseInt(req.params.hostel_id, 10);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    if (!await staffCanManageHostel(pool, req.user, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    
    const hostelResult = await pool.query("SELECT id FROM hostels WHERE id=$1", [hostelId]);
    if (hostelResult.rows.length === 0) return res.status(404).json({ error: "Hostel not found" });
    
    const roomsResult = await pool.query(
      "SELECT id, hostel_id, room_number, capacity, available_capacity, room_type, created_at FROM rooms WHERE hostel_id=$1 ORDER BY room_number ASC",
      [hostelId]
    );
    res.json(roomsResult.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch rooms: " + err.message }); }
});

// POST /api/rooms
app.post("/api/rooms", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const { hostel_id, room_number, capacity, room_type } = req.body;
    const hostelId = parseInt(hostel_id, 10);
    const capacityInt = parseInt(capacity, 10);
    
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    if (!room_number) return res.status(400).json({ error: "room_number is required" });
    if (!isValidInt(capacityInt) || capacityInt <= 0) return res.status(400).json({ error: "capacity must be a positive integer" });
    
    if (!await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    
    const roomResult = await pool.query(
      "INSERT INTO rooms (hostel_id, room_number, capacity, available_capacity, room_type) VALUES ($1,$2,$3,$4,$5) RETURNING *",
      [hostelId, room_number, capacityInt, capacityInt, room_type || "shared"]
    );
    
    res.status(201).json({ message: "Room created ✅", room: roomResult.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to create room: " + err.message }); }
});

// PATCH /api/rooms/:id
app.patch("/api/rooms/:id", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const roomId = parseInt(req.params.id, 10);
    if (!isValidInt(roomId)) return res.status(400).json({ error: "Invalid room id" });
    
    const { room_number, capacity, available_capacity, room_type } = req.body;
    
    const roomResult = await pool.query("SELECT id, hostel_id, capacity, available_capacity FROM rooms WHERE id=$1", [roomId]);
    if (roomResult.rows.length === 0) return res.status(404).json({ error: "Room not found" });
    if (!await ownerHasHostel(pool, req.user.id, roomResult.rows[0].hostel_id)) return res.status(404).json({ error: "Room not found" });
    
    let updates = [];
    let params = [];
    let paramCount = 1;
    
    if (room_number != null) {
      updates.push(`room_number=$${paramCount}`);
      params.push(room_number);
      paramCount++;
    }
    if (capacity != null) {
      const capacityInt = parseInt(capacity, 10);
      if (!isValidInt(capacityInt)) return res.status(400).json({ error: "capacity must be a positive integer" });
      if (available_capacity == null && capacityInt < Number(roomResult.rows[0].available_capacity)) return res.status(409).json({ error: "capacity cannot be below occupied beds" });
      updates.push(`capacity=$${paramCount}`);
      params.push(capacityInt);
      paramCount++;
    }
    if (available_capacity != null) {
      const availInt = parseInt(available_capacity, 10);
      const resultingCapacity = capacity == null ? Number(roomResult.rows[0].capacity) : parseInt(capacity, 10);
      if (availInt < 0 || availInt > resultingCapacity) return res.status(400).json({ error: "available_capacity must be between zero and capacity" });
      updates.push(`available_capacity=$${paramCount}`);
      params.push(availInt);
      paramCount++;
    }
    if (room_type != null) {
      updates.push(`room_type=$${paramCount}`);
      params.push(room_type);
      paramCount++;
    }
    
    if (updates.length === 0) return res.status(400).json({ error: "No fields to update" });
    
    params.push(roomId);
    const updateQuery = `UPDATE rooms SET ${updates.join(", ")} WHERE id=$${paramCount} RETURNING *`;
    const updatedResult = await pool.query(updateQuery, params);
    
    res.json({ message: "Room updated ✅", room: updatedResult.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to update room: " + err.message }); }
});

// POST /api/bookings/:id/assign-room (assign a specific room to a booking)
app.post("/api/bookings/:id/assign-room", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = parseInt(req.params.id, 10);
    const roomId = parseInt(req.body.room_id, 10);
    
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    if (!isValidInt(roomId)) return res.status(400).json({ error: "room_id must be a positive integer" });
    
    await client.query("BEGIN");
    
    const bookingResult = await client.query(
      "SELECT id, hostel_id, room_id, status FROM bookings WHERE id=$1 FOR UPDATE",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Booking not found" });
    }
    
    const booking = bookingResult.rows[0];
    if (!await wardenHasHostel(client, req.user.id, booking.hostel_id)) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Booking not found" });
    }
    
    const roomResult = await client.query(
      "SELECT id, hostel_id, available_capacity FROM rooms WHERE id=$1 FOR UPDATE",
      [roomId]
    );
    if (roomResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Room not found" });
    }
    
    const room = roomResult.rows[0];
    if (Number(room.hostel_id) !== Number(booking.hostel_id)) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "Room does not belong to the booking's hostel" });
    }
    
    if (Number(booking.room_id) === roomId) {
      await client.query("ROLLBACK");
      return res.json({ message: "Room is already assigned to this booking" });
    }

    if (Number(room.available_capacity) <= 0) {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "Room has no available capacity" });
    }
    
    const bookingStatus = normalizeBookingStatus(booking.status);
    if (bookingStatus !== "confirmed") {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "Only confirmed bookings can be assigned a room" });
    }
    
    const oldRoomId = booking.room_id;
    
    if (oldRoomId != null) {
      await client.query(
        "UPDATE rooms SET available_capacity = LEAST(available_capacity + 1, capacity) WHERE id=$1",
        [oldRoomId]
      );
    }
    
    await client.query(
      "UPDATE rooms SET available_capacity = available_capacity - 1 WHERE id=$1",
      [roomId]
    );
    
    const updatedBooking = await client.query(
      "UPDATE bookings SET room_id=$1 WHERE id=$2 RETURNING *",
      [roomId, bookingId]
    );
    await client.query(
      "INSERT INTO room_assignment_history (booking_id,previous_room_id,next_room_id,assigned_by_user_id) VALUES ($1,$2,$3,$4)",
      [bookingId, oldRoomId, roomId, req.user.id]
    );
    
    await client.query("COMMIT");
    res.json({ message: "Room assigned to booking ✅", booking: updatedBooking.rows[0] });
  } catch (err) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to assign room: " + err.message });
  } finally {
    client.release();
  }
});

// GET /api/bookings/:id/available-rooms
app.get("/api/bookings/:id/available-rooms", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    const bookingId = parseInt(req.params.id, 10);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    
    const bookingResult = await pool.query(
      "SELECT id, hostel_id FROM bookings WHERE id=$1",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) return res.status(404).json({ error: "Booking not found" });
    
    const booking = bookingResult.rows[0];
    if (!await wardenHasHostel(pool, req.user.id, booking.hostel_id)) return res.status(404).json({ error: "Booking not found" });
    const roomsResult = await pool.query(
      "SELECT id, room_number, capacity, available_capacity, room_type FROM rooms WHERE hostel_id=$1 AND available_capacity > 0 ORDER BY room_number ASC",
      [booking.hostel_id]
    );
    
    res.json(roomsResult.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch available rooms: " + err.message }); }
});

app.post("/api/warden/bookings/:id/check-in", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    await client.query("BEGIN");
    const booking = await client.query("SELECT b.* FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id WHERE b.id=$1 AND wa.warden_user_id=$2 FOR UPDATE", [bookingId, req.user.id]);
    if (booking.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    if (normalizeBookingStatus(booking.rows[0].status) !== "confirmed" || booking.rows[0].room_id == null) { await client.query("ROLLBACK"); return res.status(409).json({ error: "A confirmed booking with an assigned room is required" }); }
    const stay = await client.query("INSERT INTO booking_stays (booking_id,checked_in_at,checked_in_by_user_id) VALUES ($1,NOW(),$2) ON CONFLICT (booking_id) DO UPDATE SET checked_in_at=COALESCE(booking_stays.checked_in_at,EXCLUDED.checked_in_at), checked_in_by_user_id=COALESCE(booking_stays.checked_in_by_user_id,EXCLUDED.checked_in_by_user_id) WHERE booking_stays.checked_out_at IS NULL RETURNING *", [bookingId, req.user.id]);
    if (stay.rowCount === 0) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Stay has already been checked out" }); }
    await client.query("COMMIT");
    res.json({ stay: stay.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to check in booking" }); } finally { client.release(); }
});

app.post("/api/warden/bookings/:id/check-out", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    await client.query("BEGIN");
    const booking = await client.query("SELECT b.* FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id WHERE b.id=$1 AND wa.warden_user_id=$2 FOR UPDATE", [bookingId, req.user.id]);
    if (booking.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    const stay = await client.query("SELECT * FROM booking_stays WHERE booking_id=$1 FOR UPDATE", [bookingId]);
    if (stay.rowCount === 0 || stay.rows[0].checked_in_at == null || stay.rows[0].checked_out_at != null) { await client.query("ROLLBACK"); return res.status(409).json({ error: "An active checked-in stay is required" }); }
    if (!await releaseHostelCapacity(client, booking.rows[0].hostel_id)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Hostel availability is already fully restored" }); }
    await client.query("UPDATE rooms SET available_capacity=LEAST(available_capacity+1,capacity) WHERE id=$1", [booking.rows[0].room_id]);
    const updatedStay = await client.query("UPDATE booking_stays SET checked_out_at=NOW(), checked_out_by_user_id=$1 WHERE booking_id=$2 RETURNING *", [req.user.id, bookingId]);
    await client.query("UPDATE bookings SET status='completed' WHERE id=$1", [bookingId]);
    await createNotification(client, booking.rows[0].user_id, "booking_status", "Move-out recorded", "Your hostel stay has been checked out.", { booking_id: bookingId, status: "completed" }, "booking_updates");
    await client.query("COMMIT");
    res.json({ stay: updatedStay.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to check out booking" }); } finally { client.release(); }
});

app.get("/api/bookings/:id/stay", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    const result = await pool.query("SELECT bs.* FROM booking_stays bs JOIN bookings b ON b.id=bs.booking_id WHERE bs.booking_id=$1 AND b.user_id=$2", [bookingId, req.user.id]);
    if (result.rowCount === 0) return res.status(404).json({ error: "Stay not found" });
    res.json({ stay: result.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to fetch stay" }); }
});

app.post("/api/payments/intents", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    if (!process.env.STRIPE_SECRET_KEY) return res.status(503).json({ error: "Online payments are not configured" });
    const bookingId = Number(req.body.booking_id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "booking_id is required" });
    const booking = await pool.query(
      `SELECT b.id, b.user_id, b.status, h.name, COALESCE(h.single_room_price,h.double_room_price,h.dorm_room_price) AS price
       FROM bookings b JOIN hostels h ON h.id=b.hostel_id WHERE b.id=$1 AND b.user_id=$2`, [bookingId, req.user.id]
    );
    if (booking.rowCount === 0) return res.status(404).json({ error: "Booking not found" });
    if (normalizeBookingStatus(booking.rows[0].status) !== "confirmed") return res.status(409).json({ error: "Only confirmed bookings can be paid" });
    const amount = Number(booking.rows[0].price);
    if (!Number.isFinite(amount) || amount <= 0) return res.status(409).json({ error: "The hostel has no configured online payment amount" });
    const payload = new URLSearchParams({ amount: String(Math.round(amount * 100)), currency: "pkr", "metadata[booking_id]": String(bookingId) });
    const stripeApiBase = process.env.STRIPE_API_BASE || "https://api.stripe.com";
    const stripeResponse = await fetch(`${stripeApiBase}/v1/payment_intents`, {
      method: "POST", headers: { Authorization: `Bearer ${process.env.STRIPE_SECRET_KEY}`, "Content-Type": "application/x-www-form-urlencoded" }, body: payload,
    });
    const intent = await stripeResponse.json();
    if (!stripeResponse.ok) return res.status(502).json({ error: "Payment provider rejected the request" });
    const payment = await pool.query("INSERT INTO payments (booking_id,user_id,provider,provider_payment_id,amount,currency,status) VALUES ($1,$2,'stripe',$3,$4,'PKR','pending') RETURNING *", [bookingId, req.user.id, intent.id, amount]);
    res.status(201).json({ payment: payment.rows[0], client_secret: intent.client_secret });
  } catch (error) { res.status(500).json({ error: "Failed to create payment intent" }); }
});

app.get("/api/payments", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query("SELECT * FROM payments WHERE user_id=$1 ORDER BY created_at DESC", [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch payments" }); }
});

const PORT = process.env.PORT || 5000;
if (require.main === module) {
  app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
}

module.exports = { app, pool };