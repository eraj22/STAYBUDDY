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
// Express middleware: requires a valid signed token and attaches req.user = { id, role }.
function requireAuth(req, res, next) {
  const auth = req.headers.authorization || "";
  const token = auth.replace("Bearer ", "").trim();
  const user = verifyAuthToken(token);
  if (!user) return res.status(401).json({ error: "Missing or invalid auth token" });
  req.user = user;
  next();
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
      "SELECT id, name, email, role, phone FROM public.users WHERE email=$1",
      [email]
    );

    if (result.rows.length === 0)
      return res.status(404).json({ error: "No account found with this email." });

    const user = result.rows[0];

    // NOTE: passwords stored as plain text / dummyhash in dev DB.
    // In production replace with: bcrypt.compare(password, user.password_hash)
    // For now accept any password so warden can log in during testing.
    // To require real password comparison uncomment the check below:
    // const stored = await pool.query("SELECT password_hash FROM users WHERE id=$1",[user.id]);
    // if (stored.rows[0].password_hash !== password) return res.status(401).json({ error: "Invalid password." });

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

    const exists = await pool.query("SELECT id FROM public.users WHERE email=$1", [email]);
    if (exists.rows.length > 0)
      return res.status(409).json({ error: "Email already registered." });

    const result = await pool.query(
      "INSERT INTO public.users (name, email, password_hash, role) VALUES ($1,$2,$3,$4) RETURNING id, name, email, role",
      [name, email, password, role]
    );
    const user = result.rows[0];
    const token = signAuthToken(user);
    res.json({ success: true, token, user });
  } catch (err) {
    res.status(500).json({ error: "Registration failed: " + err.message });
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

// GET /api/warden/dashboard
app.get("/api/warden/dashboard", async (req, res) => {
  try {
    // Return a basic dashboard — extend with real hostel assignment later
    const hostelResult = await pool.query(
      "SELECT id, name, city, total_capacity, available_capacity FROM hostels LIMIT 1"
    );
    const hostel = hostelResult.rows[0] || {};
    const bookings = await pool.query("SELECT COUNT(*) FROM bookings");
    const complaints = await pool.query("SELECT COUNT(*) FROM complaints WHERE status != 'resolved'");

    res.json({
      success: true,
      hostel_name:         hostel.name  || "My Hostel",
      hostel_id:           hostel.id    || 1,
      hostel_city:         hostel.city  || "",
      total_capacity:      hostel.total_capacity     || 0,
      available_capacity:  hostel.available_capacity || 0,
      total_bookings:      parseInt(bookings.rows[0].count)   || 0,
      pending_complaints:  parseInt(complaints.rows[0].count) || 0,
    });
  } catch (err) {
    res.status(500).json({ error: "Dashboard fetch failed: " + err.message });
  }
});

// ══════════════════════════════════════════════════════════════════
//  ROOM MANAGEMENT ROUTES
// ══════════════════════════════════════════════════════════════════

// GET /api/hostels/:hostel_id/rooms
app.get("/api/hostels/:hostel_id/rooms", async (req, res) => {
  try {
    const hostelId = parseInt(req.params.hostel_id, 10);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    
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
app.post("/api/rooms", async (req, res) => {
  try {
    const { hostel_id, room_number, capacity, room_type } = req.body;
    const hostelId = parseInt(hostel_id, 10);
    const capacityInt = parseInt(capacity, 10);
    
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    if (!room_number) return res.status(400).json({ error: "room_number is required" });
    if (!isValidInt(capacityInt) || capacityInt <= 0) return res.status(400).json({ error: "capacity must be a positive integer" });
    
    const hostelResult = await pool.query("SELECT id FROM hostels WHERE id=$1", [hostelId]);
    if (hostelResult.rows.length === 0) return res.status(404).json({ error: "Hostel not found" });
    
    const roomResult = await pool.query(
      "INSERT INTO rooms (hostel_id, room_number, capacity, available_capacity, room_type) VALUES ($1,$2,$3,$4,$5) RETURNING *",
      [hostelId, room_number, capacityInt, capacityInt, room_type || "shared"]
    );
    
    res.status(201).json({ message: "Room created ✅", room: roomResult.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to create room: " + err.message }); }
});

// PATCH /api/rooms/:id
app.patch("/api/rooms/:id", async (req, res) => {
  try {
    const roomId = parseInt(req.params.id, 10);
    if (!isValidInt(roomId)) return res.status(400).json({ error: "Invalid room id" });
    
    const { room_number, capacity, available_capacity, room_type } = req.body;
    
    const roomResult = await pool.query("SELECT id FROM rooms WHERE id=$1", [roomId]);
    if (roomResult.rows.length === 0) return res.status(404).json({ error: "Room not found" });
    
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
      updates.push(`capacity=$${paramCount}`);
      params.push(capacityInt);
      paramCount++;
    }
    if (available_capacity != null) {
      const availInt = parseInt(available_capacity, 10);
      if (availInt < 0) return res.status(400).json({ error: "available_capacity cannot be negative" });
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
app.post("/api/bookings/:id/assign-room", async (req, res) => {
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
    if (!isCapacityHoldingStatus(bookingStatus)) {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "Booking is not pending or confirmed" });
    }
    
    const oldRoomId = booking.room_id;
    
    if (oldRoomId != null) {
      await client.query(
        "UPDATE rooms SET available_capacity = available_capacity + 1 WHERE id=$1",
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
app.get("/api/bookings/:id/available-rooms", async (req, res) => {
  try {
    const bookingId = parseInt(req.params.id, 10);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    
    const bookingResult = await pool.query(
      "SELECT id, hostel_id FROM bookings WHERE id=$1",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) return res.status(404).json({ error: "Booking not found" });
    
    const booking = bookingResult.rows[0];
    const roomsResult = await pool.query(
      "SELECT id, room_number, capacity, available_capacity, room_type FROM rooms WHERE hostel_id=$1 AND available_capacity > 0 ORDER BY room_number ASC",
      [booking.hostel_id]
    );
    
    res.json(roomsResult.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch available rooms: " + err.message }); }
});

const PORT = process.env.PORT || 5000;
if (require.main === module) {
  app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
}

module.exports = { app, pool };