require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");

const app = express();
app.use(cors());
app.use(express.json());

// Connect PostgreSQL
const pool = new Pool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: Number(process.env.DB_PORT),
});

// ---------- helpers ----------
function isValidInt(n) {
  return Number.isInteger(n) && n > 0;
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

// ✅ Get all hostels
app.get("/api/hostels", async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT id, name, address, city, total_capacity, available_capacity, description FROM hostels ORDER BY id ASC"
    );
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch hostels" });
  }
});

// ✅ Get hostel by id (Phase 1 core endpoint)
app.get("/api/hostels/:id", async (req, res) => {
  try {
    const hostelId = parseInt(req.params.id, 10);
    if (!isValidInt(hostelId)) {
      return res.status(400).json({ error: "Invalid hostel id" });
    }

    const hostelQ = await pool.query(
      "SELECT id, name, address, city, total_capacity, available_capacity, description FROM hostels WHERE id = $1",
      [hostelId]
    );

    if (hostelQ.rows.length === 0) {
      return res.status(404).json({ error: "Hostel not found" });
    }

    const reviewsQ = await pool.query(
      "SELECT id, user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review, created_at FROM reviews WHERE hostel_id = $1 ORDER BY created_at DESC",
      [hostelId]
    );

    res.json({
      ...hostelQ.rows[0],
      reviews: reviewsQ.rows,
    });
  } catch (err) {
    console.error("Error fetching hostel details:", err);
    res.status(500).json({ error: "Failed to fetch hostel details" });
  }
});

// ✅ Get user profile by id (Phase 1 core endpoint)
app.get("/api/users/:id", async (req, res) => {
  try {
    const userId = parseInt(req.params.id, 10);
    if (!isValidInt(userId)) {
      return res.status(400).json({ error: "Invalid user id" });
    }

    // NOTE: adjust fields if your users table has different columns
    const userQ = await pool.query(
      "SELECT id, name, email, role, created_at FROM users WHERE id = $1",
      [userId]
    );

    if (userQ.rows.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }

    res.json(userQ.rows[0]);
  } catch (err) {
    console.error("Error fetching user:", err);
    // If your users table doesn't have name/created_at, you'll get an error.
    // Tell me your users columns and I’ll adjust.
    res.status(500).json({ error: "Failed to fetch user profile" });
  }
});

// ✅ Get all interactions
app.get("/api/interactions", async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT id, user_id, hostel_id, action_type, created_at FROM interactions ORDER BY created_at DESC"
    );
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch interactions" });
  }
});

// ✅ Get all bookings
app.get("/api/bookings", async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT id, user_id, hostel_id, check_in, check_out, status, created_at FROM bookings ORDER BY id ASC"
    );
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch bookings" });
  }
});

// ✅ Add a new interaction
app.post("/api/interactions", async (req, res) => {
  try {
    const { user_id, hostel_id, action_type } = req.body;

    const uid = parseInt(user_id, 10);
    const hid = hostel_id === null || hostel_id === undefined ? null : parseInt(hostel_id, 10);

    if (!isValidInt(uid)) {
      return res.status(400).json({ error: "user_id must be a positive integer" });
    }
    if (hid !== null && !isValidInt(hid)) {
      return res.status(400).json({ error: "hostel_id must be a positive integer or null" });
    }
    if (!action_type || typeof action_type !== "string") {
      return res.status(400).json({ error: "action_type is required" });
    }

    const result = await pool.query(
      "INSERT INTO interactions (user_id, hostel_id, action_type) VALUES ($1, $2, $3) RETURNING *",
      [uid, hid, action_type]
    );

    res.status(201).json({ message: "Interaction saved ✅", interaction: result.rows[0] });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to save interaction" });
  }
});

// ✅ Get all reviews
app.get("/api/reviews", async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT id, user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review, created_at FROM reviews ORDER BY id ASC"
    );
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch reviews" });
  }
});

// ✅ Add a new review
app.post("/api/reviews", async (req, res) => {
  try {
    const { user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review } = req.body;

    const uid = parseInt(user_id, 10);
    const hid = parseInt(hostel_id, 10);

    if (!isValidInt(uid) || !isValidInt(hid)) {
      return res.status(400).json({ error: "user_id and hostel_id must be positive integers" });
    }

    const result = await pool.query(
      `INSERT INTO reviews (user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review)
       VALUES ($1,$2,$3,$4,$5,$6,$7)
       RETURNING *`,
      [uid, hid, overall_rating, cleanliness, facilities, management, text_review || null]
    );

    res.status(201).json({ message: "Review saved ✅", review: result.rows[0] });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to save review" });
  }
});

// ✅ Get all complaints
app.get("/api/complaints", async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT id, user_id, hostel_id, category, severity, status, assigned_to, description, created_at, resolved_at
       FROM complaints
       ORDER BY id ASC`
    );
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch complaints" });
  }
});

// ✅ Add new complaint
app.post("/api/complaints", async (req, res) => {
  try {
    const { user_id, hostel_id, category, severity, status, assigned_to, description } = req.body;

    const uid = parseInt(user_id, 10);
    const hid = parseInt(hostel_id, 10);
    const assigned = assigned_to === null || assigned_to === undefined ? null : parseInt(assigned_to, 10);

    if (!isValidInt(uid) || !isValidInt(hid)) {
      return res.status(400).json({ error: "user_id and hostel_id must be positive integers" });
    }
    if (assigned !== null && !isValidInt(assigned)) {
      return res.status(400).json({ error: "assigned_to must be a positive integer or null" });
    }
    if (!category || typeof category !== "string") {
      return res.status(400).json({ error: "category is required" });
    }
    if (!severity || typeof severity !== "string") {
      return res.status(400).json({ error: "severity is required" });
    }
    if (!description || typeof description !== "string") {
      return res.status(400).json({ error: "description is required" });
    }

    const result = await pool.query(
      `INSERT INTO complaints (user_id, hostel_id, category, severity, status, assigned_to, description)
       VALUES ($1,$2,$3,$4,$5,$6,$7)
       RETURNING *`,
      [uid, hid, category, severity, status || "open", assigned, description]
    );

    res.status(201).json({ message: "Complaint saved ✅", complaint: result.rows[0] });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to save complaint" });
  }
});

// ---------------- Phase 1 ML Integration endpoints (placeholders) ----------------

// ✅ Recommendation endpoint (placeholder baseline)
app.post("/api/recommendations", async (req, res) => {
  try {
    const { student_id, top_k } = req.body;

    const sid = parseInt(student_id, 10);
    if (!isValidInt(sid)) {
      return res.status(400).json({ error: "student_id must be a positive integer" });
    }

    const k = Number.isInteger(top_k) && top_k > 0 ? top_k : 5;

    // Simple baseline: recommend by highest availability
    const recQ = await pool.query(
      "SELECT id, name, city, available_capacity FROM hostels ORDER BY available_capacity DESC LIMIT $1",
      [k]
    );

    res.json({
      student_id: sid,
      model: "baseline_placeholder",
      recommendations: recQ.rows,
    });
  } catch (err) {
    console.error("Error generating recommendations:", err);
    res.status(500).json({ error: "Failed to generate recommendations" });
  }
});

// ✅ Chat endpoint (placeholder for chatbot integration)
app.post("/api/chat", async (req, res) => {
  try {
    const { message, student_id } = req.body;

    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "message is required" });
    }

    const sid = student_id ? parseInt(student_id, 10) : null;

    // Simple helpful response for Phase 1
    const hostelsQ = await pool.query(
      "SELECT id, name, city, available_capacity FROM hostels ORDER BY available_capacity DESC LIMIT 3"
    );

    res.json({
      student_id: isValidInt(sid) ? sid : null,
      reply:
        "I can help you with hostel availability, reviews, and complaints. Here are some hostels with availability right now:",
      suggestions: hostelsQ.rows,
    });
  } catch (err) {
    console.error("Error in chat endpoint:", err);
    res.status(500).json({ error: "Chat service failed" });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));

