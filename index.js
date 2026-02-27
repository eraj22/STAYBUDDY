require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");

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

// ✅ Get hostel by id (with reviews)
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
      reviews: reviewsQ.rows
    });
  } catch (err) {
    console.error("Error fetching hostel details:", err);
    res.status(500).json({ error: "Failed to fetch hostel details" });
  }
});

// ✅ Get user profile by id
// NOTE: If your users table doesn't have name/created_at, see note below.
app.get("/api/users/:id", async (req, res) => {
  try {
    const userId = parseInt(req.params.id, 10);
    if (!isValidInt(userId)) {
      return res.status(400).json({ error: "Invalid user id" });
    }

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
    res.status(500).json({ error: "Failed to fetch user profile (check users columns)" });
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

// ✅ Add a new interaction
app.post("/api/interactions", async (req, res) => {
  try {
    const { user_id, hostel_id, action_type } = req.body;

    const uid = parseInt(user_id, 10);
    const hid =
      hostel_id === null || hostel_id === undefined ? null : parseInt(hostel_id, 10);

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
    const {
      user_id,
      hostel_id,
      overall_rating,
      cleanliness,
      facilities,
      management,
      text_review
    } = req.body;

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
    const assigned =
      assigned_to === null || assigned_to === undefined ? null : parseInt(assigned_to, 10);

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

// ---------------- Phase 1 ML Integration endpoints ----------------

// ✅ Intelligent hybrid recommender (uses interactions + reviews + availability + city preference)
// Includes validation + fallback + explainability for a strong prelim demo.
app.post("/api/recommendations", async (req, res) => {
  try {
    const { student_id, top_k } = req.body;

    const sid = parseInt(student_id, 10);
    if (!isValidInt(sid)) {
      return res.status(400).json({ error: "student_id must be a positive integer" });
    }

    const k = Number.isInteger(top_k) && top_k > 0 ? top_k : 5;

    // 1) Pull all hostels
    const hostelsQ = await pool.query(
      "SELECT id, name, city, available_capacity, total_capacity, description FROM hostels ORDER BY id ASC"
    );

    // 2) Ratings summary
    const ratingsQ = await pool.query(
      `SELECT hostel_id, AVG(overall_rating)::float AS avg_rating, COUNT(*)::int AS n_reviews
       FROM reviews
       GROUP BY hostel_id`
    );

    // 3) Student interactions (implicit feedback)
    const interQ = await pool.query(
      `SELECT user_id, hostel_id, action_type, created_at
       FROM interactions
       WHERE user_id = $1
       ORDER BY created_at DESC`,
      [sid]
    );

    const hostels = hostelsQ.rows;
    const hostelMap = new Map(hostels.map(h => [h.id, h]));

    const ratingMap = new Map();
    for (const r of ratingsQ.rows) ratingMap.set(r.hostel_id, r);

    const userInteractions = interQ.rows;

    // Inferred preference from history (revealed preference)
    const userPref = buildUserPreferenceFromHistory(userInteractions, hostelMap);

    // Content signals: rating, availability
    const avgRatings = hostels.map(h => (ratingMap.get(h.id)?.avg_rating ?? 0));
    const avail = hostels.map(h => (h.available_capacity ?? 0));

    const ratingNormArr = softNormalize(avgRatings);
    const availNormArr = softNormalize(avail);

    // Collaborative-like affinity: sum weighted actions per hostel
    const affinityByHostel = new Map();
    for (const it of userInteractions) {
      const w = actionWeight(it.action_type);
      affinityByHostel.set(it.hostel_id, (affinityByHostel.get(it.hostel_id) || 0) + w);
    }
    const affinityArr = hostels.map(h => affinityByHostel.get(h.id) || 0);
    const affinityNormArr = softNormalize(affinityArr);

    // Hybrid weighting (easy to explain in demo)
    const W = {
      userAffinity: 0.45,
      rating: 0.30,
      availability: 0.15,
      city: 0.10
    };

    const scored = hostels.map((h, idx) => {
      const ratingNorm = ratingNormArr[idx];
      const availNorm = availNormArr[idx];
      const userAffinity = affinityNormArr[idx];

      const cityMatch =
        userPref.topCity && h.city === userPref.topCity ? clamp01(userPref.cityStrength) : 0;

      const score =
        W.userAffinity * userAffinity +
        W.rating * ratingNorm +
        W.availability * availNorm +
        W.city * cityMatch;

      return {
        hostel: {
          id: h.id,
          name: h.name,
          city: h.city,
          available_capacity: h.available_capacity,
          avg_rating: ratingMap.get(h.id)?.avg_rating ?? null,
          n_reviews: ratingMap.get(h.id)?.n_reviews ?? 0
        },
        score: Number(score.toFixed(4)),
        signals: {
          userAffinity: Number(userAffinity.toFixed(4)),
          ratingNorm: Number(ratingNorm.toFixed(4)),
          availNorm: Number(availNorm.toFixed(4)),
          cityMatch: Number(cityMatch.toFixed(4))
        }
      };
    });

    scored.sort((a, b) => b.score - a.score);

    // Validation + fallback (prelim “smart behavior” when signals are weak)
    const hasEnoughSignals = userInteractions.length >= 3;
    const validation = validateAndFallback(scored, { hasEnoughSignals });

    let finalList = scored;
    if (validation.usedFallback) {
      // safe fallback: rating then availability
      finalList = scored
        .slice()
        .sort(
          (a, b) =>
            (b.hostel.avg_rating ?? 0) - (a.hostel.avg_rating ?? 0) ||
            (b.hostel.available_capacity ?? 0) - (a.hostel.available_capacity ?? 0)
        );
    }

    const recommendations = finalList.slice(0, k).map((rec, i) => ({
      rank: i + 1,
      ...rec.hostel,
      score: rec.score,
      why: explain(rec, userPref),
      debug_signals: rec.signals
    }));

    res.json({
      student_id: sid,
      model: validation.usedFallback ? "hybrid_v1_fallback" : "hybrid_v1",
      personalization: {
        inferred_top_city: userPref.topCity,
        confidence_interactions: userInteractions.length
      },
      validation,
      recommendations
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
