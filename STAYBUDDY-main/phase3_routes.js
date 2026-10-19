const crypto = require("crypto");

function parsePositiveNumber(value) {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : NaN;
}

function parseCoordinate(value) {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : NaN;
}

function parseTranscript(transcript) {
  const normalized = String(transcript || "").trim().replace(/\s+/g, " ");
  const lower = normalized.toLowerCase();
  const filters = {};
  const unsupportedTerms = [];
  const clarificationQuestions = [];

  const price = lower.match(/(?:under|below|maximum|max|budget(?: of)?|up to)\s*(?:pkr|rs\.?|₨)?\s*([\d,]+)/i);
  if (price) filters.max_price = Number(price[1].replace(/,/g, ""));
  const radius = lower.match(/(?:within|under)\s*(\d+(?:\.\d+)?)\s*(?:km|kilomet(?:er|re)s?)/i);
  if (radius) filters.radius_km = Number(radius[1]);
  if (/\bsingle\b/.test(lower)) filters.room_type = "single";
  else if (/\bdouble\b|two sharing|2 sharing/.test(lower)) filters.room_type = "double";
  else if (/\bdorm(?:itory)?\b|shared room/.test(lower)) filters.room_type = "dorm";

  const amenityDictionary = [
    ["wifi", ["wifi", "wi-fi", "internet"]],
    ["air_conditioning", ["air conditioning", "ac room", "a/c"]],
    ["mess", ["mess", "food", "meals"]],
    ["laundry", ["laundry"]],
    ["security", ["security", "cctv"]],
    ["generator", ["generator", "backup power"]],
  ];
  const amenities = amenityDictionary
    .filter(([, terms]) => terms.some((term) => lower.includes(term)))
    .map(([name]) => name);
  if (amenities.length) filters.amenities = amenities;

  const university = normalized.match(/(?:near|close to|around)\s+([A-Za-z][A-Za-z .&'-]+?(?:university|college|institute))/i);
  if (university) filters.university = university[1].trim();
  const city = normalized.match(
    /(?:in|at)\s+([A-Za-z][A-Za-z .'-]{2,40}?)(?=\s+(?:under|below|maximum|max|budget|up to|within|with|near|close to|around|single|double|shared|dorm|girls?|boys?)\b|[,.]|$)/i
  );
  if (city && !/university|college|institute/i.test(city[1])) filters.city = city[1].trim();

  if (/girls?(?: only| hostel)|boys?(?: only| hostel)|female hostel|male hostel/.test(lower)) {
    filters.hostel_type = /girls|female/.test(lower) ? "girls" : "boys";
  }
  if (/pet friendly|gym|parking|furnished/.test(lower)) unsupportedTerms.push("One or more requested facilities may not use a standardized catalogue value.");
  if (!filters.city && !filters.university && !filters.radius_km) {
    clarificationQuestions.push("Which city or university should the search use?");
  }
  if (!filters.max_price) clarificationQuestions.push("What is your maximum monthly budget in PKR?");

  const extractedCount = Object.keys(filters).length;
  return {
    transcript: normalized,
    normalized_search_text: normalized,
    filters,
    confidence: Number(Math.min(0.95, 0.35 + extractedCount * 0.1).toFixed(2)),
    unsupported_terms: unsupportedTerms,
    clarification_questions: clarificationQuestions,
    speech_recognition_performed: false,
  };
}

function timeoutSignal(milliseconds) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), milliseconds);
  return { signal: controller.signal, clear: () => clearTimeout(timer) };
}

function registerPhase3Routes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, hashPassword }) {
  app.get("/api/health", async (_req, res) => {
    try {
      await pool.query("SELECT 1");
      res.json({ status: "healthy", database: "connected", notification_delivery: process.env.NOTIFICATION_DELIVERY_MODE || "fake", generated_at: new Date().toISOString() });
    } catch (_) {
      res.status(503).json({ status: "unhealthy", database: "unavailable", generated_at: new Date().toISOString() });
    }
  });

  app.get("/api/hostels/search", async (req, res) => {
    const latitude = parseCoordinate(req.query.latitude);
    const longitude = parseCoordinate(req.query.longitude);
    const radius = req.query.radius_km == null ? 5 : Number(req.query.radius_km);
    const minPrice = parsePositiveNumber(req.query.min_price);
    const maxPrice = parsePositiveNumber(req.query.max_price);
    const roomType = req.query.room_type ? String(req.query.room_type).toLowerCase() : null;
    const sort = String(req.query.sort || (latitude != null ? "distance" : "recommended"));
    const allowedRoomTypes = new Set(["single", "double", "dorm", "shared"]);
    if ((latitude != null && (!Number.isFinite(latitude) || latitude < -90 || latitude > 90)) || (longitude != null && (!Number.isFinite(longitude) || longitude < -180 || longitude > 180)) || ((latitude == null) !== (longitude == null))) return res.status(400).json({ error: "latitude and longitude must be supplied together and be valid coordinates" });
    if (!Number.isFinite(radius) || radius <= 0 || radius > 50) return res.status(400).json({ error: "radius_km must be between 0 and 50" });
    if (Number.isNaN(minPrice) || Number.isNaN(maxPrice) || (minPrice != null && maxPrice != null && minPrice > maxPrice)) return res.status(400).json({ error: "min_price and max_price must be valid ordered PKR amounts" });
    if (roomType && !allowedRoomTypes.has(roomType)) return res.status(400).json({ error: "room_type must be single, double, dorm, or shared" });
    if (!["distance", "price", "rating", "recommended"].includes(sort)) return res.status(400).json({ error: "sort must be distance, price, rating, or recommended" });

    const amenities = String(req.query.amenities || "").split(",").map((value) => value.trim()).filter(Boolean);
    const params = [];
    const add = (value) => { params.push(value); return `$${params.length}`; };
    const conditions = ["h.verified=true", "h.publication_status='published'", "h.available_capacity>0"];
    if (req.query.city) conditions.push(`h.city ILIKE ${add(`%${String(req.query.city).trim()}%`)}`);
    if (req.query.university) conditions.push(`h.university ILIKE ${add(`%${String(req.query.university).trim()}%`)}`);
    if (req.query.hostel_type) conditions.push(`h.hostel_type ILIKE ${add(`%${String(req.query.hostel_type).trim()}%`)}`);
    if (req.query.search_text) {
      const pattern = add(`%${String(req.query.search_text).trim()}%`);
      conditions.push(`(h.name ILIKE ${pattern} OR h.address ILIKE ${pattern} OR h.area ILIKE ${pattern} OR h.description ILIKE ${pattern})`);
    }
    if (amenities.length) conditions.push(`h.amenities @> ${add(JSON.stringify(amenities))}::jsonb`);
    if (roomType) conditions.push(`EXISTS (SELECT 1 FROM rooms rr WHERE rr.hostel_id=h.id AND rr.available_capacity>0 AND lower(rr.room_type)=${add(roomType)})`);
    const priceExpression = roomType
      ? `(SELECT MIN(rr.monthly_rent) FROM rooms rr WHERE rr.hostel_id=h.id AND rr.available_capacity>0 AND lower(rr.room_type)=${add(roomType)})`
      : `(SELECT MIN(rr.monthly_rent) FROM rooms rr WHERE rr.hostel_id=h.id AND rr.available_capacity>0)`;
    if (minPrice != null) conditions.push(`${priceExpression} >= ${add(minPrice)}`);
    if (maxPrice != null) conditions.push(`${priceExpression} <= ${add(maxPrice)}`);

    let distanceExpression = "NULL::numeric";
    if (latitude != null) {
      const latParam = add(latitude); const lonParam = add(longitude); const radiusParam = add(radius);
      distanceExpression = `(6371 * acos(LEAST(1,GREATEST(-1,cos(radians(${latParam})) * cos(radians(h.latitude)) * cos(radians(h.longitude)-radians(${lonParam})) + sin(radians(${latParam})) * sin(radians(h.latitude))))))`;
      conditions.push("h.latitude IS NOT NULL AND h.longitude IS NOT NULL");
      conditions.push(`${distanceExpression} <= ${radiusParam}`);
    }
    const order = sort === "distance" && latitude != null ? "distance_km ASC" : sort === "price" ? "display_price ASC NULLS LAST" : sort === "rating" ? "average_rating DESC NULLS LAST" : "average_rating DESC NULLS LAST, h.available_capacity DESC, h.name";
    try {
      const result = await pool.query(`SELECT h.id,h.external_id,h.name,h.address,h.city,h.area,h.university,h.hostel_type,h.latitude,h.longitude,h.available_capacity,h.total_capacity,h.amenities,h.facilities,h.rules,h.mess_info,h.media,h.room_types_available,
        ${priceExpression} AS display_price, (${priceExpression} IS NOT NULL) AS price_available, 'PKR' AS currency,
        ROUND(AVG(rv.overall_rating)::numeric,2) AS average_rating,COUNT(rv.id)::int AS review_count,${distanceExpression} AS distance_km
        FROM hostels h LEFT JOIN reviews rv ON rv.hostel_id=h.id
        WHERE ${conditions.join(" AND ")}
        GROUP BY h.id ORDER BY ${order} LIMIT 100`, params);
      res.json({ count: result.rowCount, filters: { city: req.query.city || null, university: req.query.university || null, radius_km: latitude != null ? radius : null, room_type: roomType, min_price: minPrice, max_price: maxPrice, amenities }, hostels: result.rows });
    } catch (error) {
      res.status(500).json({ error: "Failed to search published hostels", ...(process.env.RUN_DB_TESTS === "1" ? { detail: error.message } : {}) });
    }
  });

  app.post("/api/voice/search", requireAuth, requireRole("student"), async (req, res) => {
    if (!String(req.body?.transcript || "").trim() || String(req.body.transcript).length > 1000) return res.status(400).json({ error: "A transcript of up to 1000 characters is required" });
    const parsed = parseTranscript(req.body.transcript);
    await pool.query("INSERT INTO hostel_search_history (user_id,source,query_text,filters) VALUES ($1,'voice',$2,$3::jsonb)", [req.user.id, parsed.transcript, JSON.stringify(parsed.filters)]);
    res.json(parsed);
  });

  app.post("/api/roommate-matches", requireAuth, requireRole("student"), async (req, res) => {
    const profile = await pool.query("SELECT * FROM student_profiles WHERE user_id=$1", [req.user.id]);
    if (!profile.rowCount || !profile.rows[0].roommate_profile_sharing) return res.status(409).json({ error: "Enable roommate profile sharing before requesting matches" });
    const hostelId = req.body?.hostel_id == null ? null : Number(req.body.hostel_id);
    if (hostelId != null && !isValidInt(hostelId)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    const candidates = await pool.query(`SELECT sp.user_id,u.name,sp.university,sp.department,sp.study_preference,sp.food_preference,
      ((CASE WHEN sp.university IS NOT DISTINCT FROM $2 THEN 1 ELSE 0 END)+(CASE WHEN sp.department IS NOT DISTINCT FROM $3 THEN 1 ELSE 0 END)+(CASE WHEN sp.food_preference IS NOT DISTINCT FROM $4 THEN 1 ELSE 0 END))::int AS compatibility_signals
      FROM student_profiles sp JOIN users u ON u.id=sp.user_id
      WHERE sp.user_id<>$1 AND sp.roommate_profile_sharing=true AND u.account_status='active'
      AND ($5::int IS NULL OR EXISTS (SELECT 1 FROM bookings b WHERE b.user_id=sp.user_id AND b.hostel_id=$5 AND b.status='confirmed'))
      ORDER BY compatibility_signals DESC,u.name LIMIT 10`, [req.user.id, profile.rows[0].university, profile.rows[0].department, profile.rows[0].food_preference, hostelId]);
    let matches=candidates.rows,provider="database_fallback",serviceUnavailable=false;
    const timeout=timeoutSignal(Number(process.env.AI_TIMEOUT_MS||5000));
    try{const response=await fetch(process.env.AI_ROOMMATE_URL||"http://127.0.0.1:8002/match",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({subject:profile.rows[0],candidates:candidates.rows,limit:10}),signal:timeout.signal});if(response.ok){const body=await response.json(),order=new Map((body.matches||[]).map((item,index)=>[Number(item.user_id),{index,signals:item.compatibility_signals}]));matches=candidates.rows.filter(item=>order.has(Number(item.user_id))).sort((a,b)=>order.get(Number(a.user_id)).index-order.get(Number(b.user_id)).index).map(item=>({...item,compatibility_signals:order.get(Number(item.user_id)).signals}));provider="roommate_service";}else serviceUnavailable=true;}catch(_){serviceUnavailable=true;}finally{timeout.clear();}
    res.json({ suggestions_only: true, allocation_requires_warden: true, provider, service_unavailable:serviceUnavailable, matches });
  });

  app.post("/api/complaints/:id/accept", requireAuth, requireRole("student"), async (req, res) => {
    const complaintId = Number(req.params.id);
    if (!isValidInt(complaintId)) return res.status(400).json({ error: "Invalid complaint id" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const current = await client.query("SELECT * FROM complaints WHERE id=$1 AND user_id=$2 FOR UPDATE", [complaintId,req.user.id]);
      if (!current.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Complaint not found" }); }
      if (current.rows[0].status !== "resolved") { await client.query("ROLLBACK"); return res.status(409).json({ error: "Only a resolved complaint can be accepted" }); }
      const updated = await client.query("UPDATE complaints SET status='closed',resolution_due_at=NULL WHERE id=$1 RETURNING *", [complaintId]);
      await client.query("INSERT INTO complaint_status_history (complaint_id,previous_status,next_status,changed_by_user_id) VALUES ($1,'resolved','closed',$2)", [complaintId,req.user.id]);
      await client.query("COMMIT"); res.json({ complaint: updated.rows[0] });
    } catch (_) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to accept resolution" }); } finally { client.release(); }
  });

  app.post("/api/complaints/:id/reopen", requireAuth, requireRole("student"), async (req, res) => {
    const complaintId = Number(req.params.id); const reason = String(req.body?.reason || "").trim();
    if (!isValidInt(complaintId) || !reason || reason.length > 2000) return res.status(400).json({ error: "A reopen reason of up to 2000 characters is required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const current = await client.query("SELECT * FROM complaints WHERE id=$1 AND user_id=$2 FOR UPDATE", [complaintId,req.user.id]);
      if (!current.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Complaint not found" }); }
      if (current.rows[0].status !== "resolved" || Number(current.rows[0].reopen_count) >= 1 || !current.rows[0].resolution_due_at || new Date(current.rows[0].resolution_due_at) <= new Date()) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Complaint cannot be reopened or the 72-hour window has expired" }); }
      const updated = await client.query("UPDATE complaints SET status='in_progress',resolved_at=NULL,resolution_due_at=NULL,reopened_at=NOW(),reopen_reason=$1,reopen_count=reopen_count+1 WHERE id=$2 RETURNING *", [reason,complaintId]);
      await client.query("INSERT INTO complaint_status_history (complaint_id,previous_status,next_status,changed_by_user_id) VALUES ($1,'resolved','in_progress',$2)", [complaintId,req.user.id]);
      await client.query("COMMIT"); res.json({ complaint: updated.rows[0] });
    } catch (_) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to reopen complaint" }); } finally { client.release(); }
  });

  app.patch("/api/owner/wardens/:id", requireAuth, requireRole("owner"), async (req, res) => {
    const wardenId = Number(req.params.id); const hostelId = Number(req.body.hostel_id); const status = req.body.account_status == null ? null : String(req.body.account_status).toLowerCase();
    if (!isValidInt(wardenId) || (req.body.hostel_id != null && !isValidInt(hostelId)) || (status && !["active","disabled"].includes(status))) return res.status(400).json({ error: "Valid warden, hostel and account status are required" });
    const owned = await pool.query("SELECT wa.hostel_id FROM warden_assignments wa JOIN hostel_owners ho ON ho.hostel_id=wa.hostel_id WHERE wa.warden_user_id=$1 AND ho.owner_user_id=$2", [wardenId, req.user.id]);
    if (!owned.rowCount) return res.status(404).json({ error: "Warden not found" });
    if (req.body.hostel_id != null && !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      if (req.body.hostel_id != null) await client.query("UPDATE warden_assignments SET hostel_id=$1,assigned_by_user_id=$2,assigned_at=NOW() WHERE warden_user_id=$3", [hostelId, req.user.id, wardenId]);
      if (status) {
        await client.query("UPDATE users SET account_status=$1 WHERE id=$2 AND role='warden'", [status, wardenId]);
        await client.query("UPDATE warden_profiles SET account_status=CASE WHEN $1='active' THEN 'Active' ELSE 'Disabled' END WHERE warden_user_id=$2", [status, wardenId]);
        if (status === "disabled") await client.query("UPDATE auth_sessions SET revoked_at=NOW() WHERE user_id=$1 AND revoked_at IS NULL", [wardenId]);
      }
      const result = await client.query("SELECT u.id,u.name,u.email,u.phone,u.account_status,wa.hostel_id FROM users u JOIN warden_assignments wa ON wa.warden_user_id=u.id WHERE u.id=$1", [wardenId]);
      await client.query("COMMIT"); res.json({ warden: result.rows[0] });
    } catch (_) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to update warden" }); } finally { client.release(); }
  });

  app.post("/api/owner/wardens/:id/reset-password", requireAuth, requireRole("owner"), async (req, res) => {
    const wardenId = Number(req.params.id);
    const allowed = await pool.query("SELECT 1 FROM warden_assignments wa JOIN hostel_owners ho ON ho.hostel_id=wa.hostel_id WHERE wa.warden_user_id=$1 AND ho.owner_user_id=$2", [wardenId, req.user.id]);
    if (!isValidInt(wardenId) || !allowed.rowCount) return res.status(404).json({ error: "Warden not found" });
    const temporaryPassword = `SB@${crypto.randomBytes(6).toString("hex").toUpperCase()}x`;
    await pool.query("UPDATE users SET password_hash=$1 WHERE id=$2 AND role='warden'", [await hashPassword(temporaryPassword), wardenId]);
    await pool.query("UPDATE auth_sessions SET revoked_at=NOW() WHERE user_id=$1 AND revoked_at IS NULL", [wardenId]);
    res.json({ success: true, temporary_password: temporaryPassword, returned_once: true });
  });
}

module.exports = { registerPhase3Routes, parseTranscript };
