const crypto = require("crypto");

function registerDiscoveryCompletionRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel }) {
  const tokenHash = (token) => crypto.createHash("sha256").update(token).digest("hex");
  async function managesHostel(user, hostelId) {
    if (user.role === "owner") return ownerHasHostel(pool, user.id, hostelId);
    if (user.role === "warden") return wardenHasHostel(pool, user.id, hostelId);
    return false;
  }
  async function ownedList(userId, listId) {
    const result = await pool.query("SELECT * FROM favorite_lists WHERE id=$1 AND user_id=$2", [listId, userId]);
    return result.rows[0] || null;
  }

  app.patch("/api/favorites/:id", requireAuth, requireRole("student"), async (req, res) => {
    const id = Number(req.params.id), note = req.body.note == null ? null : String(req.body.note).trim();
    if (!isValidInt(id) || (note != null && note.length > 500)) return res.status(400).json({ error: "A valid favorite id and note up to 500 characters are required" });
    const result = await pool.query("UPDATE favorites SET note=$1,updated_at=NOW() WHERE id=$2 AND user_id=$3 RETURNING *", [note || null, id, req.user.id]);
    if (!result.rowCount) return res.status(404).json({ error: "Favorite not found" });
    res.json({ favorite: result.rows[0] });
  });

  app.post("/api/favorite-lists", requireAuth, requireRole("student"), async (req, res) => {
    const name = String(req.body.name || "").trim();
    if (!name || name.length > 120) return res.status(400).json({ error: "A list name up to 120 characters is required" });
    try {
      const result = await pool.query("INSERT INTO favorite_lists (user_id,name) VALUES ($1,$2) RETURNING *", [req.user.id, name]);
      res.status(201).json({ list: result.rows[0] });
    } catch (error) {
      if (error.code === "23505") return res.status(409).json({ error: "A favorite list with this name already exists" });
      throw error;
    }
  });
  app.get("/api/favorite-lists", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query(
      `SELECT l.*,COALESCE(jsonb_agg(jsonb_build_object('favorite_id',f.id,'hostel_id',f.hostel_id,'note',f.note,'added_at',i.added_at,'hostel_name',h.name,'city',h.city)) FILTER (WHERE f.id IS NOT NULL),'[]'::jsonb) AS items
       FROM favorite_lists l LEFT JOIN favorite_list_items i ON i.list_id=l.id LEFT JOIN favorites f ON f.id=i.favorite_id LEFT JOIN hostels h ON h.id=f.hostel_id
       WHERE l.user_id=$1 GROUP BY l.id ORDER BY l.updated_at DESC`,
      [req.user.id]
    );
    res.json({ lists: result.rows });
  });
  app.post("/api/favorite-lists/:id/items", requireAuth, requireRole("student"), async (req, res) => {
    const listId = Number(req.params.id), favoriteId = Number(req.body.favorite_id);
    if (!isValidInt(listId) || !isValidInt(favoriteId) || !await ownedList(req.user.id, listId)) return res.status(404).json({ error: "Favorite list not found" });
    const favorite = await pool.query("SELECT id FROM favorites WHERE id=$1 AND user_id=$2", [favoriteId, req.user.id]);
    if (!favorite.rowCount) return res.status(404).json({ error: "Favorite not found" });
    const result = await pool.query("INSERT INTO favorite_list_items (list_id,favorite_id) VALUES ($1,$2) ON CONFLICT DO NOTHING RETURNING *", [listId, favoriteId]);
    if (!result.rowCount) return res.status(409).json({ error: "Favorite is already in this list" });
    await pool.query("UPDATE favorite_lists SET updated_at=NOW() WHERE id=$1", [listId]);
    res.status(201).json({ item: result.rows[0] });
  });
  app.delete("/api/favorite-lists/:id/items/:favoriteId", requireAuth, requireRole("student"), async (req, res) => {
    const listId = Number(req.params.id), favoriteId = Number(req.params.favoriteId);
    if (!isValidInt(listId) || !isValidInt(favoriteId) || !await ownedList(req.user.id, listId)) return res.status(404).json({ error: "Favorite list item not found" });
    const result = await pool.query("DELETE FROM favorite_list_items WHERE list_id=$1 AND favorite_id=$2 RETURNING *", [listId, favoriteId]);
    if (!result.rowCount) return res.status(404).json({ error: "Favorite list item not found" });
    res.json({ item: result.rows[0] });
  });
  app.post("/api/favorite-lists/:id/shares", requireAuth, requireRole("student"), async (req, res) => {
    const listId = Number(req.params.id), durationDays = Number(req.body.duration_days || 7);
    if (!isValidInt(listId) || !await ownedList(req.user.id, listId)) return res.status(404).json({ error: "Favorite list not found" });
    if (![1, 7, 30].includes(durationDays)) return res.status(400).json({ error: "duration_days must be 1, 7, or 30" });
    const token = crypto.randomBytes(32).toString("base64url"), id = crypto.randomUUID();
    const result = await pool.query("INSERT INTO favorite_share_links (id,list_id,token_hash,expires_at) VALUES ($1,$2,$3,NOW()+($4::int * INTERVAL '1 day')) RETURNING id,list_id,expires_at,created_at", [id, listId, tokenHash(token), durationDays]);
    res.status(201).json({ share: result.rows[0], share_token: token });
  });
  app.post("/api/favorite-shares/:id/revoke", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query("UPDATE favorite_share_links s SET revoked_at=NOW() FROM favorite_lists l WHERE s.id=$1 AND s.list_id=l.id AND l.user_id=$2 AND s.revoked_at IS NULL RETURNING s.id,s.revoked_at", [req.params.id, req.user.id]);
    if (!result.rowCount) return res.status(404).json({ error: "Active share not found" });
    res.json({ share: result.rows[0] });
  });
  app.get("/api/public/favorite-shares/:token", async (req, res) => {
    const share = await pool.query("SELECT s.id,l.name,l.user_id FROM favorite_share_links s JOIN favorite_lists l ON l.id=s.list_id WHERE s.token_hash=$1 AND s.revoked_at IS NULL AND s.expires_at>NOW()", [tokenHash(req.params.token)]);
    if (!share.rowCount) return res.status(404).json({ error: "Shared favorite list not found" });
    const items = await pool.query(
      `SELECT f.id AS favorite_id,f.note,h.id AS hostel_id,h.name,h.city,h.address,h.amenities,h.available_capacity,
              (SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id=h.id AND available_capacity>0) AS display_price
       FROM favorite_share_links s JOIN favorite_list_items i ON i.list_id=s.list_id JOIN favorites f ON f.id=i.favorite_id JOIN hostels h ON h.id=f.hostel_id
       WHERE s.id=$1 AND h.publication_status='published' ORDER BY i.added_at`, [share.rows[0].id]
    );
    const comments = await pool.query("SELECT c.id,c.favorite_id,c.author_name,c.comment,c.created_at FROM favorite_share_comments c WHERE c.share_id=$1 ORDER BY c.created_at", [share.rows[0].id]);
    res.json({ list: { id: share.rows[0].id, name: share.rows[0].name }, items: items.rows, comments: comments.rows });
  });
  app.post("/api/public/favorite-shares/:token/comments", async (req, res) => {
    const author = String(req.body.author_name || "").trim(), comment = String(req.body.comment || "").trim(), favoriteId = Number(req.body.favorite_id);
    if (!author || author.length > 120 || !comment || comment.length > 1000 || !isValidInt(favoriteId)) return res.status(400).json({ error: "Author, favorite_id, and comment up to 1000 characters are required" });
    const share = await pool.query("SELECT s.id,s.list_id FROM favorite_share_links s WHERE s.token_hash=$1 AND s.revoked_at IS NULL AND s.expires_at>NOW()", [tokenHash(req.params.token)]);
    if (!share.rowCount) return res.status(404).json({ error: "Shared favorite list not found" });
    const item = await pool.query("SELECT 1 FROM favorite_list_items WHERE list_id=$1 AND favorite_id=$2", [share.rows[0].list_id, favoriteId]);
    if (!item.rowCount) return res.status(404).json({ error: "Favorite not found in shared list" });
    const result = await pool.query("INSERT INTO favorite_share_comments (share_id,favorite_id,author_name,comment) VALUES ($1,$2,$3,$4) RETURNING *", [share.rows[0].id, favoriteId, author, comment]);
    res.status(201).json({ comment: result.rows[0] });
  });

  app.post("/api/student/availability-alerts", requireAuth, requireRole("student"), async (req, res) => {
    const university = String(req.body.university || "").trim() || null, city = String(req.body.city || "").trim() || null, hostelType = String(req.body.hostel_type || "").trim() || null;
    const maxPrice = req.body.max_price == null ? null : Number(req.body.max_price), radius = req.body.radius_km == null ? null : Number(req.body.radius_km), latitude = req.body.latitude == null ? null : Number(req.body.latitude), longitude = req.body.longitude == null ? null : Number(req.body.longitude);
    if (!university && !city && (latitude == null || longitude == null)) return res.status(400).json({ error: "University, city, or a coordinate pair is required" });
    if ((maxPrice != null && (!Number.isFinite(maxPrice) || maxPrice < 0)) || ((latitude == null) !== (longitude == null)) || ((latitude == null) !== (radius == null)) || (latitude != null && (!Number.isFinite(latitude) || latitude < -90 || latitude > 90 || !Number.isFinite(longitude) || longitude < -180 || longitude > 180 || !Number.isFinite(radius) || radius <= 0 || radius > 50))) return res.status(400).json({ error: "Valid optional price and location radius up to 50 km are required" });
    const result = await pool.query("INSERT INTO hostel_availability_alerts (student_user_id,university,city,hostel_type,max_price,radius_km,latitude,longitude) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING *", [req.user.id, university, city, hostelType, maxPrice, radius, latitude, longitude]);
    res.status(201).json({ alert: result.rows[0] });
  });
  app.get("/api/student/availability-alerts", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query("SELECT a.*,COUNT(m.hostel_id)::int AS matched_hostels FROM hostel_availability_alerts a LEFT JOIN hostel_availability_alert_matches m ON m.alert_id=a.id WHERE a.student_user_id=$1 GROUP BY a.id ORDER BY a.created_at DESC", [req.user.id]);
    res.json({ alerts: result.rows });
  });
  app.delete("/api/student/availability-alerts/:id", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query("UPDATE hostel_availability_alerts SET active=false,disabled_at=NOW() WHERE id=$1 AND student_user_id=$2 AND active=true RETURNING *", [Number(req.params.id), req.user.id]);
    if (!result.rowCount) return res.status(404).json({ error: "Active alert not found" });
    res.json({ alert: result.rows[0] });
  });

  app.post("/api/staff/hostels/:hostelId/nearby-places", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const hostelId = Number(req.params.hostelId), category = String(req.body.category || ""), name = String(req.body.name || "").trim(), latitude = Number(req.body.latitude), longitude = Number(req.body.longitude), distance = req.body.distance_m == null ? null : Number(req.body.distance_m);
    if (!isValidInt(hostelId) || !await managesHostel(req.user, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (!["transport", "market", "hospital", "restaurant", "education", "other"].includes(category) || !name || name.length > 180 || !Number.isFinite(latitude) || latitude < -90 || latitude > 90 || !Number.isFinite(longitude) || longitude < -180 || longitude > 180 || (distance != null && (!Number.isInteger(distance) || distance < 0))) return res.status(400).json({ error: "Valid category, name, coordinates, and optional distance_m are required" });
    const result = await pool.query("INSERT INTO hostel_nearby_places (hostel_id,category,name,latitude,longitude,distance_m,created_by_user_id) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *", [hostelId, category, name, latitude, longitude, distance, req.user.id]);
    res.status(201).json({ place: result.rows[0] });
  });
  app.get("/api/hostels/:hostelId/nearby-places", async (req, res) => {
    const hostelId = Number(req.params.hostelId), category = req.query.category ? String(req.query.category) : null;
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    const hostel = await pool.query("SELECT 1 FROM hostels WHERE id=$1 AND publication_status='published'", [hostelId]);
    if (!hostel.rowCount) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query("SELECT id,category,name,latitude,longitude,distance_m FROM hostel_nearby_places WHERE hostel_id=$1 AND active=true AND ($2::varchar IS NULL OR category=$2) ORDER BY category,distance_m NULLS LAST,name", [hostelId, category]);
    res.json({ places: result.rows });
  });

  app.post("/api/search-history", requireAuth, requireRole("student"), async (req, res) => {
    const query = req.body.query_text == null ? null : String(req.body.query_text).trim(), filters = req.body.filters ?? {}, count = req.body.result_count == null ? null : Number(req.body.result_count);
    if ((query != null && query.length > 1000) || typeof filters !== "object" || Array.isArray(filters) || (count != null && (!Number.isInteger(count) || count < 0))) return res.status(400).json({ error: "Valid query, filter object, and result count are required" });
    const result = await pool.query("INSERT INTO hostel_search_history (user_id,source,query_text,filters,result_count) VALUES ($1,'text',$2,$3::jsonb,$4) RETURNING *", [req.user.id, query || null, JSON.stringify(filters), count]);
    res.status(201).json({ search: result.rows[0] });
  });
  app.get("/api/search-history", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query("SELECT * FROM hostel_search_history WHERE user_id=$1 ORDER BY created_at DESC LIMIT 100", [req.user.id]);
    res.json({ searches: result.rows });
  });
}

module.exports = { registerDiscoveryCompletionRoutes };
