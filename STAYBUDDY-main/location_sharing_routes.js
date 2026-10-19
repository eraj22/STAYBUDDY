function registerLocationSharingRoutes({ app, pool, requireAuth, requireRole, isValidInt, createNotification }) {
  const durations = new Set([60, 480, 1440]);
  const sessionResponse = (row) => ({ id: row.id, student_user_id: row.student_user_id, started_at: row.started_at, expires_at: row.expires_at, revoked_at: row.revoked_at, history_enabled: row.history_enabled, geofencing_enabled: row.geofencing_enabled });
  async function activeLink(parentId, studentId) { const r = await pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'", [parentId, studentId]); return r.rowCount > 0; }
  function distanceMeters(latitudeOne, longitudeOne, latitudeTwo, longitudeTwo) {
    const radians = (degrees) => degrees * Math.PI / 180;
    const latitudeDelta = radians(latitudeTwo - latitudeOne);
    const longitudeDelta = radians(longitudeTwo - longitudeOne);
    const value = Math.sin(latitudeDelta / 2) ** 2 + Math.cos(radians(latitudeOne)) * Math.cos(radians(latitudeTwo)) * Math.sin(longitudeDelta / 2) ** 2;
    return 6371000 * 2 * Math.atan2(Math.sqrt(value), Math.sqrt(1 - value));
  }
  async function evaluateGeofences(client, session, update) {
    if (!session.geofencing_enabled) return [];
    const geofences = await client.query(
      `SELECT g.* FROM location_geofences g
       JOIN parent_student_links l ON l.parent_user_id=g.parent_user_id AND l.student_user_id=g.student_user_id AND l.status='active'
       WHERE g.student_user_id=$1 AND g.active=true`,
      [session.student_user_id]
    );
    const transitions = [];
    for (const geofence of geofences.rows) {
      const inside = distanceMeters(Number(update.latitude), Number(update.longitude), Number(geofence.latitude), Number(geofence.longitude)) <= Number(geofence.radius_m);
      const prior = await client.query("SELECT last_inside FROM location_geofence_states WHERE geofence_id=$1 AND session_id=$2 FOR UPDATE", [geofence.id, session.id]);
      const changed = prior.rowCount > 0 && prior.rows[0].last_inside !== inside;
      await client.query(
        `INSERT INTO location_geofence_states (geofence_id,session_id,last_inside,last_evaluated_at,last_transition_at)
         VALUES ($1,$2,$3,NOW(),CASE WHEN $4 THEN NOW() ELSE NULL END)
         ON CONFLICT (geofence_id,session_id) DO UPDATE SET
           last_inside=EXCLUDED.last_inside,
           last_evaluated_at=NOW(),
           last_transition_at=CASE WHEN $4 THEN NOW() ELSE location_geofence_states.last_transition_at END`,
        [geofence.id, session.id, inside, changed]
      );
      if (!changed) continue;
      const eventType = inside ? "entered" : "exited";
      const event = await client.query(
        `INSERT INTO location_geofence_events (geofence_id,session_id,location_update_id,event_type)
         VALUES ($1,$2,$3,$4) ON CONFLICT (geofence_id,location_update_id) DO NOTHING RETURNING *`,
        [geofence.id, session.id, update.id, eventType]
      );
      if (!event.rowCount) continue;
      await createNotification(client, geofence.parent_user_id, "location_geofence", `Geofence ${eventType}`, `Your linked child ${eventType} ${geofence.name}.`, { student_user_id: session.student_user_id, geofence_id: geofence.id, event_id: event.rows[0].id, event_type: eventType }, "location_updates");
      transitions.push({ ...event.rows[0], geofence_name: geofence.name });
    }
    return transitions;
  }
  async function expireSessions(client) {
    const expired = await client.query("UPDATE location_sharing_sessions SET revoked_at=NOW() WHERE revoked_at IS NULL AND expires_at<=NOW() RETURNING id,student_user_id");
    for (const row of expired.rows) await client.query("INSERT INTO location_sharing_audit_events (session_id,event_type,details) VALUES ($1,'expired',$2::jsonb)", [row.id, JSON.stringify({ student_user_id: row.student_user_id })]);
  }
  app.post("/api/student/location-sharing/sessions", requireAuth, requireRole("student"), async (req, res) => {
    const duration = Number(req.body.duration_minutes);
    const historyEnabled = req.body.history_enabled === true;
    const geofencingEnabled = req.body.geofencing_enabled === true;
    if (!durations.has(duration) || (req.body.history_enabled != null && typeof req.body.history_enabled !== "boolean") || (req.body.geofencing_enabled != null && typeof req.body.geofencing_enabled !== "boolean")) return res.status(400).json({ error: "duration_minutes must be 60, 480, or 1440 and consent flags must be boolean" });
    const client = await pool.connect();
    try { await client.query("BEGIN"); await expireSessions(client); const active = await client.query("SELECT id FROM location_sharing_sessions WHERE student_user_id=$1 AND revoked_at IS NULL FOR UPDATE", [req.user.id]); if (active.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Stop the current location-sharing session before starting another" }); }
      const created = await client.query("INSERT INTO location_sharing_sessions (student_user_id,expires_at,history_enabled,geofencing_enabled) VALUES ($1,NOW()+($2::int * INTERVAL '1 minute'),$3,$4) RETURNING *", [req.user.id, duration, historyEnabled, geofencingEnabled]);
      await client.query("INSERT INTO location_sharing_audit_events (session_id,event_type,actor_user_id,details) VALUES ($1,'consent_granted',$2,$3::jsonb)", [created.rows[0].id, req.user.id, JSON.stringify({ duration_minutes: duration, history_enabled: historyEnabled, geofencing_enabled: geofencingEnabled })]); await client.query("COMMIT"); res.status(201).json({ session: sessionResponse(created.rows[0]) });
    } catch (error) { console.error("Location-sharing session start failed:", error); await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to start location sharing" }); } finally { client.release(); }
  });
  app.post("/api/student/location-sharing/sessions/:id/updates", requireAuth, requireRole("student"), async (req, res) => {
    const id = Number(req.params.id), latitude = Number(req.body.latitude), longitude = Number(req.body.longitude), accuracy = req.body.accuracy_m == null ? null : Number(req.body.accuracy_m), key = String(req.headers["idempotency-key"] || "");
    if (!isValidInt(id) || !key || key.length > 160 || !Number.isFinite(latitude) || latitude < -90 || latitude > 90 || !Number.isFinite(longitude) || longitude < -180 || longitude > 180 || (accuracy != null && (!Number.isFinite(accuracy) || accuracy < 0))) return res.status(400).json({ error: "Valid coordinates, optional accuracy_m, and an idempotency-key are required" });
    const client = await pool.connect();
    try { await client.query("BEGIN"); await expireSessions(client); const session = await client.query("SELECT * FROM location_sharing_sessions WHERE id=$1 AND student_user_id=$2 AND revoked_at IS NULL AND expires_at>NOW() FOR UPDATE", [id, req.user.id]); if (!session.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Active location-sharing session not found" }); }
      const update = await client.query("INSERT INTO location_sharing_updates (session_id,latitude,longitude,accuracy_m,client_event_id) VALUES ($1,$2,$3,$4,$5) ON CONFLICT (session_id,client_event_id) DO NOTHING RETURNING *", [id, latitude, longitude, accuracy, key]);
      if (!update.rowCount) { const prior = await client.query("SELECT * FROM location_sharing_updates WHERE session_id=$1 AND client_event_id=$2", [id, key]); await client.query("COMMIT"); return res.json({ location: prior.rows[0], duplicate: true }); }
      await client.query("INSERT INTO location_sharing_audit_events (session_id,event_type,actor_user_id) VALUES ($1,'location_updated',$2)", [id, req.user.id]);
      const geofenceEvents = await evaluateGeofences(client, session.rows[0], update.rows[0]);
      await client.query("COMMIT"); res.status(201).json({ location: update.rows[0], geofence_events: geofenceEvents });
    } catch (_) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to record location" }); } finally { client.release(); }
  });
  app.post("/api/student/location-sharing/sessions/:id/revoke", requireAuth, requireRole("student"), async (req, res) => {
    const r = await pool.query("UPDATE location_sharing_sessions SET revoked_at=NOW(),revoked_by_user_id=$1 WHERE id=$2 AND student_user_id=$1 AND revoked_at IS NULL RETURNING *", [req.user.id, Number(req.params.id)]);
    if (!r.rowCount) return res.status(404).json({ error: "Active location-sharing session not found" }); await pool.query("INSERT INTO location_sharing_audit_events (session_id,event_type,actor_user_id) VALUES ($1,'revoked',$2)", [r.rows[0].id, req.user.id]); res.json({ session: sessionResponse(r.rows[0]) });
  });
  app.get("/api/student/location-sharing/sessions", requireAuth, requireRole("student"), async (req, res) => { const r = await pool.query("SELECT * FROM location_sharing_sessions WHERE student_user_id=$1 ORDER BY started_at DESC", [req.user.id]); res.json({ sessions: r.rows.map(sessionResponse) }); });
  app.get("/api/parent/children/:studentId/location", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId); if (!isValidInt(studentId) || !await activeLink(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    const session = await pool.query("SELECT * FROM location_sharing_sessions WHERE student_user_id=$1 AND revoked_at IS NULL AND expires_at>NOW() ORDER BY started_at DESC LIMIT 1", [studentId]); if (!session.rowCount) return res.status(404).json({ error: "No active location sharing session" });
    const location = await pool.query("SELECT latitude,longitude,accuracy_m,reported_at,EXTRACT(EPOCH FROM (NOW()-reported_at))::int AS age_seconds,(reported_at<NOW()-INTERVAL '5 minutes') AS stale FROM location_sharing_updates WHERE session_id=$1 ORDER BY reported_at DESC LIMIT 1", [session.rows[0].id]); if (!location.rowCount) return res.status(404).json({ error: "No location update available" });
    await pool.query("INSERT INTO location_sharing_audit_events (session_id,event_type,actor_user_id) VALUES ($1,'viewed',$2)", [session.rows[0].id, req.user.id]); res.json({ session: sessionResponse(session.rows[0]), location: location.rows[0], live_tracking_claimed: false });
  });
  app.get("/api/parent/children/:studentId/location/history", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId) || !await activeLink(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    const session = await pool.query("SELECT * FROM location_sharing_sessions WHERE student_user_id=$1 AND revoked_at IS NULL AND expires_at>NOW() AND history_enabled=true ORDER BY started_at DESC LIMIT 1", [studentId]);
    if (!session.rowCount) return res.status(404).json({ error: "No active location-history consent" });
    const limit = Math.min(250, Math.max(1, Number(req.query.limit) || 100));
    const history = await pool.query("SELECT id,latitude,longitude,accuracy_m,reported_at FROM location_sharing_updates WHERE session_id=$1 ORDER BY reported_at DESC LIMIT $2", [session.rows[0].id, limit]);
    await pool.query("INSERT INTO location_sharing_audit_events (session_id,event_type,actor_user_id,details) VALUES ($1,'viewed',$2,$3::jsonb)", [session.rows[0].id, req.user.id, JSON.stringify({ view: "history", count: history.rowCount })]);
    res.json({ session: sessionResponse(session.rows[0]), locations: history.rows });
  });
  app.get("/api/parent/children/:studentId/geofences", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId) || !await activeLink(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    const result = await pool.query("SELECT * FROM location_geofences WHERE parent_user_id=$1 AND student_user_id=$2 ORDER BY created_at DESC", [req.user.id, studentId]);
    res.json({ geofences: result.rows });
  });
  app.post("/api/parent/children/:studentId/geofences", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId), name = String(req.body.name || "").trim(), latitude = Number(req.body.latitude), longitude = Number(req.body.longitude), radius = Number(req.body.radius_m);
    if (!isValidInt(studentId) || !await activeLink(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    if (!name || name.length > 120 || !Number.isFinite(latitude) || latitude < -90 || latitude > 90 || !Number.isFinite(longitude) || longitude < -180 || longitude > 180 || !Number.isInteger(radius) || radius < 50 || radius > 5000) return res.status(400).json({ error: "A name, valid coordinates, and radius_m from 50 to 5000 are required" });
    const consent = await pool.query("SELECT id FROM location_sharing_sessions WHERE student_user_id=$1 AND revoked_at IS NULL AND expires_at>NOW() AND geofencing_enabled=true", [studentId]);
    if (!consent.rowCount) return res.status(409).json({ error: "The student has not enabled geofencing for an active sharing session" });
    const created = await pool.query("INSERT INTO location_geofences (student_user_id,parent_user_id,name,latitude,longitude,radius_m) VALUES ($1,$2,$3,$4,$5,$6) RETURNING *", [studentId, req.user.id, name, latitude, longitude, radius]);
    res.status(201).json({ geofence: created.rows[0] });
  });
  app.delete("/api/parent/children/:studentId/geofences/:id", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId), id = Number(req.params.id);
    if (!isValidInt(studentId) || !isValidInt(id) || !await activeLink(req.user.id, studentId)) return res.status(404).json({ error: "Geofence not found" });
    const disabled = await pool.query("UPDATE location_geofences SET active=false,disabled_at=NOW(),updated_at=NOW() WHERE id=$1 AND parent_user_id=$2 AND student_user_id=$3 AND active=true RETURNING *", [id, req.user.id, studentId]);
    if (!disabled.rowCount) return res.status(404).json({ error: "Geofence not found" });
    res.json({ geofence: disabled.rows[0] });
  });
  app.get("/api/parent/children/:studentId/geofence-events", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId) || !await activeLink(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    const consent = await pool.query("SELECT 1 FROM location_sharing_sessions WHERE student_user_id=$1 AND revoked_at IS NULL AND expires_at>NOW() AND geofencing_enabled=true", [studentId]);
    if (!consent.rowCount) return res.status(404).json({ error: "No active geofencing consent" });
    const result = await pool.query(`SELECT e.*,g.name AS geofence_name FROM location_geofence_events e JOIN location_geofences g ON g.id=e.geofence_id WHERE g.parent_user_id=$1 AND g.student_user_id=$2 ORDER BY e.occurred_at DESC LIMIT 100`, [req.user.id, studentId]);
    res.json({ events: result.rows });
  });
}
module.exports = { registerLocationSharingRoutes };
