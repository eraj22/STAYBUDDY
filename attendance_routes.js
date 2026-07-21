const crypto = require("crypto");

function hashCredential(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function timeInZone(date, timezone) {
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: timezone, hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(date);
  return `${parts.find((part) => part.type === "hour").value}:${parts.find((part) => part.type === "minute").value}`;
}

function registerAttendanceRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification }) {
  app.get("/api/owner/hostels/:id/attendance-settings", requireAuth, requireRole("owner"), async (req, res) => {
    const hostelId = Number(req.params.id);
    if (!isValidInt(hostelId) || !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query("INSERT INTO hostel_attendance_settings (hostel_id,updated_by_user_id) VALUES ($1,$2) ON CONFLICT (hostel_id) DO UPDATE SET hostel_id=EXCLUDED.hostel_id RETURNING *", [hostelId, req.user.id]);
    res.json({ settings: result.rows[0] });
  });

  app.put("/api/owner/hostels/:id/attendance-settings", requireAuth, requireRole("owner"), async (req, res) => {
    const hostelId = Number(req.params.id); const { curfew_time, timezone, enabled } = req.body;
    if (!isValidInt(hostelId) || !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (curfew_time != null && !/^([01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$/.test(String(curfew_time))) return res.status(400).json({ error: "curfew_time must be HH:MM or HH:MM:SS" });
    if (enabled != null && typeof enabled !== "boolean") return res.status(400).json({ error: "enabled must be boolean" });
    const result = await pool.query(`INSERT INTO hostel_attendance_settings (hostel_id,curfew_time,timezone,enabled,updated_by_user_id)
      VALUES ($1,COALESCE($2::time,'22:00'),COALESCE($3,'Asia/Karachi'),COALESCE($4,true),$5)
      ON CONFLICT (hostel_id) DO UPDATE SET curfew_time=COALESCE($2,hostel_attendance_settings.curfew_time), timezone=COALESCE($3,hostel_attendance_settings.timezone), enabled=COALESCE($4,hostel_attendance_settings.enabled), updated_by_user_id=$5, updated_at=NOW() RETURNING *`, [hostelId, curfew_time || null, timezone || null, enabled ?? null, req.user.id]);
    res.json({ settings: result.rows[0] });
  });

  app.post("/api/owner/hostels/:id/attendance-credentials", requireAuth, requireRole("owner"), async (req, res) => {
    const hostelId = Number(req.params.id); const label = String(req.body.label || "").trim();
    if (!isValidInt(hostelId) || !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (!label) return res.status(400).json({ error: "Credential label is required" });
    const credential = crypto.randomBytes(32).toString("base64url");
    const result = await pool.query("INSERT INTO attendance_integration_credentials (hostel_id,label,credential_hash,created_by_user_id) VALUES ($1,$2,$3,$4) RETURNING id,hostel_id,label,active,created_at", [hostelId, label, hashCredential(credential), req.user.id]);
    res.status(201).json({ credential: result.rows[0], integration_key: credential });
  });

  app.post("/api/warden/attendance/identifiers", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.body.hostel_id); const studentId = Number(req.body.student_id); const identifier = String(req.body.external_identifier || "").trim();
    if (!isValidInt(hostelId) || !isValidInt(studentId) || !identifier) return res.status(400).json({ error: "hostel_id, student_id, and external_identifier are required" });
    if (!await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const resident = await pool.query("SELECT 1 FROM bookings WHERE hostel_id=$1 AND user_id=$2 AND status IN ('pending','confirmed')", [hostelId, studentId]);
    if (resident.rowCount === 0) return res.status(409).json({ error: "Student has no active booking at this hostel" });
    const result = await pool.query(`INSERT INTO resident_attendance_identifiers (hostel_id,student_user_id,external_identifier,enrolled_by_user_id)
      VALUES ($1,$2,$3,$4) ON CONFLICT (hostel_id,external_identifier) DO UPDATE SET student_user_id=EXCLUDED.student_user_id, active=true, revoked_at=NULL, enrolled_by_user_id=EXCLUDED.enrolled_by_user_id, enrolled_at=NOW() RETURNING *`, [hostelId, studentId, identifier, req.user.id]);
    res.status(201).json({ identifier: result.rows[0] });
  });

  app.post("/api/attendance/events", async (req, res) => {
    const key = String(req.headers["x-attendance-key"] || ""); const idempotencyKey = String(req.headers["idempotency-key"] || "");
    const { external_identifier, direction, occurred_at, source, entry_point, external_event_id, metadata } = req.body;
    if (!key || !idempotencyKey || !external_identifier || !["in", "out"].includes(direction) || !["biometric", "rfid"].includes(source) || Number.isNaN(Date.parse(occurred_at))) return res.status(400).json({ error: "Valid scanner credentials, idempotency key, identifier, direction, source, and occurred_at are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const credential = await client.query("SELECT hostel_id FROM attendance_integration_credentials WHERE credential_hash=$1 AND active=true", [hashCredential(key)]);
      if (credential.rowCount === 0) { await client.query("ROLLBACK"); return res.status(401).json({ error: "Invalid attendance integration credential" }); }
      const hostelId = credential.rows[0].hostel_id;
      const resident = await client.query("SELECT student_user_id FROM resident_attendance_identifiers WHERE hostel_id=$1 AND external_identifier=$2 AND active=true", [hostelId, String(external_identifier)]);
      if (resident.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Active resident identifier not found" }); }
      const inserted = await client.query(`INSERT INTO attendance_events (hostel_id,student_user_id,direction,occurred_at,source,entry_point,external_event_id,idempotency_key,metadata)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb) ON CONFLICT DO NOTHING RETURNING *`, [hostelId, resident.rows[0].student_user_id, direction, occurred_at, source, entry_point || null, external_event_id || null, idempotencyKey, JSON.stringify(metadata || {})]);
      if (inserted.rowCount === 0) { const existing = await client.query("SELECT * FROM attendance_events WHERE idempotency_key=$1", [idempotencyKey]); await client.query("ROLLBACK"); return res.json({ event: existing.rows[0], duplicate: true }); }
      const event = inserted.rows[0];
      const settings = await client.query("SELECT curfew_time::text,timezone,enabled FROM hostel_attendance_settings WHERE hostel_id=$1", [hostelId]);
      if (settings.rowCount && settings.rows[0].enabled && timeInZone(new Date(event.occurred_at), settings.rows[0].timezone) >= settings.rows[0].curfew_time.slice(0, 5)) {
        const exceptionType = direction === "out" ? "late_exit" : "late_return";
        await client.query(`INSERT INTO attendance_exceptions (hostel_id,student_user_id,policy_date,exception_type,details)
          VALUES ($1,$2,($3::timestamptz AT TIME ZONE $4)::date,$5,$6::jsonb) ON CONFLICT DO NOTHING`, [hostelId, event.student_user_id, event.occurred_at, settings.rows[0].timezone, exceptionType, JSON.stringify({ attendance_event_id: event.id, occurred_at: event.occurred_at })]);
      }
      const parents = await client.query("SELECT parent_user_id FROM parent_student_links WHERE student_user_id=$1 AND status='active'", [event.student_user_id]);
      for (const parent of parents.rows) await createNotification(client, parent.parent_user_id, "attendance", `Student checked ${direction}`, `Your linked student checked ${direction} at ${entry_point || "the hostel"}.`, { attendance_event_id: event.id, student_id: event.student_user_id, hostel_id: hostelId, direction }, "attendance_updates");
      await client.query("COMMIT"); res.status(201).json({ event });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to record attendance event" }); } finally { client.release(); }
  });

  app.get("/api/warden/attendance", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.query.hostel_id); if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query(`SELECT e.*,u.name AS student_name FROM attendance_events e JOIN users u ON u.id=e.student_user_id WHERE e.hostel_id=$1 AND e.occurred_at >= COALESCE($2::timestamptz,NOW()-INTERVAL '1 day') AND e.occurred_at <= COALESCE($3::timestamptz,NOW()) ORDER BY e.occurred_at DESC`, [hostelId, req.query.from || null, req.query.to || null]);
    res.json({ events: result.rows });
  });

  app.post("/api/warden/attendance/evaluate-curfew", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.body.hostel_id); const asOf = req.body.as_of ? new Date(req.body.as_of) : new Date();
    if (!isValidInt(hostelId) || Number.isNaN(asOf.getTime())) return res.status(400).json({ error: "A valid hostel_id and optional as_of timestamp are required" });
    if (!await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const settings = await pool.query("SELECT curfew_time::text,timezone,enabled FROM hostel_attendance_settings WHERE hostel_id=$1", [hostelId]);
    if (settings.rowCount === 0 || !settings.rows[0].enabled) return res.status(409).json({ error: "Attendance curfew is not enabled for this hostel" });
    const setting = settings.rows[0];
    if (timeInZone(asOf, setting.timezone) < setting.curfew_time.slice(0, 5)) return res.status(409).json({ error: "Curfew time has not been reached" });
    const students = await pool.query("SELECT DISTINCT user_id FROM bookings WHERE hostel_id=$1 AND status='confirmed'", [hostelId]);
    const client = await pool.connect();
    try {
      await client.query("BEGIN"); let created = 0;
      for (const student of students.rows) {
        const latest = await client.query("SELECT direction,id FROM attendance_events WHERE hostel_id=$1 AND student_user_id=$2 AND occurred_at <= $3 ORDER BY occurred_at DESC,id DESC LIMIT 1", [hostelId, student.user_id, asOf]);
        if (latest.rowCount && latest.rows[0].direction === "out") {
          const exception = await client.query(`INSERT INTO attendance_exceptions (hostel_id,student_user_id,policy_date,exception_type,details)
            VALUES ($1,$2,($3::timestamptz AT TIME ZONE $4)::date,'absent_at_curfew',$5::jsonb) ON CONFLICT DO NOTHING RETURNING id`, [hostelId, student.user_id, asOf, setting.timezone, JSON.stringify({ last_attendance_event_id: latest.rows[0].id, evaluated_at: asOf.toISOString() })]);
          if (exception.rowCount) {
            created += 1;
            const parents = await client.query("SELECT parent_user_id FROM parent_student_links WHERE student_user_id=$1 AND status='active'", [student.user_id]);
            for (const parent of parents.rows) await createNotification(client, parent.parent_user_id, "attendance_exception", "Student absent at curfew", "Your linked student is recorded outside the hostel after curfew.", { attendance_exception_id: exception.rows[0].id, student_id: student.user_id, hostel_id: hostelId }, "attendance_updates");
          }
        }
      }
      await client.query("COMMIT"); res.json({ created_exceptions: created });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to evaluate curfew" }); } finally { client.release(); }
  });

  app.get("/api/parent/children/:studentId/attendance", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    const linked = await pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'", [req.user.id, studentId]);
    if (!isValidInt(studentId) || linked.rowCount === 0) return res.status(404).json({ error: "Linked student not found" });
    const result = await pool.query(`SELECT e.*,h.name AS hostel_name FROM attendance_events e JOIN hostels h ON h.id=e.hostel_id WHERE e.student_user_id=$1 AND e.occurred_at >= COALESCE($2::timestamptz,NOW()-INTERVAL '7 days') AND e.occurred_at <= COALESCE($3::timestamptz,NOW()) ORDER BY e.occurred_at DESC`, [studentId, req.query.from || null, req.query.to || null]);
    res.json({ events: result.rows });
  });

  app.post("/api/warden/attendance/events/:id/corrections", requireAuth, requireRole("warden"), async (req, res) => {
    const eventId = Number(req.params.id); const { direction, occurred_at, reason } = req.body;
    if (!Number.isInteger(eventId) || !["in", "out"].includes(direction) || Number.isNaN(Date.parse(occurred_at)) || !String(reason || "").trim()) return res.status(400).json({ error: "Valid direction, occurred_at, and reason are required" });
    const event = await pool.query("SELECT hostel_id FROM attendance_events WHERE id=$1", [eventId]);
    if (event.rowCount === 0 || !await wardenHasHostel(pool, req.user.id, event.rows[0].hostel_id)) return res.status(404).json({ error: "Attendance event not found" });
    const result = await pool.query("INSERT INTO attendance_corrections (attendance_event_id,corrected_direction,corrected_occurred_at,reason,corrected_by_user_id) VALUES ($1,$2,$3,$4,$5) RETURNING *", [eventId, direction, occurred_at, String(reason).trim(), req.user.id]);
    res.status(201).json({ correction: result.rows[0] });
  });
}

module.exports = { registerAttendanceRoutes };
