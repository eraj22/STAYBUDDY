const crypto = require("crypto");
const { policyResult, countConsecutiveLateDays, applyAttendanceConsequences } = require("./posted_gate_attendance_routes");

function hashCredential(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

async function applyEventConsequences(client, event, createNotification) {
  const configured = await client.query("SELECT * FROM hostel_attendance_settings WHERE hostel_id=$1", [event.hostel_id]);
  const settings = configured.rows[0] || {
    enabled: false, timezone: "Asia/Karachi", earliest_exit_time: "07:00", curfew_time: "21:00",
    grace_minutes: 10, escalation_minutes: 30, consecutive_warning_days: 3, attendance_fine_amount: 0,
  };
  const result = await applyAttendanceConsequences(client, event, settings, createNotification);
  return { late: Boolean(result.kind), policy: result };
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

  app.get("/api/owner/hostels/:id/attendance-credentials", requireAuth, requireRole("owner"), async (req, res) => {
    const hostelId = Number(req.params.id);
    if (!isValidInt(hostelId) || !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query("SELECT id,hostel_id,label,active,created_at,revoked_at FROM attendance_integration_credentials WHERE hostel_id=$1 ORDER BY created_at DESC,id DESC", [hostelId]);
    res.json({ credentials: result.rows });
  });

  app.delete("/api/owner/hostels/:id/attendance-credentials/:credentialId", requireAuth, requireRole("owner"), async (req, res) => {
    const hostelId = Number(req.params.id), credentialId = Number(req.params.credentialId);
    if (!isValidInt(hostelId) || !isValidInt(credentialId) || !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Attendance credential not found" });
    const result = await pool.query(`UPDATE attendance_integration_credentials SET active=false,revoked_at=COALESCE(revoked_at,NOW())
      WHERE id=$1 AND hostel_id=$2 RETURNING id,hostel_id,label,active,created_at,revoked_at`, [credentialId, hostelId]);
    if (!result.rowCount) return res.status(404).json({ error: "Attendance credential not found" });
    res.json({ credential: result.rows[0] });
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
      if (inserted.rowCount === 0) { const existing = await client.query("SELECT * FROM attendance_events WHERE hostel_id=$1 AND source=$2 AND idempotency_key=$3", [hostelId,source,idempotencyKey]); await client.query("ROLLBACK"); return res.json({ event: existing.rows[0], duplicate: true }); }
      const event = inserted.rows[0];
      const consequences = await applyEventConsequences(client, event, createNotification);
      await client.query("COMMIT"); res.status(201).json({ event, ...consequences });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to record attendance event" }); } finally { client.release(); }
  });

  app.post("/api/warden/attendance/events", requireAuth, requireRole("warden"), async (req,res)=>{
    const hostelId=Number(req.body.hostel_id),studentId=Number(req.body.student_id),direction=String(req.body.direction||""),occurredAt=req.body.occurred_at||new Date().toISOString(),idempotencyKey=String(req.headers["idempotency-key"]||"").trim(),entryPoint=String(req.body.entry_point||"Manual desk entry").trim();
    if(!isValidInt(hostelId)||!isValidInt(studentId)||!["in","out"].includes(direction)||Number.isNaN(Date.parse(occurredAt))||!idempotencyKey||idempotencyKey.length>160)return res.status(400).json({error:"hostel_id, student_id, direction, occurred_at, and Idempotency-Key are required"});
    if(!await wardenHasHostel(pool,req.user.id,hostelId))return res.status(404).json({error:"Hostel not found"});
    const resident=await pool.query(`SELECT 1 FROM bookings b WHERE b.hostel_id=$1 AND b.user_id=$2 AND b.status='confirmed' AND b.room_id IS NOT NULL AND NULLIF(TRIM(COALESCE(b.bed_number,'')),'') IS NOT NULL`,[hostelId,studentId]);
    if(!resident.rowCount)return res.status(409).json({error:"Student does not have a confirmed room and bed assignment at this hostel"});
    const client=await pool.connect();
    try{
      await client.query("BEGIN");
      const latest=await client.query(`SELECT COALESCE(c.corrected_direction,e.direction) AS direction FROM attendance_events e LEFT JOIN LATERAL(SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true WHERE e.hostel_id=$1 AND e.student_user_id=$2 ORDER BY COALESCE(c.corrected_occurred_at,e.occurred_at) DESC,e.id DESC LIMIT 1`,[hostelId,studentId]);
      if(latest.rowCount && latest.rows[0].direction===direction){await client.query("ROLLBACK");return res.status(409).json({error:`Student is already recorded ${direction.toUpperCase()}`});}
      const result=await client.query(`INSERT INTO attendance_events (hostel_id,student_user_id,direction,occurred_at,source,entry_point,idempotency_key,metadata,created_by_user_id) VALUES ($1,$2,$3,$4,'manual',$5,$6,$7::jsonb,$8) ON CONFLICT DO NOTHING RETURNING *`,[hostelId,studentId,direction,occurredAt,entryPoint,idempotencyKey,JSON.stringify({manual:true}),req.user.id]);
      if(!result.rowCount){const existing=await client.query("SELECT * FROM attendance_events WHERE hostel_id=$1 AND source='manual' AND idempotency_key=$2",[hostelId,idempotencyKey]);await client.query("ROLLBACK");return res.json({event:existing.rows[0],duplicate:true});}
      const consequences=await applyEventConsequences(client,result.rows[0],createNotification);
      await client.query("COMMIT");res.status(201).json({event:result.rows[0],duplicate:false,...consequences});
    }catch(error){await client.query("ROLLBACK");res.status(500).json({error:"Failed to record manual attendance event"});}finally{client.release();}
  });

  // Daily roll-call attendance is separate from gate in/out events.  It lets
  // wardens mark every active resident for a chosen calendar date.
  app.get("/api/warden/daily-attendance", requireAuth, requireRole("warden"), async (req, res) => {
    const date = String(req.query.date || '').trim();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return res.status(400).json({ error: 'A valid date is required' });
    const result = await pool.query(
      `SELECT u.id AS student_id, u.name, r.room_number, COALESCE(r.floor_label, 'Ground Floor') AS floor_label, b.bed_number,
              COALESCE(a.status, 'absent') AS status, (a.id IS NOT NULL) AS saved
         FROM warden_assignments wa
         JOIN bookings b ON b.hostel_id=wa.hostel_id AND b.status='confirmed'
         JOIN users u ON u.id=b.user_id AND u.role='student'
         LEFT JOIN rooms r ON r.id=b.room_id
         LEFT JOIN daily_warden_attendance a
           ON a.hostel_id=wa.hostel_id AND a.student_user_id=u.id AND a.attendance_date=$2::date
        WHERE wa.warden_user_id=$1
        ORDER BY u.name`,
      [req.user.id, date],
    );
    res.json(result.rows);
  });

  app.post("/api/warden/daily-attendance", requireAuth, requireRole("warden"), async (req, res) => {
    const date = String(req.body.date || '').trim();
    const records = Array.isArray(req.body.records) ? req.body.records : [];
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !records.length) return res.status(400).json({ error: 'A date and attendance records are required' });
    const hostel = await pool.query('SELECT hostel_id FROM warden_assignments WHERE warden_user_id=$1 LIMIT 1', [req.user.id]);
    if (!hostel.rowCount) return res.status(404).json({ error: 'Assigned hostel not found' });
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      for (const record of records) {
        const studentId = Number(record.student_id);
        const status = String(record.status || '');
        if (!isValidInt(studentId) || !['present', 'absent'].includes(status)) throw new Error('Invalid attendance record');
        const resident = await client.query("SELECT 1 FROM bookings WHERE hostel_id=$1 AND user_id=$2 AND status='confirmed'", [hostel.rows[0].hostel_id, studentId]);
        if (!resident.rowCount) throw new Error('A listed student is not assigned to this hostel');
        await client.query(
          `INSERT INTO daily_warden_attendance (hostel_id,student_user_id,attendance_date,status,marked_by_warden_id)
           VALUES ($1,$2,$3::date,$4,$5)
           ON CONFLICT (hostel_id,student_user_id,attendance_date)
           DO UPDATE SET status=EXCLUDED.status,marked_by_warden_id=EXCLUDED.marked_by_warden_id,updated_at=NOW()`,
          [hostel.rows[0].hostel_id, studentId, date, status, req.user.id],
        );
      }
      await client.query('COMMIT');
      res.json({ success: true, date, saved: records.length });
    } catch (error) { await client.query('ROLLBACK'); res.status(400).json({ error: error.message || 'Failed to save attendance' }); }
    finally { client.release(); }
  });

  app.get("/api/student/attendance", requireAuth, requireRole("student"), async (req, res) => {
    const hostelId = req.query.hostel_id == null ? null : Number(req.query.hostel_id);
    const to = req.query.to == null ? new Date() : new Date(req.query.to);
    const from = req.query.from == null ? new Date(to.getTime() - 7 * 86400000) : new Date(req.query.from);
    if ((hostelId != null && !isValidInt(hostelId)) || Number.isNaN(from.getTime()) || Number.isNaN(to.getTime()) ||
      from >= to || to.getTime() - from.getTime() > 366 * 86400000)
      return res.status(400).json({ error: "Valid hostel_id and a date range of up to 366 days are required" });
    const result = await pool.query(`SELECT e.id,e.hostel_id,h.name AS hostel_name,e.direction,e.occurred_at,e.source,e.entry_point,
      COALESCE(c.corrected_direction,e.direction) AS effective_direction,
      COALESCE(c.corrected_occurred_at,e.occurred_at) AS effective_occurred_at,
      (c.id IS NOT NULL) AS corrected,c.reason AS correction_reason
      FROM attendance_events e JOIN hostels h ON h.id=e.hostel_id
      LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
      WHERE e.student_user_id=$1 AND ($2::int IS NULL OR e.hostel_id=$2)
        AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$3
        AND COALESCE(c.corrected_occurred_at,e.occurred_at)<=$4
      ORDER BY effective_occurred_at DESC,e.id DESC LIMIT 501`, [req.user.id, hostelId, from, to]);
    const stays = await pool.query(`SELECT DISTINCT b.hostel_id,h.name AS hostel_name,l.direction,l.occurred_at,l.source
      FROM bookings b JOIN booking_stays s ON s.booking_id=b.id JOIN hostels h ON h.id=b.hostel_id
      LEFT JOIN LATERAL (SELECT COALESCE(c.corrected_direction,e.direction) AS direction,
        COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at,e.source
        FROM attendance_events e LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac
          WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
        WHERE e.hostel_id=b.hostel_id AND e.student_user_id=$1 AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=s.checked_in_at
        ORDER BY occurred_at DESC,e.id DESC LIMIT 1)l ON true
      WHERE b.user_id=$1 AND b.status='confirmed' AND s.checked_in_at IS NOT NULL AND s.checked_out_at IS NULL
        AND ($2::int IS NULL OR b.hostel_id=$2) ORDER BY b.hostel_id`, [req.user.id, hostelId]);
    res.json({ events: result.rows.slice(0, 500), truncated: result.rows.length > 500,
      current_statuses: stays.rows.map(stay => ({ ...stay, status: stay.direction === 'in' ? 'inside' : stay.direction === 'out' ? 'outside' : 'unknown' })) });
  });

  app.get("/api/warden/attendance", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.query.hostel_id); if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query(`SELECT e.*,COALESCE(c.corrected_direction,e.direction) effective_direction,COALESCE(c.corrected_occurred_at,e.occurred_at) effective_occurred_at,(c.id IS NOT NULL) corrected,c.reason correction_reason,u.name student_name FROM attendance_events e JOIN users u ON u.id=e.student_user_id LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1) c ON true WHERE e.hostel_id=$1 AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=COALESCE($2::timestamptz,NOW()-INTERVAL '1 day') AND COALESCE(c.corrected_occurred_at,e.occurred_at)<=COALESCE($3::timestamptz,NOW()) ORDER BY effective_occurred_at DESC,e.id DESC`, [hostelId, req.query.from || null, req.query.to || null]);
    res.json({ events: result.rows });
  });

  app.get("/api/warden/attendance/daily-summary", requireAuth, requireRole("warden"), async (req,res)=>{
    const hostelId=Number(req.query.hostel_id),date=String(req.query.date||new Date().toISOString().slice(0,10));
    if(!isValidInt(hostelId)||!/^\d{4}-\d{2}-\d{2}$/.test(date)||!await wardenHasHostel(pool,req.user.id,hostelId))return res.status(404).json({error:"Hostel or date not found"});
    const counts=await pool.query(`WITH effective AS (SELECT e.student_user_id,e.id,COALESCE(c.corrected_direction,e.direction) direction,COALESCE(c.corrected_occurred_at,e.occurred_at) occurred_at,(c.id IS NOT NULL) corrected FROM attendance_events e LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true WHERE e.hostel_id=$1),daily AS (SELECT * FROM effective WHERE occurred_at::date=$2::date),latest AS (SELECT DISTINCT ON(student_user_id) student_user_id,direction FROM effective WHERE occurred_at<($2::date+1) ORDER BY student_user_id,occurred_at DESC,id DESC) SELECT COUNT(*) FILTER(WHERE latest.direction='in')::int present,COUNT(*) FILTER(WHERE latest.direction='out')::int outside,(SELECT COUNT(*) FROM attendance_exceptions WHERE hostel_id=$1 AND policy_date=$2::date AND exception_type='late_return')::int late_return,(SELECT COUNT(*) FROM attendance_exceptions WHERE hostel_id=$1 AND policy_date=$2::date AND exception_type='late_exit')::int late_exit,(SELECT COUNT(*) FROM attendance_exceptions WHERE hostel_id=$1 AND policy_date=$2::date AND exception_type='absent_at_curfew')::int absent_at_curfew,(SELECT COUNT(*) FROM daily WHERE corrected)::int corrected_event_count FROM latest`,[hostelId,date]);
    res.json({date,hostel_id:hostelId,...counts.rows[0]});
  });

  app.post("/api/warden/attendance/evaluate-curfew", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.body.hostel_id); const asOf = req.body.as_of ? new Date(req.body.as_of) : new Date();
    if (!isValidInt(hostelId) || Number.isNaN(asOf.getTime())) return res.status(400).json({ error: "A valid hostel_id and optional as_of timestamp are required" });
    if (!await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const settings = await pool.query(`SELECT timezone,enabled,
      ((($2::timestamptz AT TIME ZONE timezone)::date + curfew_time) AT TIME ZONE timezone) AS curfew_at
      FROM hostel_attendance_settings WHERE hostel_id=$1`, [hostelId, asOf]);
    if (settings.rowCount === 0 || !settings.rows[0].enabled) return res.status(409).json({ error: "Attendance curfew is not enabled for this hostel" });
    const setting = settings.rows[0];
    if (asOf < new Date(setting.curfew_at)) return res.status(409).json({ error: "Curfew time has not been reached" });
    const students = await pool.query(`SELECT b.user_id,MAX(s.checked_in_at) AS checked_in_at FROM bookings b
      JOIN booking_stays s ON s.booking_id=b.id WHERE b.hostel_id=$1 AND b.status='confirmed'
      AND s.checked_in_at IS NOT NULL AND s.checked_out_at IS NULL GROUP BY b.user_id`, [hostelId]);
    const client = await pool.connect();
    try {
      await client.query("BEGIN"); let created = 0;
      for (const student of students.rows) {
        const latest = await client.query(`SELECT e.id,COALESCE(c.corrected_direction,e.direction) direction,
          COALESCE(c.corrected_occurred_at,e.occurred_at) occurred_at FROM attendance_events e
          LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
          WHERE e.hostel_id=$1 AND e.student_user_id=$2 AND COALESCE(c.corrected_occurred_at,e.occurred_at)<=$3
            AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$4 ORDER BY occurred_at DESC,e.id DESC LIMIT 1`,
        [hostelId, student.user_id, setting.curfew_at, student.checked_in_at]);
        if (latest.rowCount && latest.rows[0].direction === "out") {
          const current = await client.query(`SELECT COALESCE(c.corrected_direction,e.direction) AS direction FROM attendance_events e
            LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
            WHERE e.hostel_id=$1 AND e.student_user_id=$2 AND COALESCE(c.corrected_occurred_at,e.occurred_at)<=$3
              AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$4
            ORDER BY COALESCE(c.corrected_occurred_at,e.occurred_at) DESC,e.id DESC LIMIT 1`,
          [hostelId, student.user_id, asOf, student.checked_in_at]);
          if (!current.rowCount || current.rows[0].direction !== "out") continue;
          const exception = await client.query(`INSERT INTO attendance_exceptions (hostel_id,student_user_id,policy_date,exception_type,details)
            VALUES ($1,$2,($3::timestamptz AT TIME ZONE $4)::date,'absent_at_curfew',$5::jsonb) ON CONFLICT DO NOTHING RETURNING id`, [hostelId, student.user_id, setting.curfew_at, setting.timezone, JSON.stringify({ last_attendance_event_id: latest.rows[0].id, evaluated_at: asOf.toISOString() })]);
          if (exception.rowCount) {
            created += 1;
            const parents = await client.query(`SELECT l.parent_user_id FROM parent_student_links l JOIN student_attendance_notification_permissions permission ON permission.parent_student_link_id=l.id AND permission.allowed=true LEFT JOIN parent_attendance_preferences p ON p.parent_user_id=l.parent_user_id AND p.student_user_id=l.student_user_id WHERE l.student_user_id=$1 AND l.status='active' AND COALESCE(p.notify_curfew,true)`, [student.user_id]);
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
    const result = await pool.query(`SELECT e.*,COALESCE(c.corrected_direction,e.direction) effective_direction,COALESCE(c.corrected_occurred_at,e.occurred_at) effective_occurred_at,(c.id IS NOT NULL) corrected,h.name hostel_name FROM attendance_events e JOIN hostels h ON h.id=e.hostel_id LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true WHERE e.student_user_id=$1 AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=COALESCE($2::timestamptz,NOW()-INTERVAL '7 days') AND COALESCE(c.corrected_occurred_at,e.occurred_at)<=COALESCE($3::timestamptz,NOW()) ORDER BY effective_occurred_at DESC,e.id DESC`, [studentId, req.query.from || null, req.query.to || null]);
    res.json({ events: result.rows });
  });

  app.post("/api/warden/attendance/events/:id/corrections", requireAuth, requireRole("warden"), async (req, res) => {
    const eventId = Number(req.params.id); const { direction, occurred_at, reason } = req.body;
    if (!Number.isInteger(eventId) || !["in", "out"].includes(direction) || Number.isNaN(Date.parse(occurred_at)) || !String(reason || "").trim()) return res.status(400).json({ error: "Valid direction, occurred_at, and reason are required" });
    const event = await pool.query("SELECT * FROM attendance_events WHERE id=$1", [eventId]);
    if (event.rowCount === 0 || !await wardenHasHostel(pool, req.user.id, event.rows[0].hostel_id)) return res.status(404).json({ error: "Attendance event not found" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query("INSERT INTO attendance_corrections (attendance_event_id,corrected_direction,corrected_occurred_at,reason,corrected_by_user_id) VALUES ($1,$2,$3,$4,$5) RETURNING *", [eventId, direction, occurred_at, String(reason).trim(), req.user.id]);
      const settings = await client.query("SELECT * FROM hostel_attendance_settings WHERE hostel_id=$1", [event.rows[0].hostel_id]);
      const classification = settings.rowCount && settings.rows[0].enabled
        ? policyResult(new Date(occurred_at), direction, settings.rows[0]) : null;
      const old = await client.query(`SELECT id,exception_type,policy_date::text AS policy_date FROM attendance_exceptions
        WHERE hostel_id=$1 AND details->>'attendance_event_id'=$2 AND exception_type IN ('early_exit','late_exit','late_return')
        FOR UPDATE`, [event.rows[0].hostel_id, String(eventId)]);
      for (const item of old.rows) {
        if (classification?.kind === item.exception_type && (!classification.policyDate || classification.policyDate === item.policy_date)) continue;
        await client.query("UPDATE attendance_exceptions SET status='resolved',resolved_by_user_id=$2,resolved_at=NOW() WHERE id=$1", [item.id, req.user.id]);
        await client.query(`UPDATE attendance_disciplinary_actions SET status='waived',changed_by_user_id=$2,
          changed_at=NOW(),change_reason='Underlying attendance event was corrected'
          WHERE exception_id=$1 AND status='issued'`, [item.id, req.user.id]);
      }
      if (classification?.kind) {
        const policyDateSql = classification.policyDate ? "$2::date" : "($3::timestamptz AT TIME ZONE $4)::date";
        await client.query(`INSERT INTO attendance_exceptions(hostel_id,student_user_id,policy_date,exception_type,details)
          SELECT hostel_id,student_user_id,${policyDateSql},$5,$6::jsonb FROM attendance_events WHERE id=$1
          ON CONFLICT(hostel_id,student_user_id,policy_date,exception_type) DO UPDATE SET
          status='open',resolved_at=NULL,resolved_by_user_id=NULL,details=EXCLUDED.details`,
        [eventId, classification.policyDate, occurred_at, settings.rows[0].timezone, classification.kind,
          JSON.stringify({ attendance_event_id: eventId, corrected: true, occurred_at })]);
      }
      if (settings.rowCount && settings.rows[0].enabled) {
        const curfewExceptions = await client.query(`SELECT id,
          ((policy_date + $3::time) AT TIME ZONE $4) AS curfew_at
          FROM attendance_exceptions WHERE hostel_id=$1 AND student_user_id=$2
            AND exception_type='absent_at_curfew' AND status='open'`,
        [event.rows[0].hostel_id, event.rows[0].student_user_id, settings.rows[0].curfew_time, settings.rows[0].timezone]);
        for (const exception of curfewExceptions.rows) {
          const latestAtCurfew = await client.query(`SELECT COALESCE(c.corrected_direction,e.direction) AS direction
            FROM attendance_events e LEFT JOIN LATERAL(SELECT * FROM attendance_corrections ac
              WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
            WHERE e.hostel_id=$1 AND e.student_user_id=$2
              AND COALESCE(c.corrected_occurred_at,e.occurred_at)<=$3
            ORDER BY COALESCE(c.corrected_occurred_at,e.occurred_at) DESC,e.id DESC LIMIT 1`,
          [event.rows[0].hostel_id, event.rows[0].student_user_id, exception.curfew_at]);
          if (!latestAtCurfew.rowCount || latestAtCurfew.rows[0].direction !== 'out')
            await client.query("UPDATE attendance_exceptions SET status='resolved',resolved_by_user_id=$2,resolved_at=NOW() WHERE id=$1", [exception.id, req.user.id]);
        }
      }
      const actions = await client.query(`SELECT a.id,a.action_type,a.status,x.status AS exception_status,x.policy_date::text AS policy_date
        FROM attendance_disciplinary_actions a JOIN attendance_exceptions x ON x.id=a.exception_id
        WHERE a.hostel_id=$1 AND a.student_user_id=$2 AND a.status IN ('issued','paid')
        ORDER BY x.policy_date,a.id FOR UPDATE OF a`, [event.rows[0].hostel_id, event.rows[0].student_user_id]);
      const autoWaived = [], paidNeedsReview = [];
      for (const action of actions.rows) {
        const streak = action.exception_status === 'open'
          ? await countConsecutiveLateDays(client, event.rows[0].hostel_id, event.rows[0].student_user_id, action.policy_date) : 0;
        const required = Number(settings.rows[0]?.consecutive_warning_days || 3);
        const valid = action.action_type === 'warning' ? streak >= 1 : streak > required;
        if (valid) continue;
        if (action.status === 'paid') {
          paidNeedsReview.push(action.id);
          continue;
        }
        await client.query(`UPDATE attendance_disciplinary_actions SET status='waived',changed_by_user_id=$2,
          changed_at=NOW(),change_reason='Attendance correction invalidated the consecutive-day streak'
          WHERE id=$1`, [action.id, req.user.id]);
        autoWaived.push(action.id);
      }
      await client.query("INSERT INTO attendance_audit_events (hostel_id,student_user_id,attendance_event_id,event_type,actor_user_id,details) VALUES ($1,$2,$3,'attendance_corrected',$4,$5::jsonb)", [event.rows[0].hostel_id, event.rows[0].student_user_id, eventId, req.user.id, JSON.stringify({ corrected_direction: direction, corrected_occurred_at: occurred_at, reason: String(reason).trim() })]);
      for (const actionId of autoWaived) {
        await client.query("INSERT INTO attendance_audit_events(hostel_id,student_user_id,event_type,actor_user_id,details) VALUES($1,$2,'discipline_auto_waived',$3,$4::jsonb)",
          [event.rows[0].hostel_id, event.rows[0].student_user_id, req.user.id, JSON.stringify({ action_id: actionId, attendance_event_id: eventId })]);
        await createNotification(client, event.rows[0].student_user_id, 'attendance_discipline', 'Attendance action reversed',
          'An attendance correction removed a warning or outstanding fine.', { action_id: actionId, attendance_event_id: eventId }, 'attendance_updates');
      }
      for (const actionId of paidNeedsReview) {
        const wardens = await client.query('SELECT warden_user_id FROM warden_assignments WHERE hostel_id=$1', [event.rows[0].hostel_id]);
        for (const warden of wardens.rows) await createNotification(client, warden.warden_user_id,
          'attendance_discipline', 'Paid attendance fine needs review',
          'A corrected attendance record may require a refund for a paid fine.',
          { action_id: actionId, attendance_event_id: eventId }, 'attendance_updates');
      }
      const parents = await client.query(`SELECT l.parent_user_id FROM parent_student_links l JOIN student_attendance_notification_permissions permission ON permission.parent_student_link_id=l.id AND permission.allowed=true LEFT JOIN parent_attendance_preferences p ON p.parent_user_id=l.parent_user_id AND p.student_user_id=l.student_user_id WHERE l.student_user_id=$1 AND l.status='active' AND COALESCE(p.notify_unusual,true)`, [event.rows[0].student_user_id]);
      for (const parent of parents.rows) await createNotification(client, parent.parent_user_id, "attendance_correction", "Attendance record corrected", "A hostel attendance record for your linked student was corrected by authorized staff.", { attendance_event_id: eventId, student_id: event.rows[0].student_user_id, hostel_id: event.rows[0].hostel_id, direction, occurred_at }, "attendance_updates");
      await client.query("COMMIT");
      res.status(201).json({ correction: result.rows[0], curfew_recalculated: true });
    } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
  });
}

module.exports = { registerAttendanceRoutes };
