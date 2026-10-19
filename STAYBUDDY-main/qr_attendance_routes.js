const crypto = require("node:crypto");
const path = require("node:path");
const QRCode = require("qrcode");
const { applyAttendanceConsequences } = require("./posted_gate_attendance_routes");

function digest(value) {
  return crypto.createHash("sha256").update(String(value)).digest("hex");
}

async function activeLinkedChild(pool, parentId, studentId) {
  const result = await pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'", [parentId, studentId]);
  return result.rowCount === 1;
}

async function attendanceAnalytics(pool, { hostelId = null, studentId = null, from, to }) {
  const result = await pool.query(
    `WITH effective AS (
       SELECT e.id,e.hostel_id,e.student_user_id,e.source,
              COALESCE(c.corrected_direction,e.direction) AS direction,
              COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at
       FROM attendance_events e
       LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
       WHERE ($1::int IS NULL OR e.hostel_id=$1) AND ($2::int IS NULL OR e.student_user_id=$2)
         AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$3::timestamptz
         AND COALESCE(c.corrected_occurred_at,e.occurred_at)<$4::timestamptz
     )
     SELECT occurred_at::date AS date,
            COUNT(*)::int AS movements,
            COUNT(*) FILTER(WHERE direction='in')::int AS entries,
            COUNT(*) FILTER(WHERE direction='out')::int AS exits,
            COUNT(*) FILTER(WHERE source='qr')::int AS qr_scans,
            MIN(occurred_at) AS first_movement,
            MAX(occurred_at) AS last_movement
     FROM effective GROUP BY occurred_at::date ORDER BY date`,
    [hostelId, studentId, from, to]
  );
  const exceptions = await pool.query(
    `SELECT exception_type,COUNT(*)::int AS count FROM attendance_exceptions
     WHERE ($1::int IS NULL OR hostel_id=$1) AND ($2::int IS NULL OR student_user_id=$2)
       AND policy_date >= ($3::timestamptz)::date AND policy_date < ($4::timestamptz)::date
     GROUP BY exception_type ORDER BY exception_type`, [hostelId, studentId, from, to]
  );
  return { daily: result.rows, exceptions: exceptions.rows };
}

function registerQrAttendanceRoutes({ app, pool, requireAuth, requireRole, isValidInt, wardenHasHostel, createNotification }) {
  const demoFiles = {
    "/attendance-qr-demo": "attendance-qr-demo.html",
    "/attendance-qr-demo.js": "attendance-qr-demo.js",
    "/attendance-qr-gate": "attendance-qr-gate.html",
    "/attendance-qr-gate.js": "attendance-qr-gate.js",
  };
  for (const [url, filename] of Object.entries(demoFiles)) {
    app.get(url, (_req, res) => {
      res.set({
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'",
      });
      res.sendFile(path.join(__dirname, "public", filename));
    });
  }

  app.get("/api/student/attendance/parent-permissions", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query(`SELECT l.parent_user_id AS parent_id,u.name AS parent_name,
      COALESCE(p.allowed,false) AS notifications_allowed,p.updated_at
      FROM parent_student_links l JOIN users u ON u.id=l.parent_user_id
      LEFT JOIN student_attendance_notification_permissions p ON p.parent_student_link_id=l.id
      WHERE l.student_user_id=$1 AND l.status='active' ORDER BY u.name,l.id`, [req.user.id]);
    res.json({ parents: result.rows });
  });

  app.put("/api/student/attendance/parent-permissions/:parentId", requireAuth, requireRole("student"), async (req, res) => {
    const parentId = Number(req.params.parentId);
    if (!isValidInt(parentId) || typeof req.body?.allowed !== "boolean")
      return res.status(400).json({ error: "A valid parentId and boolean allowed value are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const link = await client.query("SELECT id FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active' FOR UPDATE", [parentId, req.user.id]);
      if (!link.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Active parent link not found" }); }
      const permission = await client.query(`INSERT INTO student_attendance_notification_permissions(parent_student_link_id,allowed)
        VALUES($1,$2) ON CONFLICT(parent_student_link_id) DO UPDATE SET allowed=EXCLUDED.allowed,updated_at=NOW() RETURNING *`,
      [link.rows[0].id, req.body.allowed]);
      if (!req.body.allowed) await client.query("UPDATE attendance_return_watches SET active=false WHERE parent_user_id=$1 AND student_user_id=$2 AND active=true", [parentId, req.user.id]);
      await client.query("COMMIT");
      res.json({ parent_id: parentId, notifications_allowed: permission.rows[0].allowed, updated_at: permission.rows[0].updated_at });
    } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
  });

  app.post("/api/student/attendance/qr-token", requireAuth, requireRole("student"), async (req, res) => {
    const hostelId = req.body?.hostel_id == null ? null : Number(req.body.hostel_id);
    if (hostelId != null && !isValidInt(hostelId)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    const stay = await pool.query(
      `SELECT b.hostel_id,h.name AS hostel_name,b.room_id
       FROM bookings b JOIN hostels h ON h.id=b.hostel_id
       WHERE b.user_id=$1 AND b.status='confirmed'
         AND b.room_id IS NOT NULL AND NULLIF(TRIM(COALESCE(b.bed_number,'')),'') IS NOT NULL
         AND ($2::int IS NULL OR b.hostel_id=$2)
       ORDER BY b.updated_at DESC,b.id DESC LIMIT 2`, [req.user.id, hostelId]
    );
    if (!stay.rowCount) return res.status(409).json({ error: "A confirmed room and bed assignment is required" });
    if (hostelId == null && stay.rowCount > 1) return res.status(409).json({ error: "Choose hostel_id because more than one confirmed assignment exists" });
    const opaque = crypto.randomBytes(32).toString("base64url"), tokenId = crypto.randomUUID();
    const demo = req.query.demo === "1";
    const configuredTtl = Number(process.env.ATTENDANCE_QR_TTL_SECONDS || 60);
    const ttlSeconds = demo ? 120 : Number.isFinite(configuredTtl) ? Math.max(30, Math.min(configuredTtl, 120)) : 60;
    const expiresAt = new Date(Date.now() + ttlSeconds * 1000);
    let demoQrImage = null, demoScanUrl = null;
    if (demo) {
      try {
        const host = String(req.get("host") || "");
        if (!/^[A-Za-z0-9.\-:\[\]]{1,255}$/.test(host)) return res.status(400).json({ error: "A valid local host is required for the phone QR demo" });
        demoScanUrl = new URL(`/attendance-qr-gate?token=${encodeURIComponent(opaque)}`, `${req.protocol}://${host}`).toString();
        const svg = await QRCode.toString(demoScanUrl, { type: "svg", errorCorrectionLevel: "M", margin: 2, width: 320 });
        demoQrImage = `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`;
      } catch (_) { return res.status(400).json({ error: "Unable to render a phone-scannable QR for this host" }); }
    }
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      await client.query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", [`attendance-qr:${req.user.id}:${stay.rows[0].hostel_id}`]);
      await client.query("UPDATE attendance_qr_tokens SET revoked_at=NOW() WHERE student_user_id=$1 AND hostel_id=$2 AND used_at IS NULL AND revoked_at IS NULL", [req.user.id, stay.rows[0].hostel_id]);
      await client.query("INSERT INTO attendance_qr_tokens (id,token_hash,hostel_id,student_user_id,expires_at) VALUES ($1,$2,$3,$4,$5)", [tokenId, digest(opaque), stay.rows[0].hostel_id, req.user.id, expiresAt]);
      await client.query("COMMIT");
    } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
    res.status(201).json({ qr: { token: opaque, token_id: tokenId, hostel_id: stay.rows[0].hostel_id, hostel_name: stay.rows[0].hostel_name, expires_at: expiresAt, one_time: true, render_value: `staybuddy-attendance:${opaque}`, ...(demo ? { demo_scan_url: demoScanUrl, demo_qr_image: demoQrImage } : {}) } });
  });

  app.post("/api/attendance/qr/scan", async (req, res) => {
    const integrationKey = String(req.headers["x-attendance-key"] || "").trim();
    const idempotencyKey = String(req.headers["idempotency-key"] || "").trim();
    const rawToken = String(req.body?.qr_token || "").replace(/^staybuddy-attendance:/, "").trim();
    const direction = String(req.body?.direction || "").toLowerCase();
    const entryPoint = String(req.body?.entry_point || "Main Gate").trim().slice(0, 120) || "Main Gate";
    const scannerTime = req.body?.occurred_at;
    const scannerReportedAt = scannerTime == null ? new Date() : new Date(scannerTime);
    const occurredAt = new Date(); // The verified server receive time is authoritative for a live QR scan.
    if (!integrationKey || !idempotencyKey || idempotencyKey.length > 160 || !/^[A-Za-z0-9_-]{43}$/.test(rawToken) || !["in", "out"].includes(direction) ||
      (scannerTime != null && (typeof scannerTime !== "string" || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(scannerTime))) ||
      Number.isNaN(scannerReportedAt.getTime())) return res.status(400).json({ error: "Scanner key, Idempotency-Key, QR token, direction, and a timezone-aware occurred_at are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const credential = await client.query("SELECT id,hostel_id FROM attendance_integration_credentials WHERE credential_hash=$1 AND active=true", [digest(integrationKey)]);
      if (!credential.rowCount) { await client.query("ROLLBACK"); return res.status(401).json({ error: "Invalid attendance integration credential" }); }
      const { id: credentialId, hostel_id: credentialHostelId } = credential.rows[0];
      const token = await client.query("SELECT * FROM attendance_qr_tokens WHERE token_hash=$1 FOR UPDATE", [digest(rawToken)]);
      if (!token.rowCount) {
        await client.query("INSERT INTO attendance_qr_scan_attempts (hostel_id,integration_credential_id,direction,entry_point,outcome,reason,remote_address) VALUES ($1,$2,$3,$4,'invalid_token','Token digest not found',$5)", [credentialHostelId, credentialId, direction, entryPoint, req.ip]);
        await client.query("COMMIT"); return res.status(401).json({ error: "Invalid QR attendance token" });
      }
      const row = token.rows[0];
      const reject = async (outcome, status, message) => {
        await client.query("INSERT INTO attendance_qr_scan_attempts (token_id,hostel_id,integration_credential_id,direction,entry_point,outcome,reason,remote_address) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)", [row.id, credentialHostelId, credentialId, direction, entryPoint, outcome, message, req.ip]);
        await client.query("COMMIT");
        return res.status(status).json({ error: message });
      };
      // Attendance idempotency keys are unique across all hostels, so resolve
      // retries against that same global key before attempting an insert.
      const prior = await client.query("SELECT * FROM attendance_events WHERE idempotency_key=$1", [idempotencyKey]);
      if (prior.rowCount) {
        const event = prior.rows[0];
        if (event.hostel_id !== credentialHostelId || event.source !== "qr" || event.metadata?.qr_token_id !== row.id || event.metadata?.integration_credential_id !== credentialId ||
          event.direction !== direction || event.entry_point !== entryPoint)
          return reject("idempotency_conflict", 409, "Idempotency-Key was already used for a different QR scan");
        await client.query("ROLLBACK");
        return res.json({ event, duplicate: true });
      }
      if (Math.abs(Date.now() - scannerReportedAt.getTime()) > 2 * 60 * 1000)
        return reject("out_of_order", 400, "QR scan timestamp must be within two minutes of server time; use manual correction for older events");
      if (row.hostel_id !== credentialHostelId) return reject("wrong_hostel", 403, "QR token belongs to another hostel");
      if (row.used_at || row.revoked_at) return reject("already_used", 409, "QR token has already been used or replaced");
      if (new Date(row.expires_at) <= new Date()) return reject("expired", 410, "QR token has expired");
      const activeStay = await client.query(`SELECT MAX(COALESCE(b.check_in,b.updated_at,b.created_at)) AS assigned_at FROM bookings b
        WHERE b.user_id=$1 AND b.hostel_id=$2 AND b.status='confirmed'
          AND b.room_id IS NOT NULL AND NULLIF(TRIM(COALESCE(b.bed_number,'')),'') IS NOT NULL`, [row.student_user_id, credentialHostelId]);
      if (!activeStay.rows[0].assigned_at) return reject("inactive_assignment", 409, "Student no longer has a confirmed room and bed assignment");
      const latest = await client.query(`SELECT COALESCE(c.corrected_direction,e.direction) AS direction,COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at FROM attendance_events e LEFT JOIN LATERAL(SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true WHERE e.hostel_id=$1 AND e.student_user_id=$2 AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$3 ORDER BY occurred_at DESC,e.id DESC LIMIT 1`, [credentialHostelId, row.student_user_id, activeStay.rows[0].assigned_at]);
      if (latest.rowCount && occurredAt < new Date(latest.rows[0].occurred_at)) return reject("out_of_order", 409, "QR scan occurred before the student's latest attendance event");
      if (latest.rowCount && latest.rows[0].direction === direction) return reject("duplicate_direction", 409, `Student is already recorded ${direction}`);
      const inserted = await client.query(`INSERT INTO attendance_events (hostel_id,student_user_id,direction,occurred_at,source,entry_point,external_event_id,idempotency_key,metadata) VALUES ($1,$2,$3,$4,'qr',$5,$6,$7,$8::jsonb) RETURNING *`, [credentialHostelId, row.student_user_id, direction, occurredAt, entryPoint, row.id, idempotencyKey, JSON.stringify({ qr_token_id: row.id, integration_credential_id: credentialId, scanner_reported_at: scannerTime || null })]);
      const event = inserted.rows[0];
      await client.query("UPDATE attendance_qr_tokens SET used_at=NOW() WHERE id=$1", [row.id]);
      await client.query("INSERT INTO attendance_qr_scan_attempts (token_id,hostel_id,integration_credential_id,direction,entry_point,outcome,remote_address,occurred_at) VALUES ($1,$2,$3,$4,$5,'accepted',$6,$7)", [row.id, credentialHostelId, credentialId, direction, entryPoint, req.ip, occurredAt]);
      const configured = await client.query("SELECT * FROM hostel_attendance_settings WHERE hostel_id=$1", [credentialHostelId]);
      const settings = configured.rows[0] || { enabled: false, timezone: "Asia/Karachi", earliest_exit_time: "07:00",
        curfew_time: "21:00", grace_minutes: 10, escalation_minutes: 30, consecutive_warning_days: 3, attendance_fine_amount: 0 };
      const policy = await applyAttendanceConsequences(client, event, settings, createNotification);
      const late = Boolean(policy.kind);
      await client.query("COMMIT");
      res.status(201).json({ event, duplicate: false, late, current_status: direction === "in" ? "inside" : "outside" });
    } catch (error) {
      await client.query("ROLLBACK");
      res.status(500).json({ error: "Failed to process QR attendance scan", ...(process.env.RUN_DB_TESTS === "1" ? { detail: error.message } : {}) });
    } finally { client.release(); }
  });

  app.get("/api/parent/children/:studentId/attendance-preferences", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId) || !await activeLinkedChild(pool, req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    const result = await pool.query(`INSERT INTO parent_attendance_preferences (parent_user_id,student_user_id) VALUES ($1,$2) ON CONFLICT (parent_user_id,student_user_id) DO UPDATE SET parent_user_id=EXCLUDED.parent_user_id RETURNING *`, [req.user.id, studentId]);
    res.json({ preferences: result.rows[0] });
  });

  app.put("/api/parent/children/:studentId/attendance-preferences", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId), fields = ["notify_entry", "notify_exit", "notify_late", "notify_curfew", "notify_unusual", "daily_summary"];
    if (!isValidInt(studentId) || !await activeLinkedChild(pool, req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    if (fields.some((field) => req.body[field] != null && typeof req.body[field] !== "boolean")) return res.status(400).json({ error: "Attendance preference values must be boolean" });
    const values = fields.map((field) => req.body[field] ?? null);
    const result = await pool.query(`INSERT INTO parent_attendance_preferences (parent_user_id,student_user_id,notify_entry,notify_exit,notify_late,notify_curfew,notify_unusual,daily_summary) VALUES ($1,$2,COALESCE($3,true),COALESCE($4,true),COALESCE($5,true),COALESCE($6,true),COALESCE($7,true),COALESCE($8,false)) ON CONFLICT (parent_user_id,student_user_id) DO UPDATE SET notify_entry=COALESCE($3,parent_attendance_preferences.notify_entry),notify_exit=COALESCE($4,parent_attendance_preferences.notify_exit),notify_late=COALESCE($5,parent_attendance_preferences.notify_late),notify_curfew=COALESCE($6,parent_attendance_preferences.notify_curfew),notify_unusual=COALESCE($7,parent_attendance_preferences.notify_unusual),daily_summary=COALESCE($8,parent_attendance_preferences.daily_summary),updated_at=NOW() RETURNING *`, [req.user.id, studentId, ...values]);
    res.json({ preferences: result.rows[0] });
  });

  app.post("/api/parent/children/:studentId/attendance/notify-when-returned", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId)) return res.status(404).json({ error: "Linked student not found" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      // The student revocation endpoint locks this same link before cancelling
      // watches, preventing a new watch from appearing after consent is revoked.
      const link = await client.query("SELECT id FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active' FOR UPDATE", [req.user.id, studentId]);
      if (!link.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Linked student not found" }); }
      const consent = await client.query("SELECT 1 FROM student_attendance_notification_permissions WHERE parent_student_link_id=$1 AND allowed=true", [link.rows[0].id]);
      if (!consent.rowCount) { await client.query("ROLLBACK"); return res.status(403).json({ error: "Student has not allowed attendance notifications for this parent" }); }
      const hostel = await client.query(`SELECT b.hostel_id,s.checked_in_at FROM bookings b JOIN booking_stays s ON s.booking_id=b.id WHERE b.user_id=$1 AND b.status='confirmed' AND s.checked_in_at IS NOT NULL AND s.checked_out_at IS NULL ORDER BY s.checked_in_at DESC LIMIT 1`, [studentId]);
      if (!hostel.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Student has no active checked-in stay" }); }
      const latest = await client.query(`SELECT COALESCE(c.corrected_direction,e.direction) AS direction FROM attendance_events e LEFT JOIN LATERAL(SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true WHERE e.student_user_id=$1 AND e.hostel_id=$2 AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$3 ORDER BY COALESCE(c.corrected_occurred_at,e.occurred_at) DESC,e.id DESC LIMIT 1`, [studentId, hostel.rows[0].hostel_id, hostel.rows[0].checked_in_at]);
      if (!latest.rowCount || latest.rows[0].direction !== "out") { await client.query("ROLLBACK"); return res.status(409).json({ error: "A return alert can be created only while the student is recorded outside" }); }
      const result = await client.query(`INSERT INTO attendance_return_watches (parent_user_id,student_user_id,hostel_id) VALUES ($1,$2,$3) ON CONFLICT (parent_user_id,student_user_id,hostel_id) WHERE active DO UPDATE SET created_at=NOW() RETURNING *`, [req.user.id, studentId, hostel.rows[0].hostel_id]);
      await client.query("COMMIT");
      res.status(201).json({ watch: result.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
  });

  app.post("/api/parent/children/:studentId/attendance/exceptions/:id/acknowledge", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId), exceptionId = Number(req.params.id), note = String(req.body?.note || "").trim().slice(0, 1000) || null;
    if (!isValidInt(studentId) || !isValidInt(exceptionId) || !await activeLinkedChild(pool, req.user.id, studentId)) return res.status(404).json({ error: "Attendance exception not found" });
    const exists = await pool.query("SELECT 1 FROM attendance_exceptions WHERE id=$1 AND student_user_id=$2", [exceptionId, studentId]);
    if (!exists.rowCount) return res.status(404).json({ error: "Attendance exception not found" });
    const result = await pool.query("INSERT INTO attendance_exception_acknowledgements (exception_id,parent_user_id,note) VALUES ($1,$2,$3) ON CONFLICT (exception_id,parent_user_id) DO UPDATE SET note=EXCLUDED.note,acknowledged_at=NOW() RETURNING *", [exceptionId, req.user.id, note]);
    res.json({ acknowledgement: result.rows[0] });
  });

  app.get("/api/warden/attendance/dashboard", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.query.hostel_id);
    const selectedDate = req.query.date == null ? null : String(req.query.date).trim();
    if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (selectedDate != null && !/^\d{4}-\d{2}-\d{2}$/.test(selectedDate))
      return res.status(400).json({ error: "date must use YYYY-MM-DD format" });
    const residents = await pool.query(
      `WITH residents AS (
         SELECT DISTINCT ON (b.user_id) b.user_id,b.room_id,b.bed_number,r.room_number,
           COALESCE(r.floor_label,'Ground Floor') AS floor_label,
          -- Attendance is available after the room/bed assignment.  check_in
          -- may be a future scheduled arrival date, so it must not hide a
          -- real QR scan made during an authorised test or early arrival.
          b.created_at AS assigned_at
         FROM bookings b JOIN rooms r ON r.id=b.room_id
         WHERE b.hostel_id=$1 AND b.status='confirmed'
           AND b.room_id IS NOT NULL AND NULLIF(TRIM(COALESCE(b.bed_number,'')),'') IS NOT NULL
         ORDER BY b.user_id,b.updated_at DESC,b.id DESC
       ),
       latest AS (SELECT r.user_id,l.direction,l.occurred_at,l.source,l.entry_point FROM residents r LEFT JOIN LATERAL(SELECT COALESCE(c.corrected_direction,e.direction) AS direction,COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at,e.source,e.entry_point FROM attendance_events e LEFT JOIN LATERAL(SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true WHERE e.hostel_id=$1 AND e.student_user_id=r.user_id AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=r.assigned_at ORDER BY occurred_at DESC,e.id DESC LIMIT 1)l ON true)
       SELECT u.id AS student_id,u.name,r.floor_label,r.room_number,r.bed_number,l.direction,l.occurred_at,l.source,l.entry_point
       FROM latest l JOIN residents r ON r.user_id=l.user_id JOIN users u ON u.id=l.user_id ORDER BY u.name`, [hostelId]
    );
    const summary = residents.rows.reduce((acc, row) => { acc.total += 1; if (row.direction === "in") acc.inside += 1; else if (row.direction === "out") acc.outside += 1; else acc.unknown += 1; return acc; }, { total: 0, inside: 0, outside: 0, unknown: 0 });
    const exceptions = await pool.query("SELECT COUNT(*)::int AS open FROM attendance_exceptions WHERE hostel_id=$1 AND status='open'", [hostelId]);
    const recentEvents = await pool.query(
      `SELECT e.id,e.student_user_id,u.name AS student_name,
         COALESCE(c.corrected_direction,e.direction) AS direction,
         COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at,
         e.source,e.entry_point,
         assignment.floor_label,assignment.room_number,assignment.bed_number
       FROM attendance_events e
       JOIN users u ON u.id=e.student_user_id
       LEFT JOIN LATERAL (
         SELECT COALESCE(r.floor_label,'Ground Floor') AS floor_label,r.room_number,b.bed_number
         FROM bookings b LEFT JOIN rooms r ON r.id=b.room_id
         WHERE b.user_id=e.student_user_id AND b.hostel_id=e.hostel_id
           AND b.status IN ('confirmed','completed') AND b.room_id IS NOT NULL
         ORDER BY b.updated_at DESC,b.id DESC LIMIT 1
       ) assignment ON true
       LEFT JOIN LATERAL (
         SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id
         ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1
       ) c ON true
      WHERE e.hostel_id=$1
        AND ($2::date IS NULL OR (COALESCE(c.corrected_occurred_at,e.occurred_at) AT TIME ZONE 'Asia/Karachi')::date=$2::date)
      ORDER BY COALESCE(c.corrected_occurred_at,e.occurred_at) DESC,e.id DESC LIMIT 30`, [hostelId, selectedDate]
    );
    const dailyAttendance = selectedDate == null ? { rows: [] } : await pool.query(
      `WITH effective AS (
         SELECT e.id,e.student_user_id,
           COALESCE(c.corrected_direction,e.direction) AS direction,
           COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at,
           e.source,e.entry_point
         FROM attendance_events e
         LEFT JOIN LATERAL (
           SELECT * FROM attendance_corrections ac
           WHERE ac.attendance_event_id=e.id
           ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1
         ) c ON true
         WHERE e.hostel_id=$1
           AND (COALESCE(c.corrected_occurred_at,e.occurred_at) AT TIME ZONE 'Asia/Karachi')::date=$2::date
       )
       SELECT DISTINCT ON (student_user_id)
         student_user_id,direction,occurred_at,source,entry_point,
         COUNT(*) OVER (PARTITION BY student_user_id)::int AS event_count
       FROM effective
       ORDER BY student_user_id,occurred_at DESC,id DESC`,
      [hostelId, selectedDate],
    );
    const residentIds = new Set(residents.rows.map((resident) => Number(resident.student_id)));
    const residentDailyAttendance = dailyAttendance.rows.filter((event) => residentIds.has(Number(event.student_user_id)));
    const dailySummary = {
      total: residents.rows.length,
      attended: residentDailyAttendance.length,
      absent: Math.max(0, residents.rows.length - residentDailyAttendance.length),
      percentage: residents.rows.length === 0 ? 0 : Math.round((residentDailyAttendance.length / residents.rows.length) * 100),
    };
    res.json({ hostel_id: hostelId, selected_date: selectedDate, summary: { ...summary, open_exceptions: exceptions.rows[0].open }, daily_summary: dailySummary, residents: residents.rows, daily_attendance: residentDailyAttendance, recent_events: recentEvents.rows });
  });

  app.get("/api/warden/attendance/analytics", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.query.hostel_id), to = req.query.to ? new Date(req.query.to) : new Date(), from = req.query.from ? new Date(req.query.from) : new Date(to.getTime() - 30 * 86400000);
    if (!isValidInt(hostelId) || Number.isNaN(from.getTime()) || Number.isNaN(to.getTime()) || from >= to || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel or date range not found" });
    res.json({ hostel_id: hostelId, from, to, ...(await attendanceAnalytics(pool, { hostelId, from, to })) });
  });

  app.get("/api/parent/children/:studentId/attendance/analytics", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId), to = req.query.to ? new Date(req.query.to) : new Date(), from = req.query.from ? new Date(req.query.from) : new Date(to.getTime() - 30 * 86400000);
    if (!isValidInt(studentId) || Number.isNaN(from.getTime()) || Number.isNaN(to.getTime()) || from >= to || !await activeLinkedChild(pool, req.user.id, studentId)) return res.status(404).json({ error: "Linked student or date range not found" });
    const latest = await pool.query(`SELECT COALESCE(c.corrected_direction,e.direction) AS direction,COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at,e.source,e.entry_point,h.name AS hostel_name FROM attendance_events e JOIN hostels h ON h.id=e.hostel_id LEFT JOIN LATERAL(SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true WHERE e.student_user_id=$1 ORDER BY occurred_at DESC,e.id DESC LIMIT 1`, [studentId]);
    res.json({ student_id: studentId, from, to, current_status: latest.rows[0] || null, ...(await attendanceAnalytics(pool, { studentId, from, to })) });
  });

  app.get("/api/warden/attendance/qr-scan-attempts", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.query.hostel_id);
    if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query("SELECT id,token_id,direction,entry_point,outcome,reason,occurred_at FROM attendance_qr_scan_attempts WHERE hostel_id=$1 ORDER BY occurred_at DESC LIMIT 200", [hostelId]);
    res.json({ attempts: result.rows });
  });
}

module.exports = { registerQrAttendanceRoutes };
