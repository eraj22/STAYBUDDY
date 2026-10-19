const crypto = require("node:crypto");
const path = require("node:path");
const QRCode = require("qrcode");

const hash = (value) => crypto.createHash("sha256").update(String(value)).digest("hex");
const validTime = (value) => /^([01]\d|2[0-3]):[0-5]\d$/.test(value);
const minuteOf = (value) => Number(value.slice(0, 2)) * 60 + Number(value.slice(3, 5));

function distanceMeters(latitudeA, longitudeA, latitudeB, longitudeB) {
  const radians = (value) => value * Math.PI / 180;
  const earthRadius = 6371000;
  const latitudeDelta = radians(latitudeB - latitudeA);
  const longitudeDelta = radians(longitudeB - longitudeA);
  const a = Math.sin(latitudeDelta / 2) ** 2 + Math.cos(radians(latitudeA)) *
    Math.cos(radians(latitudeB)) * Math.sin(longitudeDelta / 2) ** 2;
  return 2 * earthRadius * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function parseAttendanceBoundary(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  const points = value.map((point) => ({
    latitude: Number(point?.latitude),
    longitude: Number(point?.longitude),
  }));
  return points.every((point) => Number.isFinite(point.latitude) && Number.isFinite(point.longitude) &&
    point.latitude >= -90 && point.latitude <= 90 && point.longitude >= -180 && point.longitude <= 180)
    ? points
    : null;
}

// Ray-casting against the hostel's ordered boundary vertices.  Longitude is
// the horizontal axis and latitude is the vertical axis for this small area.
function pointInAttendanceBoundary(latitude, longitude, boundary) {
  let inside = false;
  for (let current = 0, previous = boundary.length - 1; current < boundary.length; previous = current++) {
    const a = boundary[current], b = boundary[previous];
    const crosses = (a.latitude > latitude) !== (b.latitude > latitude);
    const intersection = (b.longitude - a.longitude) * (latitude - a.latitude) /
      (b.latitude - a.latitude) + a.longitude;
    if (crosses && longitude < intersection) inside = !inside;
  }
  return inside;
}

function localParts(date, timezone) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: timezone, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(date);
  return Object.fromEntries(parts.map(({ type, value }) => [type, value]));
}

function policyResult(date, direction, settings) {
  const local = localParts(date, settings.timezone);
  const minute = Number(local.hour) * 60 + Number(local.minute);
  const earliest = minuteOf(settings.earliest_exit_time);
  const curfew = minuteOf(settings.curfew_time);
  let policyDate = `${local.year}-${local.month}-${local.day}`;
  let kind = null, minutes = 0;
  if (direction === "out" && minute < earliest) {
    kind = "early_exit";
    minutes = earliest - minute;
  } else if (direction === "out" && minute >= curfew) {
    kind = "late_exit";
    minutes = minute - curfew;
  } else if (direction === "in" && (minute > curfew || minute < earliest)) {
    minutes = minute < earliest ? 1440 - curfew + minute : minute - curfew;
    if (minute < earliest) {
      const previous = new Date(Date.UTC(Number(local.year), Number(local.month) - 1, Number(local.day)) - 86400000);
      policyDate = previous.toISOString().slice(0, 10);
    }
    if (minutes > Number(settings.grace_minutes)) kind = "late_return";
  }
  return { kind, minutes, policyDate, severe: kind === "late_return" && minutes >= Number(settings.escalation_minutes) };
}

async function permittedParents(client, studentId, preference) {
  const allowed = new Set(["notify_entry", "notify_exit", "notify_late", "notify_curfew", "notify_unusual", "return_watch"]);
  if (!allowed.has(preference)) throw new Error("Invalid attendance preference");
  const preferenceCondition = preference === "return_watch" ? "true" : `COALESCE(p.${preference},true)`;
  const result = await client.query(`SELECT l.parent_user_id FROM parent_student_links l
    JOIN student_attendance_notification_permissions c ON c.parent_student_link_id=l.id AND c.allowed=true
    LEFT JOIN parent_attendance_preferences p ON p.parent_user_id=l.parent_user_id AND p.student_user_id=l.student_user_id
    WHERE l.student_user_id=$1 AND l.status='active' AND ${preferenceCondition}`, [studentId]);
  return result.rows.map((row) => row.parent_user_id);
}

async function notifyParents(client, createNotification, studentId, preference, type, title, body, data) {
  for (const parentId of await permittedParents(client, studentId, preference))
    await createNotification(client, parentId, type, title, body, data, "attendance_updates");
}

async function notifyWardens(client, createNotification, hostelId, type, title, body, data) {
  const result = await client.query("SELECT warden_user_id FROM warden_assignments WHERE hostel_id=$1", [hostelId]);
  for (const row of result.rows)
    await createNotification(client, row.warden_user_id, type, title, body, data, "attendance_updates");
}

async function countConsecutiveLateDays(client, hostelId, studentId, policyDate) {
  const result = await client.query(`SELECT policy_date::text AS day FROM attendance_exceptions
    WHERE hostel_id=$1 AND student_user_id=$2 AND exception_type='late_return' AND status='open'
      AND policy_date<=$3::date AND policy_date>$3::date-INTERVAL '30 days'
    ORDER BY policy_date DESC`, [hostelId, studentId, policyDate]);
  let expected = new Date(`${policyDate}T00:00:00Z`).getTime(), count = 0;
  for (const row of result.rows) {
    if (row.day !== new Date(expected).toISOString().slice(0, 10)) break;
    count += 1;
    expected -= 86400000;
  }
  return count;
}

function disciplineForStreak(consecutive, settings) {
  if (consecutive < 1) return null;
  const warningDays = Number(settings.consecutive_warning_days);
  const fineAmount = Number(settings.attendance_fine_amount);
  return consecutive > warningDays && fineAmount > 0
    ? { actionType: "fine", amount: fineAmount }
    : { actionType: "warning", amount: 0 };
}

async function applyAttendanceConsequences(client, event, settings, createNotification) {
  const { hostel_id: hostelId, student_user_id: studentId, direction } = event;
  const data = { hostel_id: hostelId, student_id: studentId, attendance_event_id: event.id, direction, source: event.source };
  const classification = settings.enabled ? policyResult(new Date(event.occurred_at), direction, settings) : { kind: null, policyDate: null, severe: false };
  const auditType = event.metadata?.gate_request_id ? "gate_scan_recorded"
    : event.metadata?.qr_token_id ? "qr_scan_accepted" : "attendance_recorded";
  await client.query(`INSERT INTO attendance_audit_events
    (hostel_id,student_user_id,attendance_event_id,event_type,actor_user_id,details)
    VALUES ($1,$2,$3,$4,$5,$6::jsonb)`,
  [hostelId, studentId, event.id, auditType, event.created_by_user_id,
    JSON.stringify({ gate_request_id: event.metadata?.gate_request_id || null, qr_token_id: event.metadata?.qr_token_id || null,
      direction, source: event.source, policy: classification })]);
  await notifyParents(client, createNotification, studentId, direction === "in" ? "notify_entry" : "notify_exit",
    "attendance", `Student checked ${direction}`, `A ${direction.toUpperCase()} event was recorded at ${event.entry_point}.`, data);
  if (direction === "in") {
    const watches = await client.query(`UPDATE attendance_return_watches SET active=false,fulfilled_by_event_id=$1,fulfilled_at=NOW()
      WHERE student_user_id=$2 AND hostel_id=$3 AND active=true RETURNING parent_user_id`, [event.id, studentId, hostelId]);
    const allowed = new Set(await permittedParents(client, studentId, "return_watch"));
    for (const watch of watches.rows) if (allowed.has(watch.parent_user_id))
      await createNotification(client, watch.parent_user_id, "attendance_return", "Student returned safely",
        `A return was recorded at ${event.entry_point}.`, data, "attendance_updates");
  }
  const movements = await client.query(`SELECT COUNT(*)::int AS count FROM attendance_events
    WHERE hostel_id=$1 AND student_user_id=$2 AND occurred_at BETWEEN $3::timestamptz-INTERVAL '6 hours' AND $3::timestamptz`,
  [hostelId, studentId, event.occurred_at]);
  if (movements.rows[0].count >= 8) {
    const unusual = await client.query(`INSERT INTO attendance_exceptions(hostel_id,student_user_id,policy_date,exception_type,details)
      VALUES($1,$2,($3::timestamptz AT TIME ZONE $4)::date,'unusual_pattern',$5::jsonb)
      ON CONFLICT DO NOTHING RETURNING id`, [hostelId, studentId, event.occurred_at, settings.timezone,
      JSON.stringify({ attendance_event_id: event.id, movements_in_six_hours: movements.rows[0].count })]);
    if (unusual.rowCount) await notifyParents(client, createNotification, studentId, "notify_unusual",
      "attendance_pattern", "Unusual attendance pattern", `${movements.rows[0].count} gate movements were recorded within six hours.`,
      { ...data, attendance_exception_id: unusual.rows[0].id });
  }
  if (!classification.kind) return classification;
  const inserted = await client.query(`INSERT INTO attendance_exceptions(hostel_id,student_user_id,policy_date,exception_type,details)
    VALUES($1,$2,$3,$4,$5::jsonb) ON CONFLICT DO NOTHING RETURNING id`,
  [hostelId, studentId, classification.policyDate, classification.kind,
      JSON.stringify({ attendance_event_id: event.id, minutes: classification.minutes, source: event.source })]);
  if (!inserted.rowCount) return classification;
  const exceptionId = inserted.rows[0].id;
  const exceptionData = { ...data, attendance_exception_id: exceptionId, minutes: classification.minutes };
  await notifyWardens(client, createNotification, hostelId, "attendance_exception",
    "Attendance policy exception", `A resident has a ${classification.kind.replace("_", " ")} event.`, exceptionData);
  await notifyParents(client, createNotification, studentId, "notify_late", "attendance_exception",
    "Attendance policy exception", `A ${classification.kind.replace("_", " ")} was recorded.`, exceptionData);
  if (classification.severe) {
    await notifyParents(client, createNotification, studentId, "notify_curfew", "attendance_escalation",
      "Return is over the escalation threshold", `The recorded return was ${classification.minutes} minutes after curfew.`, exceptionData);
    await notifyWardens(client, createNotification, hostelId, "attendance_escalation",
      "Resident returned well after curfew", `The recorded return was ${classification.minutes} minutes after curfew.`, exceptionData);
  }
  if (classification.kind !== "late_return") return classification;
  const consecutive = await countConsecutiveLateDays(client, hostelId, studentId, classification.policyDate);
  const discipline = disciplineForStreak(consecutive, settings);
  if (discipline) {
    const { actionType, amount } = discipline;
    const action = await client.query(`INSERT INTO attendance_disciplinary_actions
      (hostel_id,student_user_id,exception_id,action_type,consecutive_days,amount)
      VALUES($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING RETURNING id`,
    [hostelId, studentId, exceptionId, actionType, consecutive, amount]);
    if (action.rowCount) {
      const actionData = { ...exceptionData, action_id: action.rows[0].id, consecutive_days: consecutive, amount };
      const title = actionType === "warning" ? "Consecutive late-return warning" : "Attendance fine issued";
      const body = actionType === "warning"
        ? `${consecutive} consecutive late-return days have been recorded.`
        : `A PKR ${amount.toFixed(2)} fine was issued after ${consecutive} consecutive late-return days.`;
      await createNotification(client, studentId, "attendance_discipline", title, body, actionData, "attendance_updates");
      await notifyParents(client, createNotification, studentId, "notify_late", "attendance_discipline", title, body, actionData);
      await notifyWardens(client, createNotification, hostelId, "attendance_discipline", title, body, actionData);
    }
  }
  return { ...classification, consecutive };
}

function registerPostedGateAttendanceRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification }) {
  // Older StayBuddy databases were created before the printable QR image was
  // stored with a gate record.  Keep this migration beside the feature that
  // needs it so an existing installation can use QR attendance after an update.
  const gateQrSchemaReady = pool.query(
    "ALTER TABLE hostel_gate_qr_codes ADD COLUMN IF NOT EXISTS qr_image TEXT"
  );
  const ensureGateQrSchema = async () => {
    await gateQrSchemaReady;
  };
  const canManage = (req, hostelId) => req.user.role === "warden"
    ? wardenHasHostel(pool, req.user.id, hostelId)
    : req.user.role === "owner" && ownerHasHostel(pool, req.user.id, hostelId);
  const policyFields = ["earliest_exit_time", "curfew_time", "grace_minutes", "escalation_minutes", "consecutive_warning_days", "attendance_fine_amount", "timezone", "enabled"];

  app.get("/api/warden/assigned-hostel", requireAuth, requireRole("warden"), async (req, res) => {
    const result = await pool.query(`SELECT h.id,h.name,h.city FROM warden_assignments wa
      JOIN hostels h ON h.id=wa.hostel_id WHERE wa.warden_user_id=$1`, [req.user.id]);
    if (!result.rowCount) return res.status(404).json({ error: "No hostel is assigned to this warden" });
    res.json({ hostel: result.rows[0] });
  });

  app.get("/api/staff/hostels/:id/attendance-policy", requireAuth, requireRole("warden", "owner"), async (req, res) => {
    const hostelId = Number(req.params.id);
    if (!isValidInt(hostelId) || !await canManage(req, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query("SELECT * FROM hostel_attendance_settings WHERE hostel_id=$1", [hostelId]);
    res.json({ policy: result.rows[0] || { hostel_id: hostelId, earliest_exit_time: "07:00", curfew_time: "21:00", grace_minutes: 10, escalation_minutes: 30, consecutive_warning_days: 3, attendance_fine_amount: "0.00", timezone: "Asia/Karachi", enabled: false } });
  });

  app.put("/api/staff/hostels/:id/attendance-policy", requireAuth, requireRole("warden", "owner"), async (req, res) => {
    const hostelId = Number(req.params.id);
    if (!isValidInt(hostelId) || !await canManage(req, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (Object.keys(req.body || {}).some((key) => !policyFields.includes(key))) return res.status(400).json({ error: "Unknown attendance policy field" });
    if (["earliest_exit_time", "curfew_time"].some((key) => req.body?.[key] != null &&
      (typeof req.body[key] !== "string" || !validTime(req.body[key]))) ||
      (req.body?.timezone != null && (typeof req.body.timezone !== "string" || req.body.timezone.length > 64)))
      return res.status(400).json({ error: "Times must be HH:MM and timezone must be a valid IANA name" });
    if (["grace_minutes", "escalation_minutes", "consecutive_warning_days"].some((key) => req.body?.[key] != null && !Number.isInteger(req.body[key])) ||
      (req.body?.attendance_fine_amount != null && (typeof req.body.attendance_fine_amount !== "number" ||
        !Number.isInteger(req.body.attendance_fine_amount * 100))))
      return res.status(400).json({ error: "Thresholds must be whole minutes or days; fine amount must be a PKR number with at most two decimals" });
    const current = await pool.query("SELECT * FROM hostel_attendance_settings WHERE hostel_id=$1", [hostelId]);
    const old = current.rows[0] || {};
    const next = {
      earliest_exit_time: req.body.earliest_exit_time ?? old.earliest_exit_time ?? "07:00",
      curfew_time: req.body.curfew_time ?? old.curfew_time ?? "21:00",
      grace_minutes: req.body.grace_minutes ?? old.grace_minutes ?? 10,
      escalation_minutes: req.body.escalation_minutes ?? old.escalation_minutes ?? 30,
      consecutive_warning_days: req.body.consecutive_warning_days ?? old.consecutive_warning_days ?? 3,
      attendance_fine_amount: req.body.attendance_fine_amount ?? old.attendance_fine_amount ?? 0,
      enabled: req.body.enabled ?? old.enabled ?? true,
      timezone: req.body.timezone ?? old.timezone ?? "Asia/Karachi",
    };
    const early = String(next.earliest_exit_time).slice(0, 5), curfew = String(next.curfew_time).slice(0, 5);
    try { new Intl.DateTimeFormat("en", { timeZone: next.timezone }).format(new Date()); }
    catch (_) { return res.status(400).json({ error: "Invalid IANA timezone" }); }
    if (!validTime(early) || !validTime(curfew) || minuteOf(early) >= minuteOf(curfew) ||
      !Number.isInteger(Number(next.grace_minutes)) || Number(next.grace_minutes) < 0 || Number(next.grace_minutes) > 120 ||
      !Number.isInteger(Number(next.escalation_minutes)) || Number(next.escalation_minutes) <= Number(next.grace_minutes) || Number(next.escalation_minutes) > 240 ||
      !Number.isInteger(Number(next.consecutive_warning_days)) || Number(next.consecutive_warning_days) < 2 || Number(next.consecutive_warning_days) > 30 ||
      !Number.isFinite(Number(next.attendance_fine_amount)) || Number(next.attendance_fine_amount) < 0 || Number(next.attendance_fine_amount) > 100000 ||
      typeof next.enabled !== "boolean")
      return res.status(400).json({ error: "Invalid attendance policy or threshold order" });
    const saved = await pool.query(`INSERT INTO hostel_attendance_settings
      (hostel_id,earliest_exit_time,curfew_time,grace_minutes,escalation_minutes,consecutive_warning_days,attendance_fine_amount,timezone,enabled,updated_by_user_id)
      VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
      ON CONFLICT(hostel_id) DO UPDATE SET earliest_exit_time=$2,curfew_time=$3,grace_minutes=$4,escalation_minutes=$5,
      consecutive_warning_days=$6,attendance_fine_amount=$7,timezone=$8,enabled=$9,updated_by_user_id=$10,updated_at=NOW()
      RETURNING *`, [hostelId, early, curfew, next.grace_minutes, next.escalation_minutes, next.consecutive_warning_days,
      next.attendance_fine_amount, next.timezone, next.enabled, req.user.id]);
    res.json({ policy: saved.rows[0] });
  });

  app.post("/api/warden/hostels/:id/gate-qr", requireAuth, requireRole("warden"), async (req, res) => {
    await ensureGateQrSchema();
    const hostelId = Number(req.params.id), label = String(req.body?.gate_label || "Main Gate").trim();
    if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (!label || label.length > 120) return res.status(400).json({ error: "gate_label must be 1-120 characters" });
    const secret = crypto.randomBytes(32).toString("base64url");
    const base = String(process.env.ATTENDANCE_PUBLIC_BASE_URL || `${req.protocol}://${req.get("host") || ""}`);
    let scanUrl;
    try {
      const parsed = new URL(base);
      if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash) throw new Error("Invalid origin");
      const target = new URL('/attendance-gate', parsed);
      target.hash = `code=${encodeURIComponent(secret)}`;
      scanUrl = target.toString();
    } catch (_) { return res.status(400).json({ error: "Configure ATTENDANCE_PUBLIC_BASE_URL to a reachable app URL" }); }
    const qrImage = await QRCode.toDataURL(scanUrl, { width: 480, margin: 2 });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      await client.query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", [`posted-gate:${hostelId}:${label}`]);
      await client.query("UPDATE hostel_gate_qr_codes SET active=false,revoked_at=NOW() WHERE hostel_id=$1 AND gate_label=$2 AND active=true", [hostelId, label]);
      await client.query(`UPDATE attendance_gate_requests SET status='expired' WHERE hostel_id=$1 AND status='pending'
        AND gate_code_id IN (SELECT id FROM hostel_gate_qr_codes WHERE hostel_id=$1 AND gate_label=$2 AND active=false)`, [hostelId, label]);
      const saved = await client.query(`INSERT INTO hostel_gate_qr_codes(hostel_id,gate_label,code_hash,qr_image,created_by_user_id)
        VALUES($1,$2,$3,$4,$5) RETURNING id,hostel_id,gate_label,active,created_at,qr_image`, [hostelId, label, hash(secret), qrImage, req.user.id]);
      await client.query("COMMIT");
      res.status(201).set("Cache-Control", "no-store").json({ gate: saved.rows[0], scan_url: scanUrl, qr_image: qrImage });
    } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
  });

  app.get("/api/warden/hostels/:id/gate-qr", requireAuth, requireRole("warden"), async (req, res) => {
    await ensureGateQrSchema();
    const hostelId = Number(req.params.id);
    if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query("SELECT id,hostel_id,gate_label,active,created_at,revoked_at,CASE WHEN active THEN qr_image ELSE NULL END AS qr_image FROM hostel_gate_qr_codes WHERE hostel_id=$1 ORDER BY id DESC", [hostelId]);
    res.json({ gates: result.rows });
  });

  app.delete("/api/warden/hostels/:id/gate-qr/:gateId", requireAuth, requireRole("warden"), async (req, res) => {
    await ensureGateQrSchema();
    const hostelId = Number(req.params.id), gateId = Number(req.params.gateId);
    if (!isValidInt(hostelId) || !isValidInt(gateId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Gate not found" });
    const result = await pool.query("UPDATE hostel_gate_qr_codes SET active=false,revoked_at=COALESCE(revoked_at,NOW()) WHERE id=$1 AND hostel_id=$2 RETURNING id,active,revoked_at", [gateId, hostelId]);
    if (!result.rowCount) return res.status(404).json({ error: "Gate not found" });
    await pool.query("UPDATE attendance_gate_requests SET status='expired' WHERE gate_code_id=$1 AND status='pending'", [gateId]);
    res.json({ gate: result.rows[0] });
  });

  app.get("/attendance-gate", (_req, res) => {
    res.set({ "Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
      "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'" });
    res.sendFile(path.join(__dirname, "public", "attendance-gate.html"));
  });
  app.get("/attendance-gate.js", (_req, res) => {
    res.set({ "Cache-Control": "no-store", "Referrer-Policy": "no-referrer" });
    res.sendFile(path.join(__dirname, "public", "attendance-gate.js"));
  });
  app.get("/attendance-warden-demo", (_req, res) => {
    res.redirect(302, "/qr-test");
  });
  app.get("/attendance-warden-demo.js", (_req, res) => {
    res.set({ "Cache-Control": "no-store", "Referrer-Policy": "no-referrer" });
    res.sendFile(path.join(__dirname, "public", "attendance-warden-demo.js"));
  });
  app.get("/qr-test", (_req, res) => {
    res.set({ "Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
      "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'" });
    res.sendFile(path.join(__dirname, "public", "qr-test.html"));
  });
  app.get("/qr-console.js", (_req, res) => {
    res.set({ "Cache-Control": "no-store", "Referrer-Policy": "no-referrer" });
    res.sendFile(path.join(__dirname, "public", "qr-console.js"));
  });

  // The mobile app uses this before enabling its scanner. A confirmed
  // floor/room/bed assignment is the resident eligibility signal; a later
  // operational check-in must not prevent a resident from using attendance.
  app.get("/api/student/attendance/qr-access", requireAuth, requireRole("student"), async (req, res) => {
    const stay = await pool.query(`SELECT b.hostel_id,h.name,h.latitude,h.longitude,g.boundary AS attendance_boundary
      FROM bookings b JOIN hostels h ON h.id=b.hostel_id
      LEFT JOIN hostel_attendance_geofences g ON g.hostel_id=h.id AND g.enabled=true
      WHERE b.user_id=$1 AND b.status='confirmed'
        AND b.room_id IS NOT NULL AND NULLIF(TRIM(COALESCE(b.bed_number,'')),'') IS NOT NULL
      ORDER BY b.updated_at DESC,b.id DESC LIMIT 1`, [req.user.id]);
    if (!stay.rowCount) return res.status(409).json({ error: "A confirmed room and bed assignment is required" });
    const hostel = stay.rows[0];
    if (hostel.latitude == null || hostel.longitude == null)
      return res.status(409).json({ error: "Hostel location is not configured for QR attendance" });
    const boundary = parseAttendanceBoundary(hostel.attendance_boundary);
    res.json({
      hostel: { id: hostel.hostel_id, name: hostel.name, latitude: Number(hostel.latitude), longitude: Number(hostel.longitude) },
      radius_m: Number(process.env.ATTENDANCE_GEOFENCE_RADIUS_METERS || 5),
      attendance_boundary: boundary,
    });
  });

  app.post("/api/student/attendance/gate-requests", requireAuth, requireRole("student"), async (req, res) => {
    const code = String(req.body?.gate_code || "").trim();
    const direction = String(req.body?.direction || "").toLowerCase();
    const idempotencyKey = String(req.headers["idempotency-key"] || "").trim();
    const location = req.body?.location;
    if (!/^[A-Za-z0-9_-]{43}$/.test(code) || !["in", "out"].includes(direction) || !idempotencyKey || idempotencyKey.length > 160)
      return res.status(400).json({ error: "Gate code, direction, and Idempotency-Key are required" });
    if (location != null && (req.body.location_consent !== true || typeof location !== "object" || Array.isArray(location) ||
      !Number.isFinite(location.latitude) || !Number.isFinite(location.longitude) || !Number.isFinite(location.accuracy_m) ||
      location.latitude < -90 || location.latitude > 90 || location.longitude < -180 ||
      location.longitude > 180 || location.accuracy_m < 0 || location.accuracy_m > 100000))
      return res.status(400).json({ error: "Location requires consent and valid coordinates with accuracy" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const gate = await client.query("SELECT g.id,g.hostel_id,g.gate_label,h.latitude,h.longitude,b.boundary AS attendance_boundary FROM hostel_gate_qr_codes g JOIN hostels h ON h.id=g.hostel_id LEFT JOIN hostel_attendance_geofences b ON b.hostel_id=h.id AND b.enabled=true WHERE g.code_hash=$1 AND g.active=true FOR SHARE OF g,h", [hash(code)]);
      if (!gate.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Gate QR is invalid or revoked" }); }
      const { id: gateId, hostel_id: hostelId } = gate.rows[0];
      const radiusMeters = Number(process.env.ATTENDANCE_GEOFENCE_RADIUS_METERS || 5);
      if (location == null || gate.rows[0].latitude == null || gate.rows[0].longitude == null) {
        await client.query("ROLLBACK");
        return res.status(403).json({ error: "Location permission and a configured hostel location are required for QR attendance" });
      }
      const boundary = parseAttendanceBoundary(gate.rows[0].attendance_boundary);
      const insideBoundary = boundary != null && pointInAttendanceBoundary(
        Number(location.latitude), Number(location.longitude), boundary,
      );
      const distance = distanceMeters(Number(location.latitude), Number(location.longitude), Number(gate.rows[0].latitude), Number(gate.rows[0].longitude));
      // A reading less accurate than the required radius cannot prove that the
      // scan was made at the hostel gate. Do not expand the allowed area using
      // GPS accuracy; that would allow a distant proxy scan.
      if (Number(location.accuracy_m) > radiusMeters) {
        await client.query("ROLLBACK");
        return res.status(403).json({ error: `GPS accuracy must be ${radiusMeters}m or better to scan this QR code` });
      }
      if (boundary != null && !insideBoundary) {
        await client.query("ROLLBACK");
        return res.status(403).json({ error: "You must be inside the hostel building, gate, or entrance attendance boundary to scan this QR code" });
      }
      if (boundary == null && distance > radiusMeters) {
        await client.query("ROLLBACK");
        return res.status(403).json({ error: `You must be within ${radiusMeters}m of the hostel to scan this QR code` });
      }
      await client.query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", [`attendance-gate-key:${req.user.id}:${idempotencyKey}`]);
      const prior = await client.query("SELECT * FROM attendance_gate_requests WHERE student_user_id=$1 AND idempotency_key=$2", [req.user.id, idempotencyKey]);
      if (prior.rowCount) {
        await client.query("ROLLBACK");
        const same = prior.rows[0].direction === direction && Number(prior.rows[0].gate_code_id) === Number(gateId);
        return res.status(same ? 200 : 409).json(same ? { request: prior.rows[0], recorded: prior.rows[0].status === "recorded", duplicate: true } : { error: "Idempotency-Key already used" });
      }
      await client.query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", [`attendance-gate:${hostelId}:${req.user.id}`]);
      const stay = await client.query(`SELECT MAX(COALESCE(b.check_in,b.updated_at,b.created_at)) AS assigned_at FROM bookings b
        WHERE b.user_id=$1 AND b.hostel_id=$2 AND b.status='confirmed'
          AND b.room_id IS NOT NULL AND NULLIF(TRIM(COALESCE(b.bed_number,'')),'') IS NOT NULL`, [req.user.id, hostelId]);
      if (!stay.rows[0].assigned_at) { await client.query("ROLLBACK"); return res.status(403).json({ error: "A confirmed room and bed assignment at this hostel is required" }); }
      const latest = await client.query(`SELECT COALESCE(c.corrected_direction,e.direction) AS direction FROM attendance_events e
        LEFT JOIN LATERAL(SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
        WHERE e.student_user_id=$1 AND e.hostel_id=$2 AND COALESCE(c.corrected_occurred_at,e.occurred_at)>=$3
        ORDER BY COALESCE(c.corrected_occurred_at,e.occurred_at) DESC,e.id DESC LIMIT 1`, [req.user.id, hostelId, stay.rows[0].assigned_at]);
      if (latest.rowCount && latest.rows[0].direction === direction) { await client.query("ROLLBACK"); return res.status(409).json({ error: `Student is already recorded ${direction}` }); }
      const scanId = crypto.randomUUID();
      const saved = await client.query(`INSERT INTO attendance_gate_requests
        (id,gate_code_id,hostel_id,student_user_id,direction,idempotency_key,expires_at,latitude,longitude,accuracy_m,status)
        VALUES($1,$2,$3,$4,$5,$6,NOW()+INTERVAL '5 minutes',$7,$8,$9,'recorded') RETURNING *`,
      [scanId, gateId, hostelId, req.user.id, direction, idempotencyKey,
        location?.latitude ?? null, location?.longitude ?? null, location?.accuracy_m ?? null]);
      const event = await client.query(`INSERT INTO attendance_events
        (hostel_id,student_user_id,direction,occurred_at,source,entry_point,external_event_id,idempotency_key,metadata,created_by_user_id)
        VALUES($1,$2,$3,$4,'qr',$5,$6,$7,$8::jsonb,$9) RETURNING *`,
      [hostelId, req.user.id, direction, saved.rows[0].submitted_at, gate.rows[0].gate_label,
        scanId, `gate-scan:${scanId}`, JSON.stringify({ gate_request_id: scanId,
          phone_location_submitted: location != null, phone_location_is_unverified: true }), req.user.id]);
      const recorded = await client.query("UPDATE attendance_gate_requests SET attendance_event_id=$2 WHERE id=$1 RETURNING *",
        [scanId, event.rows[0].id]);
      await client.query("INSERT INTO hostel_attendance_settings(hostel_id,curfew_time,enabled,updated_by_user_id) VALUES($1,'21:00',false,$2) ON CONFLICT DO NOTHING", [hostelId, req.user.id]);
      const settings = await client.query("SELECT * FROM hostel_attendance_settings WHERE hostel_id=$1", [hostelId]);
      const policy = await applyAttendanceConsequences(client, event.rows[0], settings.rows[0], createNotification);
      await client.query("COMMIT");
      res.status(201).json({ request: recorded.rows[0], event: event.rows[0], policy, recorded: true });
    } catch (error) {
      await client.query("ROLLBACK");
      console.error("Gate attendance scan failed", error);
      res.status(500).json({ error: "Unable to record this gate scan. Please retry." });
    } finally { client.release(); }
  });

  app.get("/api/student/attendance/gate-requests", requireAuth, requireRole("student"), async (req, res) => {
    await pool.query("UPDATE attendance_gate_requests SET status='expired' WHERE student_user_id=$1 AND status='pending' AND expires_at<=NOW()", [req.user.id]);
    const result = await pool.query(`SELECT id,hostel_id,direction,submitted_at,expires_at,status,reviewed_at,review_reason,attendance_event_id
      FROM attendance_gate_requests WHERE student_user_id=$1 ORDER BY submitted_at DESC LIMIT 50`, [req.user.id]);
    res.json({ requests: result.rows });
  });

  app.get("/api/warden/attendance/gate-requests", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.query.hostel_id);
    if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    await pool.query("UPDATE attendance_gate_requests SET status='expired' WHERE hostel_id=$1 AND status='pending' AND expires_at<=NOW()", [hostelId]);
    const result = await pool.query(`SELECT r.*,u.name AS student_name,g.gate_label FROM attendance_gate_requests r
      JOIN users u ON u.id=r.student_user_id JOIN hostel_gate_qr_codes g ON g.id=r.gate_code_id
      WHERE r.hostel_id=$1 ORDER BY r.submitted_at DESC LIMIT 100`, [hostelId]);
    res.json({ requests: result.rows });
  });

  app.post("/api/warden/attendance/gate-requests/:requestId/review", requireAuth, requireRole("warden"), (_req, res) => {
    res.status(410).json({ error: "Gate scans record attendance immediately; only warning and fine appeals need review" });
  });

  app.get("/api/student/attendance/discipline", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query(`SELECT a.*,COALESCE(s.consecutive_warning_days,3) AS warning_appeal_threshold,
      COUNT(*) FILTER(WHERE a.action_type='warning') OVER(PARTITION BY a.hostel_id)::int AS warning_count
      FROM attendance_disciplinary_actions a LEFT JOIN hostel_attendance_settings s ON s.hostel_id=a.hostel_id
      WHERE a.student_user_id=$1 ORDER BY a.issued_at DESC`, [req.user.id]);
    res.json({ actions: result.rows });
  });
  app.get("/api/parent/children/:studentId/attendance/discipline", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId)) return res.status(404).json({ error: "Linked student not found" });
    const linked = await pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'", [req.user.id, studentId]);
    if (!linked.rowCount) return res.status(404).json({ error: "Linked student not found" });
    const result = await pool.query("SELECT * FROM attendance_disciplinary_actions WHERE student_user_id=$1 ORDER BY issued_at DESC", [studentId]);
    res.json({ actions: result.rows });
  });
  app.get("/api/warden/attendance/discipline", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.query.hostel_id);
    if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query("SELECT * FROM attendance_disciplinary_actions WHERE hostel_id=$1 ORDER BY issued_at DESC LIMIT 200", [hostelId]);
    res.json({ actions: result.rows });
  });
  app.get("/api/student/attendance/discipline-appeals", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query(`SELECT p.*,a.action_type,a.amount,a.status AS action_status
      FROM attendance_discipline_appeals p JOIN attendance_disciplinary_actions a ON a.id=p.action_id
      WHERE p.student_user_id=$1 ORDER BY p.requested_at DESC,p.id DESC LIMIT 100`, [req.user.id]);
    res.json({ appeals: result.rows });
  });
  app.post("/api/student/attendance/discipline/:id/appeals", requireAuth, requireRole("student"), async (req, res) => {
    const id = Number(req.params.id), reason = String(req.body?.reason || "").trim();
    if (!isValidInt(id) || reason.length < 10 || reason.length > 1000)
      return res.status(400).json({ error: "An action and a 10-1000 character appeal reason are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const action = await client.query("SELECT * FROM attendance_disciplinary_actions WHERE id=$1 FOR UPDATE", [id]);
      if (!action.rowCount || action.rows[0].student_user_id !== req.user.id) {
        await client.query("ROLLBACK"); return res.status(404).json({ error: "Attendance action not found" });
      }
      if (action.rows[0].status !== "issued") {
        await client.query("ROLLBACK"); return res.status(409).json({ error: "This warning or fine has already been resolved" });
      }
      if (action.rows[0].action_type === "warning") {
        const warningCount = await client.query(`SELECT COUNT(*)::int AS count FROM attendance_disciplinary_actions
          WHERE hostel_id=$1 AND student_user_id=$2 AND action_type='warning'`,
        [action.rows[0].hostel_id, req.user.id]);
        const setting = await client.query("SELECT consecutive_warning_days FROM hostel_attendance_settings WHERE hostel_id=$1", [action.rows[0].hostel_id]);
        const required = Number(setting.rows[0]?.consecutive_warning_days || 3);
        if (warningCount.rows[0].count < required) {
          await client.query("ROLLBACK"); return res.status(409).json({ error: `Warning appeals become available after ${required} warnings` });
        }
      }
      const existing = await client.query("SELECT id FROM attendance_discipline_appeals WHERE action_id=$1 AND status='pending'", [id]);
      if (existing.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "An appeal is already pending" }); }
      const appeal = await client.query(`INSERT INTO attendance_discipline_appeals
        (action_id,student_user_id,hostel_id,reason) VALUES($1,$2,$3,$4) RETURNING *`,
      [id, req.user.id, action.rows[0].hostel_id, reason]);
      await client.query("INSERT INTO attendance_audit_events(hostel_id,student_user_id,event_type,actor_user_id,details) VALUES($1,$2,'discipline_appeal_requested',$3,$4::jsonb)",
        [action.rows[0].hostel_id, req.user.id, req.user.id, JSON.stringify({ appeal_id: appeal.rows[0].id, action_id: id })]);
      await notifyWardens(client, createNotification, action.rows[0].hostel_id, "attendance_discipline",
        "Attendance action appeal", `A student requested review of an attendance ${action.rows[0].action_type}.`,
        { appeal_id: appeal.rows[0].id, action_id: id });
      await client.query("COMMIT");
      res.status(201).json({ appeal: appeal.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
  });
  app.get("/api/warden/attendance/discipline-appeals", requireAuth, requireRole("warden"), async (req, res) => {
    const hostelId = Number(req.query.hostel_id);
    if (!isValidInt(hostelId) || !await wardenHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query(`SELECT p.*,a.action_type,a.amount,a.status AS action_status,u.name AS student_name
      FROM attendance_discipline_appeals p JOIN attendance_disciplinary_actions a ON a.id=p.action_id
      JOIN users u ON u.id=p.student_user_id WHERE p.hostel_id=$1
      ORDER BY (p.status='pending') DESC,p.requested_at DESC,p.id DESC LIMIT 100`, [hostelId]);
    res.json({ appeals: result.rows });
  });
  app.post("/api/warden/attendance/discipline-appeals/:id/review", requireAuth, requireRole("warden"), async (req, res) => {
    const id = Number(req.params.id), decision = String(req.body?.decision || "");
    const reason = String(req.body?.reason || "").trim();
    if (!isValidInt(id) || !["approve", "reject"].includes(decision) || reason.length < 3 || reason.length > 1000)
      return res.status(400).json({ error: "A decision and a 3-1000 character review reason are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const appeal = await client.query("SELECT * FROM attendance_discipline_appeals WHERE id=$1 FOR UPDATE", [id]);
      if (!appeal.rowCount || !await wardenHasHostel(client, req.user.id, appeal.rows[0].hostel_id)) {
        await client.query("ROLLBACK"); return res.status(404).json({ error: "Appeal not found" });
      }
      if (appeal.rows[0].status !== "pending") {
        await client.query("ROLLBACK"); return res.status(409).json({ error: "Appeal already reviewed" });
      }
      const action = await client.query("SELECT * FROM attendance_disciplinary_actions WHERE id=$1 FOR UPDATE", [appeal.rows[0].action_id]);
      if (!action.rowCount || action.rows[0].status !== "issued") {
        await client.query("ROLLBACK"); return res.status(409).json({ error: "The warning or fine is no longer outstanding" });
      }
      if (decision === "approve") await client.query(`UPDATE attendance_disciplinary_actions
        SET status='waived',changed_by_user_id=$2,changed_at=NOW(),change_reason=$3 WHERE id=$1`,
      [action.rows[0].id, req.user.id, `Student appeal approved: ${reason}`]);
      const updated = await client.query(`UPDATE attendance_discipline_appeals
        SET status=$2,reviewed_by_user_id=$3,reviewed_at=NOW(),review_reason=$4 WHERE id=$1 RETURNING *`,
      [id, decision === "approve" ? "approved" : "rejected", req.user.id, reason]);
      await client.query("INSERT INTO attendance_audit_events(hostel_id,student_user_id,event_type,actor_user_id,details) VALUES($1,$2,$3,$4,$5::jsonb)",
        [appeal.rows[0].hostel_id, appeal.rows[0].student_user_id,
          decision === "approve" ? "discipline_appeal_approved" : "discipline_appeal_rejected", req.user.id,
          JSON.stringify({ appeal_id: id, action_id: action.rows[0].id, reason })]);
      const title = decision === "approve" ? "Attendance action removed" : "Attendance appeal rejected";
      const body = decision === "approve"
        ? `The warden removed your attendance ${action.rows[0].action_type}. Reason: ${reason}`
        : `Your attendance ${action.rows[0].action_type} appeal was rejected. Reason: ${reason}`;
      await createNotification(client, appeal.rows[0].student_user_id, "attendance_discipline", title, body,
        { appeal_id: id, action_id: action.rows[0].id, decision }, "attendance_updates");
      await notifyParents(client, createNotification, appeal.rows[0].student_user_id, "notify_late",
        "attendance_discipline", title, body, { appeal_id: id, action_id: action.rows[0].id, decision });
      await client.query("COMMIT");
      res.json({ appeal: updated.rows[0], action_status: decision === "approve" ? "waived" : "issued" });
    } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
  });
  app.patch("/api/warden/attendance/discipline/:id", requireAuth, requireRole("warden"), async (req, res) => {
    const id = Number(req.params.id), status = String(req.body?.status || ""), reason = String(req.body?.reason || "").trim();
    const reference = String(req.body?.payment_reference || "").trim();
    if (!isValidInt(id) || status !== "paid" || !reason || reason.length > 1000 || !reference || reference.length > 180)
      return res.status(400).json({ error: "Valid paid status, reason, and payment reference are required; removal uses a student appeal" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const action = await client.query("SELECT * FROM attendance_disciplinary_actions WHERE id=$1 FOR UPDATE", [id]);
      if (!action.rowCount || action.rows[0].action_type !== "fine" || !await wardenHasHostel(client, req.user.id, action.rows[0].hostel_id)) {
        await client.query("ROLLBACK"); return res.status(404).json({ error: "Fine not found" });
      }
      if (action.rows[0].status !== "issued") { await client.query("ROLLBACK"); return res.status(409).json({ error: "Fine has already been resolved" }); }
      const pendingAppeal = await client.query("SELECT id FROM attendance_discipline_appeals WHERE action_id=$1 AND status='pending'", [id]);
      if (pendingAppeal.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Review the student's appeal before marking this fine paid" }); }
      const updated = await client.query(`UPDATE attendance_disciplinary_actions SET status=$2,changed_by_user_id=$3,
        changed_at=NOW(),change_reason=$4,payment_reference=$5 WHERE id=$1 RETURNING *`,
      [id, status, req.user.id, reason, status === "paid" ? reference : null]);
      await client.query("INSERT INTO attendance_audit_events(hostel_id,student_user_id,event_type,actor_user_id,details) VALUES($1,$2,$3,$4,$5::jsonb)",
        [action.rows[0].hostel_id, action.rows[0].student_user_id, status === "paid" ? "fine_paid" : "fine_waived",
          req.user.id, JSON.stringify({ action_id: id, reason, payment_reference: status === "paid" ? reference : null })]);
      await createNotification(client, action.rows[0].student_user_id, "attendance_discipline", `Attendance fine ${status}`,
        `An attendance fine was ${status} by authorized hostel staff.`, { action_id: id, status }, "attendance_updates");
      await notifyParents(client, createNotification, action.rows[0].student_user_id, "notify_late",
        "attendance_discipline", `Attendance fine ${status}`, `An attendance fine was ${status} by authorized hostel staff.`,
        { action_id: id, status });
      await client.query("COMMIT");
      res.json({ action: updated.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
  });
}

module.exports = { registerPostedGateAttendanceRoutes, policyResult, countConsecutiveLateDays, disciplineForStreak, applyAttendanceConsequences };
