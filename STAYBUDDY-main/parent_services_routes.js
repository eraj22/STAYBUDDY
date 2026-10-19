function registerParentServicesRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification }) {
  async function managesHostel(user, hostelId) {
    if (user.role === "owner") return ownerHasHostel(pool, user.id, hostelId);
    if (user.role === "warden") return wardenHasHostel(pool, user.id, hostelId);
    return false;
  }
  async function linkedParent(parentId, studentId) {
    const result = await pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'", [parentId, studentId]);
    return result.rowCount > 0;
  }
  async function activeResident(studentId, hostelId) {
    const result = await pool.query(
      `SELECT 1 FROM bookings b LEFT JOIN booking_stays s ON s.booking_id=b.id
       WHERE b.user_id=$1 AND b.hostel_id=$2 AND b.status='confirmed'
         AND (s.booking_id IS NULL OR s.checked_out_at IS NULL) LIMIT 1`,
      [studentId, hostelId]
    );
    return result.rowCount > 0;
  }
  async function notifyLinkedParents(client, studentId, type, title, body, data, preference) {
    const parents = await client.query("SELECT parent_user_id FROM parent_student_links WHERE student_user_id=$1 AND status='active'", [studentId]);
    for (const parent of parents.rows) await createNotification(client, parent.parent_user_id, type, title, body, data, preference);
  }

  app.post("/api/staff/hostels/:hostelId/transport-services", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const hostelId = Number(req.params.hostelId), name = String(req.body.name || "").trim(), mode = String(req.body.mode || "");
    const days = req.body.operating_days ?? [], safety = req.body.safety_features ?? [], cost = req.body.monthly_cost == null ? null : Number(req.body.monthly_cost);
    if (!isValidInt(hostelId) || !await managesHostel(req.user, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (!name || name.length > 160 || !["shuttle", "bus", "van", "partner", "information_only"].includes(mode) || !Array.isArray(days) || !Array.isArray(safety) || (cost != null && (!Number.isFinite(cost) || cost < 0))) return res.status(400).json({ error: "Valid name, mode, operating_days, safety_features, and optional PKR cost are required" });
    const result = await pool.query(
      `INSERT INTO hostel_transport_services (hostel_id,name,mode,vehicle_number,driver_name,driver_phone,operating_days,safety_features,monthly_cost,created_by_user_id)
       VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9,$10) RETURNING *`,
      [hostelId, name, mode, req.body.vehicle_number || null, req.body.driver_name || null, req.body.driver_phone || null, JSON.stringify(days), JSON.stringify(safety), cost, req.user.id]
    );
    res.status(201).json({ service: result.rows[0] });
  });

  app.post("/api/staff/transport-services/:id/stops", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const id = Number(req.params.id), name = String(req.body.name || "").trim(), order = Number(req.body.stop_order);
    const service = await pool.query("SELECT * FROM hostel_transport_services WHERE id=$1", [id]);
    if (!service.rowCount || !await managesHostel(req.user, service.rows[0].hostel_id)) return res.status(404).json({ error: "Transport service not found" });
    const latitude = req.body.latitude == null ? null : Number(req.body.latitude), longitude = req.body.longitude == null ? null : Number(req.body.longitude);
    if (!name || name.length > 160 || !Number.isInteger(order) || order < 0 || ((latitude == null) !== (longitude == null)) || (latitude != null && (!Number.isFinite(latitude) || latitude < -90 || latitude > 90 || !Number.isFinite(longitude) || longitude < -180 || longitude > 180))) return res.status(400).json({ error: "Valid stop name, order, and optional coordinate pair are required" });
    const result = await pool.query("INSERT INTO hostel_transport_stops (service_id,name,latitude,longitude,stop_order,scheduled_time) VALUES ($1,$2,$3,$4,$5,$6) RETURNING *", [id, name, latitude, longitude, order, req.body.scheduled_time || null]);
    res.status(201).json({ stop: result.rows[0] });
  });

  app.post("/api/staff/transport-services/:id/assignments", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const serviceId = Number(req.params.id), studentId = Number(req.body.student_user_id);
    const service = await pool.query("SELECT * FROM hostel_transport_services WHERE id=$1 AND active=true", [serviceId]);
    if (!service.rowCount || !await managesHostel(req.user, service.rows[0].hostel_id)) return res.status(404).json({ error: "Transport service not found" });
    if (!isValidInt(studentId) || !await activeResident(studentId, service.rows[0].hostel_id)) return res.status(409).json({ error: "An active hostel resident is required" });
    const pickup = req.body.pickup_stop_id == null ? null : Number(req.body.pickup_stop_id), dropoff = req.body.dropoff_stop_id == null ? null : Number(req.body.dropoff_stop_id);
    if ((pickup != null && !isValidInt(pickup)) || (dropoff != null && !isValidInt(dropoff))) return res.status(400).json({ error: "Stop identifiers must be valid" });
    const stops = await pool.query("SELECT id FROM hostel_transport_stops WHERE service_id=$1 AND id=ANY($2::bigint[])", [serviceId, [pickup, dropoff].filter(Boolean)]);
    if (stops.rowCount !== new Set([pickup, dropoff].filter(Boolean)).size) return res.status(400).json({ error: "Stops must belong to the transport service" });
    try {
      const result = await pool.query("INSERT INTO student_transport_assignments (service_id,student_user_id,pickup_stop_id,dropoff_stop_id,assigned_by_user_id) VALUES ($1,$2,$3,$4,$5) RETURNING *", [serviceId, studentId, pickup, dropoff, req.user.id]);
      res.status(201).json({ assignment: result.rows[0] });
    } catch (error) {
      if (error.code === "23505") return res.status(409).json({ error: "Student is already assigned to this service" });
      throw error;
    }
  });

  app.post("/api/staff/transport-assignments/:id/events", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const id = Number(req.params.id), type = String(req.body.event_type || ""), occurredAt = req.body.occurred_at, key = String(req.headers["idempotency-key"] || ""), note = String(req.body.note || "").trim();
    const assignment = await pool.query("SELECT a.*,s.hostel_id,s.name AS service_name FROM student_transport_assignments a JOIN hostel_transport_services s ON s.id=a.service_id WHERE a.id=$1 AND a.active=true", [id]);
    if (!assignment.rowCount || !await managesHostel(req.user, assignment.rows[0].hostel_id)) return res.status(404).json({ error: "Transport assignment not found" });
    if (!key || key.length > 160 || !["departed", "picked_up", "arrived", "dropped_off", "delayed", "cancelled"].includes(type) || !occurredAt || Number.isNaN(Date.parse(occurredAt)) || note.length > 2000) return res.status(400).json({ error: "Valid event_type, occurred_at, note, and idempotency-key are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query("INSERT INTO transport_trip_events (assignment_id,event_type,occurred_at,note,idempotency_key,created_by_user_id) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (assignment_id,idempotency_key) DO NOTHING RETURNING *", [id, type, occurredAt, note || null, key, req.user.id]);
      if (!result.rowCount) {
        const prior = await client.query("SELECT * FROM transport_trip_events WHERE assignment_id=$1 AND idempotency_key=$2", [id, key]);
        await client.query("COMMIT");
        return res.json({ event: prior.rows[0], duplicate: true });
      }
      await notifyLinkedParents(client, assignment.rows[0].student_user_id, "transport_update", `Transport ${type}`, `${assignment.rows[0].service_name} reported ${type}.`, { student_user_id: assignment.rows[0].student_user_id, assignment_id: id, event_id: result.rows[0].id, event_type: type }, "transport_updates");
      await client.query("COMMIT");
      res.status(201).json({ event: result.rows[0] });
    } catch (error) {
      await client.query("ROLLBACK");
      res.status(500).json({ error: "Failed to record transport event" });
    } finally { client.release(); }
  });

  async function transportForStudent(studentId) {
    return pool.query(
      `SELECT a.*,s.name AS service_name,s.mode,s.vehicle_number,s.driver_name,s.driver_phone,s.operating_days,s.safety_features,s.monthly_cost,s.currency,
              ps.name AS pickup_stop,ps.scheduled_time AS pickup_time,ds.name AS dropoff_stop,ds.scheduled_time AS dropoff_time,
              COALESCE((SELECT jsonb_agg(to_jsonb(e) ORDER BY e.occurred_at DESC) FROM (SELECT id,event_type,occurred_at,note FROM transport_trip_events WHERE assignment_id=a.id ORDER BY occurred_at DESC LIMIT 20) e),'[]'::jsonb) AS recent_events
       FROM student_transport_assignments a JOIN hostel_transport_services s ON s.id=a.service_id
       LEFT JOIN hostel_transport_stops ps ON ps.id=a.pickup_stop_id LEFT JOIN hostel_transport_stops ds ON ds.id=a.dropoff_stop_id
       WHERE a.student_user_id=$1 AND a.active=true AND s.active=true ORDER BY a.assigned_at DESC`,
      [studentId]
    );
  }

  app.get("/api/student/transport", requireAuth, requireRole("student"), async (req, res) => res.json({ assignments: (await transportForStudent(req.user.id)).rows }));
  app.get("/api/parent/children/:studentId/transport", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId) || !await linkedParent(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    res.json({ assignments: (await transportForStudent(studentId)).rows });
  });

  app.post("/api/parent/children/:studentId/meeting-requests", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId), hostelId = Number(req.body.hostel_id), staffId = req.body.requested_staff_user_id == null ? null : Number(req.body.requested_staff_user_id);
    const scheduledFor = req.body.scheduled_for, duration = Number(req.body.duration_minutes), mode = String(req.body.mode || ""), topic = String(req.body.topic || "").trim(), details = String(req.body.details || "").trim();
    if (!isValidInt(studentId) || !await linkedParent(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    if (!isValidInt(hostelId) || !await activeResident(studentId, hostelId)) return res.status(404).json({ error: "Active hostel stay not found" });
    if (!scheduledFor || Number.isNaN(Date.parse(scheduledFor)) || Date.parse(scheduledFor) <= Date.now() || !Number.isInteger(duration) || duration < 15 || duration > 120 || !["in_person", "voice"].includes(mode) || !topic || topic.length > 240 || details.length > 4000) return res.status(400).json({ error: "Future schedule, 15-120 minute duration, mode, and topic are required" });
    if (staffId != null) {
      const staff = await pool.query("SELECT 1 FROM users u WHERE u.id=$1 AND ((u.role='owner' AND EXISTS(SELECT 1 FROM hostel_owners ho WHERE ho.owner_user_id=u.id AND ho.hostel_id=$2)) OR (u.role='warden' AND EXISTS(SELECT 1 FROM warden_assignments wa WHERE wa.warden_user_id=u.id AND wa.hostel_id=$2)))", [staffId, hostelId]);
      if (!staff.rowCount) return res.status(400).json({ error: "Requested staff member is not assigned to the hostel" });
    }
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query("INSERT INTO parent_staff_meeting_requests (parent_user_id,student_user_id,hostel_id,requested_staff_user_id,scheduled_for,duration_minutes,mode,topic,details) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *", [req.user.id, studentId, hostelId, staffId, scheduledFor, duration, mode, topic, details || null]);
      await client.query("INSERT INTO parent_staff_meeting_events (meeting_id,event_type,actor_user_id) VALUES ($1,'requested',$2)", [result.rows[0].id, req.user.id]);
      await client.query("COMMIT");
      res.status(201).json({ meeting: result.rows[0] });
    } catch (error) {
      await client.query("ROLLBACK");
      res.status(500).json({ error: "Failed to request meeting" });
    } finally { client.release(); }
  });

  app.get("/api/parent/meeting-requests", requireAuth, requireRole("parent"), async (req, res) => {
    const result = await pool.query("SELECT m.*,h.name AS hostel_name,u.name AS student_name,s.name AS requested_staff_name FROM parent_staff_meeting_requests m JOIN hostels h ON h.id=m.hostel_id JOIN users u ON u.id=m.student_user_id LEFT JOIN users s ON s.id=m.requested_staff_user_id WHERE m.parent_user_id=$1 ORDER BY m.created_at DESC", [req.user.id]);
    res.json({ meetings: result.rows });
  });

  app.get("/api/staff/meeting-requests", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const relation = req.user.role === "owner" ? "hostel_owners" : "warden_assignments";
    const column = req.user.role === "owner" ? "owner_user_id" : "warden_user_id";
    const result = await pool.query(`SELECT m.*,p.name AS parent_name,s.name AS student_name,h.name AS hostel_name FROM parent_staff_meeting_requests m JOIN ${relation} r ON r.hostel_id=m.hostel_id AND r.${column}=$1 JOIN users p ON p.id=m.parent_user_id JOIN users s ON s.id=m.student_user_id JOIN hostels h ON h.id=m.hostel_id WHERE m.requested_staff_user_id IS NULL OR m.requested_staff_user_id=$1 ORDER BY m.scheduled_for`, [req.user.id]);
    res.json({ meetings: result.rows });
  });

  app.patch("/api/staff/meeting-requests/:id", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const id = Number(req.params.id), status = String(req.body.status || ""), note = String(req.body.response_note || "").trim(), scheduledFor = req.body.scheduled_for;
    if (!isValidInt(id) || !["accepted", "declined", "completed", "rescheduled"].includes(status) || (status === "rescheduled" && (!scheduledFor || Number.isNaN(Date.parse(scheduledFor)) || Date.parse(scheduledFor) <= Date.now())) || note.length > 4000) return res.status(400).json({ error: "Valid status, response note, and future schedule for rescheduling are required" });
    const meeting = await pool.query("SELECT * FROM parent_staff_meeting_requests WHERE id=$1", [id]);
    if (!meeting.rowCount || !await managesHostel(req.user, meeting.rows[0].hostel_id) || (meeting.rows[0].requested_staff_user_id && Number(meeting.rows[0].requested_staff_user_id) !== Number(req.user.id))) return res.status(404).json({ error: "Meeting request not found" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const updated = await client.query("UPDATE parent_staff_meeting_requests SET status=$1::varchar,response_note=$2,scheduled_for=CASE WHEN $1::varchar='rescheduled' THEN $3::timestamptz ELSE scheduled_for END,responded_by_user_id=$4,responded_at=NOW(),updated_at=NOW() WHERE id=$5 AND status IN ('pending','accepted','rescheduled') RETURNING *", [status, note || null, scheduledFor || null, req.user.id, id]);
      if (!updated.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Meeting request is already final" }); }
      await client.query("INSERT INTO parent_staff_meeting_events (meeting_id,event_type,actor_user_id,details) VALUES ($1,$2,$3,$4::jsonb)", [id, status, req.user.id, JSON.stringify({ response_note: note || null, scheduled_for: scheduledFor || null })]);
      await createNotification(client, updated.rows[0].parent_user_id, "meeting_update", `Meeting ${status}`, `Your hostel meeting request is ${status}.`, { meeting_id: id, status, scheduled_for: updated.rows[0].scheduled_for }, "communication_updates");
      await client.query("COMMIT");
      res.json({ meeting: updated.rows[0] });
    } catch (error) {
      await client.query("ROLLBACK");
      res.status(500).json({ error: "Failed to update meeting request", ...(process.env.RUN_DB_TESTS === "1" ? { detail: error.message } : {}) });
    } finally { client.release(); }
  });

  app.post("/api/parent/meeting-requests/:id/cancel", requireAuth, requireRole("parent"), async (req, res) => {
    const id = Number(req.params.id);
    const result = await pool.query("UPDATE parent_staff_meeting_requests SET status='cancelled',cancelled_at=NOW(),updated_at=NOW() WHERE id=$1 AND parent_user_id=$2 AND status IN ('pending','accepted','rescheduled') RETURNING *", [id, req.user.id]);
    if (!result.rowCount) return res.status(404).json({ error: "Cancellable meeting request not found" });
    await pool.query("INSERT INTO parent_staff_meeting_events (meeting_id,event_type,actor_user_id) VALUES ($1,'cancelled',$2)", [id, req.user.id]);
    res.json({ meeting: result.rows[0] });
  });
}

module.exports = { registerParentServicesRoutes };
