function registerEmergencyRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification }) {
  const emergencyTypes = new Set(["medical", "safety", "fire", "accident", "general"]);
  const staffCanManage = async (user, hostelId) => user.role === "owner" ? ownerHasHostel(pool, user.id, hostelId) : wardenHasHostel(pool, user.id, hostelId);

  app.get("/api/emergency-contacts", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query("SELECT * FROM emergency_contacts WHERE student_user_id=$1 AND active=true ORDER BY priority,id", [req.user.id]);
    res.json({ contacts: result.rows });
  });
  app.post("/api/emergency-contacts", requireAuth, requireRole("student"), async (req, res) => {
    const { name, relationship, phone, email, priority } = req.body;
    if (!name || !relationship || (!phone && !email) || (priority != null && (!Number.isInteger(Number(priority)) || Number(priority) < 1))) return res.status(400).json({ error: "name, relationship, a phone or email, and an optional positive priority are required" });
    const result = await pool.query("INSERT INTO emergency_contacts (student_user_id,name,relationship,phone,email,priority) VALUES ($1,$2,$3,$4,$5,$6) RETURNING *", [req.user.id, String(name).trim(), String(relationship).trim(), phone || null, email || null, Number(priority || 1)]);
    res.status(201).json({ contact: result.rows[0] });
  });
  app.delete("/api/emergency-contacts/:id", requireAuth, requireRole("student"), async (req, res) => {
    const result = await pool.query("UPDATE emergency_contacts SET active=false WHERE id=$1 AND student_user_id=$2 AND active=true RETURNING id", [Number(req.params.id), req.user.id]);
    if (!result.rowCount) return res.status(404).json({ error: "Emergency contact not found" }); res.status(204).end();
  });

  app.post("/api/emergencies", requireAuth, requireRole("student"), async (req, res) => {
    const key = String(req.headers["idempotency-key"] || ""); const { alert_type, description, latitude, longitude, location_accuracy_m } = req.body;
    if (!key || !emergencyTypes.has(alert_type) || (latitude != null && (!Number.isFinite(Number(latitude)) || !Number.isFinite(Number(longitude))))) return res.status(400).json({ error: "idempotency-key, valid alert_type, and complete optional coordinates are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const duplicate = await client.query("SELECT * FROM emergency_incidents WHERE idempotency_key=$1", [key]);
      if (duplicate.rowCount) { await client.query("ROLLBACK"); return res.json({ incident: duplicate.rows[0], duplicate: true }); }
      const booking = await client.query("SELECT b.id,b.hostel_id,b.room_id FROM bookings b WHERE b.user_id=$1 AND b.status IN ('pending','confirmed') ORDER BY b.created_at DESC LIMIT 1 FOR UPDATE", [req.user.id]);
      if (!booking.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "An active hostel booking is required for an emergency alert" }); }
      const incident = await client.query(`INSERT INTO emergency_incidents (student_user_id,hostel_id,booking_id,room_id,alert_type,description,latitude,longitude,location_accuracy_m,idempotency_key)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING *`, [req.user.id, booking.rows[0].hostel_id, booking.rows[0].id, booking.rows[0].room_id, alert_type, description || null, latitude ?? null, longitude ?? null, location_accuracy_m ?? null, key]);
      const row = incident.rows[0]; await client.query("INSERT INTO emergency_timeline_events (incident_id,event_type,actor_user_id,details) VALUES ($1,'created',$2,$3::jsonb)", [row.id, req.user.id, JSON.stringify({ alert_type })]);
      const recipientResult = await client.query(`SELECT parent_user_id AS user_id FROM parent_student_links WHERE student_user_id=$1 AND status='active'
        UNION SELECT warden_user_id FROM warden_assignments WHERE hostel_id=$2
        UNION SELECT owner_user_id FROM hostel_owners WHERE hostel_id=$2`, [req.user.id, row.hostel_id]);
      for (const recipient of recipientResult.rows) {
        await createNotification(client, recipient.user_id, "emergency_alert", "Emergency alert", `A student emergency alert (${alert_type}) requires attention.`, { emergency_incident_id: row.id, hostel_id: row.hostel_id, student_id: req.user.id }, "emergency_alerts");
        await client.query("INSERT INTO emergency_delivery_attempts (incident_id,recipient_user_id,channel,status,attempt_count) VALUES ($1,$2,'in_app','delivered',1)", [row.id, recipient.user_id]);
      }
      const contacts = await client.query("SELECT id FROM emergency_contacts WHERE student_user_id=$1 AND active=true", [req.user.id]);
      for (const contact of contacts.rows) await client.query("INSERT INTO emergency_delivery_attempts (incident_id,contact_id,channel,status,next_retry_at) VALUES ($1,$2,'sms','queued',NOW())", [row.id, contact.id]);
      await client.query("COMMIT"); res.status(201).json({ incident: row });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to create emergency alert" }); } finally { client.release(); }
  });

  app.get("/api/emergencies", requireAuth, requireRole("student", "owner", "warden"), async (req, res) => {
    let result;
    if (req.user.role === "student") result = await pool.query("SELECT * FROM emergency_incidents WHERE student_user_id=$1 ORDER BY created_at DESC", [req.user.id]);
    else if (req.user.role === "owner") result = await pool.query("SELECT e.* FROM emergency_incidents e JOIN hostel_owners ho ON ho.hostel_id=e.hostel_id WHERE ho.owner_user_id=$1 ORDER BY e.created_at DESC", [req.user.id]);
    else result = await pool.query("SELECT e.* FROM emergency_incidents e JOIN warden_assignments wa ON wa.hostel_id=e.hostel_id WHERE wa.warden_user_id=$1 ORDER BY e.created_at DESC", [req.user.id]);
    res.json({ incidents: result.rows });
  });

  app.patch("/api/emergencies/:id", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const incidentId = Number(req.params.id); const status = String(req.body.status || ""); const reason = String(req.body.reason || "").trim();
    if (!isValidInt(incidentId) || !["acknowledged", "escalated", "resolved"].includes(status)) return res.status(400).json({ error: "A valid incident id and staff status are required" });
    const client = await pool.connect(); try { await client.query("BEGIN"); const incident = await client.query("SELECT * FROM emergency_incidents WHERE id=$1 FOR UPDATE", [incidentId]);
      if (!incident.rowCount || !await staffCanManage(req.user, incident.rows[0].hostel_id)) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Emergency incident not found" }); }
      if (["cancelled", "resolved"].includes(incident.rows[0].status)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Emergency incident is already closed" }); }
      const updated = await client.query(`UPDATE emergency_incidents SET status=$1::varchar, resolved_by_user_id=CASE WHEN $1::varchar='resolved' THEN $2 ELSE resolved_by_user_id END, resolved_at=CASE WHEN $1::varchar='resolved' THEN NOW() ELSE resolved_at END, resolution_reason=CASE WHEN $1::varchar='resolved' THEN $3 ELSE resolution_reason END WHERE id=$4 RETURNING *`, [status, req.user.id, reason || null, incidentId]);
      await client.query("INSERT INTO emergency_timeline_events (incident_id,event_type,actor_user_id,details) VALUES ($1,$2,$3,$4::jsonb)", [incidentId, status, req.user.id, JSON.stringify({ reason: reason || null })]); await client.query("COMMIT"); res.json({ incident: updated.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to update emergency incident", ...(process.env.RUN_DB_TESTS === "1" ? { detail: error.message } : {}) }); } finally { client.release(); }
  });

  app.post("/api/emergencies/:id/cancel", requireAuth, requireRole("student"), async (req, res) => {
    const reason = String(req.body.reason || "").trim(); const incidentId = Number(req.params.id); if (!reason) return res.status(400).json({ error: "Cancellation reason is required" });
    const client = await pool.connect(); try { await client.query("BEGIN"); const incident = await client.query("SELECT * FROM emergency_incidents WHERE id=$1 AND student_user_id=$2 FOR UPDATE", [incidentId, req.user.id]);
      if (!incident.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Emergency incident not found" }); } if (incident.rows[0].status !== "active") { await client.query("ROLLBACK"); return res.status(409).json({ error: "Only an active emergency can be cancelled" }); }
      const updated = await client.query("UPDATE emergency_incidents SET status='cancelled',cancelled_by_user_id=$1,cancelled_at=NOW(),cancellation_reason=$2 WHERE id=$3 RETURNING *", [req.user.id, reason, incidentId]); await client.query("INSERT INTO emergency_timeline_events (incident_id,event_type,actor_user_id,details) VALUES ($1,'cancelled',$2,$3::jsonb)", [incidentId, req.user.id, JSON.stringify({ reason })]); await client.query("COMMIT"); res.json({ incident: updated.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to cancel emergency incident" }); } finally { client.release(); }
  });

  app.get("/api/parent/children/:studentId/emergencies", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    const link = await pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'", [req.user.id, studentId]);
    if (!isValidInt(studentId) || !link.rowCount) return res.status(404).json({ error: "Linked student not found" });
    const incidents = await pool.query("SELECT * FROM emergency_incidents WHERE student_user_id=$1 ORDER BY created_at DESC", [studentId]);
    res.json({ incidents: incidents.rows });
  });
}

module.exports = { registerEmergencyRoutes };
