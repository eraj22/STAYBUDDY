function registerParentPhase3Routes({ app, pool, requireAuth, requireRole, isValidInt }) {
  async function linkedStudent(parentId, studentId) {
    const result = await pool.query(
      `SELECT u.id,u.name FROM parent_student_links l
       JOIN users u ON u.id=l.student_user_id
       WHERE l.parent_user_id=$1 AND l.student_user_id=$2 AND l.status='active'`,
      [parentId, studentId]
    );
    return result.rows[0] || null;
  }

  async function resolveChild(parentId, requestedId) {
    if (requestedId != null) {
      const studentId = Number(requestedId);
      return isValidInt(studentId) ? linkedStudent(parentId, studentId) : null;
    }
    const result = await pool.query(
      `SELECT u.id,u.name FROM parent_student_links l
       JOIN users u ON u.id=l.student_user_id
       WHERE l.parent_user_id=$1 AND l.status='active' ORDER BY u.name`,
      [parentId]
    );
    return result.rowCount === 1 ? result.rows[0] : { needs_selection: true, children: result.rows };
  }

  async function announcementsForChild(parentId, studentId) {
    return pool.query(
      `SELECT DISTINCT a.*,aa.acknowledged_at
       FROM announcements a
       LEFT JOIN bookings b ON b.hostel_id=a.hostel_id AND b.user_id=$2 AND b.status IN ('pending','confirmed')
       LEFT JOIN favorites f ON f.hostel_id=a.hostel_id AND f.user_id=$2
       LEFT JOIN announcement_acknowledgements aa ON aa.announcement_id=a.id AND aa.user_id=$1
       WHERE (a.audience IN ('residents','booked') AND b.id IS NOT NULL)
          OR (a.audience IN ('residents','favorited') AND f.user_id IS NOT NULL)
       ORDER BY a.created_at DESC`,
      [parentId, studentId]
    );
  }

  function identifyIntent(transcript) {
    const text = transcript.toLowerCase();
    if (/\b(emergency|emergencies|sos|incident|accident|danger)\b/.test(text)) return "child_emergencies";
    if (/\b(where|location|gps|track|tracking)\b/.test(text)) return "child_location";
    if (/\b(fee|fees|invoice|payment|balance|due)\b/.test(text)) return "child_fees";
    if (/\b(attendance|entered|exited|entry|exit|curfew|inside|outside|return|returned)\b/.test(text)) return "child_attendance";
    if (/\b(announcement|announcements|notice|notices|update|updates)\b/.test(text)) return "child_announcements";
    if (/\b(hostel|room|warden|accommodation|stay)\b/.test(text)) return "child_overview";
    return "unknown";
  }

  app.post("/api/parent/voice/command", requireAuth, requireRole("parent"), async (req, res) => {
    const transcript = String(req.body?.transcript || "").trim().replace(/\s+/g, " ");
    if (!transcript || transcript.length > 1000) return res.status(400).json({ error: "A transcript of up to 1000 characters is required" });
    const child = await resolveChild(req.user.id, req.body.student_id);
    if (!child) return res.status(404).json({ error: "Linked student not found" });
    if (child.needs_selection) {
      return res.json({ intent: "select_child", transcript, speech_recognition_performed: false, clarification_questions: child.children.length ? ["Which linked child do you mean?"] : ["Link a student before using parent voice actions."], children: child.children });
    }
    const intent = identifyIntent(transcript);
    if (intent === "unknown") return res.json({ intent, transcript, student: child, speech_recognition_performed: false, clarification_questions: ["Ask about hostel details, attendance, fees, announcements, emergencies, or consented location."], data: null });

    let data;
    let responseText;
    if (intent === "child_attendance") {
      const result = await pool.query(
        `SELECT e.id,e.hostel_id,COALESCE(c.corrected_direction,e.direction) AS direction,
                COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at,e.source,e.entry_point
         FROM attendance_events e LEFT JOIN LATERAL (
           SELECT corrected_direction,corrected_occurred_at FROM attendance_corrections
           WHERE attendance_event_id=e.id ORDER BY created_at DESC LIMIT 1
         ) c ON true WHERE e.student_user_id=$1 ORDER BY occurred_at DESC LIMIT 10`,
        [child.id]
      );
      data = { events: result.rows };
      responseText = result.rowCount ? `${child.name}'s latest attendance event was ${result.rows[0].direction}.` : `No attendance events are available for ${child.name}.`;
    } else if (intent === "child_fees") {
      const result = await pool.query(
        `SELECT i.id,i.amount,i.currency,i.due_date,i.status,
                COALESCE(SUM(a.amount),0)::numeric AS paid_amount,
                GREATEST(i.amount-COALESCE(SUM(a.amount),0),0)::numeric AS balance
         FROM fee_invoices i JOIN student_fee_assignments s ON s.id=i.assignment_id
         LEFT JOIN fee_receipt_allocations a ON a.invoice_id=i.id
         WHERE s.student_user_id=$1 GROUP BY i.id ORDER BY i.due_date DESC LIMIT 20`,
        [child.id]
      );
      data = { invoices: result.rows };
      const open = result.rows.filter((row) => Number(row.balance) > 0);
      responseText = open.length ? `${child.name} has ${open.length} fee invoice${open.length === 1 ? "" : "s"} with an outstanding balance.` : `No outstanding fee balance is available for ${child.name}.`;
    } else if (intent === "child_announcements") {
      const result = await announcementsForChild(req.user.id, child.id);
      data = { announcements: result.rows.slice(0, 20) };
      responseText = result.rowCount ? `There are ${result.rowCount} hostel announcement${result.rowCount === 1 ? "" : "s"} for ${child.name}.` : `No hostel announcements are available for ${child.name}.`;
    } else if (intent === "child_emergencies") {
      const result = await pool.query("SELECT id,alert_type,description,status,created_at,resolved_at FROM emergency_incidents WHERE student_user_id=$1 ORDER BY created_at DESC LIMIT 20", [child.id]);
      data = { emergencies: result.rows };
      const active = result.rows.filter((row) => !["cancelled", "resolved"].includes(row.status));
      responseText = active.length ? `${child.name} has ${active.length} active emergency incident${active.length === 1 ? "" : "s"}.` : `There are no active emergency incidents for ${child.name}.`;
    } else if (intent === "child_location") {
      const session = await pool.query("SELECT * FROM location_sharing_sessions WHERE student_user_id=$1 AND revoked_at IS NULL AND expires_at>NOW() ORDER BY started_at DESC LIMIT 1", [child.id]);
      if (!session.rowCount) {
        data = { session: null, location: null };
        responseText = `${child.name} is not currently sharing location.`;
      } else {
        const location = await pool.query("SELECT latitude,longitude,accuracy_m,reported_at,EXTRACT(EPOCH FROM (NOW()-reported_at))::int AS age_seconds,(reported_at<NOW()-INTERVAL '5 minutes') AS stale FROM location_sharing_updates WHERE session_id=$1 ORDER BY reported_at DESC LIMIT 1", [session.rows[0].id]);
        data = { session: session.rows[0], location: location.rows[0] || null };
        responseText = location.rowCount ? `${child.name}'s latest consented location is available.` : `${child.name} is sharing location but has not sent a location update.`;
        if (location.rowCount) await pool.query("INSERT INTO location_sharing_audit_events (session_id,event_type,actor_user_id,details) VALUES ($1,'viewed',$2,$3::jsonb)", [session.rows[0].id, req.user.id, JSON.stringify({ view: "parent_voice" })]);
      }
    } else {
      const result = await pool.query(
        `SELECT b.id AS booking_id,b.status,h.id AS hostel_id,h.name AS hostel_name,h.address,h.city,
                r.room_number,r.room_type,bs.checked_in_at,bs.checked_out_at,
                COALESCE(jsonb_agg(DISTINCT jsonb_build_object('id',w.id,'name',w.name,'phone',w.phone)) FILTER (WHERE w.id IS NOT NULL),'[]'::jsonb) AS wardens
         FROM bookings b JOIN hostels h ON h.id=b.hostel_id
         LEFT JOIN rooms r ON r.id=b.room_id LEFT JOIN booking_stays bs ON bs.booking_id=b.id
         LEFT JOIN warden_assignments wa ON wa.hostel_id=h.id LEFT JOIN users w ON w.id=wa.warden_user_id
         WHERE b.user_id=$1 GROUP BY b.id,h.id,r.id,bs.booking_id ORDER BY b.created_at DESC LIMIT 10`,
        [child.id]
      );
      data = { stays: result.rows };
      responseText = result.rowCount ? `${child.name}'s latest hostel is ${result.rows[0].hostel_name}.` : `No hostel booking is available for ${child.name}.`;
    }
    res.json({ intent, transcript, student: child, response_text: responseText, data, clarification_questions: [], speech_recognition_performed: false });
  });

  app.get("/api/parent/children/:studentId/announcements", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId) || !await linkedStudent(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    const result = await announcementsForChild(req.user.id, studentId);
    res.json({ announcements: result.rows });
  });

  app.post("/api/parent/announcements/:id/acknowledge", requireAuth, requireRole("parent"), async (req, res) => {
    const announcementId = Number(req.params.id), studentId = Number(req.body.student_id);
    if (!isValidInt(announcementId) || !isValidInt(studentId) || !await linkedStudent(req.user.id, studentId)) return res.status(404).json({ error: "Announcement not found" });
    const allowed = await pool.query(
      `SELECT a.id FROM announcements a
       LEFT JOIN bookings b ON b.hostel_id=a.hostel_id AND b.user_id=$2 AND b.status IN ('pending','confirmed')
       LEFT JOIN favorites f ON f.hostel_id=a.hostel_id AND f.user_id=$2
       WHERE a.id=$1 AND ((a.audience IN ('residents','booked') AND b.id IS NOT NULL) OR (a.audience IN ('residents','favorited') AND f.user_id IS NOT NULL))`,
      [announcementId, studentId]
    );
    if (!allowed.rowCount) return res.status(404).json({ error: "Announcement not found" });
    const result = await pool.query(
      `INSERT INTO announcement_acknowledgements (announcement_id,user_id,student_user_id)
       VALUES ($1,$2,$3) ON CONFLICT (announcement_id,user_id) DO UPDATE SET acknowledged_at=announcement_acknowledgements.acknowledged_at
       RETURNING *`,
      [announcementId, req.user.id, studentId]
    );
    res.json({ acknowledgement: result.rows[0] });
  });
}

module.exports = { registerParentPhase3Routes };
