const crypto = require("node:crypto");

const MESSAGE_ATTACHMENT_TYPES = new Set(["application/pdf", "image/jpeg", "image/png"]);
const MAX_MESSAGE_ATTACHMENT_BYTES = 5 * 1024 * 1024;

function validBase64(value) {
  return typeof value === "string" && value.length > 0 && /^[A-Za-z0-9+/]+={0,2}$/.test(value) && value.length % 4 === 0;
}

function matchesDeclaredAttachmentType(bytes, contentType) {
  if (contentType === "application/pdf") return bytes.subarray(0, 5).equals(Buffer.from("%PDF-"));
  if (contentType === "image/jpeg") return bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  if (contentType === "image/png") return bytes.length >= 8 && bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  return false;
}

function safeDownloadName(value) {
  return String(value || "attachment").replace(/[\\/\r\n]/g, "_");
}

function attachmentResponse(row) {
  return {
    id: row.id, message_id: row.message_id, uploaded_by_user_id: row.uploaded_by_user_id,
    original_filename: row.original_filename, content_type: row.content_type,
    byte_size: Number(row.byte_size), scan_status: row.scan_status, created_at: row.created_at,
  };
}

function registerMessagingRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, documentStorage }) {
  const realtimeClients = new Map();
  function publishRealtime(userIds, event, payload) {
    const frame = `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
    for (const userId of new Set(userIds.filter((userId) => Number.isInteger(Number(userId)) && Number(userId) > 0).map(Number))) {
      for (const response of realtimeClients.get(userId) || []) response.write(frame);
    }
  }
  async function activeLink(parentId, studentId) {
    const r = await pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'", [parentId, studentId]);
    return r.rowCount > 0;
  }
  async function participant(conversationId, user) {
    const r = await pool.query("SELECT * FROM conversations WHERE id=$1", [conversationId]);
    if (!r.rowCount) return null;
    const c = r.rows[0];
    return [c.parent_user_id, c.student_user_id, c.staff_user_id].map(Number).includes(Number(user.id)) ? c : null;
  }
  async function messageForParticipant(messageId, user) {
    const r = await pool.query("SELECT m.*,c.parent_user_id,c.student_user_id,c.staff_user_id FROM conversation_messages m JOIN conversations c ON c.id=m.conversation_id WHERE m.id=$1", [messageId]);
    if (!r.rowCount) return null;
    const row = r.rows[0];
    return [row.parent_user_id, row.student_user_id, row.staff_user_id].map(Number).includes(Number(user.id)) ? row : null;
  }

  app.post("/api/parent/conversations", requireAuth, requireRole("parent"), async (req, res) => {
    const studentId = Number(req.body.student_id), type = String(req.body.type || "parent_student"), staffId = Number(req.body.staff_user_id);
    if (!isValidInt(studentId) || !await activeLink(req.user.id, studentId)) return res.status(404).json({ error: "Linked student not found" });
    if (!["parent_student", "parent_staff"].includes(type)) return res.status(400).json({ error: "Invalid conversation type" });
    let hostelId = null;
    if (type === "parent_staff") {
      if (!isValidInt(staffId)) return res.status(400).json({ error: "staff_user_id is required" });
      const b = await pool.query("SELECT hostel_id FROM bookings WHERE user_id=$1 AND status IN ('pending','confirmed') ORDER BY created_at DESC LIMIT 1", [studentId]);
      if (!b.rowCount) return res.status(409).json({ error: "Student has no active hostel booking" });
      hostelId = b.rows[0].hostel_id;
      const staff = await pool.query("SELECT role FROM users WHERE id=$1", [staffId]);
      const allowed = staff.rowCount && (staff.rows[0].role === "owner" ? await ownerHasHostel(pool, staffId, hostelId) : staff.rows[0].role === "warden" ? await wardenHasHostel(pool, staffId, hostelId) : false);
      if (!allowed) return res.status(404).json({ error: "Eligible hostel staff member not found" });
    }
    const existing = await pool.query("SELECT * FROM conversations WHERE conversation_type=$1 AND parent_user_id=$2 AND student_user_id=$3 AND COALESCE(hostel_id,0)=COALESCE($4,0) AND COALESCE(staff_user_id,0)=COALESCE($5,0)", [type, req.user.id, studentId, hostelId, Number.isFinite(staffId) ? staffId : null]);
    if (existing.rowCount) return res.json({ conversation: existing.rows[0], existing: true });
    const r = await pool.query("INSERT INTO conversations (conversation_type,parent_user_id,student_user_id,hostel_id,staff_user_id) VALUES ($1,$2,$3,$4,$5) RETURNING *", [type, req.user.id, studentId, hostelId, type === "parent_staff" ? staffId : null]);
    res.status(201).json({ conversation: r.rows[0] });
  });

  // A warden can start one private conversation with each student who has an
  // active booking in one of the warden's assigned hostels.  These rows reuse
  // the existing staff conversation/message tables, so delivery, read state,
  // realtime events, and the student's conversation history stay in one place.
  app.post("/api/warden/conversations", requireAuth, requireRole("warden"), async (req, res) => {
    const studentId = Number(req.body.student_id);
    if (!isValidInt(studentId)) return res.status(400).json({ error: "student_id is required" });

    const booking = await pool.query(
      `SELECT b.hostel_id
         FROM bookings b
         JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id
        WHERE b.user_id=$1
          AND wa.warden_user_id=$2
          AND b.status IN ('pending','confirmed')
        ORDER BY b.id DESC
        LIMIT 1`,
      [studentId, req.user.id],
    );
    if (!booking.rowCount) return res.status(404).json({ error: "Student is not in your assigned hostel" });

    const hostelId = booking.rows[0].hostel_id;
    const existing = await pool.query(
      `SELECT * FROM conversations
        WHERE conversation_type='parent_staff'
          AND staff_user_id=$1
          AND student_user_id=$2
          AND hostel_id=$3
        LIMIT 1`,
      [req.user.id, studentId, hostelId],
    );
    if (existing.rowCount) return res.json({ conversation: existing.rows[0], existing: true });

    // The legacy parent column is deliberately set to the student for this
    // staff-only thread. The access checks below use student_user_id and
    // staff_user_id; no parent account receives access to this conversation.
    const created = await pool.query(
      `INSERT INTO conversations
        (conversation_type,parent_user_id,student_user_id,hostel_id,staff_user_id)
       VALUES ('parent_staff',$1,$1,$2,$3)
       RETURNING *`,
      [studentId, hostelId, req.user.id],
    );
    res.status(201).json({ conversation: created.rows[0] });
  });

  app.get("/api/warden/conversations", requireAuth, requireRole("warden"), async (req, res) => {
    const r = await pool.query(
      `SELECT c.*, u.name AS student_name, u.email AS student_email,
              latest.body AS last_message, latest.created_at AS last_message_at
         FROM conversations c
         JOIN users u ON u.id=c.student_user_id
         LEFT JOIN LATERAL (
           SELECT body, created_at FROM conversation_messages
            WHERE conversation_id=c.id
            ORDER BY created_at DESC LIMIT 1
         ) latest ON TRUE
        WHERE c.conversation_type='parent_staff' AND c.staff_user_id=$1
        ORDER BY COALESCE(latest.created_at,c.updated_at) DESC`,
      [req.user.id],
    );
    res.json(r.rows);
  });

  async function wardenConversation(conversationId, wardenId) {
    const r = await pool.query(
      `SELECT * FROM conversations
        WHERE id=$1 AND conversation_type='parent_staff' AND staff_user_id=$2`,
      [conversationId, wardenId],
    );
    return r.rowCount ? r.rows[0] : null;
  }

  app.get("/api/warden/conversations/:id/messages", requireAuth, requireRole("warden"), async (req, res) => {
    const c = await wardenConversation(Number(req.params.id), req.user.id);
    if (!c) return res.status(404).json({ error: "Conversation not found" });
    const r = await pool.query(
      `SELECT *, sender_user_id=$2 AS is_mine
         FROM conversation_messages
        WHERE conversation_id=$1
        ORDER BY created_at`,
      [c.id, req.user.id],
    );
    res.json(r.rows);
  });

  app.post("/api/warden/conversations/:id/messages", requireAuth, requireRole("warden"), async (req, res) => {
    const c = await wardenConversation(Number(req.params.id), req.user.id);
    const body = String(req.body.message || "").trim();
    if (!c) return res.status(404).json({ error: "Conversation not found" });
    if (!body || body.length > 4000) return res.status(400).json({ error: "Message must be 1 to 4000 characters" });
    const r = await pool.query(
      `INSERT INTO conversation_messages (conversation_id,sender_user_id,body)
       VALUES ($1,$2,$3) RETURNING *`,
      [c.id, req.user.id, body],
    );
    await pool.query("UPDATE conversations SET updated_at=NOW() WHERE id=$1", [c.id]);
    publishRealtime([c.student_user_id, c.staff_user_id], "message.created", { message: r.rows[0] });
    res.status(201).json({ message: { ...r.rows[0], is_mine: true } });
  });

  app.get("/api/conversations", requireAuth, requireRole("parent", "student", "owner", "warden"), async (req, res) => {
    const r = await pool.query("SELECT c.* FROM conversations c WHERE c.parent_user_id=$1 OR c.student_user_id=$1 OR c.staff_user_id=$1 ORDER BY c.updated_at DESC", [req.user.id]);
    res.json({ conversations: r.rows });
  });
  app.get("/api/messaging/stream", requireAuth, requireRole("parent", "student", "owner", "warden"), (req, res) => {
    res.status(200).set({
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });
    res.flushHeaders();
    const userId = Number(req.user.id);
    const clients = realtimeClients.get(userId) || new Set();
    clients.add(res); realtimeClients.set(userId, clients);
    res.write("event: ready\ndata: {}\n\n");
    const heartbeat = setInterval(() => res.write(": keepalive\n\n"), 25000);
    req.on("close", () => {
      clearInterval(heartbeat); clients.delete(res);
      if (!clients.size) realtimeClients.delete(userId);
    });
  });
  app.get("/api/conversations/:id/messages", requireAuth, async (req, res) => {
    const c = await participant(Number(req.params.id), req.user);
    if (!c) return res.status(404).json({ error: "Conversation not found" });
    const r = await pool.query("SELECT * FROM conversation_messages WHERE conversation_id=$1 ORDER BY created_at", [c.id]);
    res.json({ messages: r.rows });
  });

  app.post("/api/conversations/:id/handover", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const id = Number(req.params.id), nextStaffId = Number(req.body.next_staff_user_id), reason = String(req.body.reason || "").trim();
    if (!isValidInt(id) || !isValidInt(nextStaffId) || !reason || reason.length > 2000) return res.status(400).json({ error: "next_staff_user_id and a reason up to 2000 characters are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const current = await client.query("SELECT * FROM conversations WHERE id=$1 AND conversation_type='parent_staff' FOR UPDATE", [id]);
      if (!current.rowCount || Number(current.rows[0].staff_user_id) !== Number(req.user.id)) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Staff conversation not found" }); }
      if (Number(current.rows[0].staff_user_id) === nextStaffId) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Conversation is already assigned to this staff member" }); }
      const staff = await client.query("SELECT role FROM users WHERE id=$1", [nextStaffId]);
      const eligible = staff.rowCount && (staff.rows[0].role === "owner" ? await ownerHasHostel(client, nextStaffId, current.rows[0].hostel_id) : staff.rows[0].role === "warden" ? await wardenHasHostel(client, nextStaffId, current.rows[0].hostel_id) : false);
      if (!eligible) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Eligible hostel staff member not found" }); }
      const updated = await client.query("UPDATE conversations SET staff_user_id=$1,updated_at=NOW() WHERE id=$2 RETURNING *", [nextStaffId, id]);
      await client.query("INSERT INTO conversation_staff_handovers (conversation_id,previous_staff_user_id,next_staff_user_id,handed_over_by_user_id,reason) VALUES ($1,$2,$3,$4,$5)", [id, req.user.id, nextStaffId, req.user.id, reason]);
      await client.query("COMMIT"); res.json({ conversation: updated.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); if (error.code === "23505") return res.status(409).json({ error: "Target staff member already has this conversation" }); res.status(500).json({ error: "Failed to hand over conversation" }); }
    finally { client.release(); }
  });
  app.post("/api/conversations/:id/messages", requireAuth, async (req, res) => {
    const c = await participant(Number(req.params.id), req.user), body = String(req.body.body || "").trim();
    if (!c) return res.status(404).json({ error: "Conversation not found" });
    if (!body || body.length > 4000) return res.status(400).json({ error: "Message body must be 1 to 4000 characters" });
    const r = await pool.query("INSERT INTO conversation_messages (conversation_id,sender_user_id,body) VALUES ($1,$2,$3) RETURNING *", [c.id, req.user.id, body]);
    await pool.query("UPDATE conversations SET updated_at=NOW() WHERE id=$1", [c.id]);
    publishRealtime([c.parent_user_id, c.student_user_id, c.staff_user_id], "message.created", { message: r.rows[0] });
    res.status(201).json({ message: r.rows[0] });
  });

  app.post("/api/conversation-messages/:id/attachments", requireAuth, async (req, res) => {
    const messageId = Number(req.params.id), filename = String(req.body.filename || ""), contentType = req.body.content_type, base64 = req.body.content_base64;
    if (!isValidInt(messageId) || !filename || filename.length > 255 || !MESSAGE_ATTACHMENT_TYPES.has(contentType) || !validBase64(base64)) return res.status(400).json({ error: "A PDF, JPEG, or PNG attachment with base64 content is required" });
    const message = await messageForParticipant(messageId, req.user);
    if (!message || Number(message.sender_user_id) !== Number(req.user.id)) return res.status(404).json({ error: "Message not found" });
    const bytes = Buffer.from(base64, "base64");
    if (!bytes.length || bytes.length > MAX_MESSAGE_ATTACHMENT_BYTES) return res.status(413).json({ error: "Attachment must be between 1 byte and 5 MiB" });
    if (!matchesDeclaredAttachmentType(bytes, contentType)) return res.status(400).json({ error: "Attachment content does not match the declared PDF, JPEG, or PNG type" });
    const key = documentStorage.createKey("message_attachment", messageId), checksum = crypto.createHash("sha256").update(bytes).digest("hex");
    try {
      await documentStorage.save(key, bytes);
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        const saved = await client.query("INSERT INTO conversation_message_attachments (message_id,uploaded_by_user_id,storage_provider,storage_key,checksum,original_filename,content_type,byte_size) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING *", [messageId, req.user.id, documentStorage.provider, key, checksum, filename, contentType, bytes.length]);
        await client.query("INSERT INTO conversation_message_attachment_audit_events (attachment_id,event_type,actor_user_id) VALUES ($1,'uploaded',$2)", [saved.rows[0].id, req.user.id]);
        await client.query("COMMIT");
        res.status(201).json({ attachment: attachmentResponse(saved.rows[0]) });
      } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
    } catch (_) { await documentStorage.remove(key); res.status(500).json({ error: "Failed to store message attachment" }); }
  });
  app.get("/api/conversation-messages/:id/attachments", requireAuth, async (req, res) => {
    const message = await messageForParticipant(Number(req.params.id), req.user);
    if (!message) return res.status(404).json({ error: "Message not found" });
    const r = await pool.query("SELECT * FROM conversation_message_attachments WHERE message_id=$1 ORDER BY created_at", [message.id]);
    res.json({ attachments: r.rows.map(attachmentResponse) });
  });
  app.get("/api/message-attachments/:id/download", requireAuth, async (req, res) => {
    const id = Number(req.params.id);
    const r = await pool.query("SELECT a.*,c.parent_user_id,c.student_user_id,c.staff_user_id FROM conversation_message_attachments a JOIN conversation_messages m ON m.id=a.message_id JOIN conversations c ON c.id=m.conversation_id WHERE a.id=$1", [id]);
    if (!r.rowCount) return res.status(404).json({ error: "Attachment not found" });
    const attachment = r.rows[0];
    if (![attachment.parent_user_id, attachment.student_user_id, attachment.staff_user_id].map(Number).includes(Number(req.user.id)) || !await documentStorage.exists(attachment.storage_key)) return res.status(404).json({ error: "Attachment not found" });
    await pool.query("INSERT INTO conversation_message_attachment_audit_events (attachment_id,event_type,actor_user_id) VALUES ($1,'downloaded',$2)", [id, req.user.id]);
    res.type(attachment.content_type); res.attachment(safeDownloadName(attachment.original_filename));
    documentStorage.createReadStream(attachment.storage_key).on("error", () => res.destroy()).pipe(res);
  });

  app.post("/api/conversation-messages/:id/read", requireAuth, async (req, res) => {
    const m = await messageForParticipant(Number(req.params.id), req.user);
    if (!m) return res.status(404).json({ error: "Message not found" });
    if (Number(m.sender_user_id) === Number(req.user.id)) return res.status(409).json({ error: "Sender cannot mark their own message read" });
    const r = await pool.query("UPDATE conversation_messages SET read_at=COALESCE(read_at,NOW()) WHERE id=$1 RETURNING *", [m.id]); res.json({ message: r.rows[0] });
  });
  app.post("/api/conversation-messages/:id/reports", requireAuth, async (req, res) => {
    const id = Number(req.params.id), reason = String(req.body.reason || ""), details = String(req.body.details || "").trim();
    if (!isValidInt(id) || !["harassment", "spam", "safety", "other"].includes(reason) || details.length > 2000) return res.status(400).json({ error: "Valid reason and details up to 2000 characters are required" });
    const message = await messageForParticipant(id, req.user);
    if (!message) return res.status(404).json({ error: "Message not found" });
    if (Number(message.sender_user_id) === Number(req.user.id)) return res.status(409).json({ error: "Senders cannot report their own messages" });
    try { const r = await pool.query("INSERT INTO conversation_message_reports (message_id,reporter_user_id,reason,details) VALUES ($1,$2,$3,$4) RETURNING *", [id, req.user.id, reason, details || null]); res.status(201).json({ report: r.rows[0] }); }
    catch (error) { if (error.code === "23505") return res.status(409).json({ error: "You have already reported this message" }); res.status(500).json({ error: "Failed to report message" }); }
  });
  app.get("/api/admin/message-reports", requireAuth, requireRole("admin"), async (req, res) => {
    const status = req.query.status || null;
    if (status && !["open", "reviewed", "dismissed"].includes(status)) return res.status(400).json({ error: "Invalid report status" });
    const r = await pool.query("SELECT r.*,m.body,m.conversation_id,m.sender_user_id FROM conversation_message_reports r JOIN conversation_messages m ON m.id=r.message_id WHERE ($1::varchar IS NULL OR r.status=$1) ORDER BY r.created_at DESC", [status]); res.json({ reports: r.rows });
  });
  app.patch("/api/admin/message-reports/:id", requireAuth, requireRole("admin"), async (req, res) => {
    const id = Number(req.params.id), status = String(req.body.status || ""), note = String(req.body.review_note || "").trim();
    if (!isValidInt(id) || !["reviewed", "dismissed"].includes(status) || !note || note.length > 2000) return res.status(400).json({ error: "Valid status and review_note are required" });
    const r = await pool.query("UPDATE conversation_message_reports SET status=$1,review_note=$2,reviewed_by_user_id=$3,reviewed_at=NOW() WHERE id=$4 AND status='open' RETURNING *", [status, note, req.user.id, id]);
    if (!r.rowCount) return res.status(404).json({ error: "Open report not found" }); res.json({ report: r.rows[0] });
  });
}
module.exports = { registerMessagingRoutes };
