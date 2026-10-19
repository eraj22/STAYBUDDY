const crypto = require("node:crypto");

const MAX_EVIDENCE_BYTES = 5 * 1024 * 1024;
const ALLOWED_CONTENT_TYPES = new Set(["application/pdf", "image/jpeg", "image/png"]);

function validBase64(value) { return typeof value === "string" && value.length > 0 && value.length % 4 === 0 && /^[A-Za-z0-9+/]+={0,2}$/.test(value); }
function validSignature(bytes,type){if(type==="application/pdf")return bytes.subarray(0,5).toString("ascii")==="%PDF-";if(type==="image/jpeg")return bytes.length>=3&&bytes[0]===255&&bytes[1]===216&&bytes[2]===255;if(type==="image/png")return bytes.length>=8&&bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]));return false;}
function evidenceResponse(row){const {storage_provider,storage_key,checksum,...safe}=row;return safe;}
function safeDownloadName(value) { return String(value || "complaint-evidence").replace(/[\\/\r\n"]/g, "_"); }
function isStaff(role) { return role === "owner" || role === "warden" || role === "admin"; }

function registerComplaintRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification, documentStorage }) {
  async function complaintForUser(client, user, complaintId, { forUpdate = false } = {}) {
    if (!isValidInt(complaintId)) return null;
    const result = await client.query(`SELECT c.* FROM complaints c WHERE c.id=$1${forUpdate ? " FOR UPDATE" : ""}`, [complaintId]);
    if (!result.rowCount) return null;
    const complaint = result.rows[0];
    if (user.role === "student" && Number(complaint.user_id) === Number(user.id)) return complaint;
    if (user.role === "admin") return complaint;
    if (user.role === "owner" && await ownerHasHostel(client, user.id, complaint.hostel_id)) return complaint;
    if (user.role === "warden" && await wardenHasHostel(client, user.id, complaint.hostel_id)) return complaint;
    return null;
  }
  async function audit(client, complaintId, eventType, userId, details = {}) { await client.query("INSERT INTO complaint_activity_events (complaint_id,event_type,actor_user_id,details) VALUES ($1,$2,$3,$4::jsonb)", [complaintId, eventType, userId, JSON.stringify(details)]); }

  app.get("/api/owner/hostels/:id/complaint-alert-rule", requireAuth, requireRole("owner"), async (req, res) => {
    const hostelId = Number(req.params.id); if (!isValidInt(hostelId) || !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    const rule = await pool.query("SELECT * FROM hostel_complaint_alert_rules WHERE hostel_id=$1", [hostelId]); res.json({ rule: rule.rows[0] || null });
  });

  app.put("/api/owner/hostels/:id/complaint-alert-rule", requireAuth, requireRole("owner"), async (req, res) => {
    const hostelId = Number(req.params.id), threshold = Number(req.body.open_complaint_threshold), cooldown = req.body.cooldown_minutes == null ? 1440 : Number(req.body.cooldown_minutes), enabled = req.body.enabled;
    if (!isValidInt(hostelId) || !await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (!Number.isInteger(threshold) || threshold < 1 || threshold > 1000 || !Number.isInteger(cooldown) || cooldown < 15 || cooldown > 10080 || (enabled != null && typeof enabled !== "boolean")) return res.status(400).json({ error: "threshold 1-1000, cooldown_minutes 15-10080, and optional enabled boolean are required" });
    const rule = await pool.query(`INSERT INTO hostel_complaint_alert_rules (hostel_id,open_complaint_threshold,cooldown_minutes,enabled,updated_by_user_id) VALUES ($1,$2,$3,COALESCE($4,true),$5)
      ON CONFLICT (hostel_id) DO UPDATE SET open_complaint_threshold=EXCLUDED.open_complaint_threshold,cooldown_minutes=EXCLUDED.cooldown_minutes,enabled=COALESCE($4,hostel_complaint_alert_rules.enabled),updated_by_user_id=EXCLUDED.updated_by_user_id,updated_at=NOW() RETURNING *`, [hostelId, threshold, cooldown, enabled ?? null, req.user.id]);
    res.json({ rule: rule.rows[0] });
  });

  app.get("/api/complaints/:id/related", requireAuth, async (req, res) => {
    const complaint = await complaintForUser(pool, req.user, Number(req.params.id)); if (!complaint) return res.status(404).json({ error: "Complaint not found" });
    const rows = await pool.query(`SELECT l.*,source.status AS source_status,primary_complaint.status AS primary_status
      FROM complaint_related_links l JOIN complaints source ON source.id=l.source_complaint_id JOIN complaints primary_complaint ON primary_complaint.id=l.primary_complaint_id
      WHERE l.source_complaint_id=$1 OR l.primary_complaint_id=$1 ORDER BY l.linked_at`, [complaint.id]);
    res.json({ related_links: rows.rows });
  });

  app.post("/api/warden/complaints/:id/related", requireAuth, requireRole("warden"), async (req, res) => {
    const sourceId = Number(req.params.id), primaryId = Number(req.body.primary_complaint_id), reason = String(req.body.reason || "").trim(), key = String(req.headers["idempotency-key"] || "");
    if (!isValidInt(sourceId) || !isValidInt(primaryId) || sourceId === primaryId || !reason || reason.length > 2000 || !key || key.length > 160) return res.status(400).json({ error: "Different source/primary complaint IDs, reason, and idempotency-key are required" });
    const client = await pool.connect();
    try { await client.query("BEGIN"); const source = await complaintForUser(client, req.user, sourceId, { forUpdate: true }); if (!source) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Source complaint not found" }); }
      const primary = await complaintForUser(client, req.user, primaryId, { forUpdate: true }); if (!primary || Number(primary.hostel_id) !== Number(source.hostel_id)) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Primary complaint not found in this hostel" }); }
      const existing = await client.query("SELECT * FROM complaint_related_links WHERE source_complaint_id=$1", [source.id]); if (existing.rowCount) { await client.query("ROLLBACK"); return res.json({ related_link: existing.rows[0], duplicate: true }); }
      const linked = await client.query("INSERT INTO complaint_related_links (source_complaint_id,primary_complaint_id,reason,idempotency_key,linked_by_user_id) VALUES ($1,$2,$3,$4,$5) RETURNING *", [source.id, primary.id, reason, key, req.user.id]);
      await audit(client, source.id, "related_to_primary", req.user.id, { related_link_id: linked.rows[0].id, primary_complaint_id: primary.id, reason }); await client.query("COMMIT"); res.status(201).json({ related_link: linked.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); if (error.code === "23505") return res.status(409).json({ error: "Complaint already has a related-group link" }); res.status(500).json({ error: "Failed to link related complaint" }); } finally { client.release(); }
  });

  app.get("/api/complaints/:id/evidence", requireAuth, async (req, res) => {
    const complaint = await complaintForUser(pool, req.user, Number(req.params.id));
    if (!complaint) return res.status(404).json({ error: "Complaint not found" });
    const rows = await pool.query("SELECT id,complaint_id,original_filename,content_type,byte_size,uploaded_by_user_id,created_at FROM complaint_evidence WHERE complaint_id=$1 AND deleted_at IS NULL ORDER BY created_at", [complaint.id]);
    res.json({ evidence: rows.rows });
  });

  app.post("/api/complaints/:id/evidence", requireAuth, async (req, res) => {
    const complaint = await complaintForUser(pool, req.user, Number(req.params.id)); const { filename, content_type, content_base64 } = req.body;
    if (!complaint) return res.status(404).json({ error: "Complaint not found" });
    if (!filename || typeof filename !== "string" || filename.length > 255 || !ALLOWED_CONTENT_TYPES.has(content_type) || !validBase64(content_base64)) return res.status(400).json({ error: "filename, allowed content_type, and valid base64 content are required" });
    const bytes = Buffer.from(content_base64, "base64"); if (!bytes.length || bytes.length > MAX_EVIDENCE_BYTES) return res.status(413).json({ error: "Evidence must be between 1 byte and 5 MiB" });
    if(!validSignature(bytes,content_type))return res.status(400).json({error:"Evidence content does not match its declared PDF/JPEG/PNG type"});
    const key = documentStorage.createKey("complaint", complaint.id), checksum = crypto.createHash("sha256").update(bytes).digest("hex");
    try {
      await documentStorage.save(key, bytes); const client = await pool.connect();
      try { await client.query("BEGIN"); const saved = await client.query("INSERT INTO complaint_evidence (complaint_id,storage_provider,storage_key,checksum,original_filename,content_type,byte_size,uploaded_by_user_id) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING *", [complaint.id, documentStorage.provider, key, checksum, filename.trim(), content_type, bytes.length, req.user.id]); await audit(client, complaint.id, "evidence_uploaded", req.user.id, { evidence_id: saved.rows[0].id, checksum, byte_size: bytes.length }); await client.query("COMMIT"); return res.status(201).json({ evidence:evidenceResponse(saved.rows[0]) }); } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
    } catch (error) { await documentStorage.remove(key); return res.status(500).json({ error: "Failed to store complaint evidence" }); }
  });

  app.get("/api/complaint-evidence/:evidenceId/download", requireAuth, async (req, res) => {
    const result = await pool.query("SELECT * FROM complaint_evidence WHERE id=$1 AND deleted_at IS NULL", [Number(req.params.evidenceId)]);
    if (!result.rowCount) return res.status(404).json({ error: "Evidence not found" }); const evidence = result.rows[0]; const complaint = await complaintForUser(pool, req.user, evidence.complaint_id);
    if (!complaint || !await documentStorage.exists(evidence.storage_key)) return res.status(404).json({ error: "Evidence not found" });
    res.type(evidence.content_type); res.attachment(safeDownloadName(evidence.original_filename)); documentStorage.createReadStream(evidence.storage_key).on("error", () => res.destroy()).pipe(res);
  });

  app.delete("/api/complaints/:id/evidence/:evidenceId", requireAuth, async (req, res) => {
    const complaint = await complaintForUser(pool, req.user, Number(req.params.id)); if (!complaint) return res.status(404).json({ error: "Complaint not found" }); const client = await pool.connect();
    try { await client.query("BEGIN"); const evidence = await client.query("SELECT * FROM complaint_evidence WHERE id=$1 AND complaint_id=$2 AND deleted_at IS NULL FOR UPDATE", [Number(req.params.evidenceId), complaint.id]);
      if (!evidence.rowCount || (!isStaff(req.user.role) && Number(evidence.rows[0].uploaded_by_user_id) !== Number(req.user.id))) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Evidence not found" }); }
      const updated = await client.query("UPDATE complaint_evidence SET deleted_at=NOW(),deleted_by_user_id=$1,retention_until=NOW()+INTERVAL '90 days' WHERE id=$2 RETURNING *", [req.user.id, evidence.rows[0].id]); await audit(client, complaint.id, "evidence_deleted", req.user.id, { evidence_id: evidence.rows[0].id }); await client.query("COMMIT"); res.json({ evidence: updated.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to delete evidence" }); } finally { client.release(); }
  });

  app.get("/api/warden/complaints/:id/investigations", requireAuth, requireRole("warden"), async (req, res) => {
    const complaint = await complaintForUser(pool, req.user, Number(req.params.id)); if (!complaint) return res.status(404).json({ error: "Complaint not found" }); const rows = await pool.query("SELECT * FROM complaint_investigations WHERE complaint_id=$1 ORDER BY created_at", [complaint.id]); res.json({ investigations: rows.rows });
  });

  app.post("/api/warden/complaints/:id/investigations", requireAuth, requireRole("warden"), async (req, res) => {
    const complaint = await complaintForUser(pool, req.user, Number(req.params.id)); const fields = ["findings", "root_cause", "immediate_actions", "preventive_measures"].map((key) => String(req.body[key] || "").trim());
    if (!complaint) return res.status(404).json({ error: "Complaint not found" }); if (!fields[0] || fields.some((value) => value.length > 4000)) return res.status(400).json({ error: "findings is required and each field must be at most 4000 characters" });
    const client = await pool.connect(); try { await client.query("BEGIN"); const row = await client.query("INSERT INTO complaint_investigations (complaint_id,findings,root_cause,immediate_actions,preventive_measures,created_by_user_id) VALUES ($1,$2,$3,$4,$5,$6) RETURNING *", [complaint.id, fields[0], fields[1] || null, fields[2] || null, fields[3] || null, req.user.id]); await audit(client, complaint.id, "investigation_added", req.user.id, { investigation_id: row.rows[0].id }); await client.query("COMMIT"); res.status(201).json({ investigation: row.rows[0] }); } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to create investigation" }); } finally { client.release(); }
  });

  app.post("/api/warden/complaints/:id/escalate", requireAuth, requireRole("warden"), async (req, res) => {
    const reason = String(req.body.reason || "").trim(), urgency = String(req.body.urgency || ""), key = String(req.headers["idempotency-key"] || "");
    if (!reason || reason.length > 2000 || !["high", "urgent"].includes(urgency) || !key || key.length > 160) return res.status(400).json({ error: "reason, high/urgent urgency, and idempotency-key are required" });
    const client = await pool.connect(); try { await client.query("BEGIN"); const complaint = await complaintForUser(client, req.user, Number(req.params.id), { forUpdate: true }); if (!complaint) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Complaint not found" }); }
      const existing = await client.query("SELECT * FROM complaint_owner_escalations WHERE complaint_id=$1", [complaint.id]); if (existing.rowCount) { await client.query("ROLLBACK"); return res.json({ escalation: existing.rows[0], duplicate: true }); }
      const created = await client.query("INSERT INTO complaint_owner_escalations (complaint_id,reason,urgency,idempotency_key,escalated_by_user_id) VALUES ($1,$2,$3,$4,$5) RETURNING *", [complaint.id, reason, urgency, key, req.user.id]); await client.query("UPDATE complaints SET escalated_at=NOW(),priority=CASE WHEN priority='urgent' THEN priority ELSE $1 END WHERE id=$2", [urgency, complaint.id]);
      const owners = await client.query("SELECT owner_user_id FROM hostel_owners WHERE hostel_id=$1", [complaint.hostel_id]); for (const owner of owners.rows) await createNotification(client, owner.owner_user_id, "complaint_owner_escalated", "Complaint escalated", `A ${urgency} complaint requires owner attention.`, { complaint_id: complaint.id, escalation_id: created.rows[0].id }, "complaint_updates"); await audit(client, complaint.id, "owner_escalated", req.user.id, { escalation_id: created.rows[0].id, urgency }); await client.query("COMMIT"); res.status(201).json({ escalation: created.rows[0] });
    } catch (error) { await client.query("ROLLBACK"); if (error.code === "23505") return res.status(409).json({ error: "Complaint escalation already exists" }); res.status(500).json({ error: "Failed to escalate complaint" }); } finally { client.release(); }
  });
}

module.exports = { registerComplaintRoutes };
