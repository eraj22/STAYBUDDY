const crypto = require("node:crypto");

const MAX_BYTES = 5 * 1024 * 1024;
const CONTENT_TYPES = new Set(["application/pdf", "image/jpeg", "image/png"]);
function validBase64(value) { return typeof value === "string" && value.length > 0 && value.length % 4 === 0 && /^[A-Za-z0-9+/]+={0,2}$/.test(value); }
function validSignature(bytes, type) {
  if (type === "application/pdf") return bytes.subarray(0, 5).toString("ascii") === "%PDF-";
  if (type === "image/jpeg") return bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
  if (type === "image/png") return bytes.length >= 8 && bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  return false;
}
function safeName(value) { return String(value || "booking-document").replace(/[\\/\r\n"]/g, "_"); }
function publicDocument(row) { const { storage_provider, storage_key, checksum, ...safe } = row; return safe; }

function registerBookingDocumentRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, documentStorage }) {
  async function managesHostel(user, hostelId) {
    if (user.role === "owner") return ownerHasHostel(pool, user.id, hostelId);
    if (user.role === "warden") return wardenHasHostel(pool, user.id, hostelId);
    return false;
  }
  async function authorizedDocument(user, documentId) {
    if (!isValidInt(documentId)) return null;
    const result = await pool.query("SELECT d.*,b.user_id,b.hostel_id FROM booking_documents d JOIN bookings b ON b.id=d.booking_id WHERE d.id=$1 AND d.deleted_at IS NULL", [documentId]);
    if (!result.rowCount) return null;
    const row = result.rows[0];
    if (user.role === "student" && Number(row.user_id) === Number(user.id)) return row;
    if ((user.role === "owner" || user.role === "warden") && await managesHostel(user, row.hostel_id)) return row;
    return null;
  }

  app.post("/api/staff/hostels/:hostelId/booking-document-requirements", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const hostelId = Number(req.params.hostelId), code = String(req.body.document_code || "").trim().toLowerCase(), label = String(req.body.label || "").trim();
    if (!isValidInt(hostelId) || !await managesHostel(req.user, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    if (!/^[a-z0-9_-]{1,80}$/.test(code) || !label || label.length > 180) return res.status(400).json({ error: "A lowercase document_code and label are required" });
    const result = await pool.query(`INSERT INTO booking_document_requirements (hostel_id,document_code,label,required,created_by_user_id)
      VALUES ($1,$2,$3,$4,$5) ON CONFLICT (hostel_id,document_code) DO UPDATE SET label=EXCLUDED.label,required=EXCLUDED.required,active=true RETURNING *`, [hostelId, code, label, req.body.required !== false, req.user.id]);
    res.status(201).json({ requirement: result.rows[0] });
  });
  app.get("/api/hostels/:hostelId/booking-document-requirements", async (req, res) => {
    const hostelId = Number(req.params.hostelId);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    const hostel = await pool.query("SELECT 1 FROM hostels WHERE id=$1 AND publication_status='published'", [hostelId]);
    if (!hostel.rowCount) return res.status(404).json({ error: "Hostel not found" });
    const result = await pool.query("SELECT id,document_code,label,required FROM booking_document_requirements WHERE hostel_id=$1 AND active=true ORDER BY required DESC,label", [hostelId]);
    res.json({ requirements: result.rows });
  });
  app.post("/api/bookings/:bookingId/documents", requireAuth, requireRole("student"), async (req, res) => {
    const bookingId = Number(req.params.bookingId), code = String(req.body.document_code || "").trim().toLowerCase(), filename = String(req.body.filename || "").trim(), type = req.body.content_type, base64 = req.body.content_base64;
    const booking = await pool.query("SELECT * FROM bookings WHERE id=$1 AND user_id=$2 AND status IN ('pending','confirmed')", [bookingId, req.user.id]);
    if (!booking.rowCount) return res.status(404).json({ error: "Booking not found" });
    if (!/^[a-z0-9_-]{1,80}$/.test(code) || !filename || filename.length > 255 || !CONTENT_TYPES.has(type) || !validBase64(base64)) return res.status(400).json({ error: "Valid document_code, filename, PDF/JPEG/PNG type, and base64 content are required" });
    const requirement = await pool.query("SELECT * FROM booking_document_requirements WHERE hostel_id=$1 AND document_code=$2 AND active=true", [booking.rows[0].hostel_id, code]);
    if (!requirement.rowCount) return res.status(400).json({ error: "Document type is not configured for this hostel" });
    const bytes = Buffer.from(base64, "base64");
    if (!bytes.length || bytes.length > MAX_BYTES) return res.status(413).json({ error: "Document must be between 1 byte and 5 MiB" });
    if (!validSignature(bytes, type)) return res.status(400).json({ error: "Document content does not match its declared type" });
    const key = documentStorage.createKey("booking_document", bookingId), checksum = crypto.createHash("sha256").update(bytes).digest("hex");
    try {
      await documentStorage.save(key, bytes);
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        const prior = await client.query("UPDATE booking_documents SET deleted_at=NOW() WHERE booking_id=$1 AND document_code=$2 AND deleted_at IS NULL AND status<>'accepted'", [bookingId, code]);
        const result = await client.query("INSERT INTO booking_documents (booking_id,requirement_id,document_code,storage_provider,storage_key,checksum,original_filename,content_type,byte_size,submitted_by_user_id) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING *", [bookingId, requirement.rows[0].id, code, documentStorage.provider, key, checksum, filename, type, bytes.length, req.user.id]);
        await client.query("INSERT INTO booking_document_events (booking_document_id,event_type,actor_user_id,details) VALUES ($1,'uploaded',$2,$3::jsonb)", [result.rows[0].id, req.user.id, JSON.stringify({ replaced_documents: prior.rowCount, checksum, byte_size: bytes.length })]);
        await client.query("COMMIT");
        res.status(201).json({ document: publicDocument(result.rows[0]) });
      } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
    } catch (error) {
      await documentStorage.remove(key);
      res.status(500).json({ error: "Failed to store booking document" });
    }
  });
  app.get("/api/bookings/:bookingId/documents", requireAuth, async (req, res) => {
    const bookingId = Number(req.params.bookingId), booking = await pool.query("SELECT * FROM bookings WHERE id=$1", [bookingId]);
    if (!booking.rowCount) return res.status(404).json({ error: "Booking not found" });
    const allowed = (req.user.role === "student" && Number(booking.rows[0].user_id) === Number(req.user.id)) || ((req.user.role === "owner" || req.user.role === "warden") && await managesHostel(req.user, booking.rows[0].hostel_id));
    if (!allowed) return res.status(404).json({ error: "Booking not found" });
    const result = await pool.query("SELECT id,booking_id,requirement_id,document_code,original_filename,content_type,byte_size,status,review_note,reviewed_at,created_at FROM booking_documents WHERE booking_id=$1 AND deleted_at IS NULL ORDER BY created_at", [bookingId]);
    res.json({ documents: result.rows });
  });
  app.get("/api/booking-documents/:id/download", requireAuth, async (req, res) => {
    const document = await authorizedDocument(req.user, Number(req.params.id));
    if (!document || !await documentStorage.exists(document.storage_key)) return res.status(404).json({ error: "Document not found" });
    res.type(document.content_type); res.attachment(safeName(document.original_filename));
    const stream = documentStorage.createReadStream(document.storage_key); stream.on("error", () => res.destroy()); stream.pipe(res);
  });
  app.patch("/api/staff/booking-documents/:id/review", requireAuth, requireRole("owner", "warden"), async (req, res) => {
    const id = Number(req.params.id), status = String(req.body.status || ""), note = String(req.body.review_note || "").trim();
    const document = await authorizedDocument(req.user, id);
    if (!document) return res.status(404).json({ error: "Document not found" });
    if (!["accepted", "rejected"].includes(status) || (status === "rejected" && !note) || note.length > 2000) return res.status(400).json({ error: "Accepted/rejected status and rejection note are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query("UPDATE booking_documents SET status=$1,review_note=$2,reviewed_by_user_id=$3,reviewed_at=NOW() WHERE id=$4 AND status='pending' AND deleted_at IS NULL RETURNING *", [status, note || null, req.user.id, id]);
      if (!result.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Document has already been reviewed" }); }
      await client.query("INSERT INTO booking_document_events (booking_document_id,event_type,actor_user_id,details) VALUES ($1,$2,$3,$4::jsonb)", [id, status, req.user.id, JSON.stringify({ review_note: note || null })]);
      await client.query("COMMIT");
      res.json({ document: publicDocument(result.rows[0]) });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to review booking document" }); } finally { client.release(); }
  });
  app.delete("/api/bookings/:bookingId/documents/:id", requireAuth, requireRole("student"), async (req, res) => {
    const bookingId = Number(req.params.bookingId), id = Number(req.params.id);
    const result = await pool.query("UPDATE booking_documents d SET deleted_at=NOW() FROM bookings b WHERE d.id=$1 AND d.booking_id=$2 AND b.id=d.booking_id AND b.user_id=$3 AND d.status<>'accepted' AND d.deleted_at IS NULL RETURNING d.*", [id, bookingId, req.user.id]);
    if (!result.rowCount) return res.status(404).json({ error: "Deletable booking document not found" });
    await pool.query("INSERT INTO booking_document_events (booking_document_id,event_type,actor_user_id) VALUES ($1,'deleted',$2)", [id, req.user.id]);
    res.json({ document: publicDocument(result.rows[0]) });
  });
}

module.exports = { registerBookingDocumentRoutes };
