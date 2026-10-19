const crypto = require("node:crypto");
const { generateFeeInvoices, isBillingMonth } = require("./fee_invoice_generator");

const MAX_RECEIPT_BYTES = 5 * 1024 * 1024;
const RECEIPT_TYPES = new Set(["application/pdf", "image/jpeg", "image/png"]);
function validBase64(value) { return typeof value === "string" && value.length > 0 && value.length % 4 === 0 && /^[A-Za-z0-9+/]+={0,2}$/.test(value); }
function validReceiptSignature(bytes, type) {
  if (type === "application/pdf") return bytes.subarray(0, 5).toString("ascii") === "%PDF-";
  if (type === "image/jpeg") return bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  if (type === "image/png") return bytes.length >= 8 && bytes.subarray(0, 8).equals(Buffer.from([0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]));
  return false;
}
function safeDownloadName(value) { return String(value || "payment-receipt").replace(/[\\/\r\n"]/g, "_"); }
function receiptResponse(row) { const { storage_key, storage_provider, checksum, ...safe } = row; return safe; }

function registerFeeRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification, enqueueOperationalJob, deliveryDedupeKey, documentStorage }) {
  async function linkedParent(parentId, studentId) { return pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'",[parentId,studentId]); }
  async function statement(studentId, from, to) {
    await ensurePaymentProofs();
    const r=await pool.query(`
      SELECT i.*,s.title,h.name AS hostel_name,COALESCE(a1.allocated_amount,0) AS paid_amount,
             GREATEST(i.amount-COALESCE(a1.allocated_amount,0),0) AS remaining_amount,
             latest_payment.occurred_at AS latest_payment_at,latest_payment.method AS latest_payment_method,
             latest_proof.created_at AS proof_uploaded_at,latest_proof.status AS proof_status
      FROM fee_invoices i
      JOIN student_fee_assignments a ON a.id=i.assignment_id
      JOIN hostel_fee_schedules s ON s.id=a.schedule_id JOIN hostels h ON h.id=s.hostel_id
      LEFT JOIN (SELECT invoice_id,SUM(amount) allocated_amount FROM (SELECT invoice_id,amount FROM fee_receipt_allocations UNION ALL SELECT invoice_id,amount FROM warden_cash_fee_payments WHERE voided_at IS NULL) payments GROUP BY invoice_id) a1 ON a1.invoice_id=i.id
      LEFT JOIN LATERAL (
        SELECT occurred_at,method FROM (
          SELECT received_on::timestamptz AS occurred_at,'Cash payment'::text AS method FROM warden_cash_fee_payments WHERE invoice_id=i.id AND voided_at IS NULL
          UNION ALL SELECT reviewed_at,'Verified receipt'::text FROM fee_receipt_submissions r JOIN fee_receipt_allocations allocation ON allocation.receipt_submission_id=r.id WHERE allocation.invoice_id=i.id
        ) payment_events ORDER BY occurred_at DESC NULLS LAST LIMIT 1
      ) latest_payment ON true
      LEFT JOIN LATERAL (SELECT created_at,status FROM student_payment_proofs WHERE invoice_id=i.id ORDER BY created_at DESC LIMIT 1) latest_proof ON true
      WHERE a.student_user_id=$1 AND ($2::date IS NULL OR i.billing_month >= $2::date) AND ($3::date IS NULL OR i.billing_month <= $3::date)
      ORDER BY i.billing_month DESC,i.id DESC`,[studentId,from||null,to||null]);
    const totals={total:0,paid:0,pending:0,overdue:0,cancelled:0,remaining:0};
    for(const invoice of r.rows){const amount=Number(invoice.amount),paid=Number(invoice.paid_amount);totals.total+=amount;totals.paid+=paid;totals.remaining+=Number(invoice.remaining_amount);if(invoice.status==='cancelled')totals.cancelled+=amount;else if(invoice.status==='overdue')totals.overdue+=Number(invoice.remaining_amount);else totals.pending+=Number(invoice.remaining_amount);}
    return {invoices:r.rows,totals,currency:"PKR",balance_source:"verified_receipts_and_warden_cash"};
  }
  async function refreshOverdue(studentId) { await pool.query("UPDATE fee_invoices i SET status='overdue',updated_at=NOW() FROM student_fee_assignments a WHERE i.assignment_id=a.id AND a.student_user_id=$1 AND i.status='pending' AND i.due_date<CURRENT_DATE",[studentId]); }
  async function recipients(client, studentId) { return client.query("SELECT $1::int AS user_id UNION SELECT parent_user_id FROM parent_student_links WHERE student_user_id=$1 AND status='active'", [studentId]); }
  async function notifyReceipt(client, studentId, hostelId, type, title, body, data) { const people = await recipients(client, studentId); const wardens = await client.query("SELECT warden_user_id AS user_id FROM warden_assignments WHERE hostel_id=$1", [hostelId]); for (const person of [...people.rows, ...wardens.rows]) await createNotification(client, person.user_id, type, title, body, data, "fee_reminders"); }
  async function paidForInvoice(client, invoiceId, excludeCashPaymentId = null) {
    const result = await client.query(
      `SELECT COALESCE(SUM(amount),0) AS amount FROM (
         SELECT amount FROM fee_receipt_allocations WHERE invoice_id=$1
         UNION ALL
         SELECT amount FROM warden_cash_fee_payments
          WHERE invoice_id=$1 AND voided_at IS NULL
            AND ($2::bigint IS NULL OR id<>$2)
       ) payments`,
      [invoiceId, excludeCashPaymentId],
    );
    return Number(result.rows[0].amount);
  }
  async function refreshInvoiceStatus(client, invoiceId) {
    const invoice = await client.query("SELECT id,amount FROM fee_invoices WHERE id=$1 FOR UPDATE", [invoiceId]);
    if (!invoice.rowCount) throw new Error("Fee invoice not found");
    const paid = await paidForInvoice(client, invoiceId);
    const updated = await client.query(
      `UPDATE fee_invoices
          SET status=CASE WHEN amount <= $1::numeric THEN 'paid'
                          WHEN due_date<CURRENT_DATE THEN 'overdue'
                          ELSE 'pending' END,
              updated_at=NOW()
        WHERE id=$2
        RETURNING status`,
      [paid, invoiceId],
    );
    return { paid, status: updated.rows[0].status };
  }
  let paymentProofsReady;
  function ensurePaymentProofs() {
    if (!paymentProofsReady) paymentProofsReady = pool.query(`
      CREATE TABLE IF NOT EXISTS student_payment_proofs (
        id BIGSERIAL PRIMARY KEY,
        invoice_id INTEGER NOT NULL REFERENCES fee_invoices(id) ON DELETE CASCADE,
        hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
        student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        submitted_by_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
        payment_reference VARCHAR(180) NOT NULL,
        paid_at TIMESTAMPTZ NOT NULL,
        storage_provider VARCHAR(40) NOT NULL,
        storage_key TEXT NOT NULL UNIQUE,
        checksum CHAR(64) NOT NULL,
        original_filename VARCHAR(255) NOT NULL,
        content_type VARCHAR(120) NOT NULL,
        byte_size BIGINT NOT NULL CHECK (byte_size > 0 AND byte_size <= 5242880),
        status VARCHAR(20) NOT NULL DEFAULT 'submitted' CHECK (status IN ('submitted','reviewed')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )
    `);
    return paymentProofsReady;
  }
  app.post("/api/owner/fee-schedules", requireAuth, requireRole("owner"), async (req,res)=>{const {hostel_id,title,amount,currency="PKR",due_day,payment_option="full_only",minimum_installment_amount}=req.body;const hostelId=Number(hostel_id), option=String(payment_option);if(!isValidInt(hostelId)||!await ownerHasHostel(pool,req.user.id,hostelId))return res.status(404).json({error:"Hostel not found"});if(!title||String(currency).toUpperCase()!=="PKR"||!Number.isFinite(Number(amount))||Number(amount)<=0||!Number.isInteger(Number(due_day))||Number(due_day)<1||Number(due_day)>28||!["full_only","installments_only","both"].includes(option)||((option==="installments_only"||option==="both")&&(!Number.isFinite(Number(minimum_installment_amount))||Number(minimum_installment_amount)<=0||Number(minimum_installment_amount)>Number(amount))))return res.status(400).json({error:"title, PKR amount, due_day, payment_option, and a valid installment minimum when enabled are required"});const r=await pool.query("INSERT INTO hostel_fee_schedules (hostel_id,title,amount,currency,due_day,payment_option,minimum_installment_amount,created_by_user_id) VALUES ($1,$2,$3,'PKR',$4,$5,$6,$7) RETURNING *",[hostelId,String(title).trim(),amount,due_day,option,(option==="full_only"?null:minimum_installment_amount),req.user.id]);res.status(201).json({schedule:r.rows[0]});});
  app.get("/api/owner/fee-schedules", requireAuth, requireRole("owner"), async (req,res)=>{const r=await pool.query("SELECT s.* FROM hostel_fee_schedules s JOIN hostel_owners ho ON ho.hostel_id=s.hostel_id WHERE ho.owner_user_id=$1 ORDER BY s.created_at DESC",[req.user.id]);res.json({schedules:r.rows});});
  // Owner-facing monthly rent register. It uses the same invoices and payment
  // allocations as the student and warden flows, so a cash entry or verified
  // receipt can never be counted twice.
  app.get("/api/owner/fee-records", requireAuth, requireRole("owner"), async (req,res)=>{
    const billingMonth=String(req.query.billing_month||"");
    if(!isBillingMonth(billingMonth)) return res.status(400).json({error:"billing_month must be YYYY-MM-01"});
    try {
      const rows=await pool.query(`SELECT b.id AS booking_id,b.user_id AS student_id,u.name AS student_name,
          r.room_number,COALESCE(r.floor_label,'Ground Floor') AS floor_label,b.bed_number,
          COALESCE(r.monthly_rent,0) AS room_monthly_rent,
          invoice.id AS invoice_id,invoice.amount AS invoice_amount,invoice.due_date,invoice.status AS invoice_status,
          COALESCE(paid.amount,0) AS paid_amount,
          GREATEST(COALESCE(invoice.amount,r.monthly_rent,0)-COALESCE(paid.amount,0),0) AS remaining_amount,
          latest_cash.id AS latest_cash_payment_id,latest_cash.amount AS latest_cash_amount,
          latest_cash.received_on AS latest_cash_received_on,latest_cash.note AS latest_cash_note
        FROM bookings b
        JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id AND ho.owner_user_id=$1
        JOIN users u ON u.id=b.user_id AND u.role='student'
        LEFT JOIN rooms r ON r.id=b.room_id
        LEFT JOIN LATERAL (
          SELECT i.* FROM student_fee_assignments a JOIN fee_invoices i ON i.assignment_id=a.id
          WHERE a.booking_id=b.id AND a.active=true AND i.billing_month=$2::date
          ORDER BY i.id DESC LIMIT 1
        ) invoice ON true
        LEFT JOIN LATERAL (
          SELECT COALESCE(SUM(amount),0) AS amount FROM (
            SELECT amount FROM fee_receipt_allocations WHERE invoice_id=invoice.id
            UNION ALL SELECT amount FROM warden_cash_fee_payments WHERE invoice_id=invoice.id AND voided_at IS NULL
          ) all_payments
        ) paid ON true
        LEFT JOIN LATERAL (
          SELECT * FROM warden_cash_fee_payments WHERE invoice_id=invoice.id AND voided_at IS NULL
          ORDER BY received_on DESC,id DESC LIMIT 1
        ) latest_cash ON true
        WHERE b.status IN ('pending','confirmed')
        ORDER BY u.name`,[req.user.id,billingMonth]);
      res.json({billing_month:billingMonth,currency:'PKR',records:rows.rows});
    } catch(error){res.status(500).json({error:'Failed to load owner fee records'});}
  });
  app.post("/api/owner/fee-records/:bookingId/setup", requireAuth, requireRole("owner"), async (req,res)=>{
    const bookingId=Number(req.params.bookingId),billingMonth=String(req.body.billing_month||"");
    if(!isValidInt(bookingId)||!isBillingMonth(billingMonth)) return res.status(400).json({error:'Booking and billing month are required'});
    const client=await pool.connect();
    try { await client.query('BEGIN');
      const booking=await client.query(`SELECT b.id,b.user_id,b.hostel_id,COALESCE(r.monthly_rent,0) AS monthly_rent
        FROM bookings b JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id AND ho.owner_user_id=$2
        LEFT JOIN rooms r ON r.id=b.room_id WHERE b.id=$1 AND b.status IN ('pending','confirmed') FOR UPDATE OF b`,[bookingId,req.user.id]);
      if(!booking.rowCount){await client.query('ROLLBACK');return res.status(404).json({error:'Registered student not found'});}
      const resident=booking.rows[0],rent=Number(resident.monthly_rent);
      if(!Number.isFinite(rent)||rent<=0){await client.query('ROLLBACK');return res.status(409).json({error:'Set the room monthly rent before creating a record'});}
      let schedule=await client.query("SELECT id FROM hostel_fee_schedules WHERE hostel_id=$1 AND active=true AND lower(title)='monthly hostel rent' AND amount=$2 ORDER BY id DESC LIMIT 1 FOR UPDATE",[resident.hostel_id,rent]);
      if(!schedule.rowCount) schedule=await client.query("INSERT INTO hostel_fee_schedules(hostel_id,title,amount,currency,due_day,payment_option,created_by_user_id) VALUES($1,'Monthly hostel rent',$2,'PKR',5,'both',$3) RETURNING id",[resident.hostel_id,rent,req.user.id]);
      let assignment=await client.query('SELECT id FROM student_fee_assignments WHERE schedule_id=$1 AND booking_id=$2 ORDER BY id DESC LIMIT 1 FOR UPDATE',[schedule.rows[0].id,bookingId]);
      if(!assignment.rowCount) assignment=await client.query('INSERT INTO student_fee_assignments(schedule_id,booking_id,student_user_id,starts_on,assigned_by_user_id) VALUES($1,$2,$3,$4::date,$5) RETURNING id',[schedule.rows[0].id,bookingId,resident.user_id,billingMonth,req.user.id]);
      const invoice=await client.query("INSERT INTO fee_invoices(assignment_id,billing_month,due_date,amount,currency) VALUES($1,$2::date,($2::date+INTERVAL '4 days')::date,$3,'PKR') ON CONFLICT(assignment_id,billing_month) DO UPDATE SET updated_at=fee_invoices.updated_at RETURNING *",[assignment.rows[0].id,billingMonth,rent]);
      await client.query('COMMIT');res.status(201).json({invoice:invoice.rows[0]});
    } catch(error){await client.query('ROLLBACK');res.status(500).json({error:'Failed to create monthly rent record'});} finally{client.release();}
  });
  app.post("/api/owner/cash-fee-payments", requireAuth, requireRole("owner"), async (req,res)=>{
    const invoiceId=Number(req.body.invoice_id),amount=Number(req.body.amount),receivedOn=String(req.body.received_on||''),note=String(req.body.note||'').trim();
    if(!isValidInt(invoiceId)||!Number.isFinite(amount)||amount<=0||!/^\d{4}-\d{2}-\d{2}$/.test(receivedOn)||note.length>1000)return res.status(400).json({error:'Valid cash payment details are required'});
    const client=await pool.connect();
    try {await client.query('BEGIN');const invoice=await client.query(`SELECT i.*,a.student_user_id,s.hostel_id FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id JOIN hostel_owners ho ON ho.hostel_id=s.hostel_id AND ho.owner_user_id=$2 WHERE i.id=$1 FOR UPDATE OF i`,[invoiceId,req.user.id]);if(!invoice.rowCount){await client.query('ROLLBACK');return res.status(404).json({error:'Fee invoice not found'});}const row=invoice.rows[0],paid=await paidForInvoice(client,invoiceId);if(paid+amount>Number(row.amount)){await client.query('ROLLBACK');return res.status(409).json({error:'Cash payment exceeds remaining balance'});}const payment=await client.query('INSERT INTO warden_cash_fee_payments(invoice_id,hostel_id,student_user_id,amount,received_on,note,received_by_warden_id) VALUES($1,$2,$3,$4,$5::date,$6,$7) RETURNING *',[invoiceId,row.hostel_id,row.student_user_id,amount,receivedOn,note||null,req.user.id]);const totals=await refreshInvoiceStatus(client,invoiceId);await client.query('COMMIT');res.status(201).json({cash_payment:payment.rows[0],paid_amount:totals.paid,invoice_status:totals.status});}catch(error){await client.query('ROLLBACK');res.status(500).json({error:'Failed to record cash payment'});}finally{client.release();}
  });
  app.post("/api/owner/fee-assignments", requireAuth, requireRole("owner"), async (req,res)=>{const {schedule_id,booking_id,amount_override,starts_on,ends_on}=req.body;const schedule=await pool.query("SELECT s.* FROM hostel_fee_schedules s JOIN hostel_owners ho ON ho.hostel_id=s.hostel_id WHERE s.id=$1 AND ho.owner_user_id=$2",[Number(schedule_id),req.user.id]);if(!schedule.rowCount)return res.status(404).json({error:"Fee schedule not found"});const booking=await pool.query("SELECT * FROM bookings WHERE id=$1 AND hostel_id=$2",[Number(booking_id),schedule.rows[0].hostel_id]);if(!booking.rowCount)return res.status(404).json({error:"Booking not found"});if(!starts_on||Number.isNaN(Date.parse(starts_on))||(ends_on&&Number.isNaN(Date.parse(ends_on))))return res.status(400).json({error:"Valid starts_on and optional ends_on are required"});const r=await pool.query("INSERT INTO student_fee_assignments (schedule_id,booking_id,student_user_id,amount_override,starts_on,ends_on,assigned_by_user_id) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *",[schedule_id,booking_id,booking.rows[0].user_id,amount_override??null,starts_on,ends_on||null,req.user.id]);res.status(201).json({assignment:r.rows[0]});});
  app.post("/api/owner/fee-invoices/generate", requireAuth, requireRole("owner"), async (req,res)=>{const scheduleId=Number(req.body.schedule_id),month=String(req.body.billing_month||"");if(!isBillingMonth(month))return res.status(400).json({error:"billing_month must be YYYY-MM-01"});const schedule=await pool.query("SELECT s.id FROM hostel_fee_schedules s JOIN hostel_owners ho ON ho.hostel_id=s.hostel_id WHERE s.id=$1 AND ho.owner_user_id=$2 AND s.active=true",[scheduleId,req.user.id]);if(!schedule.rowCount)return res.status(404).json({error:"Active fee schedule not found"});try{const generated=await generateFeeInvoices(pool,{billingMonth:month,scheduleId});res.status(201).json({created_invoices:generated.created,eligible_assignments:generated.eligibleAssignments});}catch(error){res.status(500).json({error:"Failed to generate invoices"});}});
  app.patch("/api/owner/fee-assignments/:id", requireAuth, requireRole("owner"), async (req,res)=>{const id=Number(req.params.id),active=req.body.active,endsOn=req.body.ends_on;if(!isValidInt(id)||typeof active!=="boolean")return res.status(400).json({error:"A valid assignment id and active boolean are required"});if(endsOn&&Number.isNaN(Date.parse(endsOn)))return res.status(400).json({error:"ends_on must be a valid date"});const r=await pool.query("UPDATE student_fee_assignments a SET active=$1,ends_on=COALESCE($2::date,ends_on) FROM hostel_fee_schedules s JOIN hostel_owners ho ON ho.hostel_id=s.hostel_id WHERE a.id=$3 AND a.schedule_id=s.id AND ho.owner_user_id=$4 RETURNING a.*",[active,endsOn||null,id,req.user.id]);if(!r.rowCount)return res.status(404).json({error:"Fee assignment not found"});res.json({assignment:r.rows[0]});});
  app.get("/api/fee-invoices", requireAuth, requireRole("student"), async (req,res)=>{await refreshOverdue(req.user.id);const result=await statement(req.user.id);res.json({invoices:result.invoices});});
  app.get("/api/fee-statements", requireAuth, requireRole("student"), async (req,res)=>{const from=req.query.from||null,to=req.query.to||null;if((from&&!/^\d{4}-\d{2}-\d{2}$/.test(from))||(to&&!/^\d{4}-\d{2}-\d{2}$/.test(to))||(from&&to&&from>to))return res.status(400).json({error:"from and to must be ordered YYYY-MM-DD dates"});await refreshOverdue(req.user.id);res.json(await statement(req.user.id,from,to));});
  app.post("/api/fee-invoices/:id/settle", requireAuth, requireRole("student"), (_req,res)=>res.status(410).json({error:"Payment settlement is retired. Submit a PKR receipt for warden verification."}));
  app.post("/api/fee-invoices/:id/payment-proofs", requireAuth, requireRole("student"), async (req,res)=>{
    try {
      await ensurePaymentProofs();
      const invoiceId=Number(req.params.id), amount=Number(req.body.amount), reference=String(req.body.payment_reference||"").trim(), paidAt=req.body.paid_at, filename=String(req.body.filename||""), type=req.body.content_type, base64=req.body.content_base64;
      if(!isValidInt(invoiceId)||!Number.isFinite(amount)||amount<=0||!reference||reference.length>180||!paidAt||Number.isNaN(Date.parse(paidAt))||!filename||filename.length>255||!RECEIPT_TYPES.has(type)||!validBase64(base64)) return res.status(400).json({error:"Valid payment proof details and a PDF/JPEG/PNG file are required"});
      const invoice=await pool.query("SELECT i.*,a.student_user_id,s.hostel_id FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id WHERE i.id=$1 AND i.status IN ('pending','overdue')",[invoiceId]);
      if(!invoice.rowCount || Number(invoice.rows[0].student_user_id)!==Number(req.user.id)) return res.status(404).json({error:"Open fee notice not found"});
      const bytes=Buffer.from(base64,"base64"); if(!bytes.length||bytes.length>MAX_RECEIPT_BYTES) return res.status(413).json({error:"Receipt must be between 1 byte and 5 MiB"});
      const row=invoice.rows[0], key=documentStorage.createKey("student_payment_proof",invoiceId), checksum=crypto.createHash("sha256").update(bytes).digest("hex");
      await documentStorage.save(key,bytes);
      const client=await pool.connect();
      try {
        await client.query("BEGIN");
        const saved=await client.query("INSERT INTO student_payment_proofs(invoice_id,hostel_id,student_user_id,submitted_by_user_id,amount,payment_reference,paid_at,storage_provider,storage_key,checksum,original_filename,content_type,byte_size) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) RETURNING id,invoice_id,amount,payment_reference,paid_at,original_filename,content_type,status,created_at",[invoiceId,row.hostel_id,row.student_user_id,req.user.id,amount,reference,paidAt,documentStorage.provider,key,checksum,filename,type,bytes.length]);
        await notifyReceipt(client,row.student_user_id,row.hostel_id,"payment_proof_submitted","Payment proof uploaded","Your payment proof was saved and sent to the hostel office.",{invoice_id:invoiceId,payment_proof_id:saved.rows[0].id});
        await client.query("COMMIT");
        res.status(201).json({payment_proof:saved.rows[0]});
      } catch(error) { await client.query("ROLLBACK"); await documentStorage.remove(key); throw error; } finally { client.release(); }
    } catch(error) { res.status(500).json({error:"Failed to store payment proof"}); }
  });
  app.get("/api/parent/children/:studentId/fee-invoices", requireAuth, requireRole("parent"), async (req,res)=>{const studentId=Number(req.params.studentId);const link=await pool.query("SELECT 1 FROM parent_student_links WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'",[req.user.id,studentId]);if(!isValidInt(studentId)||!link.rowCount)return res.status(404).json({error:"Linked student not found"});const r=await pool.query("SELECT i.*,s.title,h.name AS hostel_name FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id JOIN hostels h ON h.id=s.hostel_id WHERE a.student_user_id=$1 ORDER BY i.due_date DESC",[studentId]);res.json({invoices:r.rows});});
  app.get("/api/parent/children/:studentId/fee-statements", requireAuth, requireRole("parent"), async (req,res)=>{const studentId=Number(req.params.studentId),from=req.query.from||null,to=req.query.to||null,link=await linkedParent(req.user.id,studentId);if(!isValidInt(studentId)||!link.rowCount)return res.status(404).json({error:"Linked student not found"});if((from&&!/^\d{4}-\d{2}-\d{2}$/.test(from))||(to&&!/^\d{4}-\d{2}-\d{2}$/.test(to))||(from&&to&&from>to))return res.status(400).json({error:"from and to must be ordered YYYY-MM-DD dates"});await refreshOverdue(studentId);res.json(await statement(studentId,from,to));});
  app.get("/api/fee-invoices/:id/receipt", requireAuth, requireRole("student"), async (req,res)=>{const id=Number(req.params.id);const r=await pool.query("SELECT i.*,p.provider,p.provider_payment_id,p.receipt_url,p.status AS payment_status FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN payments p ON p.id=i.paid_payment_id WHERE i.id=$1 AND a.student_user_id=$2 AND i.status='paid' AND p.status='succeeded'",[id,req.user.id]);if(!r.rowCount)return res.status(404).json({error:"Paid invoice receipt not found"});res.json({receipt:r.rows[0]});});
  app.get("/api/parent/children/:studentId/fee-invoices/:id/receipt", requireAuth, requireRole("parent"), async (req,res)=>{const studentId=Number(req.params.studentId),id=Number(req.params.id),link=await linkedParent(req.user.id,studentId);if(!isValidInt(studentId)||!link.rowCount)return res.status(404).json({error:"Linked student not found"});const r=await pool.query("SELECT i.*,p.provider,p.provider_payment_id,p.receipt_url,p.status AS payment_status FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN payments p ON p.id=i.paid_payment_id WHERE i.id=$1 AND a.student_user_id=$2 AND i.status='paid' AND p.status='succeeded'",[id,studentId]);if(!r.rowCount)return res.status(404).json({error:"Paid invoice receipt not found"});res.json({receipt:r.rows[0]});});
  app.post("/api/owner/fee-invoices/:id/reminders", requireAuth, requireRole("owner"), async (req,res)=>{const id=Number(req.params.id);const invoice=await pool.query("SELECT i.*,a.student_user_id FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id JOIN hostel_owners ho ON ho.hostel_id=s.hostel_id WHERE i.id=$1 AND ho.owner_user_id=$2",[id,req.user.id]);if(!invoice.rowCount)return res.status(404).json({error:"Invoice not found"});const row=invoice.rows[0];const recipients=await pool.query("SELECT $1::int AS user_id UNION SELECT parent_user_id FROM parent_student_links WHERE student_user_id=$1 AND status='active'",[row.student_user_id]);let delivered=0,queued=0;for(const recipient of recipients.rows){await createNotification(pool,recipient.user_id,"fee_reminder","Fee payment reminder",`An invoice of ${row.amount} ${row.currency} is due on ${row.due_date}.`,{invoice_id:id},"fee_reminders");await pool.query("INSERT INTO fee_reminder_deliveries (invoice_id,recipient_user_id,channel,status,scheduled_for,delivered_at,attempt_count) VALUES ($1,$2,'in_app','delivered',NOW(),NOW(),1)",[id,recipient.user_id]);const external=await pool.query("INSERT INTO fee_reminder_deliveries (invoice_id,recipient_user_id,channel,status,scheduled_for,next_retry_at) VALUES ($1,$2,'email','queued',NOW(),NOW()) RETURNING id",[id,recipient.user_id]);await enqueueOperationalJob(pool,"fee_reminder_delivery",deliveryDedupeKey("fee-email",external.rows[0].id),{delivery_id:external.rows[0].id});delivered+=1;queued+=1;}res.json({delivered,queued});});
  app.post("/api/warden/hostels/:hostelId/payment-destinations", requireAuth, requireRole("warden"), async (req,res)=>{const hostelId=Number(req.params.hostelId),holder=String(req.body.account_holder_name||"").trim(),institution=String(req.body.institution_name||"").trim(),reference=String(req.body.account_reference||"").trim(),instructions=String(req.body.instructions||"").trim();if(!isValidInt(hostelId)||!await wardenHasHostel(pool,req.user.id,hostelId))return res.status(404).json({error:"Hostel not found"});if(!holder||!institution||!reference||holder.length>180||institution.length>180||reference.length>180||instructions.length>2000)return res.status(400).json({error:"Account holder, institution, account reference, and optional instructions are required"});const row=await pool.query("INSERT INTO hostel_payment_destinations (hostel_id,account_holder_name,institution_name,account_reference,instructions,submitted_by_user_id) VALUES ($1,$2,$3,$4,$5,$6) RETURNING *",[hostelId,holder,institution,reference,instructions||null,req.user.id]);res.status(201).json({payment_destination:row.rows[0]});});
  app.post("/api/owner/payment-destinations/:id/approve", requireAuth, requireRole("owner"), async (req,res)=>{const id=Number(req.params.id);const row=await pool.query("UPDATE hostel_payment_destinations d SET status='approved',approved_by_user_id=$1,approved_at=NOW() FROM hostel_owners ho WHERE d.id=$2 AND d.hostel_id=ho.hostel_id AND ho.owner_user_id=$1 AND d.status='pending' RETURNING d.*",[req.user.id,id]);if(!row.rowCount)return res.status(404).json({error:"Pending payment destination not found"});res.json({payment_destination:row.rows[0]});});
  app.get("/api/fee-invoices/:id/payment-destination", requireAuth, async (req,res)=>{const id=Number(req.params.id);const student=req.user.role==='student'?req.user.id:null;const invoice=await pool.query(`SELECT s.hostel_id,a.student_user_id FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id WHERE i.id=$1`,[id]);if(!invoice.rowCount)return res.status(404).json({error:"Invoice not found"});const row=invoice.rows[0];const allowed=(student&&Number(row.student_user_id)===Number(student))||(req.user.role==='parent'&&(await linkedParent(req.user.id,row.student_user_id)).rowCount)||(req.user.role==='warden'&&await wardenHasHostel(pool,req.user.id,row.hostel_id))||(req.user.role==='owner'&&await ownerHasHostel(pool,req.user.id,row.hostel_id));if(!allowed)return res.status(404).json({error:"Invoice not found"});const destination=await pool.query("SELECT id,account_holder_name,institution_name,account_reference,instructions,approved_at FROM hostel_payment_destinations WHERE hostel_id=$1 AND status='approved' ORDER BY approved_at DESC,id DESC LIMIT 1",[row.hostel_id]);res.json({payment_destination:destination.rows[0]||null});});
  app.use("/api/fee-invoices/:id/receipt-submissions", (req,res,next)=>{
    if (req.method !== "POST") return next();
    if (!validBase64(req.body?.content_base64)) return next();
    const bytes=Buffer.from(req.body.content_base64,"base64");
    if (!validReceiptSignature(bytes,req.body.content_type)) return res.status(400).json({error:"Receipt content does not match its declared PDF/JPEG/PNG type"});
    next();
  });
  app.post("/api/fee-invoices/:id/receipt-submissions", requireAuth, requireRole("student","parent"), async (req,res,next)=>{
    const supersedesId=Number(req.body.supersedes_receipt_id);
    if (!isValidInt(supersedesId)) return next();
    const invoiceId=Number(req.params.id),amount=Number(req.body.amount),reference=String(req.body.payment_reference||"").trim(),paidAt=req.body.paid_at,filename=String(req.body.filename||""),type=req.body.content_type,base64=req.body.content_base64,destinationId=Number(req.body.payment_destination_id);
    if(!isValidInt(invoiceId)||!Number.isFinite(amount)||amount<=0||!reference||reference.length>180||!paidAt||Number.isNaN(Date.parse(paidAt))||!filename||filename.length>255||!RECEIPT_TYPES.has(type)||!validBase64(base64)||!isValidInt(destinationId))return res.status(400).json({error:"Valid replacement receipt details are required"});
    const prior=await pool.query(`SELECT r.*,s.payment_option,s.minimum_installment_amount FROM fee_receipt_submissions r JOIN fee_invoices i ON i.id=r.invoice_id JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id WHERE r.id=$1 AND r.invoice_id=$2 AND r.status IN ('rejected','needs_clarification')`,[supersedesId,invoiceId]);
    if(!prior.rowCount)return res.status(404).json({error:"Rejected or clarification receipt to replace was not found"});
    const row=prior.rows[0],allowed=req.user.role==='student'?Number(row.student_user_id)===Number(req.user.id):(await linkedParent(req.user.id,row.student_user_id)).rowCount;
    if(!allowed)return res.status(404).json({error:"Receipt to replace was not found"});
    const destination=await pool.query("SELECT 1 FROM hostel_payment_destinations WHERE id=$1 AND hostel_id=$2 AND status='approved'",[destinationId,row.hostel_id]);
    if(!destination.rowCount)return res.status(400).json({error:"An approved hostel payment destination is required"});
    const bytes=Buffer.from(base64,"base64");if(!bytes.length||bytes.length>MAX_RECEIPT_BYTES)return res.status(413).json({error:"Receipt must be between 1 byte and 5 MiB"});
    const key=documentStorage.createKey("fee_receipt",invoiceId),checksum=crypto.createHash("sha256").update(bytes).digest("hex");
    try{await documentStorage.save(key,bytes);const client=await pool.connect();try{await client.query("BEGIN");const saved=await client.query(`INSERT INTO fee_receipt_submissions (invoice_id,hostel_id,student_user_id,submitted_by_user_id,payment_destination_id,amount,payment_reference,paid_at,storage_provider,storage_key,checksum,original_filename,content_type,byte_size,supersedes_receipt_id) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15) RETURNING *`,[invoiceId,row.hostel_id,row.student_user_id,req.user.id,destinationId,amount,reference,paidAt,documentStorage.provider,key,checksum,filename,type,bytes.length,supersedesId]);await client.query("INSERT INTO fee_receipt_audit_events (receipt_submission_id,event_type,actor_user_id,details) VALUES ($1,'replacement_submitted',$2,$3::jsonb)",[saved.rows[0].id,req.user.id,JSON.stringify({supersedes_receipt_id:supersedesId})]);await client.query("COMMIT");res.status(201).json({receipt_submission:receiptResponse(saved.rows[0])});}catch(error){await client.query("ROLLBACK");throw error;}finally{client.release();}}catch(error){await documentStorage.remove(key);res.status(500).json({error:"Failed to store replacement receipt"});}
  });
  app.post("/api/fee-invoices/:id/receipt-submissions", requireAuth, requireRole("student","parent"), async (req,res)=>{const invoiceId=Number(req.params.id),amount=Number(req.body.amount),reference=String(req.body.payment_reference||"").trim(),paidAt=req.body.paid_at,filename=String(req.body.filename||""),type=req.body.content_type,base64=req.body.content_base64,destinationId=Number(req.body.payment_destination_id);if(!isValidInt(invoiceId)||!Number.isFinite(amount)||amount<=0||!reference||reference.length>180||!paidAt||Number.isNaN(Date.parse(paidAt))||!filename||filename.length>255||!RECEIPT_TYPES.has(type)||!validBase64(base64)||!isValidInt(destinationId))return res.status(400).json({error:"Valid receipt details, payment reference, approved destination, PDF/JPEG/PNG file, and base64 content are required"});const invoice=await pool.query("SELECT i.*,a.student_user_id,s.hostel_id,s.payment_option,s.minimum_installment_amount FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id WHERE i.id=$1 AND i.status IN ('pending','overdue')",[invoiceId]);if(!invoice.rowCount)return res.status(404).json({error:"Open fee notice not found"});const row=invoice.rows[0];const allowed=req.user.role==='student'?Number(row.student_user_id)===Number(req.user.id):(await linkedParent(req.user.id,row.student_user_id)).rowCount;if(!allowed)return res.status(404).json({error:"Open fee notice not found"});const destination=await pool.query("SELECT id FROM hostel_payment_destinations WHERE id=$1 AND hostel_id=$2 AND status='approved'",[destinationId,row.hostel_id]);if(!destination.rowCount)return res.status(400).json({error:"An approved hostel payment destination is required"});const bytes=Buffer.from(base64,"base64");if(!bytes.length||bytes.length>MAX_RECEIPT_BYTES)return res.status(413).json({error:"Receipt must be between 1 byte and 5 MiB"});const key=documentStorage.createKey("fee_receipt",invoiceId),checksum=crypto.createHash("sha256").update(bytes).digest("hex");try{await documentStorage.save(key,bytes);const client=await pool.connect();try{await client.query("BEGIN");const saved=await client.query("INSERT INTO fee_receipt_submissions (invoice_id,hostel_id,student_user_id,submitted_by_user_id,payment_destination_id,amount,payment_reference,paid_at,storage_provider,storage_key,checksum,original_filename,content_type,byte_size) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14) RETURNING *",[invoiceId,row.hostel_id,row.student_user_id,req.user.id,destinationId,amount,reference,paidAt,documentStorage.provider,key,checksum,filename,type,bytes.length]);await client.query("INSERT INTO fee_receipt_audit_events (receipt_submission_id,event_type,actor_user_id) VALUES ($1,'submitted',$2)",[saved.rows[0].id,req.user.id]);await notifyReceipt(client,row.student_user_id,row.hostel_id,"fee_receipt_submitted","Payment receipt submitted","A payment receipt is awaiting hostel verification.",{invoice_id:invoiceId,receipt_submission_id:saved.rows[0].id});await client.query("COMMIT");res.status(201).json({receipt_submission:receiptResponse(saved.rows[0])});}catch(error){await client.query("ROLLBACK");throw error;}finally{client.release();}}catch(error){await documentStorage.remove(key);res.status(500).json({error:"Failed to store payment receipt"});}});
  app.get("/api/fee-receipts", requireAuth, async (req,res)=>{const query=req.user.role==='student'?"SELECT r.* FROM fee_receipt_submissions r WHERE r.student_user_id=$1":req.user.role==='parent'?"SELECT r.* FROM fee_receipt_submissions r JOIN parent_student_links p ON p.student_user_id=r.student_user_id WHERE p.parent_user_id=$1 AND p.status='active'":null;if(!query)return res.status(403).json({error:"Unsupported role"});const rows=await pool.query(query,[req.user.id]);res.json({receipts:rows.rows.map(receiptResponse)});});
  app.get("/api/owner/fee-receipt-report", requireAuth, requireRole("owner"), async (req,res)=>{
    const hostelId=req.query.hostel_id==null?null:Number(req.query.hostel_id);
    if(hostelId!=null&&(!isValidInt(hostelId)||!await ownerHasHostel(pool,req.user.id,hostelId)))return res.status(404).json({error:"Hostel not found"});
    const totals=await pool.query(`SELECT COALESCE(SUM(i.amount),0) invoiced,COALESCE(SUM(a.allocated),0) collected,COALESCE(SUM(GREATEST(i.amount-COALESCE(a.allocated,0),0)) FILTER (WHERE i.due_date>=CURRENT_DATE),0) pending,COALESCE(SUM(GREATEST(i.amount-COALESCE(a.allocated,0),0)) FILTER (WHERE i.due_date<CURRENT_DATE),0) overdue FROM fee_invoices i JOIN student_fee_assignments fa ON fa.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=fa.schedule_id JOIN hostel_owners ho ON ho.hostel_id=s.hostel_id LEFT JOIN (SELECT invoice_id,SUM(amount) allocated FROM (SELECT invoice_id,amount FROM fee_receipt_allocations UNION ALL SELECT invoice_id,amount FROM warden_cash_fee_payments WHERE voided_at IS NULL) payments GROUP BY invoice_id) a ON a.invoice_id=i.id WHERE ho.owner_user_id=$1 AND ($2::int IS NULL OR s.hostel_id=$2) AND i.status<>'cancelled'`,[req.user.id,hostelId]);
    const audit=await pool.query(`SELECT r.id,r.invoice_id,r.hostel_id,r.amount,r.status,r.payment_reference,r.paid_at,r.reviewed_at,r.review_note,r.supersedes_receipt_id,r.created_at,u.name student_name FROM fee_receipt_submissions r JOIN hostel_owners ho ON ho.hostel_id=r.hostel_id JOIN users u ON u.id=r.student_user_id WHERE ho.owner_user_id=$1 AND ($2::int IS NULL OR r.hostel_id=$2) ORDER BY r.created_at DESC LIMIT 250`,[req.user.id,hostelId]);
    res.json({currency:"PKR",balance_source:"verified_receipts_and_warden_cash",totals:totals.rows[0],receipts:audit.rows});
  });
  app.patch("/api/warden/fee-receipts/:id", requireAuth, requireRole("warden"), async (req,res)=>{
    const id=Number(req.params.id),status=String(req.body.status||""),note=String(req.body.review_note||"").trim();
    if(!isValidInt(id)||!["verified","rejected","needs_clarification"].includes(status)||!note||note.length>2000)return res.status(400).json({error:"verified/rejected/needs_clarification status and review_note are required"});
    const client=await pool.connect();
    try{await client.query("BEGIN");const receipt=await client.query(`SELECT r.*,i.amount invoice_amount,s.payment_option,s.minimum_installment_amount FROM fee_receipt_submissions r JOIN fee_invoices i ON i.id=r.invoice_id JOIN student_fee_assignments fa ON fa.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=fa.schedule_id JOIN warden_assignments wa ON wa.hostel_id=r.hostel_id WHERE r.id=$1 AND wa.warden_user_id=$2 FOR UPDATE OF r,i`,[id,req.user.id]);
      if(!receipt.rowCount){await client.query("ROLLBACK");return res.status(404).json({error:"Receipt submission not found"});}
      const row=receipt.rows[0];if(row.status!=="submitted"){await client.query("ROLLBACK");return res.status(409).json({error:"Receipt submission has already been reviewed"});}
      if(status==="verified"){const allocated=await client.query("SELECT COALESCE(SUM(amount),0) amount FROM (SELECT amount FROM fee_receipt_allocations WHERE invoice_id=$1 UNION ALL SELECT amount FROM warden_cash_fee_payments WHERE invoice_id=$1 AND voided_at IS NULL) payments",[row.invoice_id]);const paid=Number(allocated.rows[0].amount),remaining=Number(row.invoice_amount)-paid,amount=Number(row.amount),minimum=Number(row.minimum_installment_amount||0);if(amount>remaining||remaining<=0){await client.query("ROLLBACK");return res.status(409).json({error:"Receipt exceeds the remaining invoice balance"});}if(row.payment_option==="full_only"&&amount!==remaining){await client.query("ROLLBACK");return res.status(409).json({error:"This fee notice requires the full remaining amount"});}if(row.payment_option!=="full_only"&&amount<minimum&&amount!==remaining){await client.query("ROLLBACK");return res.status(409).json({error:"Receipt amount is below the installment minimum"});}await client.query("INSERT INTO fee_receipt_allocations (receipt_submission_id,invoice_id,amount,allocated_by_user_id) VALUES ($1,$2,$3,$4)",[id,row.invoice_id,amount,req.user.id]);await client.query("UPDATE fee_invoices SET status=CASE WHEN amount <= $1::numeric THEN 'paid' WHEN due_date<CURRENT_DATE THEN 'overdue' ELSE 'pending' END,updated_at=NOW() WHERE id=$2",[paid+amount,row.invoice_id]);}
      const updated=await client.query("UPDATE fee_receipt_submissions SET status=$1,reviewed_by_user_id=$2,reviewed_at=NOW(),review_note=$3 WHERE id=$4 RETURNING *",[status,req.user.id,note,id]);await client.query("INSERT INTO fee_receipt_audit_events (receipt_submission_id,event_type,actor_user_id,details) VALUES ($1,$2,$3,$4::jsonb)",[id,status,req.user.id,JSON.stringify({review_note:note,allocation_created:status==='verified'})]);await client.query("COMMIT");res.json({receipt_submission:receiptResponse(updated.rows[0]),balance_source:"verified_receipt_allocations"});
    }catch(error){await client.query("ROLLBACK");res.status(500).json({error:"Failed to review payment receipt",...(process.env.RUN_DB_TESTS==="1"?{detail:error.message}:{})});}finally{client.release();}
  });
  app.get("/api/warden/fee-receipts", requireAuth, requireRole("warden"), async (req,res)=>{const status=req.query.status; if(status&&!["submitted","verified","rejected","needs_clarification"].includes(status))return res.status(400).json({error:"Invalid receipt status"});const rows=await pool.query(`SELECT r.*,u.name AS student_name,i.due_date,i.status AS invoice_status FROM fee_receipt_submissions r JOIN warden_assignments wa ON wa.hostel_id=r.hostel_id JOIN users u ON u.id=r.student_user_id JOIN fee_invoices i ON i.id=r.invoice_id WHERE wa.warden_user_id=$1 AND ($2::varchar IS NULL OR r.status=$2) ORDER BY r.created_at DESC`,[req.user.id,status||null]);res.json({receipts:rows.rows.map(receiptResponse)});});
  app.patch("/api/warden/fee-receipts/:id", requireAuth, requireRole("warden"), async (req,res)=>{const id=Number(req.params.id),status=String(req.body.status||""),note=String(req.body.review_note||"").trim();if(!isValidInt(id)||!["verified","rejected","needs_clarification"].includes(status)||!note||note.length>2000)return res.status(400).json({error:"verified/rejected/needs_clarification status and review_note are required"});const client=await pool.connect();try{await client.query("BEGIN");const receipt=await client.query("SELECT r.*,i.amount AS invoice_amount,i.status AS invoice_status,s.payment_option,s.minimum_installment_amount FROM fee_receipt_submissions r JOIN warden_assignments wa ON wa.hostel_id=r.hostel_id JOIN fee_invoices i ON i.id=r.invoice_id JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id WHERE r.id=$1 AND wa.warden_user_id=$2 FOR UPDATE",[id,req.user.id]);if(!receipt.rowCount){await client.query("ROLLBACK");return res.status(404).json({error:"Receipt submission not found"});}const row=receipt.rows[0];if(row.status!=="submitted"){await client.query("ROLLBACK");return res.status(409).json({error:"Receipt submission has already been reviewed"});}if(status==='verified'){const total=await client.query("SELECT COALESCE(sum(amount),0) AS amount FROM fee_receipt_submissions WHERE invoice_id=$1 AND status='verified'",[row.invoice_id]);const remaining=Number(row.invoice_amount)-Number(total.rows[0].amount);if(row.payment_option==='full_only'&&Number(row.amount)!==remaining){await client.query("ROLLBACK");return res.status(409).json({error:"This fee notice requires the full remaining amount"});}if(row.payment_option!=="full_only"&&(Number(row.amount)<Number(row.minimum_installment_amount)||Number(row.amount)>remaining)){await client.query("ROLLBACK");return res.status(409).json({error:"Receipt amount does not meet the installment policy or remaining balance"});}await client.query("UPDATE fee_invoices SET status=CASE WHEN amount <= $1::numeric + $2::numeric THEN 'paid' ELSE status END,updated_at=NOW() WHERE id=$3",[Number(total.rows[0].amount),Number(row.amount),row.invoice_id]);}const updated=await client.query("UPDATE fee_receipt_submissions SET status=$1,reviewed_by_user_id=$2,reviewed_at=NOW(),review_note=$3 WHERE id=$4 RETURNING *",[status,req.user.id,note,id]);await client.query("INSERT INTO fee_receipt_audit_events (receipt_submission_id,event_type,actor_user_id,details) VALUES ($1,$2,$3,$4::jsonb)",[id,status,req.user.id,JSON.stringify({review_note:note})]);await notifyReceipt(client,row.student_user_id,row.hostel_id,`fee_receipt_${status}`,"Payment receipt updated",`Your payment receipt was ${status.replace(/_/g," ")}.`,{invoice_id:row.invoice_id,receipt_submission_id:id,status});await client.query("COMMIT");res.json({receipt_submission:receiptResponse(updated.rows[0])});}catch(error){await client.query("ROLLBACK");res.status(500).json({error:"Failed to review payment receipt",...(process.env.RUN_DB_TESTS==="1"?{detail:error.message}:{})});}finally{client.release();}});
  app.get("/api/warden/fee-records", requireAuth, requireRole("warden"), async (req, res) => {
    const billingMonth = String(req.query.billing_month || "");
    if (!isBillingMonth(billingMonth)) return res.status(400).json({ error: "billing_month must be YYYY-MM-01" });
    try {
      const rows = await pool.query(
        `WITH residents AS (
           SELECT DISTINCT ON (student_id) * FROM (
             SELECT b.id AS booking_id, b.user_id AS student_id, u.name AS student_name,
                    r.room_number, COALESCE(r.floor_label, 'Ground Floor') AS floor_label,
                    b.bed_number, COALESCE(r.monthly_rent, 0) AS room_monthly_rent
               FROM warden_assignments wa
               JOIN bookings b ON b.hostel_id=wa.hostel_id AND b.status IN ('pending','confirmed')
               JOIN users u ON u.id=b.user_id AND u.role='student'
               LEFT JOIN rooms r ON r.id=b.room_id
              WHERE wa.warden_user_id=$1
             UNION ALL
             SELECT NULL::integer AS booking_id, d.student_user_id AS student_id, u.name AS student_name,
                    NULL::varchar AS room_number, NULL::varchar AS floor_label,
                    NULL::varchar AS bed_number, 0::numeric AS room_monthly_rent
               FROM warden_student_records d
               JOIN users u ON u.id=d.student_user_id AND u.role='student'
               JOIN warden_assignments wa ON wa.hostel_id=d.hostel_id
              WHERE wa.warden_user_id=$1
           ) registered_students
           ORDER BY student_id, booking_id DESC NULLS LAST
         )
         SELECT residents.*, invoice.id AS invoice_id, invoice.amount AS invoice_amount,
                invoice.due_date, invoice.status AS invoice_status, schedule.title AS fee_title,
                COALESCE(receipt_paid.amount,0) AS receipt_paid_amount,
                COALESCE(cash_paid.amount,0) AS cash_paid_amount,
                COALESCE(receipt_paid.amount,0)+COALESCE(cash_paid.amount,0) AS paid_amount,
                GREATEST(COALESCE(invoice.amount,residents.room_monthly_rent)-COALESCE(receipt_paid.amount,0)-COALESCE(cash_paid.amount,0),0) AS remaining_amount,
                cash_latest.id AS latest_cash_payment_id, cash_latest.amount AS latest_cash_amount,
                cash_latest.received_on AS latest_cash_received_on, cash_latest.note AS latest_cash_note,
                receipt.id AS receipt_id, receipt.status AS receipt_status, receipt.amount AS receipt_amount,
                receipt.original_filename AS receipt_filename, receipt.content_type AS receipt_content_type,
                receipt.paid_at AS receipt_paid_at
           FROM residents
           LEFT JOIN LATERAL (
             SELECT i.*, a.schedule_id
               FROM student_fee_assignments a
               JOIN fee_invoices i ON i.assignment_id=a.id AND i.billing_month=$2::date
              WHERE a.booking_id=residents.booking_id AND a.active=true
              ORDER BY i.created_at DESC, i.id DESC LIMIT 1
           ) invoice ON true
           LEFT JOIN hostel_fee_schedules schedule ON schedule.id=invoice.schedule_id
           LEFT JOIN LATERAL (SELECT SUM(amount) AS amount FROM fee_receipt_allocations WHERE invoice_id=invoice.id) receipt_paid ON true
           LEFT JOIN LATERAL (SELECT SUM(amount) AS amount FROM warden_cash_fee_payments WHERE invoice_id=invoice.id AND voided_at IS NULL) cash_paid ON true
           LEFT JOIN LATERAL (SELECT * FROM warden_cash_fee_payments WHERE invoice_id=invoice.id AND voided_at IS NULL ORDER BY received_on DESC,id DESC LIMIT 1) cash_latest ON true
           LEFT JOIN LATERAL (SELECT * FROM fee_receipt_submissions WHERE invoice_id=invoice.id ORDER BY created_at DESC,id DESC LIMIT 1) receipt ON true
          ORDER BY residents.student_name`,
        [req.user.id, billingMonth],
      );
      res.json({ billing_month: billingMonth, currency: "PKR", records: rows.rows });
    } catch (error) {
      res.status(500).json({ error: "Failed to load warden fee records" });
    }
  });

  app.get("/api/warden/students/:studentId/rent-records", requireAuth, requireRole("warden"), async (req, res) => {
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId)) return res.status(400).json({ error: "Valid student id is required" });
    try {
      const student = await pool.query(
        `SELECT u.id AS student_id,u.name AS student_name,b.id AS booking_id,
                r.room_number,COALESCE(r.floor_label,'Ground Floor') AS floor_label,b.bed_number
           FROM users u
           JOIN warden_assignments wa ON wa.warden_user_id=$2
           LEFT JOIN bookings b ON b.user_id=u.id AND b.hostel_id=wa.hostel_id AND b.status IN ('pending','confirmed')
           LEFT JOIN rooms r ON r.id=b.room_id
           LEFT JOIN warden_student_records d ON d.student_user_id=u.id AND d.hostel_id=wa.hostel_id
          WHERE u.id=$1 AND u.role='student' AND (b.id IS NOT NULL OR d.student_user_id IS NOT NULL)
          ORDER BY b.id DESC NULLS LAST LIMIT 1`,
        [studentId, req.user.id],
      );
      if (!student.rowCount) return res.status(404).json({ error: "Registered student not found" });
      const records = await pool.query(
        `SELECT i.id AS invoice_id,i.billing_month,i.due_date,i.amount AS invoice_amount,i.status AS invoice_status,
                a.booking_id,s.title AS fee_title,
                COALESCE(receipt_paid.amount,0) AS receipt_paid_amount,
                COALESCE(cash_paid.amount,0) AS cash_paid_amount,
                COALESCE(receipt_paid.amount,0)+COALESCE(cash_paid.amount,0) AS paid_amount,
                GREATEST(i.amount-COALESCE(receipt_paid.amount,0)-COALESCE(cash_paid.amount,0),0) AS remaining_amount,
                cash_latest.id AS latest_cash_payment_id,cash_latest.amount AS latest_cash_amount,
                cash_latest.received_on AS latest_cash_received_on,cash_latest.note AS latest_cash_note,
                receipt.id AS receipt_id,receipt.status AS receipt_status,receipt.original_filename AS receipt_filename
           FROM fee_invoices i
           JOIN student_fee_assignments a ON a.id=i.assignment_id
           JOIN hostel_fee_schedules s ON s.id=a.schedule_id
           JOIN warden_assignments wa ON wa.hostel_id=s.hostel_id
           LEFT JOIN LATERAL (SELECT SUM(amount) AS amount FROM fee_receipt_allocations WHERE invoice_id=i.id) receipt_paid ON true
           LEFT JOIN LATERAL (SELECT SUM(amount) AS amount FROM warden_cash_fee_payments WHERE invoice_id=i.id AND voided_at IS NULL) cash_paid ON true
           LEFT JOIN LATERAL (SELECT * FROM warden_cash_fee_payments WHERE invoice_id=i.id AND voided_at IS NULL ORDER BY received_on DESC,id DESC LIMIT 1) cash_latest ON true
           LEFT JOIN LATERAL (SELECT * FROM fee_receipt_submissions WHERE invoice_id=i.id ORDER BY created_at DESC,id DESC LIMIT 1) receipt ON true
          WHERE a.student_user_id=$1 AND wa.warden_user_id=$2
          ORDER BY i.billing_month DESC,i.id DESC`,
        [studentId, req.user.id],
      );
      res.json({ student: student.rows[0], currency: "PKR", records: records.rows });
    } catch (error) {
      res.status(500).json({ error: "Failed to load student rent history" });
    }
  });

  app.post("/api/warden/fee-records/:bookingId/setup", requireAuth, requireRole("warden"), async (req, res) => {
    const bookingId = Number(req.params.bookingId);
    const billingMonth = String(req.body.billing_month || "");
    if (!isValidInt(bookingId) || !isBillingMonth(billingMonth)) return res.status(400).json({ error: "Booking and billing_month (YYYY-MM-01) are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const booking = await client.query(
        `SELECT b.id,b.user_id,b.hostel_id,COALESCE(r.monthly_rent,0) AS monthly_rent
           FROM bookings b
           JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id
           LEFT JOIN rooms r ON r.id=b.room_id
          WHERE b.id=$1 AND b.status='confirmed' AND wa.warden_user_id=$2
          FOR UPDATE OF b`,
        [bookingId, req.user.id],
      );
      if (!booking.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Registered student not found" }); }
      const resident = booking.rows[0];
      const rent = Number(resident.monthly_rent);
      if (!Number.isFinite(rent) || rent <= 0) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Set the room monthly rent before creating a rent record" }); }

      const existing = await client.query(
        `SELECT i.* FROM fee_invoices i
           JOIN student_fee_assignments a ON a.id=i.assignment_id
          WHERE a.booking_id=$1 AND i.billing_month=$2::date
          ORDER BY i.id DESC LIMIT 1`,
        [bookingId, billingMonth],
      );
      if (existing.rowCount) { await client.query("COMMIT"); return res.json({ invoice: existing.rows[0], created: false }); }

      let schedule = await client.query(
        `SELECT id FROM hostel_fee_schedules
          WHERE hostel_id=$1 AND active=true AND lower(title)='monthly hostel rent' AND amount=$2
          ORDER BY id DESC LIMIT 1 FOR UPDATE`,
        [resident.hostel_id, rent],
      );
      if (!schedule.rowCount) {
        schedule = await client.query(
          `INSERT INTO hostel_fee_schedules (hostel_id,title,amount,currency,due_day,payment_option,created_by_user_id)
           VALUES ($1,'Monthly hostel rent',$2,'PKR',5,'both',$3) RETURNING id`,
          [resident.hostel_id, rent, req.user.id],
        );
      }
      const scheduleId = schedule.rows[0].id;
      let assignment = await client.query(
        `SELECT id FROM student_fee_assignments
          WHERE schedule_id=$1 AND booking_id=$2
          ORDER BY id DESC LIMIT 1 FOR UPDATE`,
        [scheduleId, bookingId],
      );
      if (!assignment.rowCount) {
        assignment = await client.query(
          `INSERT INTO student_fee_assignments (schedule_id,booking_id,student_user_id,starts_on,assigned_by_user_id)
           VALUES ($1,$2,$3,$4::date,$5) RETURNING id`,
          [scheduleId, bookingId, resident.user_id, billingMonth, req.user.id],
        );
      }
      const invoice = await client.query(
        `INSERT INTO fee_invoices (assignment_id,billing_month,due_date,amount,currency)
         VALUES ($1,$2::date,($2::date + INTERVAL '4 days')::date,$3,'PKR')
         ON CONFLICT (assignment_id,billing_month) DO UPDATE SET updated_at=fee_invoices.updated_at
         RETURNING *`,
        [assignment.rows[0].id, billingMonth, rent],
      );
      await client.query("COMMIT");
      res.status(201).json({ invoice: invoice.rows[0], created: true });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to set up the monthly rent record" }); }
    finally { client.release(); }
  });

  app.post("/api/warden/cash-fee-payments", requireAuth, requireRole("warden"), async (req, res) => {
    const invoiceId = Number(req.body.invoice_id);
    const amount = Number(req.body.amount);
    const receivedOn = String(req.body.received_on || "");
    const note = String(req.body.note || "").trim();
    if (!isValidInt(invoiceId) || !Number.isFinite(amount) || amount <= 0 || !/^\d{4}-\d{2}-\d{2}$/.test(receivedOn) || note.length > 1000) return res.status(400).json({ error: "Invoice, positive amount, received date, and a note up to 1000 characters are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const invoice = await client.query(
        `SELECT i.*,a.student_user_id,s.hostel_id
           FROM fee_invoices i
           JOIN student_fee_assignments a ON a.id=i.assignment_id
           JOIN hostel_fee_schedules s ON s.id=a.schedule_id
           JOIN warden_assignments wa ON wa.hostel_id=s.hostel_id
          WHERE i.id=$1 AND wa.warden_user_id=$2 FOR UPDATE OF i`,
        [invoiceId, req.user.id],
      );
      if (!invoice.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Fee invoice not found" }); }
      const row = invoice.rows[0];
      const paid = await paidForInvoice(client, invoiceId);
      if (paid + amount > Number(row.amount)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Cash payment exceeds the remaining balance" }); }
      const payment = await client.query(
        `INSERT INTO warden_cash_fee_payments (invoice_id,hostel_id,student_user_id,amount,received_on,note,received_by_warden_id)
         VALUES ($1,$2,$3,$4,$5::date,$6,$7) RETURNING *`,
        [invoiceId, row.hostel_id, row.student_user_id, amount, receivedOn, note || null, req.user.id],
      );
      const totals = await refreshInvoiceStatus(client, invoiceId);
      await client.query("COMMIT");
      res.status(201).json({ cash_payment: payment.rows[0], paid_amount: totals.paid, invoice_status: totals.status });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to record cash payment" }); }
    finally { client.release(); }
  });

  app.patch("/api/warden/cash-fee-payments/:id", requireAuth, requireRole("warden"), async (req, res) => {
    const paymentId = Number(req.params.id);
    const amount = Number(req.body.amount);
    const receivedOn = String(req.body.received_on || "");
    const note = String(req.body.note || "").trim();
    if (!isValidInt(paymentId) || !Number.isFinite(amount) || amount <= 0 || !/^\d{4}-\d{2}-\d{2}$/.test(receivedOn) || note.length > 1000) return res.status(400).json({ error: "Valid cash payment details are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const payment = await client.query(
        `SELECT p.* FROM warden_cash_fee_payments p
           JOIN warden_assignments wa ON wa.hostel_id=p.hostel_id
          WHERE p.id=$1 AND wa.warden_user_id=$2 AND p.voided_at IS NULL FOR UPDATE`,
        [paymentId, req.user.id],
      );
      if (!payment.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Cash payment not found" }); }
      const current = payment.rows[0];
      const paidWithoutCurrent = await paidForInvoice(client, current.invoice_id, paymentId);
      const invoice = await client.query("SELECT amount FROM fee_invoices WHERE id=$1 FOR UPDATE", [current.invoice_id]);
      if (paidWithoutCurrent + amount > Number(invoice.rows[0].amount)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Cash payment exceeds the remaining balance" }); }
      const updated = await client.query("UPDATE warden_cash_fee_payments SET amount=$1,received_on=$2::date,note=$3,updated_at=NOW() WHERE id=$4 RETURNING *", [amount, receivedOn, note || null, paymentId]);
      const totals = await refreshInvoiceStatus(client, current.invoice_id);
      await client.query("COMMIT");
      res.json({ cash_payment: updated.rows[0], paid_amount: totals.paid, invoice_status: totals.status });
    } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to update cash payment" }); }
    finally { client.release(); }
  });

  app.get("/api/fee-receipts/:id/download", requireAuth, async (req,res)=>{const id=Number(req.params.id),r=await pool.query("SELECT * FROM fee_receipt_submissions WHERE id=$1",[id]);if(!r.rowCount)return res.status(404).json({error:"Receipt not found"});const receipt=r.rows[0];const allowed=(req.user.role==='student'&&Number(receipt.student_user_id)===Number(req.user.id))||(req.user.role==='parent'&&(await linkedParent(req.user.id,receipt.student_user_id)).rowCount)||(req.user.role==='warden'&&await wardenHasHostel(pool,req.user.id,receipt.hostel_id))||(req.user.role==='owner'&&await ownerHasHostel(pool,req.user.id,receipt.hostel_id));if(!allowed||!await documentStorage.exists(receipt.storage_key))return res.status(404).json({error:"Receipt not found"});res.type(receipt.content_type);res.attachment(safeDownloadName(receipt.original_filename));documentStorage.createReadStream(receipt.storage_key).on("error",()=>res.destroy()).pipe(res);});
}
module.exports={registerFeeRoutes};
