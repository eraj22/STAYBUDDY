const crypto = require("node:crypto");
const path = require("node:path");
const reasoning = require("./chat_reasoning");
const { registerChatSupportRoutes } = require("./chat_support_routes");

function cleanText(value, maximum = 4000) {
  return String(value || "").trim().slice(0, maximum);
}

const detectLanguage = reasoning.languageFor;
const detectIntent = reasoning.intentFor;
const validSessionId = id => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);

function actionSuggestions(intent, role) {
  if (role === "warden") {
    const staffActions = {
      booking: [{ type: "navigate", target: "warden_bookings", label: "Review hostel bookings" }],
      fees: [{ type: "navigate", target: "warden_fee_receipts", label: "Review fee receipts" }],
      complaint_status: [{ type: "navigate", target: "warden_complaints", label: "Review complaints" }],
      attendance: [{ type: "navigate", target: "warden_attendance", label: "Open hostel attendance" }],
      human_support: [{ type: "escalate", target: "human_support", label: "Escalate this chat" }],
    };
    return staffActions[intent] || [];
  }
  if (role === "owner") {
    const ownerActions = {
      booking: [{ type: "navigate", target: "owner_bookings", label: "Review hostel bookings" }],
      fees: [{ type: "navigate", target: "owner_fees", label: "Review fee operations" }],
      complaint_status: [{ type: "navigate", target: "owner_complaints", label: "Review complaints" }],
      attendance: [{ type: "navigate", target: "owner_attendance", label: "Open attendance settings" }],
      human_support: [{ type: "escalate", target: "human_support", label: "Escalate this chat" }],
    };
    return ownerActions[intent] || [];
  }
  const actions = {
    booking: [{ type: "navigate", target: "hostel_search", label: "Search hostels" }, { type: "navigate", target: "my_bookings", label: "View my bookings" }],
    hostel_search: [{ type: "navigate", target: "hostel_search", label: "Open hostel search" }],
    fees: [{ type: "navigate", target: role === "parent" ? "child_fees" : "my_fees", label: "View fee records" }],
    complaint_status: [{ type: "navigate", target: "complaints", label: "View complaints" }],
    attendance: [{ type: "navigate", target: role === "parent" ? "child_attendance" : "attendance_qr", label: "Open attendance" }],
    cancellation: [{ type: "navigate", target: "my_bookings", label: "Review booking" }],
    human_support: [{ type: "escalate", target: "human_support", label: "Escalate this chat" }],
  };
  let result = actions[intent] || [];
  if (role === "parent") result = result.filter(a => !["my_bookings","complaints","attendance_qr"].includes(a.target));
  if (role === "guest") result = result.map(a => ["my_bookings","my_fees","complaints","attendance_qr"].includes(a.target) ? {type:"navigate",target:"login",label:"Sign in to view your records"} : a);
  return result;
}

async function retrieveKnowledge(pool, query, role, limit = 6, intent = "general_support") {
  if (intent === "general_knowledge") return [];
  const tokens = reasoning.retrievalTokens(query);
  const categories = reasoning.CATEGORIES[intent] || [];
  if (!tokens.length && !categories.length) return [];
  const result = await pool.query(
    `WITH matches AS (
       SELECT c.id AS chunk_id,c.content,d.id AS document_id,d.slug,d.title,d.category,d.source_kind,d.source_reference,d.version,d.reviewed_at,
         (SELECT COUNT(*)::float FROM unnest($2::text[]) term
          WHERE to_tsvector('english',d.title) @@ plainto_tsquery('english',term)
             OR position(term in lower(d.title))>0) AS title_hits,
         (SELECT COUNT(*)::float FROM unnest($2::text[]) term
          WHERE to_tsvector('english',c.content) @@ plainto_tsquery('english',term)
             OR position(term in lower(c.content))>0) AS content_hits
       FROM chatbot_knowledge_chunks c JOIN chatbot_knowledge_documents d ON d.id=c.document_id
       WHERE d.active AND ('all'=ANY(d.audience) OR $1=ANY(d.audience))
     ), scored AS (
       SELECT *, title_hits+content_hits AS hits,
         (title_hits*4+content_hits+CASE WHEN category=ANY($3::text[]) THEN 0.5 ELSE 0 END)::float AS score
       FROM matches
     ), candidates AS (
       SELECT *, ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY score DESC,chunk_id) AS document_rank
       FROM scored
       WHERE hits>0 OR (category=ANY($3::text[]) AND NOT EXISTS (SELECT 1 FROM scored WHERE hits>0))
     ) SELECT chunk_id,content,document_id,slug,title,category,source_kind,source_reference,version,reviewed_at,hits,score
       FROM candidates WHERE document_rank<=2
       ORDER BY score DESC,reviewed_at DESC,chunk_id LIMIT $4`,
    [role,tokens,categories,limit]);
  return result.rows;
}

async function buildStudentContext(pool, userId, intent) {
  const context = [];
  if (["booking", "cancellation"].includes(intent)) {
    const bookings = await pool.query(
      `SELECT b.id,b.status,b.check_in,b.check_out,b.expires_at,b.price_snapshot,b.currency,h.id AS hostel_id,h.name AS hostel_name,
              (SELECT COUNT(*)::int FROM booking_document_requirements r WHERE r.hostel_id=b.hostel_id AND r.active AND r.required) AS required_documents,
              (SELECT COUNT(DISTINCT d.document_code)::int FROM booking_documents d JOIN booking_document_requirements r ON r.hostel_id=b.hostel_id AND r.document_code=d.document_code AND r.active AND r.required WHERE d.booking_id=b.id AND d.status='accepted' AND d.deleted_at IS NULL) AS accepted_documents,
              COALESCE((SELECT jsonb_agg(jsonb_build_object('document_code',r.document_code,'label',r.label,'required',r.required,
                'status',COALESCE((SELECT d.status FROM booking_documents d WHERE d.booking_id=b.id AND d.document_code=r.document_code AND d.deleted_at IS NULL ORDER BY d.created_at DESC,d.id DESC LIMIT 1),'missing')) ORDER BY r.id)
                FROM booking_document_requirements r WHERE r.hostel_id=b.hostel_id AND r.active),'[]'::jsonb) AS document_requirements
       FROM bookings b JOIN hostels h ON h.id=b.hostel_id WHERE b.user_id=$1 ORDER BY b.created_at DESC LIMIT 5`,
      [userId]
    );
    context.push({ type: "bookings", records: bookings.rows });
  }
  if (intent === "fees") {
    const fees = await pool.query(
      `SELECT i.id,i.amount,i.currency,i.due_date,i.status,COALESCE(p.paid_amount,0) AS paid_amount,(i.amount-COALESCE(p.paid_amount,0)) AS balance,h.name AS hostel_name
       FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id
       JOIN hostel_fee_schedules s ON s.id=a.schedule_id JOIN hostels h ON h.id=s.hostel_id
       LEFT JOIN LATERAL (SELECT SUM(amount) AS paid_amount FROM fee_receipt_allocations WHERE invoice_id=i.id) p ON true
       WHERE a.student_user_id=$1 ORDER BY i.due_date DESC LIMIT 10`, [userId]
    );
    context.push({ type: "fees", records: fees.rows });
  }
  if (intent === "complaint_status") {
    const complaints = await pool.query(`SELECT c.id,c.category,c.severity,c.status,c.created_at,c.resolved_at,
      COALESCE((SELECT jsonb_agg(jsonb_build_object('status',sh.next_status,'at',sh.changed_at) ORDER BY sh.changed_at,sh.id)
        FROM complaint_status_history sh WHERE sh.complaint_id=c.id),'[]'::jsonb) AS timeline,
      (SELECT safe_resolution_summary FROM complaint_resolution_records WHERE complaint_id=c.id) AS resolution,
      COALESCE((SELECT jsonb_agg(jsonb_build_object('content',cm.content,'at',cm.created_at) ORDER BY cm.created_at)
        FROM chatbot_complaint_comments cm WHERE cm.complaint_id=c.id),'[]'::jsonb) AS comments
      FROM complaints c WHERE c.user_id=$1 ORDER BY c.created_at DESC LIMIT 10`, [userId]);
    context.push({ type: "complaints", records: complaints.rows });
  }
  if (intent === "attendance") {
    const attendance = await pool.query(
      `SELECT e.id,COALESCE(c.corrected_direction,e.direction) AS direction,COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at,e.source,e.entry_point,h.name AS hostel_name
       FROM attendance_events e JOIN hostels h ON h.id=e.hostel_id
       LEFT JOIN LATERAL (SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true
       WHERE e.student_user_id=$1 ORDER BY occurred_at DESC,e.id DESC LIMIT 10`, [userId]
    );
    context.push({ type: "attendance", records: attendance.rows });
  }
  return context;
}

async function buildParentContext(pool, userId, intent) {
  const children = await pool.query("SELECT u.id,u.name FROM parent_student_links l JOIN users u ON u.id=l.student_user_id WHERE l.parent_user_id=$1 AND l.status='active' ORDER BY u.name", [userId]);
  const childIds = children.rows.map((row) => row.id);
  const context = [{ type: "linked_children", records: children.rows }];
  if (!childIds.length) return context;
  if (["booking", "cancellation"].includes(intent)) {
    const bookings = await pool.query("SELECT b.id,b.user_id AS student_id,u.name AS student_name,b.status,b.check_in,b.check_out,h.name AS hostel_name FROM bookings b JOIN users u ON u.id=b.user_id JOIN hostels h ON h.id=b.hostel_id WHERE b.user_id=ANY($1::int[]) ORDER BY b.created_at DESC LIMIT 10", [childIds]);
    context.push({ type: "child_bookings", records: bookings.rows });
  }
  if (intent === "fees") {
    const fees = await pool.query(`SELECT a.student_user_id AS student_id,u.name AS student_name,i.id,i.amount,i.currency,i.due_date,i.status,COALESCE(p.paid_amount,0) AS paid_amount,(i.amount-COALESCE(p.paid_amount,0)) AS balance,h.name AS hostel_name FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN hostel_fee_schedules s ON s.id=a.schedule_id JOIN hostels h ON h.id=s.hostel_id JOIN users u ON u.id=a.student_user_id LEFT JOIN LATERAL (SELECT SUM(amount) AS paid_amount FROM fee_receipt_allocations WHERE invoice_id=i.id) p ON true WHERE a.student_user_id=ANY($1::int[]) ORDER BY i.due_date DESC LIMIT 20`, [childIds]);
    context.push({ type: "child_fees", records: fees.rows });
  }
  if (intent === "attendance") {
    const attendance = await pool.query(`SELECT DISTINCT ON(e.student_user_id) e.student_user_id AS student_id,u.name AS student_name,COALESCE(c.corrected_direction,e.direction) AS direction,COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at,e.source,h.name AS hostel_name FROM attendance_events e JOIN users u ON u.id=e.student_user_id JOIN hostels h ON h.id=e.hostel_id LEFT JOIN LATERAL(SELECT * FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1)c ON true WHERE e.student_user_id=ANY($1::int[]) ORDER BY e.student_user_id,occurred_at DESC,e.id DESC`, [childIds]);
    context.push({ type: "child_attendance", records: attendance.rows });
  }
  return context;
}

async function buildWardenContext(pool, userId, intent) {
  const assigned = await pool.query(`SELECT h.id,h.name,h.city,h.rules,s.earliest_exit_time,s.curfew_time,
      s.grace_minutes,s.escalation_minutes,s.enabled AS attendance_policy_enabled
    FROM warden_assignments wa JOIN hostels h ON h.id=wa.hostel_id
    LEFT JOIN hostel_attendance_settings s ON s.hostel_id=h.id
    WHERE wa.warden_user_id=$1`, [userId]);
  const context = [{ type: "warden_hostel", records: assigned.rows }];
  if (!assigned.rowCount) return context;
  const hostelId = assigned.rows[0].id;
  if (["booking", "cancellation"].includes(intent)) {
    const rows = await pool.query(`SELECT b.id,b.status,b.check_in,b.check_out,u.name AS student_name,
        r.room_number FROM bookings b JOIN users u ON u.id=b.user_id
        LEFT JOIN rooms r ON r.id=b.room_id WHERE b.hostel_id=$1 ORDER BY b.created_at DESC LIMIT 10`, [hostelId]);
    context.push({ type: "warden_bookings", records: rows.rows });
  }
  if (intent === "complaint_status") {
    const rows = await pool.query(`SELECT c.id,c.category,c.severity,c.priority,c.status,c.created_at,
        u.name AS student_name FROM complaints c JOIN users u ON u.id=c.user_id
        WHERE c.hostel_id=$1 ORDER BY c.created_at DESC LIMIT 10`, [hostelId]);
    context.push({ type: "warden_complaints", records: rows.rows });
  }
  if (intent === "fees") {
    const rows = await pool.query(`SELECT r.id,r.amount,r.status,r.created_at,u.name AS student_name,
        i.due_date FROM fee_receipt_submissions r JOIN users u ON u.id=r.student_user_id
        JOIN fee_invoices i ON i.id=r.invoice_id WHERE r.hostel_id=$1
        ORDER BY r.created_at DESC LIMIT 10`, [hostelId]);
    context.push({ type: "warden_fee_receipts", records: rows.rows });
  }
  if (intent === "attendance") {
    const summary = await pool.query(`WITH residents AS (
        SELECT b.user_id,MAX(s.checked_in_at) AS checked_in_at FROM bookings b JOIN booking_stays s ON s.booking_id=b.id
        WHERE b.hostel_id=$1 AND b.status='confirmed' AND s.checked_in_at IS NOT NULL AND s.checked_out_at IS NULL
        GROUP BY b.user_id
      ), latest AS (
        SELECT r.user_id,(SELECT COALESCE(c.corrected_direction,e.direction) FROM attendance_events e
          LEFT JOIN LATERAL (SELECT corrected_direction FROM attendance_corrections ac
            WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1) c ON true
          WHERE e.hostel_id=$1 AND e.student_user_id=r.user_id AND e.occurred_at>=r.checked_in_at
          ORDER BY e.occurred_at DESC,e.id DESC LIMIT 1) AS direction
        FROM residents r
      ) SELECT COUNT(*)::int AS active_residents,
        COUNT(*) FILTER(WHERE direction='in')::int AS inside,
        COUNT(*) FILTER(WHERE direction='out')::int AS outside,
        COUNT(*) FILTER(WHERE direction IS NULL)::int AS unknown,
        (SELECT COUNT(*)::int FROM attendance_discipline_appeals WHERE hostel_id=$1 AND status='pending') AS pending_discipline_appeals
      FROM latest`, [hostelId]);
    context.push({ type: "warden_attendance_summary", records: summary.rows });
    const pending = await pool.query(`SELECT p.id,p.action_id,p.requested_at,a.action_type,u.name AS student_name
      FROM attendance_discipline_appeals p JOIN attendance_disciplinary_actions a ON a.id=p.action_id
      JOIN users u ON u.id=p.student_user_id WHERE p.hostel_id=$1 AND p.status='pending'
      ORDER BY p.requested_at DESC LIMIT 10`, [hostelId]);
    context.push({ type: "warden_pending_discipline_appeals", records: pending.rows });
    const events = await pool.query(`SELECT e.id,u.name AS student_name,
        COALESCE(c.corrected_direction,e.direction) AS direction,
        COALESCE(c.corrected_occurred_at,e.occurred_at) AS occurred_at
      FROM attendance_events e JOIN users u ON u.id=e.student_user_id
      LEFT JOIN LATERAL (SELECT corrected_direction,corrected_occurred_at FROM attendance_corrections ac
        WHERE ac.attendance_event_id=e.id ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1) c ON true
      WHERE e.hostel_id=$1 ORDER BY e.occurred_at DESC,e.id DESC LIMIT 10`, [hostelId]);
    context.push({ type: "warden_attendance_events", records: events.rows });
  }
  return context;
}

async function buildOwnerContext(pool, userId, intent) {
  const owned = await pool.query(`SELECT h.id,h.name,h.city,h.available_capacity,
      COALESCE(s.enabled,false) AS attendance_policy_enabled,s.earliest_exit_time,s.curfew_time
    FROM hostel_owners ho JOIN hostels h ON h.id=ho.hostel_id
    LEFT JOIN hostel_attendance_settings s ON s.hostel_id=h.id
    WHERE ho.owner_user_id=$1 ORDER BY h.id`, [userId]);
  const context = [{ type: "owner_hostels", records: owned.rows }];
  if (!owned.rowCount) return context;
  const hostelIds = owned.rows.map(row => row.id);
  if (intent === "attendance") {
    const summary = await pool.query(`WITH residents AS (
        SELECT b.hostel_id,b.user_id,MAX(s.checked_in_at) AS checked_in_at
        FROM bookings b JOIN booking_stays s ON s.booking_id=b.id
        WHERE b.hostel_id=ANY($1::int[]) AND b.status='confirmed'
          AND s.checked_in_at IS NOT NULL AND s.checked_out_at IS NULL
        GROUP BY b.hostel_id,b.user_id
      ), latest AS (
        SELECT r.hostel_id,r.user_id,(SELECT COALESCE(c.corrected_direction,e.direction)
          FROM attendance_events e LEFT JOIN LATERAL(
            SELECT corrected_direction FROM attendance_corrections ac WHERE ac.attendance_event_id=e.id
            ORDER BY ac.created_at DESC,ac.id DESC LIMIT 1) c ON true
          WHERE e.hostel_id=r.hostel_id AND e.student_user_id=r.user_id AND e.occurred_at>=r.checked_in_at
          ORDER BY e.occurred_at DESC,e.id DESC LIMIT 1) AS direction FROM residents r
      ) SELECT h.id AS hostel_id,h.name AS hostel_name,COUNT(latest.user_id)::int AS active_residents,
          COUNT(*) FILTER(WHERE latest.direction='in')::int AS inside,
          COUNT(*) FILTER(WHERE latest.direction='out')::int AS outside,
          COUNT(*) FILTER(WHERE latest.direction IS NULL)::int AS unknown
        FROM hostels h LEFT JOIN latest ON latest.hostel_id=h.id WHERE h.id=ANY($1::int[])
        GROUP BY h.id,h.name ORDER BY h.id`, [hostelIds]);
    context.push({ type: "owner_attendance_summary", records: summary.rows });
  }
  if (intent === "fees") {
    const summary = await pool.query(`SELECT h.id AS hostel_id,h.name AS hostel_name,COUNT(i.id)::int AS invoice_count,
        COALESCE(SUM(GREATEST(i.amount-COALESCE(a.paid_amount,0),0)),0) AS outstanding_balance
      FROM hostels h LEFT JOIN hostel_fee_schedules s ON s.hostel_id=h.id
      LEFT JOIN student_fee_assignments fa ON fa.schedule_id=s.id
      LEFT JOIN fee_invoices i ON i.assignment_id=fa.id AND i.status<>'cancelled'
      LEFT JOIN LATERAL(SELECT SUM(amount) AS paid_amount FROM fee_receipt_allocations WHERE invoice_id=i.id) a ON true
      WHERE h.id=ANY($1::int[]) GROUP BY h.id,h.name ORDER BY h.id`, [hostelIds]);
    context.push({ type: "owner_fee_summary", records: summary.rows });
  }
  if (intent === "booking") {
    const summary = await pool.query(`SELECT h.id AS hostel_id,h.name AS hostel_name,b.status,COUNT(*)::int AS count
      FROM hostels h LEFT JOIN bookings b ON b.hostel_id=h.id
      WHERE h.id=ANY($1::int[]) GROUP BY h.id,h.name,b.status ORDER BY h.id,b.status`, [hostelIds]);
    context.push({ type: "owner_booking_summary", records: summary.rows });
  }
  return context;
}

async function buildHostelContext(pool, filters) {
  const f=reasoning.safeFilters(filters);
  const result=await pool.query(
    `SELECT h.id,h.name,h.city,h.university,h.hostel_type,h.available_capacity,
       h.facilities,h.amenities,h.rules,'PKR' AS currency,
       (SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id=h.id AND available_capacity>0) AS display_price,
       (SELECT COALESCE(jsonb_object_agg(room_type, monthly_rent), '{}'::jsonb)
          FROM (SELECT lower(room_type) AS room_type, MIN(monthly_rent) AS monthly_rent
                  FROM rooms WHERE hostel_id=h.id AND available_capacity>0 AND monthly_rent IS NOT NULL
                  GROUP BY lower(room_type)) available_room_prices) AS room_prices,
       CASE WHEN h.description LIKE '%Sample room prices for chatbot testing.%'
            THEN 'Example monthly rates for chatbot testing; confirm actual rates with hostel staff.'
            ELSE NULL END AS price_note,
       ROUND(AVG(r.overall_rating)::numeric,2) AS average_rating
     FROM hostels h LEFT JOIN reviews r ON r.hostel_id=h.id
     WHERE h.verified AND h.publication_status='published' AND h.available_capacity>0
       AND EXISTS(SELECT 1 FROM rooms rm WHERE rm.hostel_id=h.id AND rm.available_capacity>0
           AND ($1::numeric IS NULL OR rm.monthly_rent<=$1))
       AND ($2::text IS NULL OR position(lower($2) in lower(concat_ws(' ',h.name,h.city,h.university)))>0)
       AND ($3::text IS NULL OR
         ($3='girls' AND lower(h.hostel_type) IN ('girls','female','girls hostel')) OR
         ($3='boys' AND lower(h.hostel_type) IN ('boys','male','boys hostel')))
       AND NOT EXISTS(SELECT 1 FROM unnest($4::text[]) feature
         WHERE position(lower(feature) in lower(concat_ws(' ',h.amenities::text,h.facilities::text)))=0)
     GROUP BY h.id
     ORDER BY CASE WHEN $5='price' THEN (SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id=h.id AND available_capacity>0) END ASC NULLS LAST,
       average_rating DESC NULLS LAST,display_price ASC NULLS LAST,h.id LIMIT 5`,
    [f.max_price,f.location,f.gender,f.amenities,f.sort]);
  return {type:"hostel_search",filters:f,records:result.rows,notes:"Only published verified listings. Prices PKR/month. Availability is current, not a guarantee for future dates. Wi-Fi speed and exact distance are unknown unless supplied."};
}

async function buildLiveContext(pool,user,plan,message="") {
  const context=[];
  if(user?.role === "warden" && plan.intent !== "general_knowledge" &&
    (plan.intent !== "general_support" || /\b(my|assigned|warden|hostel|appeal)\b/i.test(message)))
    context.push(...await buildWardenContext(pool,user.id,plan.intent));
  else if(user?.role === "owner" && plan.intent !== "general_knowledge" &&
    (plan.intent !== "general_support" || /\b(my|owned|owner|hostel)\b/i.test(message)))
    context.push(...await buildOwnerContext(pool,user.id,plan.intent));
  else if(user && plan.needs_personal_data) context.push(...(user.role==="parent"
    ? await buildParentContext(pool,user.id,plan.intent)
    : user.role==="student" ? await buildStudentContext(pool,user.id,plan.intent) : []));
  if(plan.intent==="hostel_search") context.push(await buildHostelContext(pool,plan.filters));
  if(!user && plan.needs_personal_data) context.push({type:"access",records:[],note:"Guest: sign in for account information."});
  return context;
}

function conciseRecord(record) {
  return Object.entries(record).filter(([, value]) => value != null).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value instanceof Date ? value.toISOString() : value}`).join(", ");
}

function actionId(target, id) { return `${target}:${id}`; }

function requestedAppealId(message) {
  const explicit = String(message).match(/(?:appeal|request)\s*(?:id\s*)?#?(\d+)/i) || String(message).match(/#(\d+)\b/);
  return explicit ? Number(explicit[1]) : null;
}

function requestedBookingId(message) {
  const explicit = String(message).match(/booking\s*(?:id\s*)?#?(\d+)/i);
  return explicit ? Number(explicit[1]) : null;
}

function personalActionSuggestions(user, plan, message, liveContext) {
  if (!user) return [];
  const actions = [];
  const text = String(message).toLowerCase();
  if (user.role === "student") {
    const bookings = liveContext.find(item => item.type === "bookings")?.records || [];
    if (plan.intent === "cancellation") for (const booking of bookings) if (["pending", "confirmed"].includes(booking.status)) {
      actions.push({ type: "api", id: actionId("cancel_booking", booking.id), target: "cancel_booking",
        label: `Cancel booking at ${booking.hostel_name}`, method: "POST", path: `/api/bookings/${booking.id}/cancel`,
        body: { reason: "Student confirmed through the AI assistant" }, requires_confirmation: true });
    }
  }
  if (user.role === "parent" && /\b(notify|alert|watch)\b.*\b(return(?:s|ed|ing)?|back)\b|\b(return(?:s|ed|ing)?|back)\b.*\b(notify|alert|watch)\b/i.test(text)) {
    const children = liveContext.find(item => item.type === "linked_children")?.records || [];
    for (const child of children) actions.push({ type: "api", id: actionId("return_watch", child.id), target: "return_watch",
      label: `Notify me when ${child.name} returns`, method: "POST", path: `/api/parent/children/${child.id}/attendance/notify-when-returned`,
      body: {}, requires_confirmation: true });
  }
  if (user.role === "warden") {
    const decision = /\b(reject|decline|deny)\b/i.test(message) ? "rejected" : /\b(approve|accept)\b/i.test(message) ? "approved" : null;
    const appealId = requestedAppealId(message);
    const pending = liveContext.find(item => item.type === "warden_pending_discipline_appeals")?.records || [];
    if (decision && appealId && pending.some(appeal => Number(appeal.id) === appealId)) {
      actions.push({ type: "api", id: actionId("review_discipline_appeal", appealId), target: "review_discipline_appeal",
        label: `${decision === "approved" ? "Approve" : "Reject"} attendance appeal #${appealId}`,
        method: "POST", path: `/api/warden/attendance/discipline-appeals/${appealId}/review`,
        body: { decision, reason: `Confirmed through the AI assistant: ${decision}.` }, requires_confirmation: true });
    }
    const bookingId = requestedBookingId(message);
    const bookings = liveContext.find(item => item.type === "warden_bookings")?.records || [];
    if (decision && bookingId && bookings.some(booking => Number(booking.id) === bookingId && booking.status === "pending")) {
      const status = decision === "approved" ? "confirmed" : "rejected";
      actions.push({ type: "api", id: actionId("review_booking", bookingId), target: "review_booking",
        label: `${status === "confirmed" ? "Confirm" : "Reject"} booking #${bookingId}`,
        method: "PATCH", path: `/api/warden/bookings/${bookingId}`,
        body: { status, reason: `Confirmed through the AI assistant: ${status}.` }, requires_confirmation: true });
    }
  }
  if (user.role === "owner" && plan.intent === "attendance" && /\b(enable|turn on|activate|disable|turn off|deactivate)\b/i.test(text)) {
    const enable = !/\b(disable|turn off|deactivate)\b/i.test(text);
    const hostels = liveContext.find(item => item.type === "owner_hostels")?.records || [];
    for (const hostel of hostels.slice(0, 5)) actions.push({ type: "api", id: actionId("set_attendance_enabled", hostel.id), target: "set_attendance_enabled",
      label: `${enable ? "Enable" : "Disable"} attendance policy for ${hostel.name}`,
      method: "PUT", path: `/api/staff/hostels/${hostel.id}/attendance-policy`, body: { enabled: enable }, requires_confirmation: true });
  }
  return actions;
}

function fallbackAnswer(chunks, liveContext, language) {
  const facts = liveContext.flatMap((item) => (item.records || []).slice(0, 5).map((record) => conciseRecord(record)));
  if (language === "ur") {
    if (facts.length) return `آپ کے موجودہ اکاؤنٹ سے یہ معلومات ملیں:\n${facts.slice(0, 6).map((fact) => `• ${fact}`).join("\n")}\n\nمزید رہنمائی کے لیے نیچے دیے گئے مستند حوالہ جات دیکھیں۔`;
    if (chunks.length) return `دستیاب مستند معلومات کے مطابق:\n${chunks.slice(0, 2).map((chunk) => `• ${chunk.content}`).join("\n")}`;
    return "مجھے موجودہ معلومات میں اس سوال کا قابل اعتماد جواب نہیں ملا۔ براہ کرم انسانی مدد کے لیے چیٹ منتقل کریں۔";
  }
  if (facts.length) return `Here is the current information from your authorized account:\n${facts.slice(0, 6).map((fact) => `• ${fact}`).join("\n")}\n\n${chunks[0]?.content || "Use the suggested action to continue."}`;
  if (chunks.length) return chunks.slice(0, 2).map((chunk) => chunk.content).join("\n\n");
  return "I could not find a reliable answer in the current knowledge base. Please escalate this chat to human support.";
}

async function generateAnswer({ query, history, role, language, intent, chunks, liveContext, appContext, modelProvider, signal }) {
  try {
    const result=await reasoning.serviceCall("",{message:query,history,role,language,intent,retrieved_context:chunks,live_context:liveContext,app_context:appContext,provider:modelProvider},
      Number(process.env.AI_CHAT_TIMEOUT_MS||process.env.AI_TIMEOUT_MS||210000),signal);
    const reply=cleanText(result.reply,8000);
    if(!reply) throw new Error("empty_answer");
    const allowed=new Set(chunks.map(c=>String(c.chunk_id)));
    const used=Array.isArray(result.retrieved_chunk_ids)?result.retrieved_chunk_ids.filter(id=>allowed.has(String(id))):[];
    return {reply,provider:cleanText(result.provider,80)||"rag_service",model:cleanText(result.model,120)||null,
      unavailable:false,confidence:null,escalation:result.escalation===true||intent==="human_support",
      grounded:result.grounded===true && (used.length>0||liveContext.length>0),
      answer_kind:result.answer_kind||"grounded",uncertain:result.uncertain===true,used, intent};
  } catch(error) {
    if(signal?.aborted) throw error;
    console.warn("Chat generation unavailable:",error.name);
    const hasReference = chunks.length>0 || liveContext.length>0;
    return {reply:hasReference ? "The AI assistant is temporarily unavailable. Here is the available reference information:\n\n"+fallbackAnswer(chunks,liveContext,language)
      : "The AI assistant is temporarily unavailable. Please try again in a moment.",
      provider:"node_rag_fallback",model:null,unavailable:true,confidence:null,escalation:hasReference,
      grounded:chunks.length>0||liveContext.length>0,answer_kind:"fallback",uncertain:true,
      used:chunks.slice(0,2).map(c=>c.chunk_id),intent};
  }
}

function citation(chunk) {
  return {chunk_id:chunk.chunk_id,document_id:chunk.document_id,title:chunk.title,category:chunk.category,
    source_kind:chunk.source_kind,source_reference:chunk.source_reference,version:chunk.version,reviewed_at:chunk.reviewed_at};
}

async function replyFor(pool,user,message,history,previous,body,signal) {
  const started=Date.now(), appContext=reasoning.safeAppContext(body.app_context);
  const modelProvider=body.provider||'ollama';
  const plan=await reasoning.understand(message,history,previous,appContext,body.language,modelProvider,signal);
  if(signal?.aborted) throw signal.reason;
  let chunks=await retrieveKnowledge(pool,plan.query,user?.role||"all",4,plan.intent);
  if(plan.provider==='faq_router') chunks=chunks.filter(c=>(reasoning.CATEGORIES[plan.intent]||[]).includes(c.category));
  const liveContext=await buildLiveContext(pool,user,plan,message);
  if(signal?.aborted) throw signal.reason;
  const generated=await generateAnswer({query:message,history,role:user?.role||"guest",language:plan.language,intent:plan.intent,chunks,liveContext,appContext,modelProvider,signal});
  if(signal?.aborted) throw signal.reason;
  const hostels=liveContext.find(c=>c.type==="hostel_search");
  const samplePriceNote=hostels?.records.find(h=>h.price_note)?.price_note;
  if(samplePriceNote && /\b(price|rent|cost|monthly)\b/i.test(message)
      && generated.answer_kind==="grounded" && !generated.unavailable && !generated.uncertain
      && !/\b(sample|example|testing|confirm)\b/i.test(generated.reply)) {
    generated.reply += `\n\n${samplePriceNote}`;
  }
  const used=new Set(generated.used.map(String));
  const actions=[...actionSuggestions(plan.intent,user?.role||"guest"),...personalActionSuggestions(user,plan,message,liveContext)];
  if(generated.escalation||plan.discrepancy) {
    if(!actions.some(a=>a.type==="escalate")) actions.push({type:"escalate",target:"human_support",label:"Ask human support",requires_confirmation:true});
  }
  for(const h of hostels?.records||[]) actions.push({type:"navigate",target:"hostel_details",label:h.name,hostel_id:h.id});
  if(user?.role==="student") for(const c of liveContext.find(s=>s.type==="complaints")?.records||[]) {
    actions.push({type:"api",id:actionId("complaint_comment",c.id),target:"complaint_comment",label:"Add a comment to complaint "+c.id,method:"POST",
      path:"/api/chat/complaints/"+c.id+"/comments",required_fields:["message"],requires_confirmation:true});
  }
  const metadata={provider:generated.provider,llm_model:generated.model,service_unavailable:generated.unavailable,
    grounded:generated.grounded,answer_kind:generated.answer_kind,uncertain:generated.uncertain,confidence:null,
    intent:plan.intent,language:plan.language,citations:chunks.filter(c=>used.has(String(c.chunk_id))).map(citation),
    retrieved_sources:chunks.map(citation),live_context_types:liveContext.map(c=>c.type),actions,
    escalation_recommended:generated.escalation||plan.discrepancy,discrepancy_reported:plan.discrepancy,
    search_results:hostels?.records||[],search_filters:hostels?.filters||null,
    planner:plan.provider,duration_ms:Date.now()-started};
  return {generated,metadata,plan,chunks};
}

function registerRagChatRoutes({ app, pool, requireAuth, requireRole, isValidInt }) {
  registerChatSupportRoutes({app,pool,requireAuth,requireRole,isValidInt});
  const requireChatRole = requireRole("student", "parent", "warden", "owner");
  app.get("/api/chat/providers", async (_req, res) => {
    try {
      const url = new URL(process.env.AI_CHAT_URL || "http://127.0.0.1:8000/chat");
      url.pathname = "/health";
      const response = await fetch(url, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error("chat_service_unavailable");
      const health = await response.json();
      res.json({ default_provider: "ollama", providers: { ollama: true, groq: health.providers?.groq === true } });
    } catch (_error) {
      res.status(503).json({ error: "Chat provider status unavailable" });
    }
  });
  app.get("/chatbot-demo", (_req, res) => {
    res.sendFile(path.join(__dirname, "public", "chatbot-demo.html"));
  });
  app.get("/chatbot-demo.js", (_req, res) => {
    res.sendFile(path.join(__dirname, "public", "chatbot-demo.js"));
  });
  app.post("/api/chat/sessions", requireAuth, requireChatRole, async (req, res) => {
    const id = crypto.randomUUID();
    const title = cleanText(req.body?.title || "Hostel support", 160);
    const result = await pool.query("INSERT INTO chat_sessions (id,user_id,title) VALUES ($1,$2,$3) RETURNING *", [id, req.user.id, title]);
    res.status(201).json({ session: result.rows[0], quick_options: req.user.role === "warden"
      ? ["My assigned hostel", "Attendance and pending appeals", "Booking queue", "Complaint queue", "Fee receipt review", "Human support"]
      : req.user.role === "owner"
        ? ["My hostels", "Current resident attendance", "Fee summary", "Booking summary", "Enable attendance policy", "Human support"]
      : ["Find hostels", "Booking help", "Fee information", "Track complaint", "Attendance help", "Human support"] });
  });

  app.get("/api/chat/sessions", requireAuth, requireChatRole, async (req, res) => {
    const query = typeof req.query.q === "string" ? req.query.q.trim() : "";
    const offset = req.query.offset == null ? 0 : Number(req.query.offset);
    if (query.length > 100 || !Number.isSafeInteger(offset) || offset < 0 || offset > 10000)
      return res.status(400).json({ error: "q must be at most 100 characters and offset must be 0–10000" });
    const result = await pool.query(`SELECT s.* FROM chat_sessions s WHERE s.user_id=$1
      AND ($2::text='' OR position(lower($2) in lower(coalesce(s.title,'')))>0
        OR EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id=s.id AND position(lower($2) in lower(m.content))>0))
      ORDER BY s.updated_at DESC,s.id DESC LIMIT 51 OFFSET $3`, [req.user.id, query, offset]);
    res.json({ sessions: result.rows.slice(0, 50), has_more: result.rows.length > 50,
      next_offset: result.rows.length > 50 ? offset + 50 : null });
  });

  app.patch("/api/chat/sessions/:id", requireAuth, requireChatRole, async (req, res) => {
    if (!validSessionId(req.params.id)) return res.status(404).json({ error: "Chat session not found" });
    const title = typeof req.body?.title === "string" ? req.body.title.trim() : "";
    if (!title || title.length > 160) return res.status(400).json({ error: "A title of up to 160 characters is required" });
    const result = await pool.query("UPDATE chat_sessions SET title=$3,updated_at=NOW() WHERE id=$1 AND user_id=$2 RETURNING *",
      [req.params.id, req.user.id, title]);
    if (!result.rowCount) return res.status(404).json({ error: "Chat session not found" });
    res.json({ session: result.rows[0] });
  });

  app.delete("/api/chat/sessions/:id", requireAuth, requireChatRole, async (req, res) => {
    if (!validSessionId(req.params.id)) return res.status(404).json({ error: "Chat session not found" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const lock = await client.query("SELECT pg_try_advisory_xact_lock(hashtextextended($1,0)) AS locked", [req.params.id]);
      if (!lock.rows[0].locked) {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "A message is being processed; try deleting this chat shortly" });
      }
      const owned = await client.query("SELECT id FROM chat_sessions WHERE id=$1 AND user_id=$2 FOR UPDATE", [req.params.id, req.user.id]);
      if (!owned.rowCount) {
        await client.query("ROLLBACK");
        return res.status(404).json({ error: "Chat session not found" });
      }
      const handoff = await client.query("SELECT 1 FROM chatbot_support_escalations WHERE session_id=$1 AND status IN ('open','in_progress') LIMIT 1", [req.params.id]);
      if (handoff.rowCount) {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "Resolve or close the human support request before deleting this chat" });
      }
      // Retrieval events use ON DELETE SET NULL; remove this chat's query text as well.
      await client.query("DELETE FROM chatbot_retrieval_events WHERE session_id=$1", [req.params.id]);
      await client.query("DELETE FROM chat_sessions WHERE id=$1 AND user_id=$2", [req.params.id, req.user.id]);
      await client.query("COMMIT");
      res.status(204).end();
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally { client.release(); }
  });

  app.get("/api/chat/sessions/:id/messages", requireAuth, requireChatRole, async (req, res) => {
    if (!validSessionId(req.params.id)) return res.status(404).json({ error: "Chat session not found" });
    const before = req.query.before_id == null ? null : Number(req.query.before_id);
    if (before !== null && !isValidInt(before)) return res.status(400).json({ error: "Invalid before_id" });
    const owned = await pool.query("SELECT 1 FROM chat_sessions WHERE id=$1 AND user_id=$2", [req.params.id, req.user.id]);
    if (!owned.rowCount) return res.status(404).json({ error: "Chat session not found" });
    const result = await pool.query(`SELECT m.*,f.helpful,f.reason AS feedback_reason FROM chat_messages m JOIN chat_sessions s ON s.id=m.session_id LEFT JOIN chatbot_message_feedback f ON f.assistant_message_id=m.id WHERE s.id=$1 AND s.user_id=$2 AND ($3::bigint IS NULL OR m.id<$3) ORDER BY m.id DESC LIMIT 201`, [req.params.id, req.user.id, before]);
    const page = result.rows.slice(0,200).reverse();
    res.json({ messages: page, next_before_id: result.rows.length>200 ? page[0].id : null });
  });

  app.post("/api/chat/sessions/:id/messages", requireAuth, requireChatRole, async (req, res) => {
    const controller = new AbortController();
    res.on("close", () => { if (!res.writableEnded) controller.abort(); });
    const message=typeof req.body?.message==="string"?req.body.message.trim():"";
    if(!message||message.length>4000) return res.status(400).json({error:"A message of up to 4000 characters is required"});
    if(req.body?.provider != null && !["ollama","groq"].includes(req.body.provider)) return res.status(400).json({error:"provider must be ollama or groq"});
    if(!validSessionId(req.params.id)) return res.status(404).json({error:"Chat session not found"});
    const requestId=req.body.request_id||req.get("Idempotency-Key")||crypto.randomUUID();
    if(typeof requestId!=="string"||!requestId.length||requestId.length>120) return res.status(400).json({error:"request_id must be 1–120 characters"});
    const client=await pool.connect();
    let locked=false;
    let createdUserMessageId=null;
    let completedTurn=false;
    try {
      const session=await client.query("SELECT * FROM chat_sessions WHERE id=$1 AND user_id=$2",[req.params.id,req.user.id]);
      if(!session.rowCount) return res.status(404).json({error:"Chat session not found"});
      locked=(await client.query("SELECT pg_try_advisory_lock(hashtextextended($1,0)) AS locked",[req.params.id])).rows[0].locked;
      if(!locked) return res.status(409).json({error:"A message is already being processed; retry shortly with the same request_id"});
      const prior=await client.query("SELECT * FROM chat_messages WHERE session_id=$1 AND request_id=$2 ORDER BY id",[req.params.id,requestId]);
      const original=prior.rows.find(m=>m.role==="user"), completed=prior.rows.find(m=>m.role==="assistant");
      if(original && original.content!==message) return res.status(409).json({error:"request_id already used for another message"});
      if(completed) return res.status(200).json({user_message:original,assistant_message:completed,...completed.metadata,replayed:true,request_id:requestId});
      const historyResult=await client.query("SELECT role,content,metadata FROM chat_messages WHERE session_id=$1 ORDER BY created_at DESC,id DESC LIMIT 20",[req.params.id]);
      const history=historyResult.rows.reverse();
      const previous=history.filter(h=>h.role==="assistant").at(-1)?.metadata?.conversation_plan||{};
      const pending=await client.query("SELECT id,status FROM chatbot_support_escalations WHERE session_id=$1 AND status IN ('open','in_progress') ORDER BY id DESC LIMIT 1",[req.params.id]);
      const userMessage=original||(await client.query("INSERT INTO chat_messages(session_id,role,content,request_id) VALUES($1,'user',$2,$3) RETURNING *",[req.params.id,message,requestId])).rows[0];
      if (!original) createdUserMessageId=userMessage.id;
      if(pending.rowCount) {
        await client.query("UPDATE chat_sessions SET updated_at=NOW() WHERE id=$1",[req.params.id]);
        return res.status(202).json({user_message:userMessage,handoff_status:pending.rows[0].status,escalation_id:pending.rows[0].id,request_id:requestId});
      }
      const {generated,metadata,plan,chunks}=await replyFor(client,req.user,message,history,previous,req.body,controller.signal);
      if (controller.signal.aborted) throw controller.signal.reason;
      metadata.conversation_plan=plan;
      await client.query("BEGIN");
      const assistant=(await client.query("INSERT INTO chat_messages(session_id,role,content,metadata,request_id) VALUES($1,'assistant',$2,$3::jsonb,$4) RETURNING *",
        [req.params.id,generated.reply,JSON.stringify(metadata),requestId])).rows[0];
      await client.query(`UPDATE chat_sessions SET updated_at=NOW(),
        title=CASE WHEN title='Hostel support' AND updated_at=created_at
          AND (SELECT COUNT(*) FROM chat_messages WHERE session_id=$1 AND role='user')=1
          THEN left($2,160) ELSE title END WHERE id=$1`,[req.params.id,message]);
      await client.query(`INSERT INTO chatbot_retrieval_events(session_id,user_id,query_text,detected_intent,language,retrieved_chunk_ids,live_context_types,provider,grounded,confidence,escalation_recommended)
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,NULL,$10)`,[req.params.id,req.user.id,message,plan.intent,plan.language,chunks.map(c=>c.chunk_id),metadata.live_context_types,generated.provider,metadata.grounded,metadata.escalation_recommended]);
      if(plan.discrepancy) await client.query("INSERT INTO chatbot_discrepancies(session_id,user_id,message_id,description) VALUES($1,$2,$3,$4)",[req.params.id,req.user.id,userMessage.id,message]);
      if (controller.signal.aborted) throw controller.signal.reason;
      await client.query("COMMIT");
      completedTurn=true;
      res.status(201).json({user_message:userMessage,assistant_message:assistant,...metadata,request_id:requestId});
    } catch(error) {
      await client.query("ROLLBACK");
      if (controller.signal.aborted && !completedTurn && createdUserMessageId) {
        await client.query("DELETE FROM chat_messages WHERE id=$1 AND session_id=$2 AND role='user'",[createdUserMessageId,req.params.id]);
      }
      if (controller.signal.aborted) return;
      console.error("Chat turn failed:",error.code||error.name);
      res.status(500).json({error:"Unable to complete the message; retry with the same request_id",request_id:requestId});
    } finally {
      if(locked) await client.query("SELECT pg_advisory_unlock(hashtextextended($1,0))",[req.params.id]);
      client.release();
    }
  });

  app.post("/api/chat/messages/:id/feedback", requireAuth, requireChatRole, async (req, res) => {
    const messageId = Number(req.params.id), helpful = req.body?.helpful;
    const reason = cleanText(req.body?.reason, 1000) || null;
    if (!isValidInt(messageId) || typeof helpful !== "boolean") return res.status(400).json({ error: "A valid assistant message id and helpful boolean are required" });
    const owned = await pool.query("SELECT m.id FROM chat_messages m JOIN chat_sessions s ON s.id=m.session_id WHERE m.id=$1 AND m.role='assistant' AND s.user_id=$2", [messageId, req.user.id]);
    if (!owned.rowCount) return res.status(404).json({ error: "Assistant message not found" });
    const feedback = await pool.query("INSERT INTO chatbot_message_feedback (assistant_message_id,user_id,helpful,reason) VALUES ($1,$2,$3,$4) ON CONFLICT (assistant_message_id) DO UPDATE SET helpful=EXCLUDED.helpful,reason=EXCLUDED.reason,created_at=NOW() RETURNING *", [messageId, req.user.id, helpful, reason]);
    res.json({ feedback: feedback.rows[0] });
  });

  app.post("/api/chat/sessions/:id/escalate", requireAuth, requireChatRole, async (req, res) => {
    if (!validSessionId(req.params.id)) return res.status(404).json({ error: "Chat session not found" });
    const reason = cleanText(req.body?.reason, 1000);
    if (!reason) return res.status(400).json({ error: "An escalation reason is required" });
    const client = await pool.connect();
    try {
    await client.query("BEGIN");
    await client.query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", [req.params.id]);
    const session = await client.query("SELECT * FROM chat_sessions WHERE id=$1 AND user_id=$2", [req.params.id, req.user.id]);
    if (!session.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Chat session not found" }); }
    const existing = await client.query("SELECT * FROM chatbot_support_escalations WHERE session_id=$1 AND status IN ('open','in_progress') ORDER BY id DESC LIMIT 1", [req.params.id]);
    if (existing.rowCount) { await client.query("COMMIT"); return res.json({ escalation: existing.rows[0], replayed: true }); }
    const messages = await client.query("SELECT id,role,content,metadata,created_at FROM chat_messages WHERE session_id=$1 ORDER BY id DESC LIMIT 200", [req.params.id]);
    const escalation = await client.query("INSERT INTO chatbot_support_escalations (session_id,user_id,reason,conversation_snapshot) VALUES ($1,$2,$3,$4::jsonb) RETURNING *", [req.params.id, req.user.id, reason, JSON.stringify({ session: session.rows[0], messages: messages.rows.reverse() })]);
    await client.query("COMMIT");
    res.status(201).json({ escalation: escalation.rows[0] });
    } catch(error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
  });

  const guestBuckets = new Map();
  app.post("/api/chat/public/query", async (req, res) => {
    const controller = new AbortController();
    res.on("close", () => { if (!res.writableEnded) controller.abort(); });
    const key = req.ip, now = Date.now(), current = guestBuckets.get(key);
    const bucket = !current || current.resetAt <= now ? { count: 0, resetAt: now + 15 * 60 * 1000 } : current;
    bucket.count += 1; guestBuckets.set(key, bucket);
    if (bucket.count > 20) return res.status(429).json({ error: "Too many guest chatbot requests; try again later" });
    if(guestBuckets.size>1000) for(const [ip,entry] of guestBuckets) if(entry.resetAt<=now) guestBuckets.delete(ip);
    const message=typeof req.body?.message==="string"?req.body.message.trim():"";
    if(!message||message.length>4000) return res.status(400).json({error:"A message of up to 4000 characters is required"});
    if(req.body?.provider != null && !["ollama","groq"].includes(req.body.provider)) return res.status(400).json({error:"provider must be ollama or groq"});
    const history=reasoning.publicHistory(req.body.history);
    // Client history helps conversation only; it never supplies personal records.
    const previous=history.filter(h=>h.role==="user").reduce((p,h)=>reasoning.fallbackPlan(h.content,[],p),{});
    let generated,metadata;
    try { ({generated,metadata}=await replyFor(pool,null,message,history,previous,req.body,controller.signal)); }
    catch (error) { if (controller.signal.aborted) return; throw error; }
    res.json({reply:generated.reply,...metadata,personal_data_accessed:false});
  });

  app.get("/api/admin/chat/knowledge", requireAuth, requireRole("admin"), async (_req, res) => {
    const result = await pool.query(`SELECT d.*,COALESCE(jsonb_agg(jsonb_build_object('id',c.id,'chunk_index',c.chunk_index,'content',c.content) ORDER BY c.chunk_index) FILTER(WHERE c.id IS NOT NULL),'[]'::jsonb) AS chunks FROM chatbot_knowledge_documents d LEFT JOIN chatbot_knowledge_chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.category,d.title`);
    res.json({ documents: result.rows });
  });

  app.post("/api/admin/chat/knowledge", requireAuth, requireRole("admin"), async (req, res) => {
    const slug = cleanText(req.body?.slug, 120).toLowerCase().replace(/[^a-z0-9-]+/g, "-");
    const title = cleanText(req.body?.title, 240), category = cleanText(req.body?.category, 80);
    const audience = Array.isArray(req.body?.audience) ? [...new Set(req.body.audience.map(String))] : ["all"];
    const chunks = Array.isArray(req.body?.chunks) ? req.body.chunks.map((item) => cleanText(item, 8000)).filter((item) => item.length >= 20) : [];
    if (!slug || !title || !category || !chunks.length || chunks.length>30 || !audience.length || audience.some((role) => !["all", "student", "parent", "warden"].includes(role)) || (req.body.source_kind && !['platform_policy','hostel_policy','help_article','faq'].includes(req.body.source_kind))) return res.status(400).json({ error: "Valid slug, title, category, audience, source kind, and 1–30 chunks are required" });
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const document = await client.query(`INSERT INTO chatbot_knowledge_documents (slug,title,category,audience,source_kind,source_reference,created_by_user_id) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *`, [slug, title, category, audience, req.body.source_kind || "help_article", cleanText(req.body.source_reference, 500) || null, req.user.id]);
      for (let index = 0; index < chunks.length; index += 1) await client.query("INSERT INTO chatbot_knowledge_chunks (document_id,chunk_index,content) VALUES ($1,$2,$3)", [document.rows[0].id, index, chunks[index]]);
      await client.query("COMMIT");
      res.status(201).json({ document: document.rows[0], chunk_count: chunks.length });
    } catch (error) {
      await client.query("ROLLBACK");
      if (error.code === "23505") return res.status(409).json({ error: "Knowledge slug already exists" });
      res.status(500).json({ error: "Failed to create knowledge document" });
    } finally { client.release(); }
  });

  app.get("/api/admin/chat/escalations", requireAuth, requireRole("admin"), async (req, res) => {
    const status = cleanText(req.query.status, 20);
    const result = await pool.query("SELECT e.*,u.name AS user_name,u.email AS user_email,u.role AS user_role FROM chatbot_support_escalations e JOIN users u ON u.id=e.user_id WHERE ($1::text='' OR e.status=$1) ORDER BY e.created_at DESC LIMIT 200", [status]);
    res.json({ escalations: result.rows });
  });

  app.patch("/api/admin/chat/escalations/:id", requireAuth, requireRole("admin"), async (req, res) => {
    const id = Number(req.params.id), status = cleanText(req.body?.status, 20), note = cleanText(req.body?.resolution_note, 4000) || null;
    if (!isValidInt(id) || !["in_progress", "resolved", "closed"].includes(status)) return res.status(400).json({ error: "Valid escalation id and status are required" });
    const result = await pool.query("UPDATE chatbot_support_escalations SET status=$1::varchar,assigned_to_user_id=$2,resolution_note=COALESCE($3,resolution_note),resolved_at=CASE WHEN $1::varchar IN ('resolved','closed') THEN NOW() ELSE NULL END,updated_at=NOW() WHERE id=$4 RETURNING *", [status, req.user.id, note, id]);
    if (!result.rowCount) return res.status(404).json({ error: "Escalation not found" });
    res.json({ escalation: result.rows[0] });
  });
}

module.exports = { registerRagChatRoutes, detectIntent, detectLanguage, retrieveKnowledge, buildHostelContext, generateAnswer, personalActionSuggestions };
