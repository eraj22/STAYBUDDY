const crypto = require("node:crypto");
const path = require("node:path");
const { Pool } = require("pg");
const { createLocalDocumentStorage } = require("../document_storage");
const { hashPassword } = require("../passwords");

const DEMO_DATE = process.env.DEMO_SEED_DATE || "2026-09-13";
const DEMO_CHAT_SESSION_ID = "10000000-0000-4000-8000-000000000001";
const DEMO_ATTENDANCE_KEY = "staybuddy-demo-attendance-key";
const DEMO_ACCOUNTS = Object.freeze([
  { key: "owner", name: "Ayesha Khan", email: "ayesha.owner@staybuddy.test", password: "DemoOwner123!", phone: "0300-0000001" },
  { key: "warden", name: "Sana Malik", email: "sana.warden@staybuddy.test", password: "DemoWarden123!", phone: "0300-0000002" },
  { key: "student", name: "Hira Ali", email: "hira.student@staybuddy.test", password: "DemoStudent123!", phone: "0300-0000003" },
  { key: "parent", name: "Farah Ali", email: "farah.parent@staybuddy.test", password: "DemoParent123!", phone: "0300-0000004" },
]);

const REQUIRED_TABLES = [
  "users",
  "student_profiles",
  "notification_preferences",
  "warden_profiles",
  "hostels",
  "hostel_owners",
  "warden_assignments",
  "rooms",
  "bookings",
  "booking_stays",
  "booking_status_history",
  "booking_inventory_holds",
  "hostel_fee_schedules",
  "student_fee_assignments",
  "fee_invoices",
  "hostel_payment_destinations",
  "fee_receipt_submissions",
  "fee_receipt_audit_events",
  "fee_receipt_allocations",
  "complaints",
  "complaint_status_history",
  "hostel_attendance_settings",
  "attendance_integration_credentials",
  "resident_attendance_identifiers",
  "attendance_events",
  "attendance_corrections",
  "location_sharing_sessions",
  "location_sharing_updates",
  "location_sharing_audit_events",
  "emergency_incidents",
  "emergency_timeline_events",
  "chat_sessions",
  "chat_messages",
  "chatbot_knowledge_documents",
  "chatbot_knowledge_chunks",
  "attendance_qr_tokens",
  "attendance_qr_scan_attempts",
  "interactions",
  "favorites",
];

function assertDemoDate() {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(DEMO_DATE) || Number.isNaN(Date.parse(`${DEMO_DATE}T00:00:00Z`))) {
    throw new Error("DEMO_SEED_DATE must be a valid YYYY-MM-DD date");
  }
}

function atDemoTime(time) {
  return `${DEMO_DATE}T${time}:00+05:00`;
}

async function assertSchemaReady(client) {
  const result = await client.query(
    "SELECT name,to_regclass(name)::text AS relation FROM unnest($1::text[]) AS required(name)",
    [REQUIRED_TABLES]
  );
  const missing = result.rows.filter((row) => row.relation == null).map((row) => row.name);
  if (missing.length) {
    throw new Error(`Database is missing required backend tables: ${missing.join(", ")}. Apply migrations through 038 first.`);
  }
}

async function upsertUser(client, account, passwordHash) {
  const result = await client.query(
    `INSERT INTO users (name,email,password_hash,role,phone,account_status)
     VALUES ($1,$2,$3,$4,$5,'active')
     ON CONFLICT (email) DO UPDATE SET
       name=EXCLUDED.name,
       password_hash=EXCLUDED.password_hash,
       role=EXCLUDED.role,
       phone=EXCLUDED.phone,
       account_status='active'
     RETURNING id,name,email,role,phone,account_status`,
    [account.name, account.email, passwordHash, account.key, account.phone]
  );
  return result.rows[0];
}

async function upsertHostel(client, definition) {
  const result = await client.query(
    `INSERT INTO hostels (
       external_id,name,address,city,university,hostel_type,area,latitude,longitude,
       total_capacity,available_capacity,source_available_capacity,description,verified,
       publication_status,amenities,facilities,rules,mess_info,media,room_types_available,
       single_room_price,double_room_price,dorm_room_price,total_rooms,source
     ) VALUES (
       $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$10,$12,true,'published',
       $13::jsonb,$14::jsonb,$15::jsonb,$16::jsonb,'[]'::jsonb,$17::jsonb,$18,$19,$20,$21,'demo_seed'
     )
     ON CONFLICT (external_id) WHERE external_id IS NOT NULL DO UPDATE SET
       name=EXCLUDED.name,
       address=EXCLUDED.address,
       city=EXCLUDED.city,
       university=EXCLUDED.university,
       hostel_type=EXCLUDED.hostel_type,
       area=EXCLUDED.area,
       latitude=EXCLUDED.latitude,
       longitude=EXCLUDED.longitude,
       total_capacity=EXCLUDED.total_capacity,
       source_available_capacity=EXCLUDED.source_available_capacity,
       description=EXCLUDED.description,
       verified=true,
       publication_status='published',
       amenities=EXCLUDED.amenities,
       facilities=EXCLUDED.facilities,
       rules=EXCLUDED.rules,
       mess_info=EXCLUDED.mess_info,
       media=EXCLUDED.media,
       room_types_available=EXCLUDED.room_types_available,
       single_room_price=EXCLUDED.single_room_price,
       double_room_price=EXCLUDED.double_room_price,
       dorm_room_price=EXCLUDED.dorm_room_price,
       total_rooms=EXCLUDED.total_rooms,
       source='demo_seed'
     RETURNING id,external_id,name,available_capacity,total_capacity`,
    [
      definition.externalId,
      definition.name,
      definition.address,
      definition.city,
      definition.university,
      definition.hostelType,
      definition.area,
      definition.latitude,
      definition.longitude,
      definition.totalCapacity,
      definition.availableCapacity,
      definition.description,
      JSON.stringify(definition.amenities),
      JSON.stringify(definition.facilities),
      JSON.stringify(definition.rules),
      JSON.stringify(definition.messInfo),
      JSON.stringify(definition.roomTypes),
      definition.singleRoomPrice,
      definition.doubleRoomPrice,
      definition.dormRoomPrice,
      definition.totalRooms,
    ]
  );
  return result.rows[0];
}

async function upsertRoom(client, hostelId, definition) {
  const result = await client.query(
    `INSERT INTO rooms (hostel_id,room_number,capacity,available_capacity,room_type,monthly_rent)
     VALUES ($1,$2,$3,$4,$5,$6)
     ON CONFLICT (hostel_id,room_number) DO UPDATE SET
       room_type=EXCLUDED.room_type,
       monthly_rent=EXCLUDED.monthly_rent
     RETURNING id,hostel_id,room_number,capacity,available_capacity,room_type,monthly_rent`,
    [hostelId, definition.number, definition.capacity, definition.availableCapacity, definition.type, definition.rent]
  );
  return result.rows[0];
}

async function ensureBooking(client, definition) {
  const existing = await client.query(
    "SELECT * FROM bookings WHERE user_id=$1 AND idempotency_key=$2",
    [definition.studentId, definition.idempotencyKey]
  );
  if (existing.rowCount) return { ...existing.rows[0], inserted: false };

  const result = await client.query(
    `INSERT INTO bookings (
       user_id,hostel_id,room_id,check_in,check_out,status,idempotency_key,expires_at,
       requested_room_type,bed_number,price_snapshot,currency,updated_at
     ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'PKR',NOW()) RETURNING *`,
    [
      definition.studentId,
      definition.hostelId,
      definition.roomId,
      definition.checkIn,
      definition.checkOut,
      definition.status,
      definition.idempotencyKey,
      definition.expiresAt,
      definition.requestedRoomType,
      definition.bedNumber,
      definition.priceSnapshot,
    ]
  );
  return { ...result.rows[0], inserted: true };
}

async function ensureFeeSchedule(client, definition) {
  const existing = await client.query(
    "SELECT id FROM hostel_fee_schedules WHERE hostel_id=$1 AND title=$2 ORDER BY id LIMIT 1",
    [definition.hostelId, definition.title]
  );
  if (existing.rowCount) {
    const updated = await client.query(
      `UPDATE hostel_fee_schedules SET amount=$1,currency='PKR',due_day=$2,active=true,
       payment_option=$3,minimum_installment_amount=$4,created_by_user_id=$5,updated_at=NOW()
       WHERE id=$6 RETURNING *`,
      [definition.amount, definition.dueDay, definition.paymentOption, definition.minimumInstallment, definition.ownerId, existing.rows[0].id]
    );
    return updated.rows[0];
  }
  const inserted = await client.query(
    `INSERT INTO hostel_fee_schedules (
       hostel_id,title,amount,currency,due_day,active,payment_option,minimum_installment_amount,created_by_user_id
     ) VALUES ($1,$2,$3,'PKR',$4,true,$5,$6,$7) RETURNING *`,
    [definition.hostelId, definition.title, definition.amount, definition.dueDay, definition.paymentOption, definition.minimumInstallment, definition.ownerId]
  );
  return inserted.rows[0];
}

async function ensurePaymentDestination(client, definition) {
  const existing = await client.query(
    "SELECT id FROM hostel_payment_destinations WHERE hostel_id=$1 AND account_reference=$2 ORDER BY id LIMIT 1",
    [definition.hostelId, definition.reference]
  );
  if (existing.rowCount) {
    const updated = await client.query(
      `UPDATE hostel_payment_destinations SET account_holder_name=$1,institution_name=$2,instructions=$3,
       status='approved',submitted_by_user_id=$4,approved_by_user_id=$5,approved_at=COALESCE(approved_at,NOW()),revoked_at=NULL
       WHERE id=$6 RETURNING *`,
      [definition.holder, definition.institution, definition.instructions, definition.wardenId, definition.ownerId, existing.rows[0].id]
    );
    return updated.rows[0];
  }
  const inserted = await client.query(
    `INSERT INTO hostel_payment_destinations (
       hostel_id,account_holder_name,institution_name,account_reference,instructions,status,
       submitted_by_user_id,approved_by_user_id,approved_at
     ) VALUES ($1,$2,$3,$4,$5,'approved',$6,$7,NOW()) RETURNING *`,
    [definition.hostelId, definition.holder, definition.institution, definition.reference, definition.instructions, definition.wardenId, definition.ownerId]
  );
  return inserted.rows[0];
}

async function ensureStoredFile(documentStorage, key, bytes, createdStorageKeys) {
  if (await documentStorage.exists(key)) return;
  try {
    await documentStorage.save(key, bytes);
    createdStorageKeys.push(key);
  } catch (error) {
    if (!(await documentStorage.exists(key))) throw error;
  }
}

async function upsertReceipt(client, definition) {
  const result = await client.query(
    `INSERT INTO fee_receipt_submissions (
       invoice_id,hostel_id,student_user_id,submitted_by_user_id,payment_destination_id,amount,currency,
       payment_reference,paid_at,storage_provider,storage_key,checksum,original_filename,content_type,
       byte_size,status,reviewed_by_user_id,reviewed_at,review_note
     ) VALUES ($1,$2,$3,$3,$4,$5,'PKR',$6,$7,'local',$8,$9,$10,$11,$12,$13,$14,$15,$16)
     ON CONFLICT (storage_key) DO UPDATE SET
       invoice_id=EXCLUDED.invoice_id,
       hostel_id=EXCLUDED.hostel_id,
       student_user_id=EXCLUDED.student_user_id,
       submitted_by_user_id=EXCLUDED.submitted_by_user_id,
       payment_destination_id=EXCLUDED.payment_destination_id,
       amount=EXCLUDED.amount,
       payment_reference=EXCLUDED.payment_reference,
       paid_at=EXCLUDED.paid_at,
       checksum=EXCLUDED.checksum,
       original_filename=EXCLUDED.original_filename,
       content_type=EXCLUDED.content_type,
       byte_size=EXCLUDED.byte_size,
       status=EXCLUDED.status,
       reviewed_by_user_id=EXCLUDED.reviewed_by_user_id,
       reviewed_at=EXCLUDED.reviewed_at,
       review_note=EXCLUDED.review_note
     RETURNING *`,
    [
      definition.invoiceId,
      definition.hostelId,
      definition.studentId,
      definition.destinationId,
      definition.amount,
      definition.reference,
      definition.paidAt,
      definition.storageKey,
      definition.checksum,
      definition.filename,
      definition.contentType,
      definition.byteSize,
      definition.status,
      definition.reviewerId,
      definition.reviewedAt,
      definition.reviewNote,
    ]
  );
  return result.rows[0];
}

async function ensureComplaint(client, definition) {
  const existing = await client.query(
    "SELECT * FROM complaints WHERE user_id=$1 AND idempotency_key=$2",
    [definition.studentId, definition.idempotencyKey]
  );
  if (existing.rowCount) {
    const updated = await client.query(
      `UPDATE complaints SET category=$1,severity=$2,priority=$3,description=$4,
       triage_source='fallback',triage_confidence=NULL,triage_model_version=NULL
       WHERE id=$5 RETURNING *`,
      [definition.category, definition.severity, definition.priority, definition.description, existing.rows[0].id]
    );
    return updated.rows[0];
  }
  const inserted = await client.query(
    `INSERT INTO complaints (
       user_id,hostel_id,category,severity,status,assigned_to,description,idempotency_key,
       priority,sla_due_at,triage_source,triage_confidence,triage_model_version,created_at
     ) VALUES ($1,$2,$3,$4,'open',NULL,$5,$6,$7,$8,'fallback',NULL,NULL,$9) RETURNING *`,
    [
      definition.studentId,
      definition.hostelId,
      definition.category,
      definition.severity,
      definition.description,
      definition.idempotencyKey,
      definition.priority,
      definition.slaDueAt,
      definition.createdAt,
    ]
  );
  return inserted.rows[0];
}

async function ensureAttendanceEvent(client, definition) {
  const existing = await client.query(
    "SELECT * FROM attendance_events WHERE hostel_id=$1 AND source=$2 AND idempotency_key=$3",
    [definition.hostelId, definition.source, definition.idempotencyKey]
  );
  if (existing.rowCount) return existing.rows[0];
  const inserted = await client.query(
    `INSERT INTO attendance_events (
       hostel_id,student_user_id,direction,occurred_at,source,entry_point,external_event_id,
       idempotency_key,metadata,created_by_user_id
     ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10) RETURNING *`,
    [
      definition.hostelId,
      definition.studentId,
      definition.direction,
      definition.occurredAt,
      definition.source,
      definition.entryPoint,
      definition.externalEventId,
      definition.idempotencyKey,
      JSON.stringify({ demo_ref: definition.demoRef }),
      definition.createdBy,
    ]
  );
  return inserted.rows[0];
}

async function buildSummary(client) {
  const result = await client.query(
    `SELECT
       (SELECT COUNT(*)::int FROM users WHERE email=ANY($1::text[])) AS users,
       (SELECT COUNT(*)::int FROM hostels WHERE external_id IN ('DEMO-HOSTEL-01','DEMO-HOSTEL-02')) AS hostels,
       (SELECT COUNT(*)::int FROM rooms r JOIN hostels h ON h.id=r.hostel_id WHERE h.external_id IN ('DEMO-HOSTEL-01','DEMO-HOSTEL-02')) AS rooms,
       (SELECT COUNT(*)::int FROM bookings WHERE idempotency_key IN ('demo-booking-001','demo-booking-002')) AS bookings,
       (SELECT COUNT(*)::int FROM hostel_fee_schedules s JOIN hostels h ON h.id=s.hostel_id WHERE h.external_id IN ('DEMO-HOSTEL-01','DEMO-HOSTEL-02')) AS fee_schedules,
       (SELECT COUNT(*)::int FROM fee_invoices i JOIN student_fee_assignments a ON a.id=i.assignment_id JOIN users u ON u.id=a.student_user_id WHERE u.email='hira.student@staybuddy.test') AS invoices,
       (SELECT COUNT(*)::int FROM fee_receipt_submissions WHERE payment_reference IN ('DEMO-BANK-01','DEMO-BANK-02')) AS receipts,
       (SELECT COUNT(*)::int FROM complaints WHERE idempotency_key IN ('demo-complaint-001','demo-complaint-002')) AS complaints,
       (SELECT COUNT(*)::int FROM attendance_events WHERE idempotency_key IN ('demo-attendance-001','demo-attendance-002')) AS attendance_events,
       (SELECT COUNT(*)::int FROM emergency_incidents WHERE idempotency_key='demo-emergency-001') AS emergencies,
       (SELECT COUNT(*)::int FROM chat_sessions WHERE id=$2) AS chat_sessions,
       (SELECT COUNT(*)::int FROM chat_messages WHERE session_id=$2) AS chat_messages`,
    [DEMO_ACCOUNTS.map((account) => account.email), DEMO_CHAT_SESSION_ID]
  );
  return result.rows[0];
}

async function seedDemoData(pool, options = {}) {
  assertDemoDate();
  const documentStorage = options.documentStorage || createLocalDocumentStorage();
  const createdStorageKeys = [];
  const passwordHashes = Object.fromEntries(
    await Promise.all(DEMO_ACCOUNTS.map(async (account) => [account.key, await hashPassword(account.password)]))
  );
  const client = await pool.connect();
  let committed = false;

  try {
    await client.query("BEGIN");
    await client.query("SET LOCAL search_path TO public");
    await assertSchemaReady(client);

    const users = {};
    for (const account of DEMO_ACCOUNTS) {
      users[account.key] = await upsertUser(client, account, passwordHashes[account.key]);
      await client.query(
        "INSERT INTO notification_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
        [users[account.key].id]
      );
    }

    await client.query(
      `INSERT INTO student_profiles (
         user_id,university,department,gender,budget_max,max_distance_km,study_preference,
         food_preference,roommate_profile_sharing,updated_at
       ) VALUES ($1,'FAST-NUCES','Computer Science','female',20000,8,0.8,'halal',true,NOW())
       ON CONFLICT (user_id) DO UPDATE SET
         university=EXCLUDED.university,
         department=EXCLUDED.department,
         gender=EXCLUDED.gender,
         budget_max=EXCLUDED.budget_max,
         max_distance_km=EXCLUDED.max_distance_km,
         study_preference=EXCLUDED.study_preference,
         food_preference=EXCLUDED.food_preference,
         roommate_profile_sharing=true,
         updated_at=NOW()`,
      [users.student.id]
    );
    await client.query(
      `INSERT INTO parent_student_links (
         parent_user_id,student_user_id,invited_by_student_id,
         invitation_token_hash,status,expires_at,accepted_at
       ) VALUES ($1,$2,$2,$3,'active',NOW() + INTERVAL '365 days',NOW())
       ON CONFLICT (parent_user_id,student_user_id) WHERE status IN ('pending','active')
       DO UPDATE SET status='active',accepted_at=NOW(),revoked_at=NULL,
         expires_at=EXCLUDED.expires_at
       RETURNING id`,
      [
        users.parent.id,
        users.student.id,
        crypto.createHash("sha256").update("staybuddy-demo-parent-link-hira").digest("hex"),
      ]
    );
    await client.query(
      `INSERT INTO warden_profiles (
         warden_user_id,cnic,city,shift,assigned_floors,permissions,account_status
       ) VALUES ($1,'DEMO-CNIC-002','Islamabad','Morning','["Ground","First"]'::jsonb,'{"attendance":true,"complaints":true,"fees":true}'::jsonb,'Active')
       ON CONFLICT (warden_user_id) DO UPDATE SET
         cnic=EXCLUDED.cnic,
         city=EXCLUDED.city,
         shift=EXCLUDED.shift,
         assigned_floors=EXCLUDED.assigned_floors,
         permissions=EXCLUDED.permissions,
         account_status='Active',
         updated_at=NOW()`,
      [users.warden.id]
    );

    const hostelOne = await upsertHostel(client, {
      externalId: "DEMO-HOSTEL-01",
      name: "Green View Girls Hostel",
      address: "House 10, Street 5, G-10",
      city: "Islamabad",
      university: "FAST-NUCES",
      hostelType: "Girls",
      area: "G-10",
      latitude: 33.676,
      longitude: 73.0115,
      totalCapacity: 3,
      availableCapacity: 2,
      description: "Small fictional hostel near campus for the local demonstration.",
      amenities: ["wifi", "laundry", "security guard"],
      facilities: ["study room", "generator"],
      rules: ["No smoking", "Quiet hours after 10 PM"],
      messInfo: { available: true, meals: ["breakfast", "dinner"] },
      roomTypes: ["single", "shared"],
      singleRoomPrice: 18000,
      doubleRoomPrice: 12000,
      dormRoomPrice: null,
      totalRooms: 2,
    });
    const hostelTwo = await upsertHostel(client, {
      externalId: "DEMO-HOSTEL-02",
      name: "Campus Comfort Residence",
      address: "House 22, Street 2, H-13",
      city: "Islamabad",
      university: "NUST",
      hostelType: "Girls",
      area: "H-13",
      latitude: 33.6508,
      longitude: 72.995,
      totalCapacity: 2,
      availableCapacity: 1,
      description: "Second fictional hostel for search, comparison and availability tests.",
      amenities: ["wifi", "security guard", "hot water"],
      facilities: ["common room", "generator"],
      rules: ["Visitors must register at reception"],
      messInfo: { available: true, optional: true },
      roomTypes: ["double"],
      singleRoomPrice: null,
      doubleRoomPrice: 15000,
      dormRoomPrice: null,
      totalRooms: 1,
    });

    for (const hostel of [hostelOne, hostelTwo]) {
      await client.query(
        `INSERT INTO hostel_owners (owner_user_id,hostel_id) VALUES ($1,$2)
         ON CONFLICT (hostel_id) DO UPDATE SET owner_user_id=EXCLUDED.owner_user_id`,
        [users.owner.id, hostel.id]
      );
    }
    await client.query("DELETE FROM warden_assignments WHERE warden_user_id=$1 AND hostel_id=$2", [users.warden.id, hostelTwo.id]);
    await client.query(
      `INSERT INTO warden_assignments (warden_user_id,hostel_id,assigned_by_user_id)
       VALUES ($1,$2,$3)
       ON CONFLICT (warden_user_id,hostel_id) DO UPDATE SET assigned_by_user_id=EXCLUDED.assigned_by_user_id`,
      [users.warden.id, hostelOne.id, users.owner.id]
    );

    const roomOne = await upsertRoom(client, hostelOne.id, { number: "A-101", type: "single", capacity: 1, availableCapacity: 1, rent: 18000 });
    const roomTwo = await upsertRoom(client, hostelOne.id, { number: "A-102", type: "shared", capacity: 2, availableCapacity: 1, rent: 12000 });
    const roomThree = await upsertRoom(client, hostelTwo.id, { number: "B-201", type: "double", capacity: 2, availableCapacity: 1, rent: 15000 });

    const bookingOne = await ensureBooking(client, {
      studentId: users.student.id,
      hostelId: hostelOne.id,
      roomId: roomTwo.id,
      checkIn: "2026-09-01",
      checkOut: "2027-03-31",
      status: "confirmed",
      idempotencyKey: "demo-booking-001",
      expiresAt: null,
      requestedRoomType: "shared",
      bedNumber: "BED-1",
      priceSnapshot: 12000,
    });
    const bookingTwo = await ensureBooking(client, {
      studentId: users.student.id,
      hostelId: hostelTwo.id,
      roomId: null,
      checkIn: "2027-04-01",
      checkOut: "2027-09-30",
      status: "pending",
      idempotencyKey: "demo-booking-002",
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
      requestedRoomType: "double",
      bedNumber: null,
      priceSnapshot: 15000,
    });

    if (bookingOne.status === "confirmed") {
      await client.query(
        `INSERT INTO booking_stays (booking_id,checked_in_at,checked_in_by_user_id)
         VALUES ($1,$2,$3)
         ON CONFLICT (booking_id) DO UPDATE SET
           checked_in_at=COALESCE(booking_stays.checked_in_at,EXCLUDED.checked_in_at),
           checked_in_by_user_id=COALESCE(booking_stays.checked_in_by_user_id,EXCLUDED.checked_in_by_user_id)`,
        [bookingOne.id, "2026-09-01T14:00:00+05:00", users.warden.id]
      );
      await client.query(
        `INSERT INTO room_assignment_history (booking_id,previous_room_id,next_room_id,assigned_by_user_id,assigned_at)
         SELECT $1,NULL,$2,$3,$4
         WHERE NOT EXISTS (SELECT 1 FROM room_assignment_history WHERE booking_id=$1 AND next_room_id=$2)`,
        [bookingOne.id, roomTwo.id, users.warden.id, "2026-09-01T13:55:00+05:00"]
      );
    }
    if (bookingTwo.status === "pending") {
      await client.query(
        `INSERT INTO booking_inventory_holds (booking_id,room_id,expires_at)
         VALUES ($1,$2,$3)
         ON CONFLICT (booking_id) DO NOTHING`,
        [bookingTwo.id, roomThree.id, bookingTwo.expires_at]
      );
    }

    const bookingHistories = [
      [bookingOne.id, null, "pending", users.student.id, "demo seed request"],
      [bookingOne.id, "pending", "confirmed", users.warden.id, "demo seed confirmation"],
      [bookingTwo.id, null, "pending", users.student.id, "demo seed request"],
    ];
    for (const history of bookingHistories) {
      await client.query(
        `INSERT INTO booking_status_history (booking_id,previous_status,next_status,actor_user_id,reason)
         SELECT $1,$2,$3,$4,$5
         WHERE NOT EXISTS (SELECT 1 FROM booking_status_history WHERE booking_id=$1 AND reason=$5)`,
        history
      );
    }

    await client.query(
      "INSERT INTO favorites (user_id,hostel_id) VALUES ($1,$2) ON CONFLICT (user_id,hostel_id) DO NOTHING",
      [users.student.id, hostelOne.id]
    );
    const interactions = [
      [hostelOne.id, "view", "DEMO_INTERACTION_01"],
      [hostelOne.id, "favorite", "DEMO_INTERACTION_02"],
      [hostelTwo.id, "search", "DEMO_INTERACTION_03"],
    ];
    for (const [hostelId, actionType, demoRef] of interactions) {
      await client.query(
        `INSERT INTO interactions (user_id,hostel_id,action_type,metadata)
         SELECT $1,$2,$3,$4::jsonb
         WHERE NOT EXISTS (
           SELECT 1 FROM interactions WHERE user_id=$1 AND metadata->>'demo_ref'=$5
         )`,
        [users.student.id, hostelId, actionType, JSON.stringify({ demo_ref: demoRef }), demoRef]
      );
    }

    const feeOne = await ensureFeeSchedule(client, {
      hostelId: hostelOne.id,
      title: "Monthly Hostel Fee",
      amount: 12000,
      dueDay: 5,
      paymentOption: "both",
      minimumInstallment: 6000,
      ownerId: users.owner.id,
    });
    await ensureFeeSchedule(client, {
      hostelId: hostelTwo.id,
      title: "Monthly Hostel Fee",
      amount: 15000,
      dueDay: 5,
      paymentOption: "full_only",
      minimumInstallment: null,
      ownerId: users.owner.id,
    });
    const feeAssignment = await client.query(
      `INSERT INTO student_fee_assignments (
         schedule_id,booking_id,student_user_id,starts_on,ends_on,active,assigned_by_user_id
       ) VALUES ($1,$2,$3,'2026-09-01','2027-03-31',true,$4)
       ON CONFLICT (schedule_id,booking_id) DO UPDATE SET
         student_user_id=EXCLUDED.student_user_id,
         starts_on=EXCLUDED.starts_on,
         ends_on=EXCLUDED.ends_on,
         active=true,
         assigned_by_user_id=EXCLUDED.assigned_by_user_id
       RETURNING *`,
      [feeOne.id, bookingOne.id, users.student.id, users.owner.id]
    );
    const invoiceOne = await client.query(
      `INSERT INTO fee_invoices (assignment_id,billing_month,due_date,amount,currency,status)
       VALUES ($1,'2026-09-01','2026-09-05',12000,'PKR','overdue')
       ON CONFLICT (assignment_id,billing_month) DO UPDATE SET
         due_date=EXCLUDED.due_date,
         amount=EXCLUDED.amount,
         currency='PKR',
         updated_at=NOW()
       RETURNING *`,
      [feeAssignment.rows[0].id]
    );
    const invoiceTwo = await client.query(
      `INSERT INTO fee_invoices (assignment_id,billing_month,due_date,amount,currency,status)
       VALUES ($1,'2026-10-01','2026-10-05',12000,'PKR','pending')
       ON CONFLICT (assignment_id,billing_month) DO UPDATE SET
         due_date=EXCLUDED.due_date,
         amount=EXCLUDED.amount,
         currency='PKR',
         updated_at=NOW()
       RETURNING *`,
      [feeAssignment.rows[0].id]
    );
    const paymentDestination = await ensurePaymentDestination(client, {
      hostelId: hostelOne.id,
      holder: "Green View Demo Account",
      institution: "StayBuddy Demo Bank",
      reference: "DEMO-BANK-ACCOUNT-01",
      instructions: "Fictional PKR destination for local testing only.",
      wardenId: users.warden.id,
      ownerId: users.owner.id,
    });

    const pngBytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64");
    const pdfBytes = Buffer.from("%PDF-1.4\n% StayBuddy fictional demo receipt\n1 0 obj\n<<>>\nendobj\n%%EOF\n");
    const receiptOneKey = `fee_receipt/${invoiceOne.rows[0].id}/00000000-0000-4000-8000-000000000001`;
    const receiptTwoKey = `fee_receipt/${invoiceTwo.rows[0].id}/00000000-0000-4000-8000-000000000002`;
    await ensureStoredFile(documentStorage, receiptOneKey, pngBytes, createdStorageKeys);
    await ensureStoredFile(documentStorage, receiptTwoKey, pdfBytes, createdStorageKeys);

    const receiptOne = await upsertReceipt(client, {
      invoiceId: invoiceOne.rows[0].id,
      hostelId: hostelOne.id,
      studentId: users.student.id,
      destinationId: paymentDestination.id,
      amount: 6000,
      reference: "DEMO-BANK-01",
      paidAt: atDemoTime("11:00"),
      storageKey: receiptOneKey,
      checksum: crypto.createHash("sha256").update(pngBytes).digest("hex"),
      filename: "receipt-001.png",
      contentType: "image/png",
      byteSize: pngBytes.length,
      status: "verified",
      reviewerId: users.warden.id,
      reviewedAt: atDemoTime("12:00"),
      reviewNote: "First installment verified.",
    });
    const receiptTwo = await upsertReceipt(client, {
      invoiceId: invoiceTwo.rows[0].id,
      hostelId: hostelOne.id,
      studentId: users.student.id,
      destinationId: paymentDestination.id,
      amount: 6000,
      reference: "DEMO-BANK-02",
      paidAt: "2026-10-04T11:00:00+05:00",
      storageKey: receiptTwoKey,
      checksum: crypto.createHash("sha256").update(pdfBytes).digest("hex"),
      filename: "receipt-002.pdf",
      contentType: "application/pdf",
      byteSize: pdfBytes.length,
      status: "submitted",
      reviewerId: null,
      reviewedAt: null,
      reviewNote: null,
    });
    for (const [receipt, eventType, actorId, details] of [
      [receiptOne, "submitted", users.student.id, { demo: true }],
      [receiptOne, "verified", users.warden.id, { review_note: "First installment verified.", allocation_created: true }],
      [receiptTwo, "submitted", users.student.id, { demo: true }],
    ]) {
      await client.query(
        `INSERT INTO fee_receipt_audit_events (receipt_submission_id,event_type,actor_user_id,details)
         SELECT $1::bigint,$2::varchar,$3::int,$4::jsonb
         WHERE NOT EXISTS (
           SELECT 1 FROM fee_receipt_audit_events WHERE receipt_submission_id=$1::bigint AND event_type=$2::varchar
         )`,
        [receipt.id, eventType, actorId, JSON.stringify(details)]
      );
    }
    await client.query(
      `INSERT INTO fee_receipt_allocations (receipt_submission_id,invoice_id,amount,allocated_by_user_id,allocated_at)
       VALUES ($1,$2,6000,$3,$4)
       ON CONFLICT (receipt_submission_id) DO UPDATE SET
         invoice_id=EXCLUDED.invoice_id,
         amount=EXCLUDED.amount,
         allocated_by_user_id=EXCLUDED.allocated_by_user_id`,
      [receiptOne.id, invoiceOne.rows[0].id, users.warden.id, atDemoTime("12:00")]
    );

    const complaintOne = await ensureComplaint(client, {
      studentId: users.student.id,
      hostelId: hostelOne.id,
      category: "maintenance",
      severity: "high",
      priority: "high",
      description: "Water is leaking from the bathroom tap in room A-102.",
      idempotencyKey: "demo-complaint-001",
      createdAt: atDemoTime("09:30"),
      slaDueAt: "2026-09-14T09:30:00+05:00",
    });
    const complaintTwo = await ensureComplaint(client, {
      studentId: users.student.id,
      hostelId: hostelOne.id,
      category: "cleanliness",
      severity: "medium",
      priority: "medium",
      description: "The shared common room needs cleaning.",
      idempotencyKey: "demo-complaint-002",
      createdAt: atDemoTime("10:30"),
      slaDueAt: "2026-09-15T10:30:00+05:00",
    });
    for (const complaint of [complaintOne, complaintTwo]) {
      await client.query(
        `INSERT INTO complaint_status_history (
           complaint_id,previous_status,next_status,changed_by_user_id,changed_at
         ) SELECT $1,NULL,'open',$2,$3
         WHERE NOT EXISTS (
           SELECT 1 FROM complaint_status_history WHERE complaint_id=$1 AND previous_status IS NULL AND next_status='open'
         )`,
        [complaint.id, users.student.id, complaint.created_at]
      );
    }

    await client.query(
      `INSERT INTO hostel_attendance_settings (hostel_id,curfew_time,timezone,enabled,updated_by_user_id)
       VALUES ($1,'22:00','Asia/Karachi',true,$2)
       ON CONFLICT (hostel_id) DO UPDATE SET
         curfew_time='22:00',timezone='Asia/Karachi',enabled=true,updated_by_user_id=EXCLUDED.updated_by_user_id,updated_at=NOW()`,
      [hostelOne.id, users.owner.id]
    );
    await client.query(
      `INSERT INTO attendance_integration_credentials (
         hostel_id,label,credential_hash,active,created_by_user_id
       ) VALUES ($1,'Demo scanner', $2,true,$3)
       ON CONFLICT (credential_hash) DO UPDATE SET
         hostel_id=EXCLUDED.hostel_id,
         label=EXCLUDED.label,
         active=true,
         revoked_at=NULL,
         created_by_user_id=EXCLUDED.created_by_user_id`,
      [hostelOne.id, crypto.createHash("sha256").update(DEMO_ATTENDANCE_KEY).digest("hex"), users.owner.id]
    );
    await client.query(
      `INSERT INTO resident_attendance_identifiers (
         hostel_id,student_user_id,external_identifier,active,enrolled_by_user_id
       ) VALUES ($1,$2,'RFID-DEMO-01',true,$3)
       ON CONFLICT (hostel_id,external_identifier) DO UPDATE SET
         student_user_id=EXCLUDED.student_user_id,
         active=true,
         revoked_at=NULL,
         enrolled_by_user_id=EXCLUDED.enrolled_by_user_id,
         enrolled_at=NOW()`,
      [hostelOne.id, users.student.id, users.warden.id]
    );
    await ensureAttendanceEvent(client, {
      hostelId: hostelOne.id,
      studentId: users.student.id,
      direction: "out",
      occurredAt: atDemoTime("08:05"),
      source: "manual",
      entryPoint: "Front desk",
      externalEventId: null,
      idempotencyKey: "demo-attendance-001",
      demoRef: "EVENT_01",
      createdBy: users.warden.id,
    });
    const attendanceIn = await ensureAttendanceEvent(client, {
      hostelId: hostelOne.id,
      studentId: users.student.id,
      direction: "in",
      occurredAt: atDemoTime("17:00"),
      source: "rfid",
      entryPoint: "Main gate simulator",
      externalEventId: "DEMO-RFID-EVENT-001",
      idempotencyKey: "demo-attendance-002",
      demoRef: "EVENT_02",
      createdBy: null,
    });
    await client.query(
      `INSERT INTO attendance_corrections (
         attendance_event_id,corrected_direction,corrected_occurred_at,reason,corrected_by_user_id
       ) SELECT $1,'in',$2,'Corrected the recorded return time.',$3
       WHERE NOT EXISTS (
         SELECT 1 FROM attendance_corrections
         WHERE attendance_event_id=$1 AND reason='Corrected the recorded return time.'
       )`,
      [attendanceIn.id, atDemoTime("16:45"), users.warden.id]
    );

    let locationSession = await client.query(
      "SELECT * FROM location_sharing_sessions WHERE student_user_id=$1 AND revoked_at IS NULL ORDER BY id DESC LIMIT 1 FOR UPDATE",
      [users.student.id]
    );
    if (!locationSession.rowCount) {
      locationSession = await client.query(
        "INSERT INTO location_sharing_sessions (student_user_id,started_at,expires_at) VALUES ($1,NOW(),NOW()+INTERVAL '60 minutes') RETURNING *",
        [users.student.id]
      );
    } else {
      locationSession = await client.query(
        "UPDATE location_sharing_sessions SET expires_at=GREATEST(expires_at,NOW()+INTERVAL '60 minutes') WHERE id=$1 RETURNING *",
        [locationSession.rows[0].id]
      );
    }
    await client.query(
      `INSERT INTO location_sharing_updates (
         session_id,latitude,longitude,accuracy_m,client_event_id,reported_at
       ) VALUES ($1,33.6761,73.0116,15,'demo-location-update-001',NOW())
       ON CONFLICT (session_id,client_event_id) DO UPDATE SET
         latitude=EXCLUDED.latitude,
         longitude=EXCLUDED.longitude,
         accuracy_m=EXCLUDED.accuracy_m,
         reported_at=NOW()`,
      [locationSession.rows[0].id]
    );
    for (const [eventType, demoRef] of [["consent_granted", "LOC_SESSION_01"], ["location_updated", "LOC_UPDATE_01"]]) {
      await client.query(
        `INSERT INTO location_sharing_audit_events (session_id,event_type,actor_user_id,details)
         SELECT $1,$2,$3,$4::jsonb
         WHERE NOT EXISTS (
           SELECT 1 FROM location_sharing_audit_events WHERE session_id=$1 AND details->>'demo_ref'=$5
         )`,
        [locationSession.rows[0].id, eventType, users.student.id, JSON.stringify({ demo_ref: demoRef }), demoRef]
      );
    }

    const emergency = await client.query(
      `INSERT INTO emergency_incidents (
         student_user_id,hostel_id,booking_id,room_id,alert_type,description,status,
         latitude,longitude,location_accuracy_m,idempotency_key
       ) VALUES ($1,$2,$3,$4,'medical','Fictional medical assistance request for testing.','active',33.6761,73.0116,15,'demo-emergency-001')
       ON CONFLICT (idempotency_key) DO UPDATE SET description=EXCLUDED.description
       RETURNING *`,
      [users.student.id, hostelOne.id, bookingOne.id, roomTwo.id]
    );
    await client.query(
      `INSERT INTO emergency_timeline_events (incident_id,event_type,actor_user_id,details)
       SELECT $1,'created',$2,$3::jsonb
       WHERE NOT EXISTS (
         SELECT 1 FROM emergency_timeline_events WHERE incident_id=$1 AND details->>'demo_ref'='EMERGENCY_01'
       )`,
      [emergency.rows[0].id, users.student.id, JSON.stringify({ demo_ref: "EMERGENCY_01", simulated: true })]
    );

    await client.query(
      `INSERT INTO chat_sessions (id,user_id,title)
       VALUES ($1,$2,'Demo hostel support')
       ON CONFLICT (id) DO UPDATE SET user_id=EXCLUDED.user_id,title=EXCLUDED.title,updated_at=NOW()`,
      [DEMO_CHAT_SESSION_ID, users.student.id]
    );
    const chatMessages = [
      ["user", "What documents are required to book a hostel?", { demo_ref: "DEMO_02" }],
      ["assistant", "For this demonstration, use the hostel details and booking flow. Contact hostel staff when a requirement is not listed.", { demo_ref: "DEMO_02_REPLY", provider: "seeded_demo" }],
    ];
    for (const [role, content, metadata] of chatMessages) {
      await client.query(
        `INSERT INTO chat_messages (session_id,role,content,metadata)
         SELECT $1,$2,$3,$4::jsonb
         WHERE NOT EXISTS (
           SELECT 1 FROM chat_messages WHERE session_id=$1 AND metadata->>'demo_ref'=$5
         )`,
        [DEMO_CHAT_SESSION_ID, role, content, JSON.stringify(metadata), metadata.demo_ref]
      );
    }

    const summary = await buildSummary(client);
    await client.query("COMMIT");
    committed = true;
    return {
      summary,
      accounts: DEMO_ACCOUNTS,
      ids: {
        ownerUserId: users.owner.id,
        wardenUserId: users.warden.id,
        studentUserId: users.student.id,
        parentUserId: users.parent.id,
        hostelOneId: hostelOne.id,
        hostelTwoId: hostelTwo.id,
        roomOneId: roomOne.id,
        roomTwoId: roomTwo.id,
        roomThreeId: roomThree.id,
        bookingOneId: bookingOne.id,
        bookingTwoId: bookingTwo.id,
        invoiceOneId: invoiceOne.rows[0].id,
        invoiceTwoId: invoiceTwo.rows[0].id,
        receiptOneId: receiptOne.id,
        receiptTwoId: receiptTwo.id,
        locationSessionId: locationSession.rows[0].id,
      },
    };
  } catch (error) {
    if (!committed) {
      try { await client.query("ROLLBACK"); } catch (_) {}
      for (const key of createdStorageKeys) {
        try { await documentStorage.remove(key); } catch (_) {}
      }
    }
    throw error;
  } finally {
    client.release();
  }
}

async function main() {
  if (!process.argv.includes("--confirm-demo")) {
    throw new Error("Refusing to seed without --confirm-demo. Use npm run seed:demo.");
  }
  require("dotenv").config({ path: path.join(__dirname, "..", ".env"), quiet: true });
  const pool = new Pool({
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  });
  pool.on("connect", (client) => client.query("SET search_path TO public"));
  try {
    const result = await seedDemoData(pool);
    console.log("StayBuddy fictional demo data is ready.");
    console.log(`Database: ${process.env.DB_NAME}`);
    console.log("Records:", result.summary);
    console.log("Demo logins:");
    for (const account of result.accounts) {
      console.log(`  ${account.key}: ${account.email} / ${account.password}`);
    }
    console.log(`Attendance simulator key: ${DEMO_ATTENDANCE_KEY}`);
    console.log("No paid API was called. AI requests remain inside the local Compose network.");
  } finally {
    await pool.end();
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`Demo seed failed: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { DEMO_ACCOUNTS, DEMO_ATTENDANCE_KEY, DEMO_CHAT_SESSION_ID, seedDemoData };
