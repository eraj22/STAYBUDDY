require("dotenv").config();

console.log("ENV CHECK:", {
  DB_HOST: process.env.DB_HOST,
  DB_PORT: process.env.DB_PORT,
  DB_USER: process.env.DB_USER,
  DB_NAME: process.env.DB_NAME,
});

const express = require("express");
const cors = require("cors");
const crypto = require("crypto");
const { Pool } = require("pg");
const nodemailer = require("nodemailer");
const { registerAttendanceRoutes } = require("./attendance_routes");
const { registerEmergencyRoutes } = require("./emergency_routes");
const { registerVerificationRoutes } = require("./verification_routes");
const { createLocalDocumentStorage } = require("./document_storage");
const { registerFeeRoutes } = require("./fee_routes");
const { registerMessagingRoutes } = require("./messaging_routes");
const { registerComplaintRoutes } = require("./complaint_routes");
const { registerAdminOperationsRoutes } = require("./admin_operations_routes");
const { registerLocationSharingRoutes } = require("./location_sharing_routes");
const { registerPhase3Routes } = require("./phase3_routes");
const { registerParentPhase3Routes } = require("./parent_phase3_routes");
const { registerParentServicesRoutes } = require("./parent_services_routes");
const { registerDiscoveryCompletionRoutes } = require("./discovery_completion_routes");
const { registerBookingDocumentRoutes } = require("./booking_document_routes");
const { registerRagChatRoutes } = require("./rag_chat_routes");
const { registerQrAttendanceRoutes } = require("./qr_attendance_routes");
const { registerPostedGateAttendanceRoutes } = require("./posted_gate_attendance_routes");
const { registerWardenOperationsRoutes } = require("./warden_operations_routes");
const { registerOwnerOperationsRoutes } = require("./owner_operations_routes");
const { enqueueOperationalJob, deliveryDedupeKey } = require("./job_queue");
const { hashPassword, isPassword, verifyPassword } = require("./passwords");
const { analyzeCsvComplaint } = require("./complaint_csv_model");
const { getLiveRecommendations } = require("./live_recommendation_engine");

const {
  buildUserPreferenceFromHistory,
  softNormalize,
  validateAndFallback,
  explain,
  actionWeight,
  clamp01
} = require("./recommender");

const app = express();
app.use(cors());
app.post("/api/payments/webhook/stripe", express.raw({ type: "application/json" }), (_req, res) => {
  res.status(410).json({ error: "In-app payment processing is disabled; submit a payment receipt instead" });
});
app.use(express.json({ limit: "8mb" }));

// Connect PostgreSQL
const pool = new Pool({
  host:     process.env.DB_HOST,
  user:     process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port:     Number(process.env.DB_PORT),
});

// Set search_path on every new connection
pool.on("connect", (client) => {
  client.query("SET search_path TO public");
});

// Test DB connection on startup
pool.connect((err, client, release) => {
  if (err) {
    console.error("âŒ DB connection failed:", err.message);
  } else {
    console.log("âœ… DB connected successfully");
    release();
  }
});

// â”€â”€ Email transporter (Gmail App Password) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});

// In-memory OTP store: email -> { otp, expiresAt }
const otpStore = new Map();

const PASSWORD_RESET_TTL_MINUTES = 15;
const PARENT_INVITATION_TTL_HOURS = 72;
let passwordResetTableReady;

async function ensurePasswordResetTable() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS password_reset_tokens (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      token_hash CHAR(64) NOT NULL UNIQUE,
      expires_at TIMESTAMPTZ NOT NULL,
      used_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE INDEX IF NOT EXISTS password_reset_tokens_active_user_idx
      ON password_reset_tokens (user_id, expires_at)
      WHERE used_at IS NULL
  `);
  console.log("âœ… password reset tokens table ready");
}
passwordResetTableReady = ensurePasswordResetTable().catch((error) => {
  console.error("âŒ password reset tokens table setup failed:", error.message);
  throw error;
});

// Existing local databases created before the Phase 3 migration can be
// missing this table. Authentication must be able to self-initialize it so a
// fresh checkout does not fail at the first login.
const authSessionsReady = pool.query(`
  CREATE TABLE IF NOT EXISTS auth_sessions (
    id UUID PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ,
    user_agent TEXT,
    ip_address TEXT
  )
`).then(() => pool.query(
  "CREATE INDEX IF NOT EXISTS auth_sessions_user_active_idx ON auth_sessions (user_id, expires_at DESC) WHERE revoked_at IS NULL"
)).catch((error) => {
  console.error("Auth sessions table setup failed:", error.message);
  throw error;
});

let operationalTablesReady;
async function ensureOperationalTables() {
  await pool.query("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check");
  await pool.query("ALTER TABLE users ADD CONSTRAINT users_role_check CHECK (role IN ('student', 'parent', 'owner', 'warden', 'admin'))");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS student_profiles (
      user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      university VARCHAR(150), department VARCHAR(150), gender VARCHAR(30),
      budget_max NUMERIC(12, 2) CHECK (budget_max IS NULL OR budget_max >= 0),
      max_distance_km NUMERIC(6, 2) CHECK (max_distance_km IS NULL OR max_distance_km >= 0),
      study_preference NUMERIC(3, 2) CHECK (study_preference IS NULL OR (study_preference >= 0 AND study_preference <= 1)),
      food_preference VARCHAR(30), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`CREATE TABLE IF NOT EXISTS user_push_tokens (
    id BIGSERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(512) NOT NULL UNIQUE, platform VARCHAR(30), active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`);


  await pool.query(`
    CREATE TABLE IF NOT EXISTS notification_preferences (
      user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      booking_updates BOOLEAN NOT NULL DEFAULT true, complaint_updates BOOLEAN NOT NULL DEFAULT true,
      announcements BOOLEAN NOT NULL DEFAULT true, email_enabled BOOLEAN NOT NULL DEFAULT true,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS hostel_owners (
      owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
      claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (owner_user_id, hostel_id), UNIQUE (hostel_id)
    )
  `);
  await pool.query(`
    ALTER TABLE hostels
      ADD COLUMN IF NOT EXISTS floors INTEGER,
      ADD COLUMN IF NOT EXISTS contact_phone VARCHAR(40),
      ADD COLUMN IF NOT EXISTS contact_email VARCHAR(254),
      ADD COLUMN IF NOT EXISTS publication_status VARCHAR(20) NOT NULL DEFAULT 'draft',
      ADD COLUMN IF NOT EXISTS facilities JSONB NOT NULL DEFAULT '[]'::jsonb,
      ADD COLUMN IF NOT EXISTS rules JSONB NOT NULL DEFAULT '[]'::jsonb,
      ADD COLUMN IF NOT EXISTS mess_info JSONB NOT NULL DEFAULT '{}'::jsonb,
      ADD COLUMN IF NOT EXISTS media JSONB NOT NULL DEFAULT '[]'::jsonb,
      ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  `);
  // Older local databases predate the resident/owner portal fields. Keep the
  // application boot self-contained by adding the backward-compatible fields
  // before any authenticated endpoint reads them.
  await pool.query(`
    ALTER TABLE rooms
      ADD COLUMN IF NOT EXISTS floor_label VARCHAR(80) NOT NULL DEFAULT 'Ground Floor',
      ADD COLUMN IF NOT EXISTS amenities JSONB NOT NULL DEFAULT '[]'::jsonb
  `);
  await pool.query(`
    ALTER TABLE bookings
      ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_sharing_sessions (
      id BIGSERIAL PRIMARY KEY,
      student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      expires_at TIMESTAMPTZ NOT NULL,
      revoked_at TIMESTAMPTZ,
      revoked_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
      history_enabled BOOLEAN NOT NULL DEFAULT false,
      geofencing_enabled BOOLEAN NOT NULL DEFAULT false,
      CHECK (expires_at > started_at)
    )
  `);
  await pool.query(`
    ALTER TABLE location_sharing_sessions
      ADD COLUMN IF NOT EXISTS history_enabled BOOLEAN NOT NULL DEFAULT false,
      ADD COLUMN IF NOT EXISTS geofencing_enabled BOOLEAN NOT NULL DEFAULT false
  `);
  await pool.query("CREATE UNIQUE INDEX IF NOT EXISTS location_sharing_one_active_session ON location_sharing_sessions (student_user_id) WHERE revoked_at IS NULL");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_sharing_updates (
      id BIGSERIAL PRIMARY KEY,
      session_id BIGINT NOT NULL REFERENCES location_sharing_sessions(id) ON DELETE CASCADE,
      latitude NUMERIC(10,8) NOT NULL CHECK (latitude BETWEEN -90 AND 90),
      longitude NUMERIC(11,8) NOT NULL CHECK (longitude BETWEEN -180 AND 180),
      accuracy_m NUMERIC(10,2) CHECK (accuracy_m IS NULL OR accuracy_m >= 0),
      client_event_id VARCHAR(160) NOT NULL,
      reported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      UNIQUE (session_id, client_event_id)
    )
  `);
  await pool.query("CREATE INDEX IF NOT EXISTS location_sharing_updates_session_time_idx ON location_sharing_updates (session_id, reported_at DESC)");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_sharing_audit_events (
      id BIGSERIAL PRIMARY KEY,
      session_id BIGINT REFERENCES location_sharing_sessions(id) ON DELETE SET NULL,
      event_type VARCHAR(40) NOT NULL,
      actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
      details JSONB NOT NULL DEFAULT '{}'::jsonb,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query("CREATE INDEX IF NOT EXISTS location_sharing_audit_events_session_idx ON location_sharing_audit_events (session_id, created_at DESC)");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS warden_assignments (
      warden_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
      assigned_by_user_id INTEGER NOT NULL REFERENCES users(id),
      assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (warden_user_id, hostel_id)
    )
  `);

    // ========================================================================
  // WARDEN PROFILE
  // Extra information entered by the owner while creating a Warden account.
  // ========================================================================
  await pool.query(`
    CREATE TABLE IF NOT EXISTS warden_profiles (
      warden_user_id INTEGER PRIMARY KEY
        REFERENCES users(id) ON DELETE CASCADE,

      cnic VARCHAR(20),
      city VARCHAR(120),

      shift VARCHAR(30) NOT NULL DEFAULT 'Morning',

      assigned_floors JSONB NOT NULL DEFAULT '[]'::jsonb,

      permissions JSONB NOT NULL DEFAULT '{}'::jsonb,

      account_status VARCHAR(20) NOT NULL DEFAULT 'Active'
        CHECK (account_status IN ('Active', 'Disabled')),

      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS warden_profiles_status_idx
    ON warden_profiles (account_status)
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS complaint_status_history (
      id SERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
      previous_status VARCHAR(30), next_status VARCHAR(30) NOT NULL,
      changed_by_user_id INTEGER NOT NULL REFERENCES users(id), changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query("ALTER TABLE complaints ADD COLUMN IF NOT EXISTS priority VARCHAR(20) NOT NULL DEFAULT 'medium', ADD COLUMN IF NOT EXISTS sla_due_at TIMESTAMPTZ, ADD COLUMN IF NOT EXISTS escalated_at TIMESTAMPTZ");
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_evidence (id BIGSERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE, storage_provider VARCHAR(40) NOT NULL, storage_key TEXT NOT NULL UNIQUE, checksum CHAR(64) NOT NULL, original_filename VARCHAR(255) NOT NULL, content_type VARCHAR(120) NOT NULL, byte_size BIGINT NOT NULL CHECK (byte_size>0 AND byte_size<=5242880), uploaded_by_user_id INTEGER NOT NULL REFERENCES users(id), deleted_at TIMESTAMPTZ, deleted_by_user_id INTEGER REFERENCES users(id), retention_until TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_investigations (id BIGSERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE, findings TEXT NOT NULL, root_cause TEXT, immediate_actions TEXT, preventive_measures TEXT, created_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_owner_escalations (id BIGSERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE, reason TEXT NOT NULL, urgency VARCHAR(20) NOT NULL CHECK (urgency IN ('high','urgent')), idempotency_key VARCHAR(160) NOT NULL UNIQUE, escalated_by_user_id INTEGER NOT NULL REFERENCES users(id), escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_activity_events (id BIGSERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE, event_type VARCHAR(50) NOT NULL, actor_user_id INTEGER NOT NULL REFERENCES users(id), details JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query("ALTER TABLE complaints DROP CONSTRAINT IF EXISTS complaints_priority_check");
  await pool.query("ALTER TABLE complaints ADD CONSTRAINT complaints_priority_check CHECK (priority IN ('low','medium','high','urgent'))");
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_triage_history (id BIGSERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE, previous_priority VARCHAR(20), next_priority VARCHAR(20) NOT NULL, previous_sla_due_at TIMESTAMPTZ, next_sla_due_at TIMESTAMPTZ, note TEXT NOT NULL, triaged_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_sla_escalations (id BIGSERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE, escalated_by_worker VARCHAR(120) NOT NULL, escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), details JSONB NOT NULL DEFAULT '{}'::jsonb)`);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS announcements (
      id SERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
      author_user_id INTEGER NOT NULL REFERENCES users(id), title VARCHAR(180) NOT NULL, body TEXT NOT NULL,
      audience VARCHAR(20) NOT NULL DEFAULT 'residents' CHECK (audience IN ('residents', 'booked', 'favorited')),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS notifications (
      id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      announcement_id INTEGER REFERENCES announcements(id) ON DELETE CASCADE, type VARCHAR(40) NOT NULL,
      title VARCHAR(180) NOT NULL, body TEXT NOT NULL, data JSONB NOT NULL DEFAULT '{}'::jsonb,
      read_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS room_assignment_history (
      id SERIAL PRIMARY KEY, booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
      previous_room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL, next_room_id INTEGER NOT NULL REFERENCES rooms(id),
      assigned_by_user_id INTEGER NOT NULL REFERENCES users(id), assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS booking_stays (
      booking_id INTEGER PRIMARY KEY REFERENCES bookings(id) ON DELETE CASCADE,
      checked_in_at TIMESTAMPTZ, checked_in_by_user_id INTEGER REFERENCES users(id),
      checked_out_at TIMESTAMPTZ, checked_out_by_user_id INTEGER REFERENCES users(id),
      CHECK (checked_out_at IS NULL OR checked_in_at IS NOT NULL),
      CHECK (checked_out_at IS NULL OR checked_out_at >= checked_in_at)
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS payments (
      id SERIAL PRIMARY KEY, booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
      user_id INTEGER NOT NULL REFERENCES users(id), provider VARCHAR(30) NOT NULL,
      provider_payment_id VARCHAR(255) UNIQUE, amount NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
      currency CHAR(3) NOT NULL DEFAULT 'PKR', status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'succeeded', 'failed', 'cancelled', 'refunded')),
      receipt_url TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query("ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS attendance_updates BOOLEAN NOT NULL DEFAULT true");
  await pool.query("ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS emergency_alerts BOOLEAN NOT NULL DEFAULT true");
  await pool.query("ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS fee_reminders BOOLEAN NOT NULL DEFAULT true");
  await pool.query("ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS location_updates BOOLEAN NOT NULL DEFAULT true");
  await pool.query("ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS transport_updates BOOLEAN NOT NULL DEFAULT true");
  await pool.query("ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS communication_updates BOOLEAN NOT NULL DEFAULT true");
  await pool.query("ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS availability_updates BOOLEAN NOT NULL DEFAULT true");
  await pool.query(`CREATE TABLE IF NOT EXISTS operational_jobs (id BIGSERIAL PRIMARY KEY, job_type VARCHAR(60) NOT NULL, dedupe_key VARCHAR(220) NOT NULL UNIQUE, payload JSONB NOT NULL DEFAULT '{}'::jsonb, status VARCHAR(20) NOT NULL DEFAULT 'queued', run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), attempt_count INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 5, locked_at TIMESTAMPTZ, locked_by VARCHAR(120), last_error TEXT, completed_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS operational_job_attempts (id BIGSERIAL PRIMARY KEY, job_id BIGINT NOT NULL REFERENCES operational_jobs(id) ON DELETE CASCADE, attempt_number INTEGER NOT NULL, outcome VARCHAR(20) NOT NULL, details JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(job_id,attempt_number))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS operational_job_manual_retries (id BIGSERIAL PRIMARY KEY, job_id BIGINT NOT NULL REFERENCES operational_jobs(id) ON DELETE CASCADE, retried_by_user_id INTEGER NOT NULL REFERENCES users(id), reason TEXT NOT NULL CHECK (char_length(trim(reason)) BETWEEN 1 AND 2000), previous_attempt_count INTEGER NOT NULL CHECK (previous_attempt_count >= 0), previous_max_attempts INTEGER NOT NULL CHECK (previous_max_attempts BETWEEN 1 AND 20), retry_max_attempts INTEGER NOT NULL CHECK (retry_max_attempts BETWEEN 1 AND 20), retried_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), CHECK (retry_max_attempts = previous_max_attempts + 1))`);
  await pool.query("CREATE INDEX IF NOT EXISTS operational_job_manual_retries_job_idx ON operational_job_manual_retries (job_id, retried_at DESC)");
  await pool.query("ALTER TABLE operational_jobs DROP CONSTRAINT IF EXISTS operational_jobs_job_type_check");
  await pool.query("ALTER TABLE operational_jobs ADD CONSTRAINT operational_jobs_job_type_check CHECK (job_type IN ('emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check','fee_invoice_generation','fee_reminder_check','complaint_sla_check','complaint_volume_check','location_retention_check','booking_expiry_check','complaint_auto_close','attendance_curfew_check','availability_alert_check'))");
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_complaint_alert_rules (hostel_id INTEGER PRIMARY KEY REFERENCES hostels(id) ON DELETE CASCADE, open_complaint_threshold INTEGER NOT NULL CHECK (open_complaint_threshold BETWEEN 1 AND 1000), cooldown_minutes INTEGER NOT NULL DEFAULT 1440 CHECK (cooldown_minutes BETWEEN 15 AND 10080), enabled BOOLEAN NOT NULL DEFAULT true, updated_by_user_id INTEGER NOT NULL REFERENCES users(id), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_volume_alerts (id BIGSERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE, threshold_count INTEGER NOT NULL, observed_open_count INTEGER NOT NULL, triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), triggered_by_worker VARCHAR(120) NOT NULL)`);
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_related_links (id BIGSERIAL PRIMARY KEY, source_complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE, primary_complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE, reason TEXT NOT NULL, idempotency_key VARCHAR(160) NOT NULL UNIQUE, linked_by_user_id INTEGER NOT NULL REFERENCES users(id), linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), CHECK (source_complaint_id <> primary_complaint_id))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS complaint_resolution_records (id BIGSERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE, final_status VARCHAR(20) NOT NULL CHECK (final_status IN ('resolved','closed')), safe_resolution_summary TEXT NOT NULL CHECK (char_length(trim(safe_resolution_summary)) BETWEEN 1 AND 4000), actions_taken TEXT NOT NULL CHECK (char_length(trim(actions_taken)) BETWEEN 1 AND 4000), prevention_follow_up_notes TEXT CHECK (prevention_follow_up_notes IS NULL OR char_length(prevention_follow_up_notes) <= 4000), resolver_user_id INTEGER NOT NULL REFERENCES users(id), resolved_at TIMESTAMPTZ NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_attendance_settings (hostel_id INTEGER PRIMARY KEY REFERENCES hostels(id) ON DELETE CASCADE, curfew_time TIME NOT NULL DEFAULT '22:00:00', timezone VARCHAR(64) NOT NULL DEFAULT 'Asia/Karachi', enabled BOOLEAN NOT NULL DEFAULT true, updated_by_user_id INTEGER REFERENCES users(id), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS attendance_integration_credentials (id SERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE, label VARCHAR(120) NOT NULL, credential_hash CHAR(64) NOT NULL UNIQUE, active BOOLEAN NOT NULL DEFAULT true, created_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), revoked_at TIMESTAMPTZ)`);
  await pool.query(`CREATE TABLE IF NOT EXISTS resident_attendance_identifiers (id SERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, external_identifier VARCHAR(160) NOT NULL, active BOOLEAN NOT NULL DEFAULT true, enrolled_by_user_id INTEGER NOT NULL REFERENCES users(id), enrolled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), revoked_at TIMESTAMPTZ, UNIQUE (hostel_id, external_identifier))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS attendance_events (id BIGSERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, direction VARCHAR(3) NOT NULL CHECK (direction IN ('in','out')), occurred_at TIMESTAMPTZ NOT NULL, source VARCHAR(20) NOT NULL CHECK (source IN ('biometric','rfid','manual')), entry_point VARCHAR(120), external_event_id VARCHAR(160), idempotency_key VARCHAR(160) NOT NULL UNIQUE, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_by_user_id INTEGER REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE (hostel_id,source,external_event_id))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS attendance_corrections (id BIGSERIAL PRIMARY KEY, attendance_event_id BIGINT NOT NULL REFERENCES attendance_events(id) ON DELETE CASCADE, corrected_direction VARCHAR(3) NOT NULL CHECK (corrected_direction IN ('in','out')), corrected_occurred_at TIMESTAMPTZ NOT NULL, reason TEXT NOT NULL, corrected_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS attendance_exceptions (id BIGSERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, policy_date DATE NOT NULL, exception_type VARCHAR(30) NOT NULL CHECK (exception_type IN ('late_exit','late_return','absent_at_curfew')), status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved')), details JSONB NOT NULL DEFAULT '{}'::jsonb, resolved_by_user_id INTEGER REFERENCES users(id), resolved_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE (hostel_id,student_user_id,policy_date,exception_type))`);
  // An authorized warden may re-open/download the currently active gate QR.
  // The image is revoked together with its code whenever a replacement is made.
  await pool.query("ALTER TABLE hostel_gate_qr_codes ADD COLUMN IF NOT EXISTS qr_image TEXT");
  await pool.query(`CREATE TABLE IF NOT EXISTS emergency_contacts (id SERIAL PRIMARY KEY, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, name VARCHAR(120) NOT NULL, relationship VARCHAR(80) NOT NULL, phone VARCHAR(30), email VARCHAR(180), priority SMALLINT NOT NULL DEFAULT 1 CHECK (priority>0), verified_at TIMESTAMPTZ, active BOOLEAN NOT NULL DEFAULT true, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), CHECK (phone IS NOT NULL OR email IS NOT NULL))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS emergency_incidents (id BIGSERIAL PRIMARY KEY, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE RESTRICT, booking_id INTEGER REFERENCES bookings(id) ON DELETE SET NULL, room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL, alert_type VARCHAR(30) NOT NULL CHECK (alert_type IN ('medical','safety','fire','accident','general')), description TEXT, status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active','acknowledged','escalated','cancelled','resolved')), latitude NUMERIC(10,8), longitude NUMERIC(11,8), location_accuracy_m NUMERIC(10,2), idempotency_key VARCHAR(160) NOT NULL UNIQUE, cancelled_by_user_id INTEGER REFERENCES users(id), cancelled_at TIMESTAMPTZ, cancellation_reason TEXT, resolved_by_user_id INTEGER REFERENCES users(id), resolved_at TIMESTAMPTZ, resolution_reason TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), CHECK ((latitude IS NULL AND longitude IS NULL) OR (latitude IS NOT NULL AND longitude IS NOT NULL)))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS emergency_timeline_events (id BIGSERIAL PRIMARY KEY, incident_id BIGINT NOT NULL REFERENCES emergency_incidents(id) ON DELETE CASCADE, event_type VARCHAR(30) NOT NULL CHECK (event_type IN ('created','acknowledged','escalated','cancelled','resolved','note')), actor_user_id INTEGER REFERENCES users(id), details JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS emergency_delivery_attempts (id BIGSERIAL PRIMARY KEY, incident_id BIGINT NOT NULL REFERENCES emergency_incidents(id) ON DELETE CASCADE, recipient_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, contact_id INTEGER REFERENCES emergency_contacts(id) ON DELETE SET NULL, channel VARCHAR(20) NOT NULL CHECK (channel IN ('in_app','push','sms','email','call')), status VARCHAR(20) NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','delivered','failed')), provider VARCHAR(60), provider_reference VARCHAR(160), attempt_count INTEGER NOT NULL DEFAULT 0, error_message TEXT, next_retry_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), CHECK (recipient_user_id IS NOT NULL OR contact_id IS NOT NULL))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_registration_applications (id BIGSERIAL PRIMARY KEY, owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, hostel_id INTEGER REFERENCES hostels(id) ON DELETE SET NULL, status VARCHAR(40) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','submitted','more_information_requested','inspection_scheduled','provisional','conditionally_approved','approved','rejected','withdrawn')), reapplication_allowed BOOLEAN NOT NULL DEFAULT true, submission_notes TEXT, submitted_at TIMESTAMPTZ, decided_at TIMESTAMPTZ, decided_by_user_id INTEGER REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_registration_documents (id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE, document_type VARCHAR(80) NOT NULL, storage_provider VARCHAR(40) NOT NULL DEFAULT 'local', storage_key TEXT NOT NULL, checksum VARCHAR(128), verification_status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (verification_status IN ('pending','accepted','rejected')), submitted_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), reviewed_at TIMESTAMPTZ, reviewed_by_user_id INTEGER REFERENCES users(id), review_note TEXT)`);
  await pool.query("ALTER TABLE hostel_registration_documents ADD COLUMN IF NOT EXISTS original_filename VARCHAR(255), ADD COLUMN IF NOT EXISTS content_type VARCHAR(120), ADD COLUMN IF NOT EXISTS byte_size BIGINT, ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ, ADD COLUMN IF NOT EXISTS deleted_by_user_id INTEGER REFERENCES users(id), ADD COLUMN IF NOT EXISTS retention_until TIMESTAMPTZ");
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_verification_events (id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE, event_type VARCHAR(40) NOT NULL, actor_user_id INTEGER REFERENCES users(id), details JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_inspections (id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE, priority VARCHAR(20) NOT NULL CHECK (priority IN ('high','standard','low')), assigned_to_user_id INTEGER REFERENCES users(id), scheduled_for TIMESTAMPTZ NOT NULL, scope JSONB NOT NULL DEFAULT '[]'::jsonb, findings TEXT, outcome VARCHAR(20) CHECK (outcome IN ('passed','failed','follow_up_required')), completed_at TIMESTAMPTZ, created_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_approval_conditions (id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE, requirement TEXT NOT NULL, due_at TIMESTAMPTZ NOT NULL, status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','completed','overdue','waived')), completed_at TIMESTAMPTZ, completed_by_user_id INTEGER REFERENCES users(id), evidence_storage_key TEXT, created_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_verification_checklist_items (id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE, item_code VARCHAR(80) NOT NULL, label VARCHAR(240) NOT NULL, required BOOLEAN NOT NULL DEFAULT true, status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','passed','failed','waived')), review_note TEXT, reviewed_by_user_id INTEGER REFERENCES users(id), reviewed_at TIMESTAMPTZ, created_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(application_id,item_code))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_verification_checklist_events (id BIGSERIAL PRIMARY KEY, checklist_item_id BIGINT NOT NULL REFERENCES hostel_verification_checklist_items(id) ON DELETE CASCADE, event_type VARCHAR(30) NOT NULL CHECK (event_type IN ('created','reviewed')), actor_user_id INTEGER REFERENCES users(id), details JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_verification_information_requests (id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE, details TEXT NOT NULL, due_at TIMESTAMPTZ NOT NULL, status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','responded','expired','cancelled')), requested_by_user_id INTEGER NOT NULL REFERENCES users(id), requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), responded_by_user_id INTEGER REFERENCES users(id), responded_at TIMESTAMPTZ, response_note TEXT)`);
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_approval_condition_evidence (id BIGSERIAL PRIMARY KEY, condition_id BIGINT NOT NULL REFERENCES hostel_approval_conditions(id) ON DELETE CASCADE, document_id BIGINT NOT NULL REFERENCES hostel_registration_documents(id) ON DELETE RESTRICT, submitted_by_user_id INTEGER NOT NULL REFERENCES users(id), note TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(condition_id,document_id))`);
  
  await pool.query(`
  CREATE TABLE IF NOT EXISTS hostel_fee_schedules (
    id BIGSERIAL PRIMARY KEY,
    hostel_id INTEGER NOT NULL
      REFERENCES hostels(id) ON DELETE CASCADE,
    title VARCHAR(120) NOT NULL,
    amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    currency CHAR(3) NOT NULL DEFAULT 'PKR',
    due_day SMALLINT NOT NULL CHECK (due_day BETWEEN 1 AND 28),
    active BOOLEAN NOT NULL DEFAULT true,
    created_by_user_id INTEGER NOT NULL
      REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )
`);

await pool.query(`
  CREATE TABLE IF NOT EXISTS student_fee_assignments (
    id BIGSERIAL PRIMARY KEY,
    schedule_id BIGINT NOT NULL
      REFERENCES hostel_fee_schedules(id) ON DELETE CASCADE,
    booking_id INTEGER NOT NULL
      REFERENCES bookings(id) ON DELETE CASCADE,
    student_user_id INTEGER NOT NULL
      REFERENCES users(id) ON DELETE CASCADE,
    amount_override NUMERIC(12,2)
      CHECK (amount_override IS NULL OR amount_override > 0),
    starts_on DATE NOT NULL,
    ends_on DATE,
    active BOOLEAN NOT NULL DEFAULT true,
    assigned_by_user_id INTEGER NOT NULL
      REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (ends_on IS NULL OR ends_on >= starts_on),
    UNIQUE (schedule_id, booking_id)
  )
`);

await pool.query(`
  CREATE TABLE IF NOT EXISTS fee_invoices (
    id BIGSERIAL PRIMARY KEY,
    assignment_id BIGINT NOT NULL
      REFERENCES student_fee_assignments(id) ON DELETE CASCADE,
    billing_month DATE NOT NULL,
    due_date DATE NOT NULL,
    amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    currency CHAR(3) NOT NULL DEFAULT 'PKR',
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
      CHECK (status IN ('pending', 'paid', 'overdue', 'cancelled')),
    paid_payment_id INTEGER
      REFERENCES payments(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (assignment_id, billing_month)
  )
`);

await pool.query(`
  CREATE TABLE IF NOT EXISTS fee_reminder_deliveries (
    id BIGSERIAL PRIMARY KEY,
    invoice_id BIGINT NOT NULL
      REFERENCES fee_invoices(id) ON DELETE CASCADE,
    recipient_user_id INTEGER NOT NULL
      REFERENCES users(id) ON DELETE CASCADE,
    channel VARCHAR(20) NOT NULL DEFAULT 'in_app'
      CHECK (channel IN ('in_app', 'email', 'sms')),
    status VARCHAR(20) NOT NULL DEFAULT 'queued'
      CHECK (status IN ('queued', 'delivered', 'failed')),
    scheduled_for TIMESTAMPTZ NOT NULL,
    delivered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )
`);

await pool.query(`
  ALTER TABLE fee_reminder_deliveries
    ADD COLUMN IF NOT EXISTS provider VARCHAR(60),
    ADD COLUMN IF NOT EXISTS provider_reference VARCHAR(160),
    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS error_message TEXT,
    ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
`);

await pool.query(`
  CREATE INDEX IF NOT EXISTS fee_reminders_delivery_idx
    ON fee_reminder_deliveries (status, scheduled_for)
`);
  
  await pool.query(`CREATE TABLE IF NOT EXISTS conversations (id BIGSERIAL PRIMARY KEY, conversation_type VARCHAR(30) NOT NULL CHECK (conversation_type IN ('parent_student','parent_staff')), parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, hostel_id INTEGER REFERENCES hostels(id) ON DELETE SET NULL, staff_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), CHECK ((conversation_type='parent_student' AND hostel_id IS NULL AND staff_user_id IS NULL) OR (conversation_type='parent_staff' AND hostel_id IS NOT NULL AND staff_user_id IS NOT NULL)))`);
  await pool.query("CREATE UNIQUE INDEX IF NOT EXISTS conversations_unique_participants ON conversations (conversation_type,parent_user_id,student_user_id,COALESCE(hostel_id,0),COALESCE(staff_user_id,0))");
  await pool.query(`CREATE TABLE IF NOT EXISTS conversation_messages (id BIGSERIAL PRIMARY KEY, conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, sender_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, body TEXT NOT NULL CHECK (char_length(trim(body)) BETWEEN 1 AND 4000), delivered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), read_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`);
  await pool.query(`CREATE TABLE IF NOT EXISTS conversation_message_reports (id BIGSERIAL PRIMARY KEY, message_id BIGINT NOT NULL REFERENCES conversation_messages(id) ON DELETE CASCADE, reporter_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, reason VARCHAR(30) NOT NULL CHECK (reason IN ('harassment','spam','safety','other')), details TEXT, status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','reviewed','dismissed')), reviewed_by_user_id INTEGER REFERENCES users(id), reviewed_at TIMESTAMPTZ, review_note TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(message_id,reporter_user_id))`);
  await pool.query(`CREATE TABLE IF NOT EXISTS conversation_staff_handovers (id BIGSERIAL PRIMARY KEY, conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, previous_staff_user_id INTEGER NOT NULL REFERENCES users(id), next_staff_user_id INTEGER NOT NULL REFERENCES users(id), handed_over_by_user_id INTEGER NOT NULL REFERENCES users(id), reason TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), CHECK(previous_staff_user_id<>next_staff_user_id))`);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS parent_student_links (
      id SERIAL PRIMARY KEY,
      parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      invited_by_student_id INTEGER NOT NULL REFERENCES users(id),
      invitation_token_hash CHAR(64) NOT NULL UNIQUE,
      status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'active', 'revoked')),
      expires_at TIMESTAMPTZ NOT NULL,
      accepted_at TIMESTAMPTZ,
      revoked_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      CHECK (parent_user_id <> student_user_id),
      CHECK ((status <> 'active') OR accepted_at IS NOT NULL),
      CHECK ((status <> 'revoked') OR revoked_at IS NOT NULL)
    )
  `);
  await pool.query("CREATE INDEX IF NOT EXISTS hostel_owners_owner_idx ON hostel_owners (owner_user_id)");
  await pool.query("CREATE INDEX IF NOT EXISTS warden_assignments_hostel_idx ON warden_assignments (hostel_id)");
  await pool.query("CREATE INDEX IF NOT EXISTS complaint_status_history_complaint_idx ON complaint_status_history (complaint_id, changed_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS complaint_resolution_records_resolver_idx ON complaint_resolution_records (resolver_user_id, resolved_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS announcements_hostel_created_idx ON announcements (hostel_id, created_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS notifications_user_created_idx ON notifications (user_id, created_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS room_assignment_history_booking_idx ON room_assignment_history (booking_id, assigned_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS payments_user_created_idx ON payments (user_id, created_at DESC)");
  await pool.query("CREATE UNIQUE INDEX IF NOT EXISTS parent_student_links_active_or_pending_unique ON parent_student_links (parent_user_id, student_user_id) WHERE status IN ('pending', 'active')");
  await pool.query("CREATE INDEX IF NOT EXISTS parent_student_links_parent_active_idx ON parent_student_links (parent_user_id, student_user_id) WHERE status = 'active'");
  await pool.query("CREATE INDEX IF NOT EXISTS parent_student_links_student_active_idx ON parent_student_links (student_user_id, parent_user_id) WHERE status = 'active'");
  await pool.query("CREATE INDEX IF NOT EXISTS attendance_events_hostel_time_idx ON attendance_events (hostel_id, occurred_at DESC)");
  await pool.query("CREATE INDEX IF NOT EXISTS attendance_events_student_time_idx ON attendance_events (student_user_id, occurred_at DESC)");
  console.log("âœ… profiles and staff ownership tables ready");
}
// Older local databases may have been created from the original schema dump.
// These additive portal fields are safe to apply at startup and let those
// databases run the current owner, warden, and resident screens without a
// manual migration step.
async function ensurePortalCompatibilityColumns() {
  await pool.query(`
    ALTER TABLE hostels
      ADD COLUMN IF NOT EXISTS contact_phone VARCHAR(40),
      ADD COLUMN IF NOT EXISTS contact_email VARCHAR(254),
      ADD COLUMN IF NOT EXISTS publication_status VARCHAR(20) NOT NULL DEFAULT 'draft',
      ADD COLUMN IF NOT EXISTS facilities JSONB NOT NULL DEFAULT '[]'::jsonb,
      ADD COLUMN IF NOT EXISTS rules JSONB NOT NULL DEFAULT '[]'::jsonb,
      ADD COLUMN IF NOT EXISTS mess_info JSONB NOT NULL DEFAULT '{}'::jsonb,
      ADD COLUMN IF NOT EXISTS media JSONB NOT NULL DEFAULT '[]'::jsonb,
      ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  `);
  await pool.query(`
    ALTER TABLE rooms
      ADD COLUMN IF NOT EXISTS floor_label VARCHAR(80) NOT NULL DEFAULT 'Ground Floor',
      ADD COLUMN IF NOT EXISTS amenities JSONB NOT NULL DEFAULT '[]'::jsonb
  `);
  await pool.query(`
    ALTER TABLE bookings
      ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_sharing_sessions (
      id BIGSERIAL PRIMARY KEY,
      student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      expires_at TIMESTAMPTZ NOT NULL,
      revoked_at TIMESTAMPTZ,
      revoked_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
      history_enabled BOOLEAN NOT NULL DEFAULT false,
      geofencing_enabled BOOLEAN NOT NULL DEFAULT false,
      CHECK (expires_at > started_at)
    )
  `);
  await pool.query(`ALTER TABLE location_sharing_sessions
    ADD COLUMN IF NOT EXISTS history_enabled BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS geofencing_enabled BOOLEAN NOT NULL DEFAULT false`);
  await pool.query("CREATE UNIQUE INDEX IF NOT EXISTS location_sharing_one_active_session ON location_sharing_sessions (student_user_id) WHERE revoked_at IS NULL");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_sharing_updates (
      id BIGSERIAL PRIMARY KEY,
      session_id BIGINT NOT NULL REFERENCES location_sharing_sessions(id) ON DELETE CASCADE,
      latitude NUMERIC(10,8) NOT NULL CHECK (latitude BETWEEN -90 AND 90),
      longitude NUMERIC(11,8) NOT NULL CHECK (longitude BETWEEN -180 AND 180),
      accuracy_m NUMERIC(10,2) CHECK (accuracy_m IS NULL OR accuracy_m >= 0),
      client_event_id VARCHAR(160) NOT NULL,
      reported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      UNIQUE (session_id, client_event_id)
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_sharing_audit_events (
      id BIGSERIAL PRIMARY KEY,
      session_id BIGINT REFERENCES location_sharing_sessions(id) ON DELETE SET NULL,
      event_type VARCHAR(40) NOT NULL,
      actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
      details JSONB NOT NULL DEFAULT '{}'::jsonb,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS parent_student_links (
      id SERIAL PRIMARY KEY,
      parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      invited_by_student_id INTEGER NOT NULL REFERENCES users(id),
      invitation_token_hash CHAR(64) NOT NULL UNIQUE,
      status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'active', 'revoked')),
      expires_at TIMESTAMPTZ NOT NULL,
      accepted_at TIMESTAMPTZ,
      revoked_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      CHECK (parent_user_id <> student_user_id)
    )
  `);
  await pool.query("CREATE UNIQUE INDEX IF NOT EXISTS parent_student_links_active_or_pending_unique ON parent_student_links (parent_user_id, student_user_id) WHERE status IN ('pending', 'active')");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS hostel_attendance_geofences (
      hostel_id INTEGER PRIMARY KEY REFERENCES hostels(id) ON DELETE CASCADE,
      boundary JSONB NOT NULL,
      enabled BOOLEAN NOT NULL DEFAULT true,
      updated_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      CHECK (jsonb_typeof(boundary) = 'array'),
      CHECK (jsonb_array_length(boundary) >= 3)
    )
  `);
  // Fee tables are also needed by the owner report on databases created from
  // the original schema dump.  Empty tables yield a valid zero-fee report;
  // they must never prevent the owner dashboard or PDF from opening.
  await pool.query(`CREATE TABLE IF NOT EXISTS hostel_fee_schedules (
    id BIGSERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
    title VARCHAR(120) NOT NULL, amount NUMERIC(12,2) NOT NULL, currency CHAR(3) NOT NULL DEFAULT 'PKR',
    due_day SMALLINT NOT NULL, active BOOLEAN NOT NULL DEFAULT true,
    created_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`);
  await pool.query(`CREATE TABLE IF NOT EXISTS student_fee_assignments (
    id BIGSERIAL PRIMARY KEY, schedule_id BIGINT NOT NULL REFERENCES hostel_fee_schedules(id) ON DELETE CASCADE,
    booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amount_override NUMERIC(12,2), starts_on DATE NOT NULL, ends_on DATE, active BOOLEAN NOT NULL DEFAULT true,
    assigned_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(schedule_id,booking_id)
  )`);
  await pool.query(`CREATE TABLE IF NOT EXISTS fee_invoices (
    id BIGSERIAL PRIMARY KEY, assignment_id BIGINT NOT NULL REFERENCES student_fee_assignments(id) ON DELETE CASCADE,
    billing_month DATE NOT NULL, due_date DATE NOT NULL, amount NUMERIC(12,2) NOT NULL, currency CHAR(3) NOT NULL DEFAULT 'PKR',
    status VARCHAR(20) NOT NULL DEFAULT 'pending', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(assignment_id,billing_month)
  )`);
  await pool.query(`CREATE TABLE IF NOT EXISTS fee_receipt_submissions (
    id BIGSERIAL PRIMARY KEY, invoice_id BIGINT NOT NULL REFERENCES fee_invoices(id) ON DELETE CASCADE,
    hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    submitted_by_user_id INTEGER NOT NULL REFERENCES users(id), amount NUMERIC(12,2) NOT NULL, status VARCHAR(30) NOT NULL DEFAULT 'submitted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`);
  await pool.query(`CREATE TABLE IF NOT EXISTS fee_receipt_allocations (
    id BIGSERIAL PRIMARY KEY, receipt_submission_id BIGINT NOT NULL UNIQUE REFERENCES fee_receipt_submissions(id) ON DELETE RESTRICT,
    invoice_id BIGINT NOT NULL REFERENCES fee_invoices(id) ON DELETE RESTRICT, amount NUMERIC(12,2) NOT NULL,
    allocated_by_user_id INTEGER NOT NULL REFERENCES users(id), allocated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`);
  await pool.query(`CREATE TABLE IF NOT EXISTS warden_cash_fee_payments (
    id BIGSERIAL PRIMARY KEY, invoice_id BIGINT NOT NULL REFERENCES fee_invoices(id) ON DELETE RESTRICT,
    hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amount NUMERIC(12,2) NOT NULL, received_on DATE NOT NULL, note VARCHAR(1000), received_by_warden_id INTEGER NOT NULL REFERENCES users(id),
    voided_at TIMESTAMPTZ, voided_by_user_id INTEGER REFERENCES users(id), void_reason VARCHAR(1000), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`);
  await pool.query(`ALTER TABLE complaints
    ADD COLUMN IF NOT EXISTS room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS bed_number VARCHAR(40)`);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_geofences (
      id BIGSERIAL PRIMARY KEY,
      student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      name VARCHAR(120) NOT NULL,
      latitude NUMERIC(10,8) NOT NULL CHECK (latitude BETWEEN -90 AND 90),
      longitude NUMERIC(11,8) NOT NULL CHECK (longitude BETWEEN -180 AND 180),
      radius_m INTEGER NOT NULL CHECK (radius_m BETWEEN 50 AND 5000),
      active BOOLEAN NOT NULL DEFAULT true,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      disabled_at TIMESTAMPTZ
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_geofence_states (
      geofence_id BIGINT NOT NULL REFERENCES location_geofences(id) ON DELETE CASCADE,
      session_id BIGINT NOT NULL REFERENCES location_sharing_sessions(id) ON DELETE CASCADE,
      last_inside BOOLEAN NOT NULL,
      last_evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      last_transition_at TIMESTAMPTZ,
      PRIMARY KEY (geofence_id, session_id)
    )
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS location_geofence_events (
      id BIGSERIAL PRIMARY KEY,
      geofence_id BIGINT NOT NULL REFERENCES location_geofences(id) ON DELETE CASCADE,
      session_id BIGINT NOT NULL REFERENCES location_sharing_sessions(id) ON DELETE CASCADE,
      location_update_id BIGINT NOT NULL REFERENCES location_sharing_updates(id) ON DELETE CASCADE,
      event_type VARCHAR(10) NOT NULL CHECK (event_type IN ('entered','exited')),
      occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      UNIQUE (geofence_id, location_update_id)
    )
  `);
  console.log("Portal database compatibility fields ready");
}
operationalTablesReady = ensurePortalCompatibilityColumns();

let hostelAccessCredentialTableReady;
function ensureHostelAccessCredentialTable() {
  if (!hostelAccessCredentialTableReady) {
    hostelAccessCredentialTableReady = pool.query(`
      CREATE TABLE IF NOT EXISTS hostel_access_credentials (
        id BIGSERIAL PRIMARY KEY,
        booking_id INTEGER NOT NULL UNIQUE REFERENCES bookings(id) ON DELETE CASCADE,
        student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
        username VARCHAR(180) NOT NULL,
        password_hash TEXT NOT NULL,
        issued_by_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
        issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        activated_at TIMESTAMPTZ,
        UNIQUE(student_user_id, booking_id)
      )
    `);
  }
  return hostelAccessCredentialTableReady;
}

let studentHubTablesReady;
function ensureStudentHubTables() {
  if (!studentHubTablesReady) {
    studentHubTablesReady = (async () => {
      await pool.query(`CREATE TABLE IF NOT EXISTS student_room_change_requests (
        id BIGSERIAL PRIMARY KEY, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
        booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
        reason TEXT NOT NULL CHECK (char_length(trim(reason)) BETWEEN 3 AND 2000),
        status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected','cancelled')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )`);
      await pool.query("ALTER TABLE student_room_change_requests ADD COLUMN IF NOT EXISTS requested_room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL");
      await pool.query("ALTER TABLE student_room_change_requests ADD COLUMN IF NOT EXISTS requested_bed_number VARCHAR(40)");
      await pool.query(`CREATE TABLE IF NOT EXISTS student_hostel_feedback (
        id BIGSERIAL PRIMARY KEY, student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
        feedback_type VARCHAR(20) NOT NULL CHECK (feedback_type IN ('food','services')),
        rating SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5), comment TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )`);
      await pool.query(`CREATE TABLE IF NOT EXISTS student_hostel_chat_messages (
        id BIGSERIAL PRIMARY KEY, hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
        sender_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        channel VARCHAR(20) NOT NULL CHECK (channel IN ('roommates','parents','warden','group')),
        body TEXT NOT NULL CHECK (char_length(trim(body)) BETWEEN 1 AND 4000),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )`);
      await pool.query(`CREATE TABLE IF NOT EXISTS student_room_preferences (
        student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
        culture_preference VARCHAR(160), caste_preference VARCHAR(160), university_preference VARCHAR(240),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (student_user_id, hostel_id)
      )`);
      await pool.query(`CREATE TABLE IF NOT EXISTS complaint_attachments (
        id BIGSERIAL PRIMARY KEY, complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
        data_uri TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )`);
      await pool.query("CREATE INDEX IF NOT EXISTS student_hostel_chat_messages_hostel_channel_idx ON student_hostel_chat_messages (hostel_id, channel, created_at DESC)");
    })();
  }
  return studentHubTablesReady;
}

let bookingApplicationTableReady;
function ensureBookingApplicationTable() {
  if (!bookingApplicationTableReady) {
    bookingApplicationTableReady = pool.query(`CREATE TABLE IF NOT EXISTS booking_applications (
      booking_id INTEGER PRIMARY KEY REFERENCES bookings(id) ON DELETE CASCADE,
      student_name VARCHAR(160), student_email VARCHAR(254), student_phone VARCHAR(32), student_cnic VARCHAR(32),
      university VARCHAR(240), department VARCHAR(240), semester VARCHAR(32), gender VARCHAR(32), occupation VARCHAR(80), degree VARCHAR(80),
      parent_name VARCHAR(160), parent_phone VARCHAR(32), parent_cnic VARCHAR(32), parent_relation VARCHAR(80), parent_city VARCHAR(240), parent_email VARCHAR(254),
      secondary_guardian_name VARCHAR(160), secondary_guardian_phone VARCHAR(32), secondary_guardian_cnic VARCHAR(32), secondary_guardian_relation VARCHAR(80), secondary_guardian_email VARCHAR(254),
      special_requirements TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )`);
  }
  return bookingApplicationTableReady;
}

let parentStudentLinkTableReady;
function ensureParentStudentLinkTable() {
  if (!parentStudentLinkTableReady) {
    parentStudentLinkTableReady = (async () => {
      await pool.query(`CREATE TABLE IF NOT EXISTS parent_student_links (
        id SERIAL PRIMARY KEY, parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        invited_by_student_id INTEGER NOT NULL REFERENCES users(id), invitation_token_hash CHAR(64) NOT NULL UNIQUE,
        status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','active','revoked')),
        expires_at TIMESTAMPTZ NOT NULL, accepted_at TIMESTAMPTZ, revoked_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CHECK (parent_user_id <> student_user_id), CHECK ((status <> 'active') OR accepted_at IS NOT NULL), CHECK ((status <> 'revoked') OR revoked_at IS NOT NULL)
      )`);
      await pool.query("CREATE UNIQUE INDEX IF NOT EXISTS parent_student_links_active_or_pending_unique ON parent_student_links (parent_user_id, student_user_id) WHERE status IN ('pending', 'active')");
    })();
  }
  return parentStudentLinkTableReady;
}

async function studentConfirmedBooking(userId, hostelId) {
  return pool.query(
    `SELECT b.id,b.hostel_id,b.room_id,b.bed_number FROM bookings b
      WHERE b.user_id=$1 AND b.hostel_id=$2 AND b.status='confirmed' AND b.room_id IS NOT NULL
      ORDER BY b.updated_at DESC LIMIT 1`,
    [userId, hostelId],
  );
}

// â”€â”€ Auth token signing (HMAC-SHA256) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Tokens are opaque and forgeable unless they carry a signature that only
// the server can produce. AUTH_SECRET must be configured outside tests.
const AUTH_SECRET = process.env.AUTH_SECRET || (process.env.NODE_ENV === "test" ? "test-only-auth-secret" : null);
if (!AUTH_SECRET) throw new Error("AUTH_SECRET is required");
const AUTH_TOKEN_TTL_MINUTES = Math.max(5, Number(process.env.AUTH_TOKEN_TTL_MINUTES || 480));

function base64UrlEncode(input) {
  return Buffer.from(input).toString("base64url");
}
async function signAuthToken(user, req = null) {
  await authSessionsReady;
  const sessionId = crypto.randomUUID();
  const issuedAt = Date.now();
  const expiresAt = issuedAt + AUTH_TOKEN_TTL_MINUTES * 60 * 1000;
  await pool.query(
    "INSERT INTO auth_sessions (id,user_id,expires_at,user_agent,ip_address) VALUES ($1,$2,$3,$4,$5)",
    [sessionId, user.id, new Date(expiresAt), req?.headers?.["user-agent"] || null, req?.ip || null]
  );
  const payload = { id: user.id, role: user.role, sid: sessionId, iat: issuedAt, exp: expiresAt };
  const encodedPayload = base64UrlEncode(JSON.stringify(payload));
  const signature = crypto
    .createHmac("sha256", AUTH_SECRET)
    .update(encodedPayload)
    .digest("base64url");
  return `${encodedPayload}.${signature}`;
}
function verifyAuthToken(token) {
  if (!token || typeof token !== "string" || !token.includes(".")) return null;
  const [encodedPayload, signature] = token.split(".");
  if (!encodedPayload || !signature) return null;
  const expectedSignature = crypto
    .createHmac("sha256", AUTH_SECRET)
    .update(encodedPayload)
    .digest("base64url");
  const providedBuffer = Buffer.from(signature);
  const expectedBuffer = Buffer.from(expectedSignature);
  if (
    providedBuffer.length !== expectedBuffer.length ||
    !crypto.timingSafeEqual(providedBuffer, expectedBuffer)
  ) {
    return null;
  }
  try {
    const payload = JSON.parse(Buffer.from(encodedPayload, "base64url").toString("utf8"));
    if (!isValidInt(Number(payload.id)) || !payload.role || !payload.sid || !Number.isFinite(payload.exp) || payload.exp <= Date.now()) return null;
    return { id: Number(payload.id), role: String(payload.role), sessionId: String(payload.sid), expiresAt: new Date(payload.exp) };
  } catch {
    return null;
  }
}
function hashPasswordResetToken(token) {
  return crypto.createHash("sha256").update(token).digest("hex");
}
function passwordResetUrl(token) {
  const baseUrl = process.env.PASSWORD_RESET_URL || "staybuddy://reset-password";
  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}token=${encodeURIComponent(token)}`;
}
function parentInvitationUrl(token) {
  const baseUrl = process.env.PARENT_INVITATION_URL || "staybuddy://parent-invitation";
  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}token=${encodeURIComponent(token)}`;
}
async function sendPasswordResetEmail(user, token) {
  await transporter.sendMail({
    from: `"StayBuddy" <${process.env.EMAIL_USER}>`,
    to: user.email,
    subject: "Reset your StayBuddy password",
    text: `Use this link within ${PASSWORD_RESET_TTL_MINUTES} minutes to reset your password: ${passwordResetUrl(token)}`,
  });
}
async function sendParentInvitationEmail(parent, student, token) {
  await transporter.sendMail({
    from: `"StayBuddy" <${process.env.EMAIL_USER}>`,
    to: parent.email,
    subject: "Accept your StayBuddy parent invitation",
    text: `${student.name} invited you to access their StayBuddy accommodation information. Accept within ${PARENT_INVITATION_TTL_HOURS} hours: ${parentInvitationUrl(token)}`,
  });
}
// Express middleware: requires a valid signed token and attaches req.user = { id, role }.
async function requireAuth(req, res, next) {
  const auth = req.headers.authorization || "";
  const token = auth.replace("Bearer ", "").trim();
  const user = verifyAuthToken(token);
  if (!user) return res.status(401).json({ error: "Missing, expired, or invalid auth token" });
  try {
    const active = await pool.query(`SELECT u.id,u.role,u.account_status
      FROM auth_sessions s JOIN users u ON u.id=s.user_id
      WHERE s.id=$1 AND s.user_id=$2 AND s.revoked_at IS NULL AND s.expires_at>NOW() AND u.account_status='active'`, [user.sessionId, user.id]);
    if (!active.rowCount || active.rows[0].role !== user.role) return res.status(401).json({ error: "Session is revoked, expired, or disabled" });
    if (user.role === "warden") {
      const warden = await pool.query("SELECT 1 FROM warden_assignments wa LEFT JOIN warden_profiles wp ON wp.warden_user_id=wa.warden_user_id WHERE wa.warden_user_id=$1 AND COALESCE(lower(wp.account_status),'active')='active'", [user.id]);
      if (!warden.rowCount) return res.status(403).json({ error: "Warden account is disabled or unassigned" });
    }
    await pool.query("UPDATE auth_sessions SET last_seen_at=NOW() WHERE id=$1", [user.sessionId]);
    req.user = user;
    next();
  } catch (error) { next(error); }
}
function requireRole(...roles) {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: "You do not have permission for this action" });
    }
    next();
  };
}
async function ownerHasHostel(client, ownerUserId, hostelId) {
  const result = await client.query(
    "SELECT 1 FROM hostel_owners WHERE owner_user_id=$1 AND hostel_id=$2",
    [ownerUserId, hostelId]
  );
  return result.rowCount === 1;
}
async function wardenHasHostel(client, wardenUserId, hostelId) {
  const result = await client.query(
    "SELECT 1 FROM warden_assignments WHERE warden_user_id=$1 AND hostel_id=$2",
    [wardenUserId, hostelId]
  );
  return result.rowCount === 1;
}
async function parentHasStudent(client, parentUserId, studentUserId) {
  const result = await client.query(
    `SELECT 1 FROM parent_student_links
     WHERE parent_user_id=$1 AND student_user_id=$2 AND status='active'`,
    [parentUserId, studentUserId]
  );
  return result.rowCount === 1;
}
async function createNotification(client, userId, type, title, body, data, preferenceColumn) {
  const preferenceColumns = new Set(["booking_updates", "complaint_updates", "announcements", "attendance_updates", "emergency_alerts", "fee_reminders", "location_updates", "transport_updates", "communication_updates", "availability_updates"]);
  if (!preferenceColumns.has(preferenceColumn)) throw new Error("Invalid notification preference");
  await client.query(
    `INSERT INTO notifications (user_id, type, title, body, data)
     SELECT $1,$2,$3,$4,$5::jsonb
     WHERE COALESCE((SELECT ${preferenceColumn} FROM notification_preferences WHERE user_id=$1), true)`,
    [userId, type, title, body, JSON.stringify(data)]
  );
}

// ---------- helpers ----------
function isValidInt(n) {
  return Number.isInteger(n) && n > 0;
}
function generateOtp() {
  return Math.floor(1000 + Math.random() * 9000).toString();
}
function isValidDateInput(value) {
  return !Number.isNaN(Date.parse(value));
}
function isValidHostelPhone(value) {
  const compact = String(value || "").trim().replace(/[\s-]/g, "");
  return /^(?:03\d{9}|\+923\d{9})$/.test(compact);
}
function isValidHostelEmail(value) {
  const email = String(value || "").trim();
  return /^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$/.test(email);
}
function normalizeBookingStatus(status) {
  return String(status || "pending").trim().toLowerCase();
}

function isBookingStatus(status) {
  return ["pending", "confirmed", "rejected", "cancelled", "expired", "completed"].includes(status);
}
function isCapacityHoldingStatus(status) {
  return status === "confirmed";
}
async function releaseHostelCapacity(client, hostelId) {
  const result = await client.query(
    `UPDATE hostels
     SET available_capacity = available_capacity + 1
     WHERE id=$1
       AND available_capacity < COALESCE(source_available_capacity, total_capacity)
     RETURNING id`,
    [hostelId]
  );
  return result.rowCount === 1;
}
function isValidComplaintStatus(status) {
  return ["open", "in_progress", "resolved", "closed"].includes(status);
}
function canTransitionComplaintStatus(currentStatus, nextStatus) {
  const allowed = { open: ["in_progress", "resolved", "closed"], in_progress: ["resolved", "closed"], resolved: ["closed"], closed: [] };
  return Boolean(allowed[currentStatus] && allowed[currentStatus].includes(nextStatus));
}
function complaintResolutionInput(body) {
  const summary = String(body.safe_resolution_summary || "").trim();
  const actions = String(body.actions_taken || "").trim();
  const prevention = body.prevention_follow_up_notes == null ? null : String(body.prevention_follow_up_notes).trim();
  if (!summary || !actions || summary.length > 4000 || actions.length > 4000 || (prevention != null && prevention.length > 4000)) return null;
  return { summary, actions, prevention: prevention || null };
}
function priorityToSeverity(priority) {
  const map = { "High": "high", "Medium": "medium", "Low": "low" };
  return map[priority] || "medium";
}
const hostelResponseColumns = [
  "id", "external_id", "name", "address", "city", "university", "hostel_type", "area",
  "verified", "latitude", "longitude", "distance_from_fast_km", "price_tier",
  "single_room_price", "double_room_price", "dorm_room_price", "meal_included", "food_type",
  "food_rating", "room_types_available", "total_rooms", "total_capacity", "available_capacity",
  "amenities", "internet_speed_mbps", "noise_level", "curfew_hour", "study_environment_score",
  "description"
].join(", ");
const hostelResponseColumnsWithAlias = hostelResponseColumns
  .split(", ")
  .map((column) => `h.${column}`)
  .join(", ");
async function callAiComplaintCategorizer(description) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), Number(process.env.AI_TIMEOUT_MS || 5000));
  try {
    const response = await fetch(process.env.AI_COMPLAINT_URL || "http://127.0.0.1:8001/categorize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: description }),
      signal: controller.signal,
    });
    if (!response.ok) {
      console.warn("AI categorizer returned", response.status);
      return analyzeCsvComplaint(description);
    }
    const result = await response.json();
    return result?.success && result.category ? result : analyzeCsvComplaint(description);
  } catch (err) {
    console.warn("AI categorizer unavailable:", err.message);
    return analyzeCsvComplaint(description);
  } finally {
    clearTimeout(timeout);
  }
}
function fallbackComplaintTriage(description){const text=String(description).toLowerCase();if(/water|electric|power|plumb|leak/.test(text))return{category:"Maintenance",severity:/fire|spark|flood|danger/.test(text)?"high":"medium"};if(/food|meal|mess|kitchen/.test(text))return{category:"Food",severity:"medium"};if(/harass|threat|security|theft|unsafe/.test(text))return{category:"Security",severity:"high"};if(/noise|roommate|clean|hygiene/.test(text))return{category:"Resident Services",severity:"medium"};return{category:"Other",severity:"medium"};}

const complaintSimilarityStopWords = new Set(["about","after","again","been","being","from","have","into","just","more","that","their","there","these","they","this","with","your"]);
function complaintTokens(value) {
  return new Set(String(value || "").toLowerCase().split(/[^a-z0-9]+/).filter((token) => token.length >= 3 && !complaintSimilarityStopWords.has(token)));
}
function complaintSimilarity(left, right) {
  const leftTokens = complaintTokens(left), rightTokens = complaintTokens(right);
  if (!leftTokens.size || !rightTokens.size) return 0;
  let intersection = 0;
  for (const token of leftTokens) if (rightTokens.has(token)) intersection += 1;
  return intersection / new Set([...leftTokens, ...rightTokens]).size;
}
async function findSimilarRoomComplaints(client, { userId, hostelId, roomId, description, category }) {
  if (!roomId) return [];
  const candidates = await client.query(`SELECT id,user_id,category,severity,status,description,created_at
    FROM complaints
    WHERE hostel_id=$1 AND room_id=$2 AND status IN ('open','in_progress','resolved')
      AND created_at>=NOW()-INTERVAL '90 days'
    ORDER BY created_at DESC LIMIT 25`, [hostelId, roomId]);
  return candidates.rows.map((row) => {
    const textScore = complaintSimilarity(description, row.description);
    const score = Math.min(1, textScore * 0.75 + (String(row.category).toLowerCase() === String(category).toLowerCase() ? 0.25 : 0));
    return { id: row.id, category: row.category, severity: row.severity, status: row.status, same_student: row.user_id === userId, description: row.user_id === userId ? row.description : null, created_at: row.created_at, similarity: Number(score.toFixed(3)) };
  }).filter((row) => row.similarity >= 0.35).sort((left, right) => right.similarity - left.similarity).slice(0, 5);
}

const sensitiveRequestBuckets = new Map();
app.use((req, res, next) => {
  const sensitive = req.method === "POST" && (
    req.path.startsWith("/api/auth/") || req.path.includes("/invitations") ||
    req.path.includes("/documents") || req.path.includes("/evidence") ||
    req.path.includes("/attachments") || req.path.includes("/receipt-submissions") ||
    req.path.includes("/favorite-shares") || req.path.includes("/attendance/qr")
  );
  if (!sensitive) return next();
  const now = Date.now(); const windowMs = 15 * 60 * 1000; const maximum = 30;
  const key = `${req.ip}:${req.path}`; const prior = sensitiveRequestBuckets.get(key);
  const bucket = !prior || prior.resetAt <= now ? { count: 0, resetAt: now + windowMs } : prior;
  bucket.count += 1; sensitiveRequestBuckets.set(key, bucket);
  res.setHeader("RateLimit-Limit", maximum); res.setHeader("RateLimit-Remaining", Math.max(0, maximum - bucket.count));
  if (bucket.count > maximum) return res.status(429).json({ error: "Too many sensitive requests; try again later" });
  next();
});

registerAttendanceRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification });
registerWardenOperationsRoutes({ app, pool, requireAuth, requireRole, isValidInt });
registerOwnerOperationsRoutes({ app, pool, requireAuth, requireRole, isValidInt });
registerEmergencyRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification, enqueueOperationalJob, deliveryDedupeKey });
registerVerificationRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, documentStorage: createLocalDocumentStorage() });
registerFeeRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification, enqueueOperationalJob, deliveryDedupeKey, documentStorage: createLocalDocumentStorage() });
registerMessagingRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, documentStorage: createLocalDocumentStorage() });
registerLocationSharingRoutes({ app, pool, requireAuth, requireRole, isValidInt, createNotification });
registerComplaintRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification, documentStorage: createLocalDocumentStorage() });
registerAdminOperationsRoutes({ app, pool, requireAuth, requireRole });
registerPhase3Routes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, hashPassword });
registerParentPhase3Routes({ app, pool, requireAuth, requireRole, isValidInt });
registerParentServicesRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification });
registerDiscoveryCompletionRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel });
registerBookingDocumentRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, documentStorage: createLocalDocumentStorage() });
registerRagChatRoutes({ app, pool, requireAuth, requireRole, isValidInt });
registerQrAttendanceRoutes({ app, pool, requireAuth, requireRole, isValidInt, wardenHasHostel, createNotification });
registerPostedGateAttendanceRoutes({ app, pool, requireAuth, requireRole, isValidInt, ownerHasHostel, wardenHasHostel, createNotification });

// âœ… Test API
app.get("/", (req, res) => {
  res.json({ message: "StayBuddy API running âœ…" });
});

// âœ… Test DB Connection
app.get("/api/db-test", async (req, res) => {
  try {
    const result = await pool.query("SELECT NOW() as now");
    res.json({ db_time: result.rows[0].now });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Database connection failed" });
  }
});

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  OWNER ROUTES
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// Create owners table on startup
async function ensureOwnerTable() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS owners (
      id          SERIAL PRIMARY KEY,
      full_name   VARCHAR(120) NOT NULL,
      email       VARCHAR(120) NOT NULL UNIQUE,
      cnic        VARCHAR(20)  NOT NULL,
      city        VARCHAR(80)  NOT NULL,
      phone       VARCHAR(20)  NOT NULL,
      is_verified BOOLEAN      NOT NULL DEFAULT false,
      created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    )
  `);
  console.log("âœ… owners table ready");
}
// owners and ownership relationships are migration-managed; no startup DDL.



// ============================================================================
// GET OWNER WARDENS
// ============================================================================

app.get(
  "/api/owner/wardens",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    try {
      await operationalTablesReady;

      const result = await pool.query(
        `
        SELECT
          u.id,
          u.name,
          u.email,
          u.phone,

          wa.hostel_id,
          wa.assigned_at,

          h.name AS hostel_name,

          wp.cnic,
          wp.city,
          COALESCE(wp.shift, 'Morning') AS shift,
          COALESCE(wp.assigned_floors, '[]'::jsonb) AS floors,
          COALESCE(wp.permissions, '{}'::jsonb) AS permissions,
          COALESCE(wp.account_status, 'Active') AS account_status

        FROM warden_assignments wa

        INNER JOIN hostel_owners ho
          ON ho.hostel_id = wa.hostel_id

        INNER JOIN users u
          ON u.id = wa.warden_user_id

        INNER JOIN hostels h
          ON h.id = wa.hostel_id

        LEFT JOIN warden_profiles wp
          ON wp.warden_user_id = u.id

        WHERE ho.owner_user_id = $1

        ORDER BY wa.assigned_at DESC
        `,
        [req.user.id]
      );

      res.json(result.rows);
    } catch (error) {
      console.error(
        "âŒ GET OWNER WARDENS FAILED:",
        error
      );

      res.status(500).json({
        error: "Failed to fetch wardens",
        detail: error.message,
      });
    }
  }
);


// ============================================================================
// CREATE WARDEN ACCOUNT
// ============================================================================

app.post(
  "/api/owner/wardens",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    const client = await pool.connect();

    try {
      await operationalTablesReady;

      // ----------------------------------------------------------------------
      // REQUEST VALUES
      // ----------------------------------------------------------------------

      const hostelId =
          Number(req.body.hostel_id);

      const name =
          String(req.body.name || "").trim();

      const email =
          String(req.body.email || "")
            .trim()
            .toLowerCase();

      const phone =
          String(req.body.phone || "").trim();

      const cnic =
          String(req.body.cnic || "").trim();

      const city =
          String(req.body.city || "").trim();

      const shift =
          String(req.body.shift || "Morning").trim();

      const floors =
          Array.isArray(req.body.floors)
              ? req.body.floors
                  .map((item) => String(item))
              : [];

      const permissions =
          req.body.permissions &&
          typeof req.body.permissions === "object" &&
          !Array.isArray(req.body.permissions)
              ? req.body.permissions
              : {};

      // ----------------------------------------------------------------------
      // VALIDATION
      // ----------------------------------------------------------------------

      if (!isValidInt(hostelId)) {
        return res.status(400).json({
          error: "A valid hostel_id is required",
        });
      }

      if (!email) {
        return res.status(400).json({
          error: "Warden email is required",
        });
      }

      const emailPattern =
          /^[^\s@]+@[^\s@]+$/;

      if (!emailPattern.test(email)) {
        return res.status(400).json({
          error: "Please provide a valid email address",
        });
      }

      if (
        ![
          "Morning",
          "Evening",
          "Night",
        ].includes(shift)
      ) {
        return res.status(400).json({
          error: "Invalid Warden shift",
        });
      }

      // ----------------------------------------------------------------------
      // OWNER MUST OWN THIS HOSTEL
      // ----------------------------------------------------------------------

      const ownershipResult =
          await client.query(
        `
        SELECT
          h.id,
          h.name,
          h.city

        FROM hostel_owners ho

        INNER JOIN hostels h
          ON h.id = ho.hostel_id

        WHERE ho.owner_user_id = $1
          AND ho.hostel_id = $2

        LIMIT 1
        `,
        [
          req.user.id,
          hostelId,
        ]
      );

      if (ownershipResult.rowCount === 0) {
        return res.status(403).json({
          error:
              "You are not allowed to create a Warden for this hostel",
        });
      }

      const hostel =
          ownershipResult.rows[0];

      // ----------------------------------------------------------------------
      // START TRANSACTION
      // ----------------------------------------------------------------------

      await client.query("BEGIN");

      // ----------------------------------------------------------------------
      // MAKE SURE EMAIL DOES NOT ALREADY EXIST
      // ----------------------------------------------------------------------

      const existingUser =
          await client.query(
        `
        SELECT
          id,
          email,
          role,
          name,
          phone

        FROM users

        WHERE LOWER(email) = LOWER($1)

        LIMIT 1
        `,
        [email]
      );

      if (existingUser.rowCount > 0 && existingUser.rows[0].role !== "warden") {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "This email belongs to a non-warden account" });
      }
      if (existingUser.rowCount > 0) {
        const currentAssignment = await client.query(
          "SELECT hostel_id FROM warden_assignments WHERE warden_user_id=$1 FOR UPDATE",
          [existingUser.rows[0].id]
        );
        if (currentAssignment.rowCount && Number(currentAssignment.rows[0].hostel_id) !== hostelId) {
          await client.query("ROLLBACK");
          return res.status(409).json({ error: "This warden is already assigned to another hostel; reassign the existing warden instead" });
        }
      }
      if (existingUser.rowCount === 0 && (!name || !phone || !cnic || !city || floors.length === 0)) {
        await client.query("ROLLBACK");
        return res.status(400).json({ error: "New wardens require name, phone, CNIC, city and at least one assigned floor" });
      }

      // ----------------------------------------------------------------------
      // GENERATE TEMP PASSWORD
      //
      // Example:
      // SB@4F8C219Ax
      // ----------------------------------------------------------------------

      const randomPart =
          crypto
            .randomBytes(5)
            .toString("hex")
            .toUpperCase();

      const temporaryPassword = existingUser.rowCount === 0 ? `SB@${randomPart}x` : null;

      // ----------------------------------------------------------------------
      // HASH TEMP PASSWORD
      // ----------------------------------------------------------------------

      const passwordHash = temporaryPassword ? await hashPassword(temporaryPassword) : null;

      // ----------------------------------------------------------------------
      // CREATE USER
      // ----------------------------------------------------------------------

      const userResult = existingUser.rowCount
        ? await client.query(
            `UPDATE users SET name=$1,phone=$2,account_status='active' WHERE id=$3
             RETURNING id,name,email,phone,role`,
            [name || existingUser.rows[0].name, phone || existingUser.rows[0].phone, existingUser.rows[0].id]
          )
        : await client.query(
            `INSERT INTO users (name,email,phone,password_hash,role)
             VALUES ($1,$2,$3,$4,'warden') RETURNING id,name,email,phone,role`,
            [name, email, phone, passwordHash]
          );

      const warden =
          userResult.rows[0];

      // ----------------------------------------------------------------------
      // CREATE WARDEN PROFILE
      // ----------------------------------------------------------------------

      await client.query(
        `
        INSERT INTO warden_profiles (
          warden_user_id,
          cnic,
          city,
          shift,
          assigned_floors,
          permissions,
          account_status
        )
        VALUES (
          $1,
          $2,
          $3,
          $4,
          $5::jsonb,
          $6::jsonb,
          'Active'
        )
        ON CONFLICT (warden_user_id) DO UPDATE SET
          cnic=COALESCE(NULLIF(EXCLUDED.cnic,''),warden_profiles.cnic),
          city=COALESCE(NULLIF(EXCLUDED.city,''),warden_profiles.city),
          shift=EXCLUDED.shift,
          assigned_floors=CASE WHEN EXCLUDED.assigned_floors='[]'::jsonb THEN warden_profiles.assigned_floors ELSE EXCLUDED.assigned_floors END,
          permissions=EXCLUDED.permissions,
          account_status='Active'
        `,
        [
          warden.id,
          cnic,
          city,
          shift,
          JSON.stringify(floors),
          JSON.stringify(permissions),
        ]
      );

      // ----------------------------------------------------------------------
      // ASSIGN WARDEN TO EXACT HOSTEL
      // ----------------------------------------------------------------------

      const assignmentResult =
          await client.query(
        `
        INSERT INTO warden_assignments (
          warden_user_id,
          hostel_id,
          assigned_by_user_id
        )
        VALUES (
          $1,
          $2,
          $3
        )
        ON CONFLICT (warden_user_id,hostel_id) DO UPDATE SET
          assigned_by_user_id=EXCLUDED.assigned_by_user_id,
          assigned_at=NOW()

        RETURNING
          warden_user_id,
          hostel_id,
          assigned_by_user_id,
          assigned_at
        `,
        [
          warden.id,
          hostelId,
          req.user.id,
        ]
      );

      // ----------------------------------------------------------------------
      // COMMIT
      // ----------------------------------------------------------------------

      await client.query("COMMIT");

      console.log(
        "âœ… WARDEN CREATED:",
        {
          wardenId: warden.id,
          email: warden.email,
          hostelId,
          hostelName: hostel.name,
          ownerUserId: req.user.id,
        }
      );

      // ----------------------------------------------------------------------
      // RESPONSE
      //
      // Plain temp password is returned ONLY HERE.
      // Database only contains password_hash.
      // ----------------------------------------------------------------------

      res.status(201).json({
        success: true,

        message:
            "Warden account created successfully",

        warden: {
          ...warden,

          cnic,
          city,
          shift,
          floors,
          permissions,

          account_status:
              "Active",
        },

        hostel,

        assignment:
            assignmentResult.rows[0],

        temp_password: temporaryPassword,
        existing_account_reused: existingUser.rowCount > 0,
      });
    } catch (error) {
      try {
        await client.query("ROLLBACK");
      } catch (_) {
        // ignore rollback error
      }

      if (error.code === "23505") return res.status(409).json({ error: "This warden already has a hostel assignment" });

      console.error(
        "âŒ CREATE WARDEN FAILED:",
        error
      );

      res.status(500).json({
        error:
            "Failed to create Warden account",

        detail:
            error.message,
      });
    } finally {
      client.release();
    }
  }
);


// POST /api/owner/send-otp
app.post("/api/owner/send-otp", async (req, res) => {
  try {
    const {
      full_name,
      email,
      password,
      cnic,
      city,
      phone,
    } = req.body;

    const cleanEmail = String(email || "").trim().toLowerCase();

    if (!full_name || !cleanEmail || !password || !cnic || !city || !phone) {
      return res.status(400).json({
        error: "All fields including password are required",
      });
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!emailRegex.test(cleanEmail)) {
      return res.status(400).json({
        error: "Invalid email address",
      });
    }

    if (!isPassword(password)) {
      return res.status(400).json({
        error: "Password must be at least 8 characters long",
      });
    }

    const existingUser = await pool.query(
      "SELECT id, role FROM users WHERE LOWER(email)=LOWER($1)",
      [cleanEmail]
    );

    if (existingUser.rows.length > 0) {
      return res.status(409).json({
        error: "This email already has an account. Please login instead.",
      });
    }

    const existingOwner = await pool.query(
      "SELECT id, is_verified FROM owners WHERE LOWER(email)=LOWER($1)",
      [cleanEmail]
    );

    if (
      existingOwner.rows.length > 0 &&
      existingOwner.rows[0].is_verified
    ) {
      return res.status(409).json({
        error: "This owner is already registered. Please login instead.",
      });
    }

    if (existingOwner.rows.length > 0) {
      await pool.query(
        `UPDATE owners
         SET full_name=$1,
             cnic=$2,
             city=$3,
             phone=$4
         WHERE LOWER(email)=LOWER($5)`,
        [
          String(full_name).trim(),
          String(cnic).trim(),
          String(city).trim(),
          String(phone).trim(),
          cleanEmail,
        ]
      );
    } else {
      await pool.query(
        `INSERT INTO owners
          (full_name, email, cnic, city, phone, is_verified)
         VALUES ($1,$2,$3,$4,$5,false)`,
        [
          String(full_name).trim(),
          cleanEmail,
          String(cnic).trim(),
          String(city).trim(),
          String(phone).trim(),
        ]
      );
    }

    const passwordHash = await hashPassword(password);

    const otp = generateOtp();

    otpStore.set(cleanEmail, {
      otp,
      passwordHash,
      expiresAt: Date.now() + 10 * 60 * 1000,
    });

    console.log(`OTP for ${cleanEmail}: ${otp}`);

    await transporter.sendMail({
      from: `"StayBuddy" <${process.env.EMAIL_USER}>`,
      to: cleanEmail,
      subject: "Your StayBuddy OTP Code",
      html: `
        <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:32px;">
          <h2 style="color:#0B7C80;">StayBuddy Owner Verification</h2>
          <p>Hello <strong>${full_name}</strong>,</p>
          <p>Your verification code is:</p>
          <h1 style="color:#0B7C80;letter-spacing:10px;">${otp}</h1>
          <p>This code expires in 10 minutes.</p>
        </div>
      `,
    });

    return res.json({
      success: true,
      message: "OTP sent to your email",
    });
  } catch (err) {
    console.error("Error sending owner OTP:", err);

    return res.status(500).json({
      error: "Failed to send OTP: " + err.message,
    });
  }
});


// POST /api/owner/verify-otp
app.post("/api/owner/verify-otp", async (req, res) => {
  const client = await pool.connect();

  try {
    const cleanEmail = String(req.body.email || "")
      .trim()
      .toLowerCase();

    const otp = String(req.body.otp || "").trim();

    if (!cleanEmail || !otp) {
      return res.status(400).json({
        error: "email and otp are required",
      });
    }

    const record = otpStore.get(cleanEmail);

    if (!record) {
      return res.status(400).json({
        error: "No OTP found. Please request a new one.",
      });
    }

    if (Date.now() > record.expiresAt) {
      otpStore.delete(cleanEmail);

      return res.status(400).json({
        error: "OTP expired. Please request a new one.",
      });
    }

    if (record.otp !== otp) {
      return res.status(400).json({
        error: "Incorrect OTP. Please try again.",
      });
    }

    await client.query("BEGIN");

    const ownerResult = await client.query(
      `UPDATE owners
       SET is_verified=true
       WHERE LOWER(email)=LOWER($1)
       RETURNING id, full_name, email, city, phone, created_at`,
      [cleanEmail]
    );

    if (ownerResult.rows.length === 0) {
      await client.query("ROLLBACK");

      return res.status(404).json({
        error: "Owner not found.",
      });
    }

    const owner = ownerResult.rows[0];

    const userResult = await client.query(
      `INSERT INTO users
         (name, email, password_hash, role, phone)
       VALUES ($1,$2,$3,'owner',$4)
       RETURNING id, name, email, role, phone`,
      [
        owner.full_name,
        cleanEmail,
        record.passwordHash,
        owner.phone,
      ]
    );

    await client.query("COMMIT");

    otpStore.delete(cleanEmail);

    const user = userResult.rows[0];
    const token = await signAuthToken(user, req);

    return res.json({
      success: true,
      message: "Owner verified and registered successfully",
      owner,
      user,
      token,
    });
  } catch (err) {
    try {
      await client.query("ROLLBACK");
    } catch (_) {}

    console.error("Error verifying owner OTP:", err);

    return res.status(500).json({
      error: "Failed to verify OTP: " + err.message,
    });
  } finally {
    client.release();
  }
});

// POST /api/owner/login
app.post("/api/owner/login", async (req, res) => {
  try {
    const { email } = req.body;
    if (!email) return res.status(400).json({ error: "email is required" });

    const result = await pool.query(
      "SELECT id, full_name, email, city, is_verified FROM owners WHERE email=$1",
      [email]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "No owner account found with this email." });
    }
    if (!result.rows[0].is_verified) {
      return res.status(403).json({ error: "Account not verified. Please complete OTP verification." });
    }

    res.json({ success: true, owner: result.rows[0] });
  } catch (err) {
    res.status(500).json({ error: "Login failed" });
  }
});

// GET /api/owners (view all owners)
app.get("/api/owners", async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT id, full_name, email, cnic, city, phone, is_verified, created_at FROM owners ORDER BY created_at DESC"
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch owners" });
  }
});

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  EXISTING ROUTES
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

app.get("/api/hostels", async (req, res) => {
  try {
    const result = await pool.query(`SELECT ${hostelResponseColumns} FROM hostels h WHERE h.publication_status='published' ORDER BY h.id ASC`);
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch hostels" }); }
});

// Discovery map data comes only from hostel records created in StayBuddy.
// A map pin is returned only when its owner saved a real map location.
app.get("/api/student/hostels", requireAuth, requireRole("student"), async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT ${hostelResponseColumnsWithAlias}
       FROM hostels h
       WHERE h.latitude IS NOT NULL
         AND h.longitude IS NOT NULL
       ORDER BY h.created_at DESC, h.id DESC`
    );
    res.json(result.rows);
  } catch (err) {
    console.error("STUDENT HOSTEL DISCOVERY FAILED:", err);
    res.status(500).json({ error: "Failed to load registered hostels" });
  }
});

app.get("/api/hostels/:id", requireAuth, requireRole("student"), async (req, res, next) => {
  try {
    if (["search", "compare"].includes(req.params.id)) return next();
    const hostelId = parseInt(req.params.id, 10);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    const hostelQ = await pool.query(
      `SELECT ${hostelResponseColumns},
              h.facilities, h.rules, h.mess_info,
              h.contact_phone, h.contact_email
         FROM hostels h
        WHERE h.id=$1`,
      [hostelId],
    );
    if (hostelQ.rows.length === 0) return res.status(404).json({ error: "Hostel not found" });
    const reviewsQ = await pool.query("SELECT id, user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review, created_at FROM reviews WHERE hostel_id=$1 ORDER BY created_at DESC", [hostelId]);
    res.json({ ...hostelQ.rows[0], reviews: reviewsQ.rows });
  } catch (err) { res.status(500).json({ error: "Failed to fetch hostel details" }); }
});

// GET /api/hostels/search/by-city
app.get("/api/hostels/search/by-city", async (req, res) => {
  try {
    const { city, min_price, max_price } = req.query;
    if (!city) return res.status(400).json({ error: "city query parameter is required" });

    const params = [`%${city}%`];
    let query = `SELECT ${hostelResponseColumns} FROM hostels WHERE publication_status='published' AND available_capacity>0 AND city ILIKE $1`;
    for (const [price, operator] of [[min_price, ">="], [max_price, "<="]]) {
      if (price == null) continue;
      const parsed = Number(price);
      if (!Number.isFinite(parsed) || parsed < 0) {
        return res.status(400).json({ error: "min_price and max_price must be non-negative numbers" });
      }
      params.push(parsed);
      query += ` AND single_room_price ${operator} $${params.length}`;
    }
    query += " ORDER BY name ASC";
    const result = await pool.query(query, params);
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to search hostels by city: " + err.message }); }
});

// GET /api/hostels/search/by-university
app.get("/api/hostels/search/by-university", async (req, res) => {
  try {
    const { university } = req.query;
    if (!university) return res.status(400).json({ error: "university query parameter is required" });
    
    const result = await pool.query(
      `SELECT ${hostelResponseColumns} FROM hostels WHERE publication_status='published' AND available_capacity>0 AND university ILIKE $1 ORDER BY name ASC`,
      [`%${university}%`]
    );
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to search hostels by university: " + err.message }); }
});

// GET /api/hostels/search/nearby (GPS proximity search)
app.get("/api/hostels/search/nearby", async (req, res) => {
  try {
    const { latitude, longitude, radius_km } = req.query;
    
    if (!latitude || !longitude) {
      return res.status(400).json({ error: "latitude and longitude query parameters are required" });
    }
    
    const lat = parseFloat(latitude);
    const lon = parseFloat(longitude);
    const radiusKm = parseFloat(radius_km) || 5;
    
    if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      return res.status(400).json({ error: "Invalid latitude or longitude values" });
    }
    
    const result = await pool.query(`
      SELECT * FROM (
        SELECT
          ${hostelResponseColumns},
          (6371 * acos(cos(radians($1)) * cos(radians(latitude)) *
          cos(radians(longitude) - radians($2)) +
          sin(radians($1)) * sin(radians(latitude)))) AS distance_km
        FROM hostels
        WHERE publication_status='published' AND available_capacity>0 AND latitude IS NOT NULL AND longitude IS NOT NULL
      ) AS nearby_hostels
      WHERE distance_km <= $3
      ORDER BY distance_km ASC
    `, [lat, lon, radiusKm]);
    
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to search nearby hostels: " + err.message }); }
});

// GET /api/hostels/search/filter
app.get("/api/hostels/search/filter", async (req, res) => {
  try {
    const { city, university, hostel_type, verified, min_capacity, max_capacity, min_price, max_price, amenities, search_text } = req.query;

    let query = `SELECT ${hostelResponseColumns} FROM hostels WHERE publication_status='published' AND available_capacity>0`;
    let params = [];
    
    if (city) {
      query += ` AND city ILIKE $${params.length + 1}`;
      params.push(`%${city}%`);
    }
    if (university) {
      query += ` AND university ILIKE $${params.length + 1}`;
      params.push(`%${university}%`);
    }
    if (hostel_type) {
      query += ` AND hostel_type ILIKE $${params.length + 1}`;
      params.push(`%${hostel_type}%`);
    }
    if (verified != null) {
      if (String(verified).toLowerCase() !== "true") return res.status(400).json({ error: "Public discovery only returns verified hostels" });
    }
    if (min_capacity) {
      const minCap = parseInt(min_capacity, 10);
      if (isValidInt(minCap)) {
        query += ` AND available_capacity >= $${params.length + 1}`;
        params.push(minCap);
      }
    }
    if (max_capacity) {
      const maxCap = parseInt(max_capacity, 10);
      if (isValidInt(maxCap)) {
        query += ` AND available_capacity <= $${params.length + 1}`;
        params.push(maxCap);
      }
    }
    for (const [price, operator] of [[min_price, ">="], [max_price, "<="]]) {
      if (price == null) continue;
      const parsed = Number(price);
      if (!Number.isFinite(parsed) || parsed < 0) {
        return res.status(400).json({ error: "min_price and max_price must be non-negative numbers" });
      }
      query += ` AND single_room_price ${operator} $${params.length + 1}`;
      params.push(parsed);
    }
    if (amenities) {
      const amenityList = String(amenities)
        .split(",")
        .map((amenity) => amenity.trim())
        .filter(Boolean);
      if (amenityList.length === 0) return res.status(400).json({ error: "amenities must contain one or more values" });
      query += ` AND amenities @> $${params.length + 1}::jsonb`;
      params.push(JSON.stringify(amenityList));
    }
    if (search_text) {
      query += ` AND (name ILIKE $${params.length + 1} OR address ILIKE $${params.length + 1} OR area ILIKE $${params.length + 1} OR description ILIKE $${params.length + 1})`;
      const searchPattern = `%${search_text}%`;
      params.push(searchPattern);
    }
    
    query += " ORDER BY name ASC";
    const result = await pool.query(query, params);
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to filter hostels: " + err.message }); }
});

// GET /api/hostels/compare?ids=1,2
app.get("/api/hostels/compare", async (req, res) => {
  try {
    const ids = String(req.query.ids || "")
      .split(",")
      .map((value) => parseInt(value.trim(), 10));

    if (ids.length < 2 || ids.length > 3 || ids.some((id) => !isValidInt(id)) || new Set(ids).size !== ids.length) {
      return res.status(400).json({ error: "ids must contain two or three unique positive hostel IDs" });
    }

    const result = await pool.query(`
      SELECT ${hostelResponseColumnsWithAlias},
             ROUND(AVG(r.overall_rating)::numeric, 2) AS average_rating,
             COUNT(r.id)::integer AS review_count
      FROM hostels h
      LEFT JOIN reviews r ON r.hostel_id = h.id
      WHERE h.id = ANY($1::int[]) AND h.publication_status='published'
      GROUP BY h.id
      ORDER BY array_position($1::int[], h.id)
    `, [ids]);

    if (result.rows.length !== ids.length) {
      return res.status(404).json({ error: "One or more hostels were not found" });
    }

    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to compare hostels: " + err.message }); }
});

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  FAVORITES ROUTES
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// POST /api/favorites (add to favorites)
app.post("/api/favorites", requireAuth, async (req, res) => {
  try {
    const { hostel_id } = req.body;
    const uid = req.user.id;
    const hid = parseInt(hostel_id, 10);
    
    if (!isValidInt(hid)) {
      return res.status(400).json({ error: "hostel_id must be a positive integer" });
    }
    
    const userResult = await pool.query("SELECT id FROM users WHERE id=$1", [uid]);
    if (userResult.rows.length === 0) return res.status(404).json({ error: "User not found" });
    
    const hostelResult = await pool.query(`SELECT h.id,h.available_capacity,
      (SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id=h.id AND available_capacity>0) AS display_price
      FROM hostels h WHERE h.id=$1 AND h.publication_status='published'`, [hid]);
    if (hostelResult.rows.length === 0) return res.status(404).json({ error: "Hostel not found" });
    const favoriteCount = await pool.query("SELECT COUNT(*)::int AS count FROM favorites WHERE user_id=$1", [uid]);
    if (favoriteCount.rows[0].count >= 100) return res.status(409).json({ error: "Favorites limit reached (100)" });
    
    const result = await pool.query(
      "INSERT INTO favorites (user_id,hostel_id,price_snapshot,availability_snapshot) VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING RETURNING *",
      [uid, hid, hostelResult.rows[0].display_price, hostelResult.rows[0].available_capacity]
    );
    
    if (result.rows.length === 0) {
      return res.status(409).json({ error: "Hostel is already in favorites" });
    }
    
    res.status(201).json({ message: "Hostel added to favorites âœ…", favorite: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to add favorite: " + err.message }); }
});

// DELETE /api/favorites/:user_id/:hostel_id (remove from favorites)
app.delete("/api/favorites/:user_id/:hostel_id", requireAuth, async (req, res) => {
  try {
    const uid = parseInt(req.params.user_id, 10);
    const hid = parseInt(req.params.hostel_id, 10);
    
    if (!isValidInt(uid) || !isValidInt(hid)) {
      return res.status(400).json({ error: "user_id and hostel_id must be positive integers" });
    }
    if (uid !== req.user.id) {
      return res.status(403).json({ error: "Cannot modify another user's favorites" });
    }
    
    const result = await pool.query(
      "DELETE FROM favorites WHERE user_id=$1 AND hostel_id=$2 RETURNING *",
      [uid, hid]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: "Favorite not found" });
    }
    
    res.json({ message: "Hostel removed from favorites âœ…", favorite: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to remove favorite: " + err.message }); }
});

// GET /api/favorites/:user_id (get user's favorites)
app.get("/api/favorites/:user_id", requireAuth, async (req, res) => {
  try {
    const uid = parseInt(req.params.user_id, 10);
    if (!isValidInt(uid)) return res.status(400).json({ error: "Invalid user id" });
    if (uid !== req.user.id) {
      return res.status(403).json({ error: "Cannot view another user's favorites" });
    }
    
    const userResult = await pool.query("SELECT id FROM users WHERE id=$1", [uid]);
    if (userResult.rows.length === 0) return res.status(404).json({ error: "User not found" });
    
    const result = await pool.query(`
            SELECT f.id AS favorite_id, f.user_id, f.created_at, f.updated_at, f.note, f.price_snapshot, f.availability_snapshot,
              h.id AS hostel_id, ${hostelResponseColumnsWithAlias}
      FROM favorites f
      JOIN hostels h ON f.hostel_id = h.id
      WHERE f.user_id=$1
      ORDER BY f.created_at DESC
    `, [uid]);
    
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch favorites: " + err.message }); }
});

app.get("/api/users/:id", requireAuth, async (req, res) => {
  try {
    const userId = parseInt(req.params.id, 10);
    if (!isValidInt(userId)) return res.status(400).json({ error: "Invalid user id" });
    if (userId !== req.user.id) return res.status(403).json({ error: "Cannot view another user's account" });
    const userQ = await pool.query("SELECT id, name, email, role, created_at FROM users WHERE id=$1", [userId]);
    if (userQ.rows.length === 0) return res.status(404).json({ error: "User not found" });
    res.json(userQ.rows[0]);
  } catch (err) { res.status(500).json({ error: "Failed to fetch user profile" }); }
});

app.get("/api/interactions", requireAuth, async (req, res) => {
  try {
    const result = await pool.query("SELECT id, user_id, hostel_id, action_type, created_at FROM interactions WHERE user_id=$1 ORDER BY created_at DESC", [req.user.id]);
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch interactions" }); }
});

app.post("/api/interactions", requireAuth, async (req, res) => {
  try {
    const { hostel_id, action_type } = req.body;
    const uid = req.user.id;
    const hid = hostel_id == null ? null : parseInt(hostel_id, 10);
    if (hid !== null && !isValidInt(hid)) return res.status(400).json({ error: "hostel_id must be a positive integer or null" });
    if (!action_type) return res.status(400).json({ error: "action_type is required" });
    const result = await pool.query("INSERT INTO interactions (user_id, hostel_id, action_type) VALUES ($1,$2,$3) RETURNING *", [uid, hid, action_type]);
    res.status(201).json({ message: "Interaction saved âœ…", interaction: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to save interaction" }); }
});

app.get("/api/bookings", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;
    const result = await pool.query(
      `SELECT b.id, b.user_id, b.hostel_id, b.room_id, b.check_in, b.check_out, b.status, b.created_at,
              h.name AS hostel_name, h.city AS hostel_city, h.area AS hostel_area, h.single_room_price
       FROM bookings b
       JOIN hostels h ON h.id = b.hostel_id
       WHERE b.user_id=$1
       ORDER BY b.created_at DESC, b.id DESC`,
      [userId]
    );
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch bookings" }); }
});

app.post("/api/bookings", requireAuth, requireRole("student"), async (req, res) => {
  const client = await pool.connect();
  try {
    const { hostel_id, check_in, check_out, requested_room_type } = req.body;
    const application = req.body?.application && typeof req.body.application === 'object' && !Array.isArray(req.body.application) ? req.body.application : null;
    const userId = req.user.id;
    const hostelId = parseInt(hostel_id, 10);
    const idempotencyKey = String(req.headers["idempotency-key"] || "").trim();

    if (!isValidInt(hostelId)) {
      return res.status(400).json({ error: "hostel_id must be a positive integer" });
    }
    if (!idempotencyKey || idempotencyKey.length > 160) return res.status(400).json({ error: "An idempotency-key header of up to 160 characters is required" });
    if (req.body.status != null && normalizeBookingStatus(req.body.status) !== "pending") return res.status(400).json({ error: "Booking status is controlled by the server" });
    if (!check_in || !check_out || !isValidDateInput(check_in) || !isValidDateInput(check_out)) {
      return res.status(400).json({ error: "check_in and check_out must be valid dates" });
    }
    if (new Date(check_out) <= new Date(check_in)) {
      return res.status(400).json({ error: "check_out must be after check_in" });
    }
    const roomType = requested_room_type == null ? null : String(requested_room_type).trim().toLowerCase();
    // Room records created by the owner may use labels such as "2 Seater".
    // Keep the value normalized, but allow those persisted labels as well as
    // the legacy canonical values used by older clients.
    const isSeaterLabel = roomType && /^\d+\s*seater$/.test(roomType);
    if (roomType && !["single","double","triple","dorm","dormitory","shared"].includes(roomType) && !isSeaterLabel) return res.status(400).json({ error: "requested_room_type must be a supported room type" });

    await ensureBookingApplicationTable();
    await client.query("BEGIN");

    const previous = await client.query("SELECT * FROM bookings WHERE user_id=$1 AND idempotency_key=$2", [userId, idempotencyKey]);
    if (previous.rowCount) { await client.query("COMMIT"); return res.json({ booking: previous.rows[0], duplicate: true }); }

    const hostelResult = await client.query(
      "SELECT h.id,h.name FROM hostels h WHERE h.id=$1 FOR UPDATE",
      [hostelId]
    );
    if (hostelResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Hostel not found" });
    }

    const userResult = await client.query(
      "SELECT id FROM users WHERE id=$1",
      [userId]
    );
    if (userResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "User not found" });
    }

    // This is an enquiry, not a room allocation.  The owner first reviews it
    // and gives the student their hostel credentials.  Only that credentialed
    // student can later select a floor, room and bed.
    const bookingResult = await client.query(
      `INSERT INTO bookings (user_id,hostel_id,check_in,check_out,status,idempotency_key,requested_room_type,currency,updated_at)
       VALUES ($1,$2,$3,$4,'pending',$5,$6,'PKR',NOW()) RETURNING *`,
      [userId, hostelId, check_in, check_out, idempotencyKey, roomType]
    );
    if (application) {
      const value = (key, max = 240) => String(application[key] || '').trim().slice(0, max) || null;
      await client.query(
        `INSERT INTO booking_applications (booking_id,student_name,student_email,student_phone,student_cnic,university,department,semester,gender,occupation,degree,parent_name,parent_phone,parent_cnic,parent_relation,parent_city,parent_email,secondary_guardian_name,secondary_guardian_phone,secondary_guardian_cnic,secondary_guardian_relation,secondary_guardian_email,special_requirements)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23)`,
        [bookingResult.rows[0].id, value('student_name',160), value('student_email',254), value('student_phone',32), value('student_cnic',32), value('university'), value('department'), value('semester',32), value('gender',32), value('occupation',80), value('degree',80), value('parent_name',160), value('parent_phone',32), value('parent_cnic',32), value('parent_relation',80), value('parent_city'), value('parent_email',254), value('secondary_guardian_name',160), value('secondary_guardian_phone',32), value('secondary_guardian_cnic',32), value('secondary_guardian_relation',80), value('secondary_guardian_email',254), value('special_requirements',2000)]
      );
    }
    await client.query("INSERT INTO booking_status_history (booking_id,previous_status,next_status,actor_user_id,reason) VALUES ($1,NULL,'pending',$2,'student hostel enquiry')", [bookingResult.rows[0].id, userId]);

    await client.query("COMMIT");
    res.status(201).json({
      message: "Booking request sent to the owner",
      booking: bookingResult.rows[0],
    });
  } catch (err) {
    await client.query("ROLLBACK");
    if (err.code === "23505") {
      const previous = await pool.query("SELECT * FROM bookings WHERE user_id=$1 AND idempotency_key=$2", [req.user.id, String(req.headers["idempotency-key"] || "")]);
      if (previous.rowCount) return res.json({ booking: previous.rows[0], duplicate: true });
    }
    res.status(500).json({ error: "Failed to create booking: " + err.message });
  } finally {
    client.release();
  }
});

app.put("/api/bookings/:id", requireAuth, async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = parseInt(req.params.id, 10);
    if (!isValidInt(bookingId)) {
      return res.status(400).json({ error: "Invalid booking id" });
    }

    const { check_in, check_out, status } = req.body;
    const nextStatus = status == null ? null : normalizeBookingStatus(status);

    if (check_in != null && !isValidDateInput(check_in)) {
      return res.status(400).json({ error: "check_in must be a valid date" });
    }
    if (check_out != null && !isValidDateInput(check_out)) {
      return res.status(400).json({ error: "check_out must be a valid date" });
    }
    if (nextStatus != null && !isBookingStatus(nextStatus)) {
      return res.status(400).json({ error: "status must be one of pending, confirmed, cancelled, completed" });
    }

    await client.query("BEGIN");

    const bookingResult = await client.query(
      "SELECT id, user_id, hostel_id, room_id, check_in, check_out, status FROM bookings WHERE id=$1 FOR UPDATE",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Booking not found" });
    }

    const booking = bookingResult.rows[0];
    if (booking.user_id !== req.user.id) {
      await client.query("ROLLBACK");
      return res.status(403).json({ error: "Cannot update another user's booking" });
    }
    if (status != null) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "Booking status is controlled by hostel staff" });
    }
    if (normalizeBookingStatus(booking.status) !== "pending") { await client.query("ROLLBACK"); return res.status(409).json({ error: "Only pending booking dates can be edited" }); }
    const updatedCheckIn = check_in || booking.check_in;
    const updatedCheckOut = check_out || booking.check_out;
    const updatedStatus = nextStatus || normalizeBookingStatus(booking.status);

    if (new Date(updatedCheckOut) <= new Date(updatedCheckIn)) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "check_out must be after check_in" });
    }

    const currentHoldsCapacity = isCapacityHoldingStatus(normalizeBookingStatus(booking.status));
    const nextHoldsCapacity = isCapacityHoldingStatus(updatedStatus);

    if (!currentHoldsCapacity && nextHoldsCapacity) {
      const hostelResult = await client.query(
        "SELECT id, available_capacity FROM hostels WHERE id=$1 FOR UPDATE",
        [booking.hostel_id]
      );
      if (hostelResult.rows.length === 0) {
        await client.query("ROLLBACK");
        return res.status(404).json({ error: "Hostel not found" });
      }
      if (Number(hostelResult.rows[0].available_capacity) <= 0) {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "No beds available for this hostel" });
      }
      await client.query(
        "UPDATE hostels SET available_capacity = available_capacity - 1 WHERE id=$1",
        [booking.hostel_id]
      );
      if (booking.room_id != null) {
        const roomResult = await client.query(
          "SELECT id, available_capacity FROM rooms WHERE id=$1 FOR UPDATE",
          [booking.room_id]
        );
        if (roomResult.rows.length === 0 || Number(roomResult.rows[0].available_capacity) <= 0) {
          await client.query("ROLLBACK");
          return res.status(409).json({ error: "Assigned room has no available capacity" });
        }
        await client.query(
          "UPDATE rooms SET available_capacity = available_capacity - 1 WHERE id=$1",
          [booking.room_id]
        );
      }
    } else if (currentHoldsCapacity && !nextHoldsCapacity) {
      if (!await releaseHostelCapacity(client, booking.hostel_id)) {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "Hostel availability is already fully restored" });
      }
      if (booking.room_id != null) {
        await client.query(
          "UPDATE rooms SET available_capacity = available_capacity + 1 WHERE id=$1",
          [booking.room_id]
        );
      }
    }

    const updatedResult = await client.query(
      "UPDATE bookings SET check_in=$1, check_out=$2, status=$3 WHERE id=$4 RETURNING *",
      [updatedCheckIn, updatedCheckOut, updatedStatus, bookingId]
    );

    await client.query("COMMIT");
    res.json({ message: "Booking updated âœ…", booking: updatedResult.rows[0] });
  } catch (err) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to update booking: " + err.message });
  } finally {
    client.release();
  }
});

app.post("/api/bookings/:id/cancel", requireAuth, async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = parseInt(req.params.id, 10);
    if (!isValidInt(bookingId)) {
      return res.status(400).json({ error: "Invalid booking id" });
    }

    await client.query("BEGIN");

    const bookingResult = await client.query(
      "SELECT id, user_id, hostel_id, room_id, status FROM bookings WHERE id=$1 FOR UPDATE",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Booking not found" });
    }

    const booking = bookingResult.rows[0];
    if (booking.user_id !== req.user.id) {
      await client.query("ROLLBACK");
      return res.status(403).json({ error: "Cannot cancel another user's booking" });
    }
    const currentStatus = normalizeBookingStatus(booking.status);
    if (currentStatus === "cancelled") {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "Booking is already cancelled" });
    }
    if (["completed","rejected","expired"].includes(currentStatus)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "This booking can no longer be cancelled" }); }

    if (isCapacityHoldingStatus(currentStatus)) {
      if (!await releaseHostelCapacity(client, booking.hostel_id)) {
        await client.query("ROLLBACK");
        return res.status(409).json({ error: "Hostel availability is already fully restored" });
      }
      const releasedHold=await client.query("UPDATE booking_inventory_holds SET released_at=NOW() WHERE booking_id=$1 AND released_at IS NULL AND converted_at IS NULL RETURNING room_id",[bookingId]);
      if(releasedHold.rowCount)await client.query("UPDATE rooms SET available_capacity=LEAST(capacity,available_capacity+1) WHERE id=$1",[releasedHold.rows[0].room_id]);
      if (booking.room_id != null && (!releasedHold.rowCount || Number(releasedHold.rows[0].room_id)!==Number(booking.room_id))) {
        await client.query(
          "UPDATE rooms SET available_capacity = available_capacity + 1 WHERE id=$1",
          [booking.room_id]
        );
      }
    }

    const updatedResult = await client.query(
      "UPDATE bookings SET status='cancelled',updated_at=NOW() WHERE id=$1 RETURNING *",
      [bookingId]
    );
    await client.query("INSERT INTO booking_status_history (booking_id,previous_status,next_status,actor_user_id,reason) VALUES ($1,$2,'cancelled',$3,$4)", [bookingId,currentStatus,req.user.id,String(req.body.reason||'student cancellation').slice(0,2000)]);

    await client.query("COMMIT");
    res.json({ message: "Booking cancelled âœ…", booking: updatedResult.rows[0] });
  } catch (err) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to cancel booking: " + err.message });
  } finally {
    client.release();
  }
});

app.get("/api/reviews", async (req, res) => {
  try {
    const result = await pool.query("SELECT id, user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review, created_at FROM reviews ORDER BY id ASC");
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch reviews" }); }
});

app.post("/api/reviews", requireAuth, requireRole("student"), async (req, res) => {
  try {
    const { hostel_id, overall_rating, cleanliness, facilities, management, text_review } = req.body;
    const uid = req.user.id; const hid = parseInt(hostel_id, 10);
    if (!isValidInt(hid)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    const ratings = [overall_rating,cleanliness,facilities,management].map(Number);
    if (ratings.some((rating) => !Number.isInteger(rating) || rating < 1 || rating > 5) || String(text_review || "").length > 4000) return res.status(400).json({ error: "Ratings must be integers from 1 to 5 and review text must be at most 4000 characters" });
    const eligible = await pool.query("SELECT 1 FROM bookings WHERE user_id=$1 AND hostel_id=$2 AND status IN ('confirmed','completed')", [uid,hid]);
    if (!eligible.rowCount) return res.status(409).json({ error: "A confirmed or completed stay is required before reviewing this hostel" });
    const result = await pool.query("INSERT INTO reviews (user_id, hostel_id, overall_rating, cleanliness, facilities, management, text_review) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *", [uid, hid, overall_rating, cleanliness, facilities, management, text_review || null]);
    res.status(201).json({ message: "Review saved âœ…", review: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to save review" }); }
});

app.get("/api/complaints", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;
    const result = await pool.query(
      `SELECT id, user_id, hostel_id, room_id, category, severity, status, assigned_to, description, created_at, resolved_at
       FROM complaints
       WHERE user_id=$1
       ORDER BY id ASC`,
      [userId]
    );
    res.json(result.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch complaints" }); }
});

app.post("/api/complaints/similar", requireAuth, requireRole("student"), async (req, res) => {
  try {
    const hostelId = parseInt(req.body.hostel_id, 10);
    const description = String(req.body.description || "").trim();
    if (!isValidInt(hostelId) || description.length < 3 || description.length > 4000) return res.status(400).json({ error: "A valid hostel_id and a description of 3 to 4000 characters are required" });
    const stay = await pool.query(`SELECT b.room_id FROM bookings b JOIN booking_stays bs ON bs.booking_id=b.id
      WHERE b.user_id=$1 AND b.hostel_id=$2 AND b.status='confirmed' AND bs.checked_in_at IS NOT NULL AND bs.checked_out_at IS NULL
      ORDER BY bs.checked_in_at DESC LIMIT 1`, [req.user.id, hostelId]);
    if (!stay.rowCount) return res.status(409).json({ error: "An active checked-in stay at this hostel is required" });
    const category = fallbackComplaintTriage(description).category;
    const similarComplaints = await findSimilarRoomComplaints(pool, { userId: req.user.id, hostelId, roomId: stay.rows[0].room_id, description, category });
    res.json({ room_id: stay.rows[0].room_id, category, duplicate_warning: similarComplaints.length > 0, similar_complaints: similarComplaints });
  } catch (err) { res.status(500).json({ error: "Failed to check similar complaints: " + err.message }); }
});

app.post("/api/complaints", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const { hostel_id, description } = req.body;
    const uid = req.user.id; const hid = parseInt(hostel_id, 10);
    const idempotencyKey = String(req.headers["idempotency-key"] || "").trim();
    if (!isValidInt(hid)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    if (!description) return res.status(400).json({ error: "description is required" });
    if (!idempotencyKey || idempotencyKey.length > 160) return res.status(400).json({ error: "An idempotency-key header of up to 160 characters is required" });
    if (req.body.status != null || req.body.assigned_to != null) return res.status(400).json({ error: "Complaint status and assignment are controlled by the server" });
    const attachments = Array.isArray(req.body.attachments) ? req.body.attachments : [];
    if (attachments.length > 3 || attachments.some((item) => typeof item !== 'string' || !item.startsWith('data:image/') || item.length > 2_000_000)) {
      return res.status(400).json({ error: "Attach up to three image files, each no larger than 1.5 MB" });
    }
    const stay = await pool.query(`SELECT b.room_id FROM bookings b
      WHERE b.user_id=$1 AND b.hostel_id=$2 AND b.status='confirmed' AND b.room_id IS NOT NULL
      ORDER BY b.updated_at DESC,b.id DESC LIMIT 1`, [uid, hid]);
    if (!stay.rowCount) return res.status(409).json({ error: "A confirmed room assignment at this hostel is required" });
    const previous = await pool.query("SELECT * FROM complaints WHERE user_id=$1 AND idempotency_key=$2", [uid,idempotencyKey]);
    if (previous.rowCount) return res.json({ complaint: previous.rows[0], duplicate: true });
    
    const aiResult = await callAiComplaintCategorizer(description);
    const fallback= fallbackComplaintTriage(description);
    const finalCategory=aiResult?.success&&aiResult.category?aiResult.category:fallback.category;
    const finalSeverity=aiResult?.success?priorityToSeverity(aiResult.priority||"Medium"):fallback.severity;
    const similarComplaints = await findSimilarRoomComplaints(pool, { userId: uid, hostelId: hid, roomId: stay.rows[0].room_id, description, category: finalCategory });
    
    const confidence = aiResult?.success && Number.isFinite(Number(aiResult.confidence)) ? Math.max(0,Math.min(1,Number(aiResult.confidence))) : null;
    const result = await pool.query(
      `INSERT INTO complaints (user_id,hostel_id,room_id,category,severity,status,assigned_to,description,idempotency_key,triage_source,triage_confidence,triage_model_version)
       VALUES ($1,$2,$3,$4,$5,'open',NULL,$6,$7,$8,$9,$10) RETURNING *`,
      [uid, hid, stay.rows[0].room_id, finalCategory, finalSeverity, description, idempotencyKey, aiResult?.success ? "ai" : "fallback", confidence, aiResult?.model_version || null]
    );
    if (attachments.length) {
      await Promise.all(attachments.map((dataUri) => pool.query(
        "INSERT INTO complaint_attachments (complaint_id,data_uri) VALUES ($1,$2)", [result.rows[0].id, dataUri]
      )));
    }
    
    res.status(201).json({
      message: "Complaint saved âœ…",
      complaint: result.rows[0],
      triage: {
        source: aiResult?.success ? "ai" : "fallback",
        fallback_used: !aiResult?.success,
        category: finalCategory,
        priority: aiResult?.priority || finalSeverity,
        confidence: aiResult?.confidence || null,
        suggestion: aiResult?.suggestion || null,
        model_version: aiResult?.model_version || null
      },
      duplicate_warning: similarComplaints.length > 0,
      similar_complaints: similarComplaints
    });
  } catch (err) { res.status(500).json({ error: "Failed to save complaint: " + err.message }); }
});

app.patch("/api/complaints/:id", requireAuth, async (req, res) => {
  return res.status(410).json({ error: "Direct complaint closure is retired. Use /api/complaints/:id/accept after warden resolution." });
  try {
    const complaintId = parseInt(req.params.id, 10);
    if (!isValidInt(complaintId)) return res.status(400).json({ error: "Invalid complaint id" });
    
    const { status, assigned_to, resolved_at } = req.body;
    if (status !== "closed" || assigned_to != null || resolved_at != null || Object.keys(req.body).some((key) => key !== "status")) {
      return res.status(400).json({ error: "Students may only close their own complaint" });
    }
    
    if (status != null && !isValidComplaintStatus(status)) {
      return res.status(400).json({ error: "status must be one of open, in_progress, resolved, closed" });
    }
    
    const complaintResult = await pool.query(
      "SELECT id, status FROM complaints WHERE id=$1 AND user_id=$2",
      [complaintId, req.user.id]
    );
    if (complaintResult.rows.length === 0) {
      return res.status(404).json({ error: "Complaint not found" });
    }
    if (["resolved", "closed"].includes(complaintResult.rows[0].status)) {
      return res.status(409).json({ error: "Complaint is already finalized" });
    }
    
    let updateQuery = "UPDATE complaints SET ";
    const params = [];
    const updates = [];
    let paramCount = 1;
    
    if (status != null) {
      updates.push(`status=$${paramCount}`);
      params.push(status);
      paramCount++;
      updates.push("resolved_at=NOW()");
    }
    if (assigned_to != null) {
      const assignedId = parseInt(assigned_to, 10);
      if (!isValidInt(assignedId)) return res.status(400).json({ error: "assigned_to must be a positive integer" });
      updates.push(`assigned_to=$${paramCount}`);
      params.push(assignedId);
      paramCount++;
    }
    if (resolved_at != null) {
      updates.push(`resolved_at=$${paramCount}`);
      params.push(resolved_at);
      paramCount++;
    }
    
    if (updates.length === 0) {
      return res.status(400).json({ error: "No fields to update" });
    }
    
    updateQuery += updates.join(", ") + ` WHERE id=$${paramCount} RETURNING *`;
    params.push(complaintId);
    
    const result = await pool.query(updateQuery, params);
    await pool.query("INSERT INTO complaint_status_history (complaint_id, previous_status, next_status, changed_by_user_id) VALUES ($1,$2,'closed',$3)", [complaintId, complaintResult.rows[0].status, req.user.id]);
    res.json({ message: "Complaint updated âœ…", complaint: result.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to update complaint: " + err.message }); }
});

app.post("/api/complaints/:id/feedback", requireAuth, async (req, res) => {
  try {
    const complaintId = Number(req.params.id), rating = Number(req.body.rating), comment = String(req.body.comment || "").trim();
    if (!isValidInt(complaintId) || !Number.isInteger(rating) || rating < 1 || rating > 5 || comment.length > 2000) return res.status(400).json({ error: "A rating from 1 to 5 and an optional comment up to 2000 characters are required" });
    const complaint = await pool.query("SELECT id FROM complaints WHERE id=$1 AND user_id=$2 AND status IN ('resolved','closed')", [complaintId, req.user.id]);
    if (!complaint.rowCount) return res.status(404).json({ error: "Finalized complaint not found" });
    const feedback = await pool.query("INSERT INTO complaint_resolution_feedback (complaint_id,student_user_id,rating,comment) VALUES ($1,$2,$3,$4) RETURNING *", [complaintId, req.user.id, rating, comment || null]);
    res.status(201).json({ feedback: feedback.rows[0] });
  } catch (error) {
    if (error.code === "23505") return res.status(409).json({ error: "Resolution feedback has already been submitted" });
    res.status(500).json({ error: "Failed to submit resolution feedback" });
  }
});

app.get("/api/complaints/:id/feedback", requireAuth, async (req, res) => {
  try {
    const complaintId = Number(req.params.id);
    if (!isValidInt(complaintId)) return res.status(400).json({ error: "Invalid complaint id" });
    const feedback = await pool.query(`SELECT f.* FROM complaint_resolution_feedback f JOIN complaints c ON c.id=f.complaint_id
      WHERE f.complaint_id=$1 AND (c.user_id=$2 OR $3='admin' OR EXISTS (SELECT 1 FROM hostel_owners ho WHERE ho.hostel_id=c.hostel_id AND ho.owner_user_id=$2) OR EXISTS (SELECT 1 FROM warden_assignments wa WHERE wa.hostel_id=c.hostel_id AND wa.warden_user_id=$2) OR EXISTS (SELECT 1 FROM parent_student_links pl WHERE pl.parent_user_id=$2 AND pl.student_user_id=c.user_id AND pl.status='active'))`, [complaintId, req.user.id, req.user.role]);
    if (!feedback.rowCount) return res.status(404).json({ error: "Resolution feedback not found" });
    res.json({ feedback: feedback.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to fetch resolution feedback" }); }
});

app.post("/api/recommendations", requireAuth, async (req, res) => {
  try {
    const { top_k } = req.body;
    const sid = req.user.id;
    const k = Number.isInteger(top_k) && top_k > 0 ? top_k : 5;
    const hostelsQ = await pool.query("SELECT id, name, city, available_capacity, total_capacity, description FROM hostels WHERE publication_status='published' AND available_capacity>0 ORDER BY id ASC");
    const ratingsQ = await pool.query("SELECT hostel_id, AVG(overall_rating)::float AS avg_rating, COUNT(*)::int AS n_reviews FROM reviews GROUP BY hostel_id");
    const interQ = await pool.query("SELECT user_id, hostel_id, action_type, created_at FROM interactions WHERE user_id=$1 ORDER BY created_at DESC", [sid]);
    const hostels = hostelsQ.rows;
    const hostelMap = new Map(hostels.map(h => [h.id, h]));
    const ratingMap = new Map(); for (const r of ratingsQ.rows) ratingMap.set(r.hostel_id, r);
    const userInteractions = interQ.rows;
    const userPref = buildUserPreferenceFromHistory(userInteractions, hostelMap);
    const ratingNormArr = softNormalize(hostels.map(h => ratingMap.get(h.id)?.avg_rating ?? 0));
    const availNormArr = softNormalize(hostels.map(h => h.available_capacity ?? 0));
    const affinityByHostel = new Map();
    for (const it of userInteractions) { const w = actionWeight(it.action_type); affinityByHostel.set(it.hostel_id, (affinityByHostel.get(it.hostel_id) || 0) + w); }
    const affinityNormArr = softNormalize(hostels.map(h => affinityByHostel.get(h.id) || 0));
    const W = { userAffinity: 0.45, rating: 0.30, availability: 0.15, city: 0.10 };
    const scored = hostels.map((h, idx) => {
      const cityMatch = userPref.topCity && h.city === userPref.topCity ? clamp01(userPref.cityStrength) : 0;
      const score = W.userAffinity * affinityNormArr[idx] + W.rating * ratingNormArr[idx] + W.availability * availNormArr[idx] + W.city * cityMatch;
      return { hostel: { id: h.id, name: h.name, city: h.city, available_capacity: h.available_capacity, avg_rating: ratingMap.get(h.id)?.avg_rating ?? null, n_reviews: ratingMap.get(h.id)?.n_reviews ?? 0 }, score: Number(score.toFixed(4)), signals: { userAffinity: Number(affinityNormArr[idx].toFixed(4)), ratingNorm: Number(ratingNormArr[idx].toFixed(4)), availNorm: Number(availNormArr[idx].toFixed(4)), cityMatch: Number(cityMatch.toFixed(4)) } };
    });
    scored.sort((a, b) => b.score - a.score);
    const validation = validateAndFallback(scored, { hasEnoughSignals: userInteractions.length >= 3 });
    let finalList = scored;
    if (validation.usedFallback) { finalList = scored.slice().sort((a, b) => (b.hostel.avg_rating ?? 0) - (a.hostel.avg_rating ?? 0) || (b.hostel.available_capacity ?? 0) - (a.hostel.available_capacity ?? 0)); }
    res.json({ student_id: sid, model: validation.usedFallback ? "hybrid_v1_fallback" : "hybrid_v1", personalization: { inferred_top_city: userPref.topCity, confidence_interactions: userInteractions.length }, validation, recommendations: finalList.slice(0, k).map((rec, i) => ({ rank: i + 1, ...rec.hostel, score: rec.score, why: explain(rec, userPref), debug_signals: rec.signals })) });
  } catch (err) { res.status(500).json({ error: "Failed to generate recommendations" }); }
});

// Live recommendation endpoint used by the Flutter app. It reads the current
// PostgreSQL catalogue, favorites, bookings and interaction history instead
// of the static CSV data used by the original notebook experiment.
app.get("/api/student/recommendations", requireAuth, requireRole("student"), async (req, res) => {
  try {
    const topK = Math.max(1, Math.min(10, Number.parseInt(req.query.top_k, 10) || 5));
    const result = await getLiveRecommendations(pool, { studentId: req.user.id, topK });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: "Failed to generate live recommendations" });
  }
});

app.post("/api/recommendations/ml", requireAuth, requireRole("student"), async (req, res) => {
  try {
    const requestedTopK = req.body?.top_k == null ? 5 : req.body.top_k;
    if (!Number.isInteger(requestedTopK) || requestedTopK < 1 || requestedTopK > 10) {
      return res.status(400).json({ error: "top_k must be an integer between 1 and 10" });
    }

    const liveResult = await getLiveRecommendations(pool, {
      studentId: req.user.id,
      topK: requestedTopK,
      budgetMax: req.body?.budget_max,
      requiredAmenities: Array.isArray(req.body?.required_amenities) ? req.body.required_amenities : [],
    });
    return res.json({
      ...liveResult,
      count: liveResult.recommendations.length,
      alpha_used: liveResult.model === "live_hybrid_v2" ? 0.55 : 1.0,
    });

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    let aiResponse;
    try {
      const candidateRows = await pool.query(`SELECT h.id,h.name,h.city,h.university,h.amenities,h.available_capacity,h.single_room_price,h.double_room_price,h.dorm_room_price,COALESCE(AVG(r.overall_rating),0)::float average_rating FROM hostels h LEFT JOIN reviews r ON r.hostel_id=h.id WHERE h.publication_status='published' AND h.available_capacity>0 GROUP BY h.id`);
      const profileRows = await pool.query("SELECT university,budget_max,max_distance_km,study_preference,food_preference FROM student_profiles WHERE user_id=$1",[req.user.id]);
      aiResponse = await fetch(process.env.AI_RECOMMENDER_URL || "http://127.0.0.1:8000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ student_id: req.user.id, student_profile: profileRows.rows[0]||{}, candidates: candidateRows.rows, constraints: { budget_max: req.body?.budget_max??profileRows.rows[0]?.budget_max??null, required_amenities: Array.isArray(req.body?.required_amenities)?req.body.required_amenities:[] }, top_k: Math.min(requestedTopK * 3, 20) }),
        signal: controller.signal
      });
    } finally {
      clearTimeout(timeout);
    }

    if (!aiResponse.ok) {
      return res.status(502).json({ error: "Recommendation service returned an error" });
    }

    const aiResult = await aiResponse.json();
    if (!aiResult.success || !Array.isArray(aiResult.recommendations)) {
      return res.status(502).json({ error: "Recommendation service returned an invalid response" });
    }

    const canonicalIds = aiResult.recommendations
      .map((recommendation) => recommendation.hostel_id)
      .map(Number).filter(Number.isInteger);

    if (canonicalIds.length === 0) {
      return res.json({
        model: "ml_hybrid_catalog_v1",
        student_type: aiResult.student_type,
        alpha_used: aiResult.alpha_used,
        count: 0,
        unavailable_count: 0,
        recommendations: []
      });
    }

    const catalogResult = await pool.query(
      `SELECT ${hostelResponseColumnsWithAlias}
       FROM hostels h
       WHERE h.id = ANY($1::int[])
         AND h.available_capacity > 0 AND h.publication_status='published'
       ORDER BY array_position($1::int[], h.id)`,
      [canonicalIds]
    );
    const catalogById = new Map(catalogResult.rows.map((hostel) => [Number(hostel.id), hostel]));
    const recommendations = aiResult.recommendations
      .filter((recommendation) => catalogById.has(Number(recommendation.hostel_id)))
      .slice(0, requestedTopK)
      .map((recommendation, index) => ({
        rank: index + 1,
        ...catalogById.get(Number(recommendation.hostel_id)),
        model_score: recommendation.hybrid_score,
        content_score: recommendation.cb_score,
        collaborative_score: recommendation.cf_score,
        student_type: recommendation.student_type,
        alpha_used: recommendation.alpha_used
      }));

    res.json({
      model: "ml_hybrid_catalog_v1",
      student_type: aiResult.student_type,
      alpha_used: aiResult.alpha_used,
      count: recommendations.length,
      unavailable_count: canonicalIds.length - catalogResult.rows.length,
      recommendations
    });
  } catch (err) {
    const fallback = await pool.query(`SELECT ${hostelResponseColumnsWithAlias} FROM hostels h WHERE h.publication_status='published' AND h.available_capacity>0 ORDER BY COALESCE(h.single_room_price,h.double_room_price,h.dorm_room_price) ASC NULLS LAST,h.available_capacity DESC,h.id LIMIT $1`,[Math.min(Number(req.body?.top_k)||5,10)]);
    res.json({ model:"database_fallback_v1",service_unavailable:true,timeout:err.name==="AbortError",count:fallback.rowCount,recommendations:fallback.rows });
  }
});

// Owner can suspend or restore a warden assigned to one of their hostels.
// Updating both profile and user status blocks existing sessions immediately.
app.patch(
  "/api/owner/wardens/:wardenId/account-status",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    const wardenId = Number(req.params.wardenId);
    const requestedStatus = String(req.body?.status || "").trim().toLowerCase();
    if (!isValidInt(wardenId) || !["active", "disabled"].includes(requestedStatus)) {
      return res.status(400).json({ error: "Choose Active or Disabled" });
    }

    const client = await pool.connect();
    try {
      await operationalTablesReady;
      await client.query("BEGIN");
      const target = await client.query(
        `SELECT u.id,u.role FROM users u
         JOIN warden_assignments wa ON wa.warden_user_id=u.id
         JOIN hostel_owners ho ON ho.hostel_id=wa.hostel_id
         WHERE u.id=$1 AND u.role='warden' AND ho.owner_user_id=$2
         FOR UPDATE OF u`,
        [wardenId, req.user.id],
      );
      if (!target.rowCount) {
        await client.query("ROLLBACK");
        return res.status(404).json({ error: "Warden not found for this hostel" });
      }

      const profileStatus = requestedStatus === "active" ? "Active" : "Disabled";
      await client.query(
        "UPDATE users SET account_status=$1 WHERE id=$2",
        [requestedStatus, wardenId],
      );
      await client.query(
        `INSERT INTO warden_profiles (warden_user_id,account_status)
         VALUES ($1,$2)
         ON CONFLICT (warden_user_id) DO UPDATE SET
           account_status=EXCLUDED.account_status,updated_at=NOW()`,
        [wardenId, profileStatus],
      );
      if (requestedStatus === "disabled") {
        await client.query(
          "UPDATE auth_sessions SET revoked_at=NOW() WHERE user_id=$1 AND revoked_at IS NULL",
          [wardenId],
        );
      }
      await client.query("COMMIT");
      res.json({ success: true, warden_id: wardenId, account_status: profileStatus });
    } catch (error) {
      await client.query("ROLLBACK");
      res.status(500).json({ error: "Failed to update warden account status" });
    } finally {
      client.release();
    }
  },
);

app.post("/api/chat", requireAuth, requireRole("student"), async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "message is required" });
    const hostelsQ = await pool.query("SELECT id, name, city, available_capacity FROM hostels WHERE publication_status='published' AND available_capacity>0 ORDER BY available_capacity DESC LIMIT 3");
    res.json({ student_id: req.user.id, deprecated: true, replacement: "/api/chat/sessions/:id/messages", reply: "Use an authenticated chat session for contextual support. These published hostels currently have availability:", suggestions: hostelsQ.rows });
  } catch (err) { res.status(500).json({ error: "Chat service failed" }); }
});


// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  AUTH ROUTES  (warden / student login)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// POST /api/auth/login
// Body: { email, password }
// Works for any role in users table (student, warden, admin)
app.post("/api/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ error: "email and password are required" });

    const loginId = String(email).trim().toLowerCase();
    const result = await pool.query(
      `SELECT id, name, email, role, phone, password_hash, account_status
       FROM public.users
       WHERE LOWER(email)=$1 OR LOWER(split_part(email, '@', 1))=$1
       ORDER BY CASE WHEN LOWER(email)=$1 THEN 0 ELSE 1 END
       LIMIT 1`,
      [loginId]
    );

    if (result.rows.length === 0)
      return res.status(401).json({ error: "Invalid email or password" });

    const user = result.rows[0];
    if (user.account_status !== "active") return res.status(403).json({ error: "Account is disabled" });
    if (!await verifyPassword(password, user.password_hash)) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    const token = await signAuthToken(user, req);

    res.json({
      success: true,
      token,
      user: {
        id:    user.id,
        name:  user.name,
        email: user.email,
        role:  user.role,
        phone: user.phone,
      }
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Login failed: " + err.message });
  }
});

// POST /api/auth/register
app.post("/api/auth/register", async (req, res) => {
  try {
    const { name, email, password, role = "student" } = req.body;
    if (!name || !email || !password)
      return res.status(400).json({ error: "name, email and password are required" });
    if (role !== "student") {
      return res.status(403).json({ error: "Public registration is available for students only" });
    }
    if (!isPassword(password)) {
      return res.status(400).json({ error: "password must be at least 8 characters long" });
    }

    const exists = await pool.query("SELECT id FROM public.users WHERE email=$1", [email]);
    if (exists.rows.length > 0)
      return res.status(409).json({ error: "Email already registered." });

    const passwordHash = await hashPassword(password);
    const result = await pool.query(
      "INSERT INTO public.users (name, email, password_hash, role) VALUES ($1,$2,$3,$4) RETURNING id, name, email, role",
      [name, email, passwordHash, role]
    );
    const user = result.rows[0];
    const token = await signAuthToken(user, req);
    res.json({ success: true, token, user });
  } catch (err) {
    res.status(500).json({ error: "Registration failed: " + err.message });
  }
});

// POST /api/auth/password-reset/request
// Body: { email }. Always returns the same message to avoid account enumeration.
app.post("/api/auth/password-reset/request", async (req, res) => {
  try {
    const email = String(req.body.email || "").trim().toLowerCase();
    const genericResponse = { success: true, message: "If an account exists, a password reset link has been sent." };
    if (!email) return res.status(400).json({ error: "email is required" });

    await passwordResetTableReady;
    const userResult = await pool.query(
      "SELECT id, name, email FROM users WHERE LOWER(email)=LOWER($1)",
      [email]
    );
    if (userResult.rowCount === 0) return res.json(genericResponse);

    const user = userResult.rows[0];
    const token = crypto.randomBytes(32).toString("base64url");
    const tokenHash = hashPasswordResetToken(token);
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      await client.query(
        "UPDATE password_reset_tokens SET used_at=NOW() WHERE user_id=$1 AND used_at IS NULL",
        [user.id]
      );
      await client.query(
        "INSERT INTO password_reset_tokens (user_id, token_hash, expires_at) VALUES ($1,$2,NOW() + ($3 * INTERVAL '1 minute'))",
        [user.id, tokenHash, PASSWORD_RESET_TTL_MINUTES]
      );
      await client.query("COMMIT");
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }

    if (process.env.TEST_PASSWORD_RESET_TOKENS === "1") {
      return res.json({ ...genericResponse, test_reset_token: token });
    }

    try {
      await sendPasswordResetEmail(user, token);
      return res.json(genericResponse);
    } catch (error) {
      await pool.query("DELETE FROM password_reset_tokens WHERE token_hash=$1", [tokenHash]);
      console.error("Password reset email failed:", error.message);
      return res.status(500).json({ error: "Unable to send password reset email. Please try again later." });
    }
  } catch (error) {
    console.error("Password reset request failed:", error);
    res.status(500).json({ error: "Unable to request a password reset" });
  }
});

// POST /api/auth/password-reset/confirm
// Body: { token, password }
app.post("/api/auth/password-reset/confirm", async (req, res) => {
  const { token, password } = req.body;
  if (!token || !isPassword(password)) {
    return res.status(400).json({ error: "A valid reset token and password of at least 8 characters are required" });
  }

  const client = await pool.connect();
  try {
    await passwordResetTableReady;
    const tokenHash = hashPasswordResetToken(token);
    await client.query("BEGIN");
    const tokenResult = await client.query(
      `SELECT id, user_id FROM password_reset_tokens
       WHERE token_hash=$1 AND used_at IS NULL AND expires_at > NOW()
       FOR UPDATE`,
      [tokenHash]
    );
    if (tokenResult.rowCount === 0) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "This password reset token is invalid or expired" });
    }

    const passwordHash = await hashPassword(password);
    const userId = tokenResult.rows[0].user_id;
    await client.query("UPDATE users SET password_hash=$1 WHERE id=$2", [passwordHash, userId]);
    await client.query("UPDATE password_reset_tokens SET used_at=NOW() WHERE user_id=$1 AND used_at IS NULL", [userId]);
    await client.query("COMMIT");
    res.json({ success: true, message: "Password reset successfully. Please sign in with your new password." });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Password reset confirmation failed:", error);
    res.status(500).json({ error: "Unable to reset password" });
  } finally {
    client.release();
  }
});

// GET /api/auth/me  (requires token in Authorization header)
app.get("/api/auth/me", requireAuth, async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT id, name, email, role, phone FROM public.users WHERE id=$1", [req.user.id]);
    if (result.rows.length === 0)
      return res.status(404).json({ error: "User not found" });

    res.json({ success: true, user: result.rows[0] });
  } catch (err) {
    res.status(500).json({ error: "Auth check failed" });
  }
});

// Student access is granted only after an owner/warden-backed booking has
// been confirmed and a real room has been assigned. Public student accounts
// can still discover hostels and submit booking requests, but this endpoint
// is the gate for internal hostel information.
app.get("/api/student/access", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT b.id AS booking_id, b.status AS booking_status,
              b.bed_number, b.check_in, b.check_out,
              h.id AS hostel_id, h.name AS hostel_name, h.city, h.address,
              h.amenities, h.facilities, h.rules, h.mess_info,
              h.contact_phone, h.contact_email,
              r.id AS room_id, r.room_number, r.floor_label, r.room_type,
              r.monthly_rent, r.amenities AS room_amenities
         FROM bookings b
         JOIN hostels h ON h.id=b.hostel_id
         JOIN rooms r ON r.id=b.room_id
        WHERE b.user_id=$1
          AND b.status='confirmed'
          AND b.room_id IS NOT NULL
        ORDER BY b.created_at DESC`,
      [req.user.id],
    );
    res.json({
      has_access: result.rowCount > 0,
      assignments: result.rows,
    });
  } catch (error) {
    console.error("Student access lookup failed:", error);
    res.status(500).json({ error: "Failed to load student hostel access" });
  }
});

// Internal hostel hub. Every endpoint below is deliberately limited to a
// confirmed, room-assigned resident; public student accounts cannot read or
// create internal hostel data just by knowing a hostel id.
app.get("/api/student/hostels/:hostelId/bed-availability", requireAuth, requireRole("student"), async (req, res) => {
  try {
    const hostelId = Number(req.params.hostelId);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    const membership = await studentConfirmedBooking(req.user.id, hostelId);
    if (!membership.rowCount) return res.status(403).json({ error: "Confirmed hostel membership is required" });
    const rooms = await pool.query(
      `SELECT r.id,r.floor_label,r.room_number,r.room_type,r.capacity,r.available_capacity,r.monthly_rent,
              COALESCE(json_agg(json_build_object('bed_number',rb.bed_number,'status',rb.status) ORDER BY rb.bed_number)
                FILTER (WHERE rb.id IS NOT NULL),'[]'::json) AS beds
         FROM rooms r LEFT JOIN room_beds rb ON rb.room_id=r.id
        WHERE r.hostel_id=$1
        GROUP BY r.id ORDER BY r.floor_label,r.room_number`, [hostelId]
    );
    res.json({ rooms: rooms.rows });
  } catch (error) { res.status(500).json({ error: "Failed to load bed availability" }); }
});

app.get("/api/student/room-change-requests", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const requests = await pool.query(
      `SELECT r.*,h.name AS hostel_name,room.floor_label,room.room_number
       FROM student_room_change_requests r
       JOIN hostels h ON h.id=r.hostel_id
       LEFT JOIN rooms room ON room.id=r.requested_room_id
       WHERE r.student_user_id=$1 ORDER BY r.created_at DESC`, [req.user.id]
    );
    res.json({ requests: requests.rows });
  } catch (error) { res.status(500).json({ error: "Failed to load room-change requests" }); }
});

app.get("/api/warden/room-change-requests", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const requests = await pool.query(
      `SELECT request.*, student.name AS student_name, room.floor_label, room.room_number
       FROM student_room_change_requests request
       JOIN warden_assignments assignment ON assignment.hostel_id=request.hostel_id
       JOIN users student ON student.id=request.student_user_id
       LEFT JOIN rooms room ON room.id=request.requested_room_id
       WHERE assignment.warden_user_id=$1
       ORDER BY CASE WHEN request.status='pending' THEN 0 ELSE 1 END, request.created_at DESC`,
      [req.user.id],
    );
    res.json({ requests: requests.rows });
  } catch (error) { res.status(500).json({ error: "Failed to load room-change requests" }); }
});

app.patch("/api/warden/room-change-requests/:id", requireAuth, requireRole("warden"), async (req, res) => {
  const requestId = Number(req.params.id);
  const decision = String(req.body?.status || '').toLowerCase();
  if (!isValidInt(requestId) || !['approved', 'rejected'].includes(decision)) return res.status(400).json({ error: 'Choose approved or rejected' });
  const client = await pool.connect();
  try {
    await ensureStudentHubTables();
    await client.query('BEGIN');
    const requested = await client.query(
      `SELECT request.* FROM student_room_change_requests request
       JOIN warden_assignments assignment ON assignment.hostel_id=request.hostel_id
       WHERE request.id=$1 AND assignment.warden_user_id=$2 FOR UPDATE OF request`,
      [requestId, req.user.id],
    );
    if (!requested.rowCount) { await client.query('ROLLBACK'); return res.status(404).json({ error: 'Room-change request not found' }); }
    const request = requested.rows[0];
    if (request.status !== 'pending') { await client.query('ROLLBACK'); return res.status(409).json({ error: 'This request has already been decided' }); }
    if (decision === 'rejected') {
      const updated = await client.query("UPDATE student_room_change_requests SET status='rejected',updated_at=NOW() WHERE id=$1 RETURNING *", [requestId]);
      await createNotification(client, request.student_user_id, 'room_change_request', 'Room change request rejected', 'Your room or bed change request was not approved.', { request_id: requestId, status: 'rejected' }, 'booking_updates');
      await client.query('COMMIT'); return res.json({ request: updated.rows[0] });
    }
    if (!request.requested_room_id || !request.requested_bed_number) { await client.query('ROLLBACK'); return res.status(409).json({ error: 'This older request has no room and bed selection' }); }
    const booking = await client.query("SELECT id,room_id,bed_number FROM bookings WHERE id=$1 AND user_id=$2 AND hostel_id=$3 AND status='confirmed' FOR UPDATE", [request.booking_id, request.student_user_id, request.hostel_id]);
    const targetRoom = await client.query("SELECT id,available_capacity FROM rooms WHERE id=$1 AND hostel_id=$2 FOR UPDATE", [request.requested_room_id, request.hostel_id]);
    const targetBed = await client.query("SELECT id,status FROM room_beds WHERE room_id=$1 AND bed_number=$2 FOR UPDATE", [request.requested_room_id, request.requested_bed_number]);
    if (!booking.rowCount || !targetRoom.rowCount || !targetBed.rowCount || targetBed.rows[0].status !== 'available' || Number(targetRoom.rows[0].available_capacity) < 1) { await client.query('ROLLBACK'); return res.status(409).json({ error: 'The requested bed is no longer available' }); }
    const current = booking.rows[0];
    if (Number(current.room_id) === Number(request.requested_room_id) && String(current.bed_number) === String(request.requested_bed_number)) { await client.query('ROLLBACK'); return res.status(409).json({ error: 'Student already has this bed' }); }
    const alreadyUsed = await client.query("SELECT 1 FROM bookings WHERE room_id=$1 AND bed_number=$2 AND status='confirmed' AND id<>$3", [request.requested_room_id, request.requested_bed_number, current.id]);
    if (alreadyUsed.rowCount) { await client.query('ROLLBACK'); return res.status(409).json({ error: 'The requested bed is no longer available' }); }
    if (current.room_id != null) {
      await client.query('UPDATE rooms SET available_capacity=LEAST(capacity,available_capacity+1) WHERE id=$1', [current.room_id]);
      await client.query("UPDATE room_beds SET status='available' WHERE room_id=$1 AND bed_number=$2", [current.room_id, current.bed_number]);
    }
    await client.query('UPDATE rooms SET available_capacity=available_capacity-1 WHERE id=$1', [request.requested_room_id]);
    await client.query("UPDATE room_beds SET status='occupied' WHERE id=$1", [targetBed.rows[0].id]);
    const moved = await client.query('UPDATE bookings SET room_id=$1,bed_number=$2,updated_at=NOW() WHERE id=$3 RETURNING *', [request.requested_room_id, request.requested_bed_number, current.id]);
    const updated = await client.query("UPDATE student_room_change_requests SET status='approved',updated_at=NOW() WHERE id=$1 RETURNING *", [requestId]);
    await createNotification(client, request.student_user_id, 'room_change_request', 'Room change request approved', 'Your room and bed allocation has been updated.', { request_id: requestId, status: 'approved', room_id: request.requested_room_id, bed_number: request.requested_bed_number }, 'booking_updates');
    await client.query('COMMIT');
    res.json({ request: updated.rows[0], booking: moved.rows[0] });
  } catch (error) { await client.query('ROLLBACK'); res.status(500).json({ error: 'Failed to decide room-change request' }); }
  finally { client.release(); }
});

app.post("/api/student/hostels/:hostelId/room-change-requests", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const hostelId = Number(req.params.hostelId);
    const reason = String(req.body?.reason || "").trim();
    const requestedRoomId = Number(req.body?.requested_room_id);
    const requestedBedNumber = String(req.body?.requested_bed_number || "").trim();
    if (!isValidInt(hostelId) || reason.length < 3 || reason.length > 2000) return res.status(400).json({ error: "Give a room-change reason between 3 and 2000 characters" });
    if (!isValidInt(requestedRoomId) || !requestedBedNumber) return res.status(400).json({ error: "Choose an available room and bed" });
    const membership = await studentConfirmedBooking(req.user.id, hostelId);
    if (!membership.rowCount) return res.status(403).json({ error: "Confirmed hostel membership is required" });
    const bed = await pool.query(
      `SELECT rb.id FROM room_beds rb JOIN rooms r ON r.id=rb.room_id
       WHERE r.id=$1 AND r.hostel_id=$2 AND rb.bed_number=$3 AND rb.status='available'`,
      [requestedRoomId, hostelId, requestedBedNumber],
    );
    if (!bed.rowCount) return res.status(409).json({ error: "That bed is no longer available. Please choose another one." });
    const open = await pool.query("SELECT id FROM student_room_change_requests WHERE student_user_id=$1 AND hostel_id=$2 AND status='pending'", [req.user.id, hostelId]);
    if (open.rowCount) return res.status(409).json({ error: "A room-change request is already pending" });
    const created = await pool.query(
      `INSERT INTO student_room_change_requests (student_user_id,hostel_id,booking_id,reason,requested_room_id,requested_bed_number)
      VALUES ($1,$2,$3,$4,$5,$6) RETURNING *`, [req.user.id, hostelId, membership.rows[0].id, reason, requestedRoomId, requestedBedNumber]
    );
    const wardens = await pool.query("SELECT warden_user_id FROM warden_assignments WHERE hostel_id=$1", [hostelId]);
    for (const warden of wardens.rows) {
      await createNotification(pool, warden.warden_user_id, "room_change_request", "Room change requested", "A resident has requested a room or bed change.", { request_id: created.rows[0].id }, "booking_updates");
    }
    res.status(201).json({ request: created.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to submit room-change request" }); }
});

app.post("/api/student/hostels/:hostelId/feedback", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const hostelId = Number(req.params.hostelId);
    const feedbackType = String(req.body?.type || "").toLowerCase();
    const rating = Number(req.body?.rating);
    const comment = String(req.body?.comment || "").trim();
    if (!isValidInt(hostelId) || !['food','services'].includes(feedbackType) || !Number.isInteger(rating) || rating < 1 || rating > 5 || comment.length > 2000) return res.status(400).json({ error: "Provide a type, 1–5 rating, and an optional comment of up to 2000 characters" });
    const membership = await studentConfirmedBooking(req.user.id, hostelId);
    if (!membership.rowCount) return res.status(403).json({ error: "Confirmed hostel membership is required" });
    const feedback = await pool.query(
      `INSERT INTO student_hostel_feedback (student_user_id,hostel_id,feedback_type,rating,comment)
       VALUES ($1,$2,$3,$4,$5) RETURNING *`, [req.user.id, hostelId, feedbackType, rating, comment || null]
    );
    res.status(201).json({ feedback: feedback.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to save feedback" }); }
});

app.get("/api/student/hostels/:hostelId/preferences", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const hostelId = Number(req.params.hostelId);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    const membership = await studentConfirmedBooking(req.user.id, hostelId);
    if (!membership.rowCount) return res.status(403).json({ error: "Confirmed hostel membership is required" });
    const preference = await pool.query("SELECT culture_preference,caste_preference,university_preference,updated_at FROM student_room_preferences WHERE student_user_id=$1 AND hostel_id=$2", [req.user.id, hostelId]);
    res.json({ preference: preference.rows[0] || null });
  } catch (error) { res.status(500).json({ error: "Failed to load room preferences" }); }
});

app.put("/api/student/hostels/:hostelId/preferences", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const hostelId = Number(req.params.hostelId);
    const values = ['culture_preference','caste_preference','university_preference'].map((key) => String(req.body?.[key] || "").trim());
    if (!isValidInt(hostelId) || values.some((value) => value.length > 240)) return res.status(400).json({ error: "Invalid room preferences" });
    const membership = await studentConfirmedBooking(req.user.id, hostelId);
    if (!membership.rowCount) return res.status(403).json({ error: "Confirmed hostel membership is required" });
    const preference = await pool.query(
      `INSERT INTO student_room_preferences (student_user_id,hostel_id,culture_preference,caste_preference,university_preference)
       VALUES ($1,$2,$3,$4,$5)
       ON CONFLICT (student_user_id,hostel_id) DO UPDATE SET culture_preference=EXCLUDED.culture_preference,caste_preference=EXCLUDED.caste_preference,university_preference=EXCLUDED.university_preference,updated_at=NOW()
       RETURNING *`, [req.user.id, hostelId, ...values]
    );
    res.json({ preference: preference.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to save room preferences" }); }
});

app.get("/api/student/hostels/:hostelId/chat/:channel", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const hostelId = Number(req.params.hostelId);
    const channel = String(req.params.channel || "").toLowerCase();
    if (!isValidInt(hostelId) || !['roommates','parents','warden','group'].includes(channel)) return res.status(400).json({ error: "Invalid chat channel" });
    const membership = await studentConfirmedBooking(req.user.id, hostelId);
    if (!membership.rowCount) return res.status(403).json({ error: "Confirmed hostel membership is required" });
    const messages = await pool.query(
      `SELECT m.id,m.channel,m.body,m.created_at,m.sender_user_id,u.name AS sender_name
       FROM student_hostel_chat_messages m JOIN users u ON u.id=m.sender_user_id
       WHERE m.hostel_id=$1 AND m.channel=$2 ORDER BY m.created_at ASC LIMIT 200`, [hostelId, channel]
    );
    res.json({ messages: messages.rows });
  } catch (error) { res.status(500).json({ error: "Failed to load messages" }); }
});

app.post("/api/student/hostels/:hostelId/chat/:channel", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureStudentHubTables();
    const hostelId = Number(req.params.hostelId);
    const channel = String(req.params.channel || "").toLowerCase();
    const body = String(req.body?.body || "").trim();
    if (!isValidInt(hostelId) || !['roommates','parents','warden','group'].includes(channel) || body.length < 1 || body.length > 4000) return res.status(400).json({ error: "A valid channel and message of up to 4000 characters are required" });
    const membership = await studentConfirmedBooking(req.user.id, hostelId);
    if (!membership.rowCount) return res.status(403).json({ error: "Confirmed hostel membership is required" });
    const message = await pool.query(
      `INSERT INTO student_hostel_chat_messages (hostel_id,sender_user_id,channel,body) VALUES ($1,$2,$3,$4) RETURNING *`,
      [hostelId, req.user.id, channel, body]
    );
    res.status(201).json({ message: message.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to send message" }); }
});

// The student keeps their normal account for discovery.  After an owner sends
// a separate hostel credential, this activates room-selection access only for
// that pending booking.
app.post("/api/student/bookings/:id/hostel-access/activate", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await ensureHostelAccessCredentialTable();
    const bookingId = Number(req.params.id);
    const password = String(req.body?.password || "");
    if (!isValidInt(bookingId) || !password) return res.status(400).json({ error: "Booking id and hostel password are required" });
    const credential = await pool.query(
      `SELECT c.id,c.password_hash,c.hostel_id,c.booking_id,b.status
         FROM hostel_access_credentials c
         JOIN bookings b ON b.id=c.booking_id
        WHERE c.booking_id=$1 AND c.student_user_id=$2`,
      [bookingId, req.user.id]
    );
    if (!credential.rowCount) return res.status(404).json({ error: "No hostel credential was issued for this request" });
    if (credential.rows[0].status !== "pending") return res.status(409).json({ error: "This booking is no longer awaiting room selection" });
    if (!await verifyPassword(password, credential.rows[0].password_hash)) return res.status(401).json({ error: "Incorrect hostel access password" });
    await pool.query("UPDATE hostel_access_credentials SET activated_at=NOW() WHERE id=$1", [credential.rows[0].id]);
    res.json({ success: true, booking_id: bookingId, hostel_id: credential.rows[0].hostel_id });
  } catch (error) { res.status(500).json({ error: "Failed to activate hostel access" }); }
});

app.post("/api/auth/logout", requireAuth, async (req, res) => {
  await pool.query("UPDATE auth_sessions SET revoked_at=NOW() WHERE id=$1 AND user_id=$2 AND revoked_at IS NULL", [req.user.sessionId, req.user.id]);
  res.json({ success: true });
});

// Mobile clients register their FCM/Expo token here. The token is only used
// by the configured push relay when a queued notification requests push.
app.put("/api/notifications/push-token", requireAuth, async (req, res) => {
  const token = String(req.body?.token || "").trim();
  const platform = String(req.body?.platform || "").trim().slice(0, 30) || null;
  if (token.length < 20 || token.length > 512) return res.status(400).json({ error: "A valid push token is required" });
  try {
    const saved = await pool.query(`INSERT INTO user_push_tokens(user_id,token,platform,active)
      VALUES($1,$2,$3,true) ON CONFLICT(token) DO UPDATE SET user_id=EXCLUDED.user_id,platform=EXCLUDED.platform,active=true,updated_at=NOW()
      RETURNING id,platform,active,updated_at`, [req.user.id, token, platform]);
    res.json({ push_token: saved.rows[0] });
  } catch (_) { res.status(500).json({ error: "Failed to register push token" }); }
});

app.get("/api/student/profile", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    await ensureBookingApplicationTable();
    const result = await pool.query(
      `SELECT u.id, u.name, u.email, u.phone, p.university, p.department, p.gender,
              p.budget_max, p.max_distance_km, p.study_preference, p.food_preference,
              application.student_cnic, application.semester, application.degree,
              application.parent_name, application.parent_phone, application.parent_cnic,
              application.parent_relation, application.parent_email, application.parent_city
       FROM users u
       LEFT JOIN student_profiles p ON p.user_id=u.id
       LEFT JOIN LATERAL (
         SELECT a.student_cnic,a.semester,a.degree,a.parent_name,a.parent_phone,a.parent_cnic,
                a.parent_relation,a.parent_email,a.parent_city
         FROM bookings b JOIN booking_applications a ON a.booking_id=b.id
         WHERE b.user_id=u.id ORDER BY b.updated_at DESC NULLS LAST,b.id DESC LIMIT 1
       ) application ON true
       WHERE u.id=$1`,
      [req.user.id]
    );
    res.json(result.rows[0]);
  } catch (error) { res.status(500).json({ error: "Failed to fetch student profile" }); }
});

app.put("/api/student/profile", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    await ensureBookingApplicationTable();
    const { name, phone, university, department, gender, budget_max, max_distance_km, study_preference, food_preference, student_cnic, semester, parent_name, parent_phone, parent_cnic, parent_relation, parent_email, parent_city } = req.body;
    for (const value of [budget_max, max_distance_km, study_preference]) {
      if (value != null && (!Number.isFinite(Number(value)) || Number(value) < 0)) {
        return res.status(400).json({ error: "Numeric profile preferences must be non-negative" });
      }
    }
    if (study_preference != null && Number(study_preference) > 1) {
      return res.status(400).json({ error: "study_preference must be between 0 and 1" });
    }
    await pool.query("UPDATE users SET name=COALESCE($1,name), phone=COALESCE($2,phone) WHERE id=$3", [name || null, phone || null, req.user.id]);
    const result = await pool.query(
      `INSERT INTO student_profiles (user_id, university, department, gender, budget_max, max_distance_km, study_preference, food_preference)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
       ON CONFLICT (user_id) DO UPDATE SET university=COALESCE(EXCLUDED.university, student_profiles.university),
         department=COALESCE(EXCLUDED.department, student_profiles.department), gender=COALESCE(EXCLUDED.gender, student_profiles.gender),
         budget_max=COALESCE(EXCLUDED.budget_max, student_profiles.budget_max), max_distance_km=COALESCE(EXCLUDED.max_distance_km, student_profiles.max_distance_km),
         study_preference=COALESCE(EXCLUDED.study_preference, student_profiles.study_preference), food_preference=COALESCE(EXCLUDED.food_preference, student_profiles.food_preference), updated_at=NOW()
       RETURNING *`,
      [req.user.id, university || null, department || null, gender || null, budget_max ?? null, max_distance_km ?? null, study_preference ?? null, food_preference || null]
    );
    const booking = await pool.query("SELECT id FROM bookings WHERE user_id=$1 ORDER BY updated_at DESC,id DESC LIMIT 1", [req.user.id]);
    if (booking.rowCount) {
      const values = [student_cnic, semester, parent_name, parent_phone, parent_cnic, parent_relation, parent_email, parent_city].map((value) => value == null ? null : String(value).trim());
      if (values.some((value) => value != null && value.length > 240)) return res.status(400).json({ error: "Personal record values are too long" });
      await pool.query(
        `INSERT INTO booking_applications (booking_id,student_cnic,semester,parent_name,parent_phone,parent_cnic,parent_relation,parent_email,parent_city)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
         ON CONFLICT (booking_id) DO UPDATE SET student_cnic=COALESCE(EXCLUDED.student_cnic,booking_applications.student_cnic),semester=COALESCE(EXCLUDED.semester,booking_applications.semester),parent_name=COALESCE(EXCLUDED.parent_name,booking_applications.parent_name),parent_phone=COALESCE(EXCLUDED.parent_phone,booking_applications.parent_phone),parent_cnic=COALESCE(EXCLUDED.parent_cnic,booking_applications.parent_cnic),parent_relation=COALESCE(EXCLUDED.parent_relation,booking_applications.parent_relation),parent_email=COALESCE(EXCLUDED.parent_email,booking_applications.parent_email),parent_city=COALESCE(EXCLUDED.parent_city,booking_applications.parent_city),updated_at=NOW()`,
        [booking.rows[0].id, ...values],
      );
    }
    res.json({ success: true, profile: result.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to update student profile" }); }
});

app.get("/api/notification-preferences", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      "INSERT INTO notification_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO UPDATE SET user_id=EXCLUDED.user_id RETURNING *", [req.user.id]
    );
    res.json(result.rows[0]);
  } catch (error) { res.status(500).json({ error: "Failed to fetch notification preferences" }); }
});

app.put("/api/notification-preferences", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const { booking_updates, complaint_updates, announcements, attendance_updates, location_updates, transport_updates, communication_updates, availability_updates, email_enabled } = req.body;
    if ([booking_updates, complaint_updates, announcements, attendance_updates, location_updates, transport_updates, communication_updates, availability_updates, email_enabled].some((value) => value != null && typeof value !== "boolean")) {
      return res.status(400).json({ error: "Notification preferences must be boolean values" });
    }
    const result = await pool.query(
      `INSERT INTO notification_preferences (user_id, booking_updates, complaint_updates, announcements, attendance_updates, location_updates, transport_updates, communication_updates, availability_updates, email_enabled)
       VALUES ($1,COALESCE($2,true),COALESCE($3,true),COALESCE($4,true),COALESCE($5,true),COALESCE($6,true),COALESCE($7,true),COALESCE($8,true),COALESCE($9,true),COALESCE($10,true))
       ON CONFLICT (user_id) DO UPDATE SET booking_updates=COALESCE($2,notification_preferences.booking_updates), complaint_updates=COALESCE($3,notification_preferences.complaint_updates), announcements=COALESCE($4,notification_preferences.announcements), attendance_updates=COALESCE($5,notification_preferences.attendance_updates), location_updates=COALESCE($6,notification_preferences.location_updates), transport_updates=COALESCE($7,notification_preferences.transport_updates), communication_updates=COALESCE($8,notification_preferences.communication_updates), availability_updates=COALESCE($9,notification_preferences.availability_updates), email_enabled=COALESCE($10,notification_preferences.email_enabled), updated_at=NOW()
       RETURNING *`, [req.user.id, booking_updates ?? null, complaint_updates ?? null, announcements ?? null, attendance_updates ?? null, location_updates ?? null, transport_updates ?? null, communication_updates ?? null, availability_updates ?? null, email_enabled ?? null]
    );
    res.json(result.rows[0]);
  } catch (error) { res.status(500).json({ error: "Failed to update notification preferences" }); }
});

// Parent accounts are created through an administrator, never public role selection.
app.post("/api/admin/parents", requireAuth, requireRole("admin"), async (req, res) => {
  try {
    await operationalTablesReady;
    const { name, email, password, phone } = req.body;
    const normalizedEmail = String(email || "").trim().toLowerCase();
    if (!name || !normalizedEmail || !isPassword(password)) {
      return res.status(400).json({ error: "name, email, and a password of at least 8 characters are required" });
    }
    const exists = await pool.query("SELECT id FROM users WHERE LOWER(email)=LOWER($1)", [normalizedEmail]);
    if (exists.rowCount > 0) return res.status(409).json({ error: "Email already registered" });
    const passwordHash = await hashPassword(password);
    const parent = await pool.query(
      "INSERT INTO users (name,email,password_hash,role,phone) VALUES ($1,$2,$3,'parent',$4) RETURNING id,name,email,role,phone",
      [String(name).trim(), normalizedEmail, passwordHash, phone || null]
    );
    res.status(201).json({ parent: parent.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to provision parent account" }); }
});

// A student invites an existing, administrator-provisioned parent account.
app.post("/api/student/parents/invitations", requireAuth, requireRole("student"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const email = String(req.body.email || "").trim().toLowerCase();
    if (!email) return res.status(400).json({ error: "Parent email is required" });
    const parentResult = await client.query(
      "SELECT id,name,email FROM users WHERE LOWER(email)=LOWER($1) AND role='parent'",
      [email]
    );
    if (parentResult.rowCount === 0) return res.status(404).json({ error: "A provisioned parent account with this email was not found" });
    const parent = parentResult.rows[0];
    const token = crypto.randomBytes(32).toString("base64url");
    const tokenHash = hashPasswordResetToken(token);
    await client.query("BEGIN");
    const existing = await client.query(
      `SELECT status FROM parent_student_links
       WHERE parent_user_id=$1 AND student_user_id=$2 AND status IN ('pending','active') FOR UPDATE`,
      [parent.id, req.user.id]
    );
    if (existing.rowCount > 0) {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: existing.rows[0].status === "active" ? "This parent is already linked" : "A parent invitation is already pending" });
    }
    const invitation = await client.query(
      `INSERT INTO parent_student_links (parent_user_id,student_user_id,invited_by_student_id,invitation_token_hash,expires_at)
       VALUES ($1,$2,$2,$3,NOW() + ($4 * INTERVAL '1 hour')) RETURNING id,parent_user_id,student_user_id,status,expires_at,created_at`,
      [parent.id, req.user.id, tokenHash, PARENT_INVITATION_TTL_HOURS]
    );
    await client.query("COMMIT");
    if (process.env.TEST_PARENT_INVITATION_TOKENS === "1") {
      return res.status(201).json({ invitation: invitation.rows[0], test_invitation_token: token });
    }
    try {
      const student = await pool.query("SELECT id,name FROM users WHERE id=$1", [req.user.id]);
      await sendParentInvitationEmail(parent, student.rows[0], token);
      res.status(201).json({ invitation: invitation.rows[0], message: "Parent invitation sent" });
    } catch (error) {
      await pool.query("UPDATE parent_student_links SET status='revoked', revoked_at=NOW() WHERE id=$1 AND status='pending'", [invitation.rows[0].id]);
      console.error("Parent invitation email failed:", error.message);
      res.status(503).json({ error: "Unable to deliver the parent invitation" });
    }
  } catch (error) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to create parent invitation" });
  } finally { client.release(); }
});

app.get("/api/student/parents", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    const links = await pool.query(
      `SELECT l.parent_user_id, l.status, l.expires_at, l.accepted_at, l.created_at,
              u.name, u.email, u.phone
       FROM parent_student_links l JOIN users u ON u.id=l.parent_user_id
       WHERE l.student_user_id=$1 AND l.status IN ('pending','active')
       ORDER BY l.created_at DESC`,
      [req.user.id]
    );
    res.json({ parents: links.rows });
  } catch (error) { res.status(500).json({ error: "Failed to fetch parent links" }); }
});

app.delete("/api/student/parents/:parentId", requireAuth, requireRole("student"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const parentId = Number(req.params.parentId);
    if (!isValidInt(parentId)) return res.status(400).json({ error: "Invalid parent id" });
    await client.query("BEGIN");
    const revoked = await client.query(
      `UPDATE parent_student_links SET status='revoked', revoked_at=NOW()
       WHERE parent_user_id=$1 AND student_user_id=$2 AND status IN ('pending','active')
       RETURNING id,parent_user_id,student_user_id,status,revoked_at`,
      [parentId, req.user.id]
    );
    if (revoked.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Parent link not found" }); }
    await client.query("UPDATE attendance_return_watches SET active=false WHERE parent_user_id=$1 AND student_user_id=$2 AND active=true", [parentId, req.user.id]);
    await client.query("COMMIT");
    res.json({ link: revoked.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to revoke parent link" }); } finally { client.release(); }
});

app.post("/api/parent/invitations/accept", requireAuth, requireRole("parent"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const token = String(req.body.token || "");
    if (!token) return res.status(400).json({ error: "Invitation token is required" });
    await client.query("BEGIN");
    const invitation = await client.query(
      `SELECT * FROM parent_student_links
       WHERE invitation_token_hash=$1 AND parent_user_id=$2 FOR UPDATE`,
      [hashPasswordResetToken(token), req.user.id]
    );
    if (invitation.rowCount === 0 || invitation.rows[0].status !== "pending" || new Date(invitation.rows[0].expires_at) <= new Date()) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "This parent invitation is invalid or expired" });
    }
    const accepted = await client.query(
      `UPDATE parent_student_links SET status='active', accepted_at=NOW()
       WHERE id=$1 RETURNING id,parent_user_id,student_user_id,status,accepted_at`,
      [invitation.rows[0].id]
    );
    await client.query("COMMIT");
    res.json({ link: accepted.rows[0] });
  } catch (error) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to accept parent invitation" });
  } finally { client.release(); }
});

app.get("/api/parent/children", requireAuth, requireRole("parent"), async (req, res) => {
  try {
    await operationalTablesReady;
    const children = await pool.query(
      `SELECT u.id, u.name, u.email, u.phone, sp.university, sp.department, l.accepted_at
       FROM parent_student_links l
       JOIN users u ON u.id=l.student_user_id
       LEFT JOIN student_profiles sp ON sp.user_id=u.id
       WHERE l.parent_user_id=$1 AND l.status='active' ORDER BY u.name`,
      [req.user.id]
    );
    res.json({ children: children.rows });
  } catch (error) { res.status(500).json({ error: "Failed to fetch linked children" }); }
});

app.get("/api/parent/children/:studentId/overview", requireAuth, requireRole("parent"), async (req, res) => {
  try {
    await operationalTablesReady;
    const studentId = Number(req.params.studentId);
    if (!isValidInt(studentId) || !await parentHasStudent(pool, req.user.id, studentId)) {
      return res.status(404).json({ error: "Linked student not found" });
    }
    const [student, bookings, payments, announcements, wardens] = await Promise.all([
      pool.query(
        `SELECT u.id,u.name,u.email,u.phone,sp.university,sp.department
         FROM users u LEFT JOIN student_profiles sp ON sp.user_id=u.id
         WHERE u.id=$1 AND u.role='student'`,
        [studentId],
      ),
      pool.query(
        `SELECT b.*, h.name AS hostel_name, h.address AS hostel_address, h.city AS hostel_city,
                h.amenities, r.room_number, r.room_type, bs.checked_in_at, bs.checked_out_at
         FROM bookings b JOIN hostels h ON h.id=b.hostel_id
         LEFT JOIN rooms r ON r.id=b.room_id LEFT JOIN booking_stays bs ON bs.booking_id=b.id
         WHERE b.user_id=$1 ORDER BY b.created_at DESC`, [studentId]
      ),
      pool.query(
        `SELECT p.* FROM payments p JOIN bookings b ON b.id=p.booking_id
         WHERE b.user_id=$1 ORDER BY p.created_at DESC`, [studentId]
      ),
      pool.query(
        `SELECT DISTINCT a.* FROM announcements a
         LEFT JOIN bookings b ON b.hostel_id=a.hostel_id AND b.user_id=$1 AND b.status IN ('pending','confirmed')
         LEFT JOIN favorites f ON f.hostel_id=a.hostel_id AND f.user_id=$1
         WHERE b.id IS NOT NULL OR f.user_id IS NOT NULL ORDER BY a.created_at DESC`, [studentId]
      ),
      pool.query(
        `SELECT DISTINCT u.id,u.name,u.email,u.phone,h.id AS hostel_id,h.name AS hostel_name
         FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id
         JOIN users u ON u.id=wa.warden_user_id JOIN hostels h ON h.id=b.hostel_id
         WHERE b.user_id=$1`, [studentId]
      ),
    ]);
    if (student.rowCount === 0) return res.status(404).json({ error: "Linked student not found" });
    res.json({ student: student.rows[0], bookings: bookings.rows, payments: payments.rows, announcements: announcements.rows, wardens: wardens.rows });
  } catch (error) { res.status(500).json({ error: "Failed to fetch child overview" }); }
});

// ============================================================================
// GET LOGGED-IN OWNER'S HOSTELS
// ============================================================================

app.get(
  "/api/owner/hostels",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    try {
      await operationalTablesReady;

      const result = await pool.query(
        `
        SELECT
          h.*,
          COALESCE(g.boundary, '[]'::jsonb) AS attendance_boundary,
          COALESCE(g.enabled, false) AS attendance_boundary_enabled

        FROM hostels h

        INNER JOIN hostel_owners ho
          ON ho.hostel_id = h.id

        LEFT JOIN hostel_attendance_geofences g
          ON g.hostel_id = h.id

        WHERE ho.owner_user_id = $1

        ORDER BY h.id ASC
        `,
        [req.user.id]
      );

      console.log(
        "OWNER HOSTELS FETCHED:",
        {
          ownerUserId: req.user.id,
          count: result.rows.length,
        }
      );

      res.json(result.rows);
    } catch (error) {
      console.error(
        "GET OWNER HOSTELS FAILED:",
        error
      );

      res.status(500).json({
        error: "Failed to fetch owner hostels",
        detail: error.message,
      });
    }
  }
);
app.patch(
  "/api/owner/hostels/:id",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    try {
      await operationalTablesReady;

      const hostelId = Number(req.params.id);

      if (!isValidInt(hostelId)) {
        return res.status(400).json({
          error: "Invalid hostel id",
        });
      }

      const ownsHostel = await ownerHasHostel(
        pool,
        req.user.id,
        hostelId
      );

      if (!ownsHostel) {
        return res.status(404).json({
          error: "Hostel not found",
        });
      }

      const jsonFields = new Set(["amenities", "facilities", "rules", "mess_info", "media", "room_types_available"]);
      const textFields = new Set(["name", "hostel_type", "city", "address", "university", "description", "area", "contact_phone", "contact_email"]);
      const numericFields = new Set(["latitude", "longitude", "single_room_price", "double_room_price", "dorm_room_price", "floors"]);
      const statusFields = new Set(["publication_status"]);
      const editableFields = new Set([...jsonFields, ...textFields, ...numericFields, ...statusFields]);
      const entries = Object.entries(req.body).filter(([key, value]) => editableFields.has(key) && value !== undefined);

      if (entries.length === 0) {
        return res.status(400).json({ error: "No editable hostel fields supplied" });
      }

      const requiredTextFields = new Set(["name", "hostel_type", "city", "address"]);
      for (const [key, rawValue] of entries) {
        if (requiredTextFields.has(key) && rawValue != null && !String(rawValue).trim()) {
          return res.status(400).json({ error: `${key} cannot be empty` });
        }
        if (numericFields.has(key) && rawValue != null && !Number.isFinite(Number(rawValue))) {
          return res.status(400).json({ error: `${key} must be numeric` });
        }
        if (key.endsWith("_price") && Number(rawValue) < 0) {
          return res.status(400).json({ error: `${key} cannot be negative` });
        }
        if (key === "floors" && (!Number.isInteger(Number(rawValue)) || Number(rawValue) < 1)) {
          return res.status(400).json({ error: "floors must be a positive integer" });
        }
        if (key === "latitude" && (Number(rawValue) < -90 || Number(rawValue) > 90)) {
          return res.status(400).json({ error: "latitude must be between -90 and 90" });
        }
        if (key === "longitude" && (Number(rawValue) < -180 || Number(rawValue) > 180)) {
          return res.status(400).json({ error: "longitude must be between -180 and 180" });
        }
        if (key === "publication_status" && !["draft", "submitted", "published", "suspended"].includes(String(rawValue))) {
          return res.status(400).json({ error: "Invalid publication_status" });
        }
        if (key === "contact_phone" && !isValidHostelPhone(rawValue)) {
          return res.status(400).json({ error: "Hostel phone must be 03XXXXXXXXX or +923XXXXXXXXX" });
        }
        if (key === "contact_email" && !isValidHostelEmail(rawValue)) {
          return res.status(400).json({ error: "Enter a valid hostel email address" });
        }
        if (jsonFields.has(key) && (rawValue == null || typeof rawValue !== "object")) {
          return res.status(400).json({ error: `${key} must be structured JSON` });
        }
      }

      const values = entries.map(([key, value]) =>
        jsonFields.has(key) ? JSON.stringify(value) : numericFields.has(key) && value != null ? Number(value) : value == null ? null : String(value).trim()
      );
      const assignments = entries.map(([key], index) => `${key} = $${index + 1}${jsonFields.has(key) ? "::jsonb" : ""}`);
      const result = await pool.query(
        `UPDATE hostels SET ${assignments.join(", ")}, updated_at = NOW() WHERE id = $${values.length + 1} RETURNING *`,
        [...values, hostelId]
      );

      if (result.rowCount === 0) {
        return res.status(404).json({
          error: "Hostel not found",
        });
      }

      res.json({
        success: true,
        hostel: result.rows[0],
      });
    } catch (error) {
      console.error(
        "UPDATE OWNER HOSTEL FAILED:",
        error
      );

      res.status(500).json({
        error: "Failed to update hostel",
        detail: error.message,
      });
    }
  }
);
// ============================================================================
// REGISTER HOSTEL FOR LOGGED-IN OWNER
//
// Creates:
//   1. hostel record
//   2. hostel_owners relationship
//
// Both happen in one transaction.
// ============================================================================

app.put(
  "/api/owner/hostels/:id/attendance-geofence",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    try {
      const hostelId = Number(req.params.id);
      const boundary = req.body?.boundary;
      if (!isValidInt(hostelId) || !Array.isArray(boundary) || boundary.length < 3 || boundary.length > 30) {
        return res.status(400).json({ error: "Boundary must contain between 3 and 30 map points" });
      }
      const points = boundary.map((point) => ({
        latitude: Number(point?.latitude),
        longitude: Number(point?.longitude),
      }));
      if (!points.every((point) => Number.isFinite(point.latitude) && Number.isFinite(point.longitude) &&
        point.latitude >= -90 && point.latitude <= 90 && point.longitude >= -180 && point.longitude <= 180)) {
        return res.status(400).json({ error: "Every boundary point requires a valid latitude and longitude" });
      }
      if (!await ownerHasHostel(pool, req.user.id, hostelId)) {
        return res.status(404).json({ error: "Hostel not found" });
      }
      const result = await pool.query(
        `INSERT INTO hostel_attendance_geofences(hostel_id,boundary,enabled,updated_by_user_id)
         VALUES($1,$2::jsonb,true,$3)
         ON CONFLICT(hostel_id) DO UPDATE SET boundary=EXCLUDED.boundary,enabled=true,
           updated_by_user_id=EXCLUDED.updated_by_user_id,updated_at=NOW()
         RETURNING hostel_id,boundary,enabled,updated_at`,
        [hostelId, JSON.stringify(points), req.user.id],
      );
      res.json({ geofence: result.rows[0] });
    } catch (error) {
      console.error("UPDATE ATTENDANCE GEOFENCE FAILED:", error);
      res.status(500).json({ error: "Failed to save attendance boundary" });
    }
  },
);

app.post(
  "/api/owner/hostels",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    const client = await pool.connect();

    try {
      await operationalTablesReady;

      console.log("OWNER HOSTEL REGISTRATION REQUEST:", {
        ownerUserId: req.user.id,
        ownerEmail: req.user.email,
        body: req.body,
      });

      const name = String(req.body.name || "").trim();
      const address = String(req.body.address || "").trim();
      const city = String(req.body.city || "").trim();
      const university = req.body.university == null
        ? null
        : String(req.body.university).trim();
      const hostelType = req.body.hostel_type == null
        ? null
        : String(req.body.hostel_type).trim();
      const description = req.body.description == null
        ? null
        : String(req.body.description).trim();
      const latitude = req.body.latitude == null ? null : Number(req.body.latitude);
      const longitude = req.body.longitude == null ? null : Number(req.body.longitude);
      const totalCapacity = Number(req.body.total_capacity);
      const totalFloors = req.body.floors == null ? null : Number(req.body.floors);
      const totalRooms = req.body.total_rooms == null ? null : Number(req.body.total_rooms);
      const contactPhone = req.body.contact_phone == null ? null : String(req.body.contact_phone).trim();
      const contactEmail = req.body.contact_email == null ? null : String(req.body.contact_email).trim().toLowerCase();
      const amenities = Array.isArray(req.body.amenities)
        ? req.body.amenities.map((item) => String(item).trim()).filter(Boolean)
        : [];
      const roomTypes = Array.isArray(req.body.room_types)
        ? req.body.room_types
        : [];
      const facilities = Array.isArray(req.body.facilities) ? req.body.facilities : [];
      const rules = Array.isArray(req.body.rules) ? req.body.rules : [];
      const messInfo = req.body.mess_info && typeof req.body.mess_info === "object" && !Array.isArray(req.body.mess_info) ? req.body.mess_info : {};
      const media = Array.isArray(req.body.media) ? req.body.media : [];
      const attendanceBoundary = req.body.attendance_boundary;
      const registrationRooms = Array.isArray(req.body.rooms) ? req.body.rooms : [];

      if (!name) return res.status(400).json({ error: "Hostel name is required" });
      if (!address) return res.status(400).json({ error: "Hostel address is required" });
      if (!city) return res.status(400).json({ error: "Hostel city is required" });
      if (contactPhone && !isValidHostelPhone(contactPhone)) {
        return res.status(400).json({ error: "Hostel phone must be 03XXXXXXXXX or +923XXXXXXXXX" });
      }
      if (contactEmail && !isValidHostelEmail(contactEmail)) {
        return res.status(400).json({ error: "Enter a valid hostel email address" });
      }
      if (!Number.isInteger(totalCapacity) || totalCapacity <= 0) {
        return res.status(400).json({ error: "A positive total_capacity is required" });
      }
      if (totalFloors != null && (!Number.isInteger(totalFloors) || totalFloors <= 0)) {
        return res.status(400).json({ error: "floors must be a positive integer" });
      }
      if (totalRooms != null && (!Number.isInteger(totalRooms) || totalRooms <= 0)) {
        return res.status(400).json({ error: "total_rooms must be a positive integer" });
      }
      if (latitude != null && !Number.isFinite(latitude)) {
        return res.status(400).json({ error: "Invalid latitude" });
      }
      if (longitude != null && !Number.isFinite(longitude)) {
        return res.status(400).json({ error: "Invalid longitude" });
      }
      if (attendanceBoundary != null && (!Array.isArray(attendanceBoundary) || attendanceBoundary.length < 3 || attendanceBoundary.length > 30)) {
        return res.status(400).json({ error: "Attendance boundary must contain between 3 and 30 map points" });
      }
      const boundaryPoints = attendanceBoundary == null ? null : attendanceBoundary.map((point) => ({
        latitude: Number(point?.latitude),
        longitude: Number(point?.longitude),
      }));
      if (boundaryPoints != null && !boundaryPoints.every((point) => Number.isFinite(point.latitude) && Number.isFinite(point.longitude) &&
        point.latitude >= -90 && point.latitude <= 90 && point.longitude >= -180 && point.longitude <= 180)) {
        return res.status(400).json({ error: "Every attendance boundary point requires a valid latitude and longitude" });
      }

      // Detailed owner onboarding data.  A room's beds are normalized here so
      // the values saved below are safe and the advertised capacity matches
      // the actual number of named beds.
      if (registrationRooms.length > 500) {
        return res.status(400).json({ error: "A hostel can contain at most 500 rooms" });
      }
      const normalizedRegistrationRooms = [];
      const roomNumbers = new Set();
      for (const rawRoom of registrationRooms) {
        const floorLabel = String(rawRoom?.floor_label || "").trim();
        const roomNumber = String(rawRoom?.room_number || "").trim();
        const capacity = Number(rawRoom?.capacity);
        const monthlyRent = Number(rawRoom?.monthly_rent);
        const roomType = String(rawRoom?.room_type || `${capacity} Seater`).trim();
        const beds = Array.isArray(rawRoom?.beds)
          ? rawRoom.beds.map((bed) => String(bed || "").trim())
          : [];
        if (!floorLabel || !roomNumber || !Number.isInteger(capacity) || capacity < 1 || capacity > 6 ||
          !Number.isInteger(monthlyRent) || monthlyRent < 0 || beds.length !== capacity || beds.some((bed) => !bed)) {
          return res.status(400).json({ error: "Each room needs a floor, room number, 1–6 beds, and a valid monthly rent" });
        }
        if (new Set(beds.map((bed) => bed.toLowerCase())).size !== beds.length) {
          return res.status(400).json({ error: `Bed numbers must be unique in room ${roomNumber}` });
        }
        const roomKey = roomNumber.toLowerCase();
        if (roomNumbers.has(roomKey)) {
          return res.status(400).json({ error: `Room number ${roomNumber} is repeated` });
        }
        roomNumbers.add(roomKey);
        normalizedRegistrationRooms.push({ floorLabel, roomNumber, capacity, monthlyRent, roomType, beds });
      }
      if (normalizedRegistrationRooms.length > 0) {
        const plannedCapacity = normalizedRegistrationRooms.reduce((sum, room) => sum + room.capacity, 0);
        if (totalRooms !== normalizedRegistrationRooms.length || totalCapacity !== plannedCapacity) {
          return res.status(400).json({ error: "Room, bed, and total capacity values do not match" });
        }
      }

      await client.query("BEGIN");

      const existingHostel = await client.query(
        `SELECT h.* FROM hostels h
         INNER JOIN hostel_owners ho ON ho.hostel_id = h.id
         WHERE ho.owner_user_id = $1 AND LOWER(h.name) = LOWER($2)
         LIMIT 1`,
        [req.user.id, name]
      );

      if (existingHostel.rowCount > 0) {
        await client.query("ROLLBACK");
        return res.status(409).json({
          error: "You have already registered a hostel with this name.",
          hostel: existingHostel.rows[0],
        });
      }

      const hostelResult = await client.query(
        `INSERT INTO hostels (
          name, address, city, university, latitude, longitude,
          total_capacity, available_capacity, description,
          hostel_type, total_rooms, amenities, room_types_available,
          facilities, rules, mess_info, media, publication_status,
          floors, contact_phone, contact_email
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$7,$8,$9,$10,$11::jsonb,$12::jsonb,
                $13::jsonb,$14::jsonb,$15::jsonb,$16::jsonb,'published',$17,$18,$19)
        RETURNING *`,
        [
          name,
          address,
          city,
          university || null,
          latitude,
          longitude,
          totalCapacity,
          description || null,
          hostelType || null,
          totalRooms,
          JSON.stringify(amenities),
          JSON.stringify(roomTypes),
          JSON.stringify(facilities),
          JSON.stringify(rules),
          JSON.stringify(messInfo),
          JSON.stringify(media),
          totalFloors,
          contactPhone || null,
          contactEmail || null,
        ]
      );

      const hostel = hostelResult.rows[0];

      if (boundaryPoints != null) {
        await client.query(
          `INSERT INTO hostel_attendance_geofences(hostel_id,boundary,enabled,updated_by_user_id)
           VALUES($1,$2::jsonb,true,$3)`,
          [hostel.id, JSON.stringify(boundaryPoints), req.user.id],
        );
      }

      for (const room of normalizedRegistrationRooms) {
        const roomResult = await client.query(
          `INSERT INTO rooms(hostel_id, room_number, floor_label, capacity, available_capacity, room_type, monthly_rent)
           VALUES($1,$2,$3,$4,$4,$5,$6)
           RETURNING id`,
          [hostel.id, room.roomNumber, room.floorLabel, room.capacity, room.roomType, room.monthlyRent],
        );
        const roomId = roomResult.rows[0].id;
        for (const bedNumber of room.beds) {
          await client.query(
            `INSERT INTO room_beds(room_id, bed_number) VALUES($1,$2)`,
            [roomId, bedNumber],
          );
        }
      }

      await client.query(
        `INSERT INTO hostel_owners (owner_user_id, hostel_id)
         VALUES ($1, $2)`,
        [req.user.id, hostel.id]
      );

      await client.query("COMMIT");

      console.log("HOSTEL CREATED AND LINKED:", {
        ownerUserId: req.user.id,
        hostelId: hostel.id,
        hostelName: hostel.name,
      });

      res.status(201).json({
        success: true,
        message: "Hostel registered successfully",
        hostel,
        roomsCreated: normalizedRegistrationRooms.length,
      });
    } catch (error) {
      try { await client.query("ROLLBACK"); } catch (_) {}
      console.error("OWNER HOSTEL REGISTRATION FAILED:", error);
      res.status(500).json({
        error: "Failed to register hostel",
        detail: error.message,
      });
    } finally {
      client.release();
    }
  }
);

app.get(
  "/api/owner/students",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    try {
      await operationalTablesReady;
      await ensureBookingApplicationTable();
      await ensureHostelAccessCredentialTable();

      const result = await pool.query(
         `SELECT DISTINCT ON (b.id)
        u.id,
u.name,
u.email,
u.phone,
u.role,
b.id AS booking_id,
b.hostel_id,
b.room_id,
b.bed_number,
b.status AS booking_status,
h.name AS hostel_name,
r.room_number AS room_number,
ba.student_cnic,ba.university,ba.department,ba.semester,ba.gender,ba.occupation,ba.degree,
ba.parent_name,ba.parent_phone,ba.parent_cnic,ba.parent_relation,ba.parent_city,ba.parent_email,
ba.secondary_guardian_name,ba.secondary_guardian_phone,ba.secondary_guardian_relation,ba.secondary_guardian_email,
ba.special_requirements,hac.id AS hostel_credential_id,
COALESCE(invoice.status,'no_invoice') AS fee_status
        FROM hostel_owners ho
        JOIN hostels h
          ON h.id = ho.hostel_id
        JOIN bookings b
          ON b.hostel_id = h.id
          LEFT JOIN rooms r
  ON r.id = b.room_id
          LEFT JOIN booking_applications ba
  ON ba.booking_id = b.id
          LEFT JOIN hostel_access_credentials hac
  ON hac.booking_id = b.id
          LEFT JOIN LATERAL (
            SELECT i.status FROM student_fee_assignments assignment
            JOIN fee_invoices i ON i.assignment_id=assignment.id
            WHERE assignment.booking_id=b.id AND assignment.active=true
            ORDER BY i.billing_month DESC,i.id DESC LIMIT 1
          ) invoice ON TRUE
        JOIN users u
          ON u.id = b.user_id
        WHERE ho.owner_user_id = $1
          AND u.role = 'student'
        ORDER BY b.id DESC
        `,
        [req.user.id]
      );

      res.json({
        success: true,
        students: result.rows,
      });
    } catch (error) {
      console.error(
        "Owner students fetch failed:",
        error
      );

      res.status(500).json({
        error: "Failed to fetch owner students",
      });
    }
  }
);
// POST /api/owner/students
app.post(
  "/api/owner/students",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    const client = await pool.connect();

    try {
      await operationalTablesReady;
      await ensureBookingApplicationTable();

      const {
  name,
  email,
  phone,
  cnic,
  city,
  university,
  department,
  semester,
  parent_name,
  parent_phone,
  parent_email,
  parent_relation,
  parent_city,
  room_number,
  bed_number,
  hostel_id,
} = req.body;
const bedNumber =
  typeof bed_number === "string" && bed_number.trim()
    ? bed_number.trim()
    : null;
      if (!name || !email) {
        return res.status(400).json({
          error: "Student name and email are required",
        });
      }

      await client.query("BEGIN");

      // --------------------------------------------------
      // 1. Determine owner's hostel
      // --------------------------------------------------
      let hostelId = hostel_id
        ? Number(hostel_id)
        : null;

      if (hostelId) {
        const ownershipCheck = await client.query(
          `
          SELECT h.id, h.name
          FROM hostel_owners ho
          JOIN hostels h
            ON h.id = ho.hostel_id
          WHERE ho.owner_user_id = $1
            AND h.id = $2
          LIMIT 1
          `,
          [req.user.id, hostelId]
        );

        if (ownershipCheck.rowCount === 0) {
          await client.query("ROLLBACK");

          return res.status(404).json({
            error: "Hostel not found",
          });
        }
      } else {
        const hostelResult = await client.query(
          `
          SELECT h.id, h.name
          FROM hostel_owners ho
          JOIN hostels h
            ON h.id = ho.hostel_id
          WHERE ho.owner_user_id = $1
          ORDER BY h.id
          LIMIT 1
          `,
          [req.user.id]
        );

        if (hostelResult.rowCount === 0) {
          await client.query("ROLLBACK");

          return res.status(404).json({
            error: "Owner has no hostel",
          });
        }

        hostelId = hostelResult.rows[0].id;
      }

      // --------------------------------------------------
      // 2. Check duplicate email
      // --------------------------------------------------
      const existingUser = await client.query(
        `
        SELECT id
        FROM users
        WHERE LOWER(email) = LOWER($1)
        LIMIT 1
        `,
        [email.trim()]
      );

      if (existingUser.rowCount > 0) {
        await client.query("ROLLBACK");

        return res.status(409).json({
          error: "Email already registered",
        });
      }

      // --------------------------------------------------
      // 3. Generate temporary student password
      // Same pattern used for Warden accounts
      // --------------------------------------------------
      const randomPart = crypto
        .randomBytes(5)
        .toString("hex")
        .toUpperCase();

      const temporaryPassword =
        `SB@${randomPart}`;

      const passwordHash =
        await hashPassword(
          temporaryPassword
        );

      // --------------------------------------------------
      // 4. Create student user
      // --------------------------------------------------
      const userResult = await client.query(
        `
        INSERT INTO users
        (
          name,
          email,
          phone,
          password_hash,
          role
        )
        VALUES ($1, $2, $3, $4, 'student')
        RETURNING
          id,
          name,
          email,
          phone,
          role,
          created_at
        `,
        [
          name.trim(),
          email.trim().toLowerCase(),
          phone?.trim() || null,
          passwordHash,
        ]
      );

      const student = userResult.rows[0];

      // --------------------------------------------------
      // 5. Optional room assignment
      // --------------------------------------------------
      let roomId = null;
      let assignedRoom = null;

      if (
        room_number &&
        room_number.toString().trim() !== "" &&
        room_number.toString().toLowerCase() !== "unassigned"
      ) {
        const roomResult = await client.query(
          `
          SELECT
            id,
            room_number,
            capacity,
            available_capacity,
            room_type,
            monthly_rent
          FROM rooms
          WHERE hostel_id = $1
            AND room_number = $2
          FOR UPDATE
          `,
          [
            hostelId,
            room_number.toString().trim(),
          ]
        );

        if (roomResult.rowCount === 0) {
          await client.query("ROLLBACK");

          return res.status(404).json({
            error: "Selected room not found",
          });
        }

        const room = roomResult.rows[0];

        if (
          Number(room.available_capacity) <= 0
        ) {
          await client.query("ROLLBACK");

          return res.status(409).json({
            error: "Selected room has no available beds",
          });
        }

        roomId = room.id;
        assignedRoom = room.room_number;

        await client.query(
          `
          UPDATE rooms
          SET available_capacity =
            available_capacity - 1
          WHERE id = $1
          `,
          [roomId]
        );
      }

      // --------------------------------------------------
      // 6. Reserve one hostel bed
      // --------------------------------------------------
      const hostelCapacityResult =
        await client.query(
          `
          UPDATE hostels
          SET available_capacity =
            available_capacity - 1
          WHERE id = $1
            AND available_capacity > 0
          RETURNING available_capacity
          `,
          [hostelId]
        );

      if (
        hostelCapacityResult.rowCount === 0
      ) {
        await client.query("ROLLBACK");

        return res.status(409).json({
          error: "Hostel has no available capacity",
        });
      }

      // --------------------------------------------------
      // 7. Create confirmed booking
      // Current Add Student form has no stay dates,
      // so default stay = today to 30 days from today.
      // --------------------------------------------------
      const bookingResult =
        await client.query(
          `
          INSERT INTO bookings
          (
           user_id,
hostel_id,
room_id,
bed_number,
check_in,
check_out,
status
          )
          VALUES
          (
           $1,
$2,
$3,
$4,
CURRENT_DATE,
CURRENT_DATE + 30,
'confirmed'
          )
          RETURNING *
          `,
          [
  student.id,
  hostelId,
  roomId,
  bedNumber,
]
        );

      await client.query(
        `INSERT INTO booking_applications
          (booking_id,student_name,student_email,student_phone,student_cnic,university,department,semester,parent_name,parent_phone,parent_relation,parent_city,parent_email)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)`,
        [
          bookingResult.rows[0].id,
          name.trim(), email.trim().toLowerCase(), phone?.trim() || null,
          cnic?.trim() || null, university?.trim() || null,
          department?.trim() || null, semester?.trim() || null,
          parent_name?.trim() || null, parent_phone?.trim() || null,
          parent_relation?.trim() || null, parent_city?.trim() || city?.trim() || null,
          parent_email?.trim().toLowerCase() || null,
        ],
      );

      await client.query("COMMIT");

      return res.status(201).json({
        success: true,

        message:
          "Student created successfully",

        student: {
          ...student,

          hostel_id: hostelId,

          booking_id:
            bookingResult.rows[0].id,

          booking_status:
            bookingResult.rows[0].status,

          room_id: roomId,

          room_number:
            assignedRoom,
        },

        booking:
          bookingResult.rows[0],

        temp_password:
          temporaryPassword,
      });
    } catch (error) {
      try {
        await client.query("ROLLBACK");
      } catch (_) {
        // Ignore rollback error
      }

      console.error(
        "OWNER CREATE STUDENT FAILED:",
        error
      );

      return res.status(500).json({
        error:
          "Failed to create student",
        detail: error.message,
      });
    } finally {
      client.release();
    }
  }
);

app.patch("/api/owner/students/:id", requireAuth, requireRole("owner"), async (req, res) => {
  const studentId = Number(req.params.id);
  if (!isValidInt(studentId)) return res.status(400).json({ error: "Invalid student id" });
  const text = (key, limit = 254) => String(req.body?.[key] || "").trim().slice(0, limit);
  const name = text("name", 160), email = text("email", 254).toLowerCase(), phone = text("phone", 32);
  if (!name || !email) return res.status(400).json({ error: "Student name and email are required" });
  const client = await pool.connect();
  try {
    await ensureBookingApplicationTable();
    await client.query("BEGIN");
    const booking = await client.query(
      `SELECT b.id FROM bookings b JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id
       WHERE b.user_id=$1 AND ho.owner_user_id=$2 ORDER BY b.id DESC LIMIT 1 FOR UPDATE`,
      [studentId, req.user.id],
    );
    if (!booking.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Student not found for this hostel" }); }
    const duplicate = await client.query("SELECT id FROM users WHERE LOWER(email)=LOWER($1) AND id<>$2", [email, studentId]);
    if (duplicate.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "This email is already in use" }); }
    const user = await client.query("UPDATE users SET name=$1,email=$2,phone=$3 WHERE id=$4 AND role='student' RETURNING id,name,email,phone", [name,email,phone || null,studentId]);
    await client.query(
      `INSERT INTO booking_applications
        (booking_id,student_name,student_email,student_phone,student_cnic,university,department,semester,parent_name,parent_phone,parent_relation,parent_city,parent_email)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
       ON CONFLICT (booking_id) DO UPDATE SET
         student_name=EXCLUDED.student_name,student_email=EXCLUDED.student_email,student_phone=EXCLUDED.student_phone,
         student_cnic=EXCLUDED.student_cnic,university=EXCLUDED.university,department=EXCLUDED.department,semester=EXCLUDED.semester,
         parent_name=EXCLUDED.parent_name,parent_phone=EXCLUDED.parent_phone,parent_relation=EXCLUDED.parent_relation,
         parent_city=EXCLUDED.parent_city,parent_email=EXCLUDED.parent_email,updated_at=NOW()`,
      [booking.rows[0].id,name,email,phone || null,text("cnic",32) || null,text("university",240) || null,text("department",240) || null,text("semester",32) || null,text("parent_name",160) || null,text("parent_phone",32) || null,text("parent_relation",80) || null,text("parent_city",240) || null,text("parent_email",254).toLowerCase() || null],
    );
    await client.query("COMMIT");
    res.json({ success: true, student: user.rows[0] });
  } catch (error) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to update student details" });
  } finally { client.release(); }
});
app.get("/api/owner/bookings", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(`SELECT b.*, u.name AS student_name, u.email AS student_email, h.name AS hostel_name FROM bookings b JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id JOIN users u ON u.id=b.user_id JOIN hostels h ON h.id=b.hostel_id WHERE ho.owner_user_id=$1 ORDER BY b.created_at DESC`, [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch owner bookings" }); }
});

// Owner accepts an enquiry by issuing a separate hostel-access credential.
// It never changes the student's normal discovery-account password.
app.post("/api/owner/bookings/:id/hostel-access-credentials", requireAuth, requireRole("owner"), async (req, res) => {
  const client = await pool.connect();
  try {
    await ensureHostelAccessCredentialTable();
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    await client.query("BEGIN");
    const booking = await client.query(
      `SELECT b.id,b.user_id,b.hostel_id,b.status,u.email,u.name
         FROM bookings b
         JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id
         JOIN users u ON u.id=b.user_id
        WHERE b.id=$1 AND ho.owner_user_id=$2
        FOR UPDATE`,
      [bookingId, req.user.id]
    );
    if (!booking.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking request not found" }); }
    if (booking.rows[0].status !== "pending") { await client.query("ROLLBACK"); return res.status(409).json({ error: "Credentials can only be issued for a pending request" }); }

    const temporaryPassword = `SBH-${crypto.randomBytes(5).toString("hex").toUpperCase()}`;
    const passwordHash = await hashPassword(temporaryPassword);
    const record = await client.query(
      `INSERT INTO hostel_access_credentials
         (booking_id,student_user_id,hostel_id,username,password_hash,issued_by_user_id,issued_at,activated_at)
       VALUES ($1,$2,$3,$4,$5,$6,NOW(),NULL)
       ON CONFLICT (booking_id) DO UPDATE
         SET username=EXCLUDED.username,password_hash=EXCLUDED.password_hash,
             issued_by_user_id=EXCLUDED.issued_by_user_id,issued_at=NOW(),activated_at=NULL
       RETURNING id,booking_id,username,issued_at`,
      [bookingId, booking.rows[0].user_id, booking.rows[0].hostel_id, booking.rows[0].email, passwordHash, req.user.id]
    );
    await client.query(
      `INSERT INTO booking_status_history (booking_id,previous_status,next_status,actor_user_id,reason)
       VALUES ($1,'pending','pending',$2,'owner issued hostel access credential')`,
      [bookingId, req.user.id]
    );
    await client.query("COMMIT");
    res.status(201).json({
      credential: record.rows[0],
      student_name: booking.rows[0].name,
      temporary_password: temporaryPassword,
    });
  } catch (error) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to generate hostel credential", detail: error.message });
  } finally { client.release(); }
});

// Parent access is generated only after the selected student's booking has
// become confirmed. The parent is linked to that exact student in the DB.
app.post("/api/owner/bookings/:id/parent-credentials", requireAuth, requireRole("owner"), async (req, res) => {
  const client = await pool.connect();
  try {
    await ensureBookingApplicationTable();
    await ensureParentStudentLinkTable();
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    await client.query("BEGIN");
    const booking = await client.query(
      `SELECT b.id,b.user_id,ba.parent_name,ba.parent_email
       FROM bookings b JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id
       LEFT JOIN booking_applications ba ON ba.booking_id=b.id
       LEFT JOIN hostel_access_credentials hac ON hac.booking_id=b.id
       WHERE b.id=$1 AND ho.owner_user_id=$2
         AND (b.status='confirmed' OR hac.id IS NOT NULL)
       FOR UPDATE OF b`,
      [bookingId, req.user.id]
    );
    if (!booking.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Approve the student's booking first, then generate parent credentials" }); }
    const parentName = String(booking.rows[0].parent_name || req.body?.parent_name || '').trim();
    const parentEmail = String(booking.rows[0].parent_email || req.body?.parent_email || '').trim().toLowerCase();
    if (!parentName || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(parentEmail)) { await client.query("ROLLBACK"); return res.status(400).json({ error: "This student's booking form needs a parent name and valid parent email" }); }
    await client.query(
      `INSERT INTO booking_applications (booking_id,parent_name,parent_email)
       VALUES ($1,$2,$3)
       ON CONFLICT (booking_id) DO UPDATE SET parent_name=EXCLUDED.parent_name,parent_email=EXCLUDED.parent_email,updated_at=NOW()`,
      [bookingId, parentName, parentEmail]
    );
    const existing = await client.query("SELECT id,role FROM users WHERE LOWER(email)=LOWER($1) FOR UPDATE", [parentEmail]);
    let parentId;
    let temporaryPassword = null;
    if (existing.rowCount) {
      if (existing.rows[0].role !== 'parent') { await client.query("ROLLBACK"); return res.status(409).json({ error: "That email already belongs to a non-parent account" }); }
      parentId = existing.rows[0].id;
    } else {
      temporaryPassword = `SBP-${crypto.randomBytes(5).toString("hex").toUpperCase()}`;
      const parent = await client.query("INSERT INTO users (name,email,password_hash,role) VALUES ($1,$2,$3,'parent') RETURNING id", [parentName, parentEmail, await hashPassword(temporaryPassword)]);
      parentId = parent.rows[0].id;
    }
    const invitationHash = crypto.createHash('sha256').update(crypto.randomBytes(32)).digest('hex');
    await client.query(
      `INSERT INTO parent_student_links (parent_user_id,student_user_id,invited_by_student_id,invitation_token_hash,status,expires_at,accepted_at)
       VALUES ($1,$2,$2,$3,'active',NOW() + INTERVAL '365 days',NOW())
       ON CONFLICT (parent_user_id,student_user_id) WHERE status IN ('pending','active') DO UPDATE SET status='active',accepted_at=NOW(),revoked_at=NULL`,
      [parentId, booking.rows[0].user_id, invitationHash]
    );
    await client.query("COMMIT");
    res.status(201).json({ parent_name: parentName, parent_email: parentEmail, temporary_password: temporaryPassword, existing_account: temporaryPassword == null });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Parent credential generation failed:", error.message);
    res.status(500).json({ error: "Failed to generate parent credentials" });
  } finally { client.release(); }
});

app.get("/api/owner/parent-credentials", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await ensureParentStudentLinkTable();
    const result = await pool.query(
      `SELECT DISTINCT ON (p.id,l.student_user_id) p.id,p.name,p.email,p.phone,u.name AS student_name,l.created_at
       FROM parent_student_links l JOIN users p ON p.id=l.parent_user_id JOIN users u ON u.id=l.student_user_id
       JOIN bookings b ON b.user_id=l.student_user_id
       LEFT JOIN hostel_access_credentials hac ON hac.booking_id=b.id
       JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id
       WHERE ho.owner_user_id=$1 AND l.status='active' AND (b.status='confirmed' OR hac.id IS NOT NULL)
       ORDER BY p.id,l.student_user_id,l.created_at DESC`, [req.user.id]
    );
    res.json({ parents: result.rows });
  } catch (error) { res.status(500).json({ error: "Failed to load parent credentials" }); }
});

app.patch("/api/owner/bookings/:id", requireAuth, requireRole("owner"), async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = Number(req.params.id); const nextStatus = normalizeBookingStatus(req.body.status);
    if (!isValidInt(bookingId) || !isBookingStatus(nextStatus)) return res.status(400).json({ error: "A valid booking id and status are required" });
    await client.query("BEGIN");
    const booking = await client.query("SELECT b.* FROM bookings b JOIN hostel_owners ho ON ho.hostel_id=b.hostel_id WHERE b.id=$1 AND ho.owner_user_id=$2 FOR UPDATE", [bookingId, req.user.id]);
    if (booking.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    const current = normalizeBookingStatus(booking.rows[0].status);
    if (current !== "pending" || !["confirmed","rejected"].includes(nextStatus)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Only pending bookings may be confirmed or rejected" }); }
    if (nextStatus === "confirmed") {
      const missing = await client.query(`SELECT r.id,r.document_code,r.label FROM booking_document_requirements r
        WHERE r.hostel_id=$1 AND r.active=true AND r.required=true AND NOT EXISTS (
          SELECT 1 FROM booking_documents d WHERE d.booking_id=$2 AND d.requirement_id=r.id AND d.status='accepted' AND d.deleted_at IS NULL
        )`, [booking.rows[0].hostel_id, bookingId]);
      if (missing.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Required booking documents are not accepted", missing_requirements: missing.rows }); }
    }
    if (isCapacityHoldingStatus(current) && !isCapacityHoldingStatus(nextStatus)) { await releaseHostelCapacity(client, booking.rows[0].hostel_id); const hold=await client.query("UPDATE booking_inventory_holds SET released_at=NOW() WHERE booking_id=$1 AND released_at IS NULL AND converted_at IS NULL RETURNING room_id",[bookingId]); if(hold.rowCount)await client.query("UPDATE rooms SET available_capacity=LEAST(capacity,available_capacity+1) WHERE id=$1",[hold.rows[0].room_id]); }
    if (!isCapacityHoldingStatus(current) && isCapacityHoldingStatus(nextStatus)) { const hostel = await client.query("SELECT available_capacity FROM hostels WHERE id=$1 FOR UPDATE", [booking.rows[0].hostel_id]); if (Number(hostel.rows[0].available_capacity) <= 0) { await client.query("ROLLBACK"); return res.status(409).json({ error: "No beds available" }); } await client.query("UPDATE hostels SET available_capacity=available_capacity-1 WHERE id=$1", [booking.rows[0].hostel_id]); }
    const updated = await client.query("UPDATE bookings SET status=$1,expires_at=NULL,updated_at=NOW() WHERE id=$2 RETURNING *", [nextStatus, bookingId]);
    await client.query("INSERT INTO booking_status_history (booking_id,previous_status,next_status,actor_user_id,reason) VALUES ($1,$2,$3,$4,$5)", [bookingId,current,nextStatus,req.user.id,String(req.body.reason||'owner decision').slice(0,2000)]);
    await client.query("COMMIT"); res.json({ booking: updated.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); console.error("Warden booking update failed:", error.message); res.status(500).json({ error: "Failed to update booking" }); } finally { client.release(); }
});

app.get("/api/owner/complaints", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(`SELECT c.*, u.name AS student_name, h.name AS hostel_name FROM complaints c JOIN hostel_owners ho ON ho.hostel_id=c.hostel_id JOIN users u ON u.id=c.user_id JOIN hostels h ON h.id=c.hostel_id WHERE ho.owner_user_id=$1 ORDER BY c.created_at DESC`, [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch owner complaints" }); }
});

app.patch("/api/owner/complaints/:id", requireAuth, requireRole("owner"), async (req, res) => {
  const complaintId = Number(req.params.id);
  if (!isValidInt(complaintId)) return res.status(400).json({ error: "Invalid complaint id" });
  const visible = await pool.query(
    "SELECT 1 FROM complaints c JOIN hostel_owners ho ON ho.hostel_id=c.hostel_id WHERE c.id=$1 AND ho.owner_user_id=$2",
    [complaintId, req.user.id]
  );
  if (!visible.rowCount) return res.status(404).json({ error: "Complaint not found" });
  return res.status(403).json({ error: "Owners have read-only complaint oversight; an assigned warden performs complaint workflow actions" });
  /* istanbul ignore next -- retained temporarily for response compatibility audit */
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const complaintId = Number(req.params.id); const status = req.body.status;
    if (!isValidInt(complaintId) || !isValidComplaintStatus(status)) return res.status(400).json({ error: "A valid complaint id and status are required" });
    if (status === "closed") return res.status(400).json({ error: "Wardens resolve complaints; students or the auto-close job close them" });
    await client.query("BEGIN");
    const complaint = await client.query("SELECT c.* FROM complaints c JOIN hostel_owners ho ON ho.hostel_id=c.hostel_id WHERE c.id=$1 AND ho.owner_user_id=$2 FOR UPDATE", [complaintId, req.user.id]);
    if (complaint.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Complaint not found" }); }
    if (!canTransitionComplaintStatus(complaint.rows[0].status, status)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Invalid complaint status transition" }); }
    const finalizing = status === "resolved";
    const existingResolution = finalizing ? await client.query("SELECT id FROM complaint_resolution_records WHERE complaint_id=$1", [complaintId]) : null;
    const resolutionFieldsSupplied = ["safe_resolution_summary", "actions_taken", "prevention_follow_up_notes"].some((key) => Object.hasOwn(req.body, key));
    if (existingResolution?.rowCount && resolutionFieldsSupplied) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Resolution record is append-only and cannot be changed" }); }
    const resolution = finalizing && !existingResolution?.rowCount ? complaintResolutionInput(req.body) : null;
    if (finalizing && !existingResolution?.rowCount && !resolution) { await client.query("ROLLBACK"); return res.status(400).json({ error: "Finalization requires safe_resolution_summary and actions_taken (each up to 4000 characters), with optional prevention_follow_up_notes" }); }
    const updated = await client.query("UPDATE complaints SET status=$1::varchar, resolved_at=CASE WHEN $1::varchar IN ('resolved','closed') THEN NOW() ELSE NULL END WHERE id=$2 RETURNING *", [status, complaintId]);
    await client.query("INSERT INTO complaint_status_history (complaint_id, previous_status, next_status, changed_by_user_id) VALUES ($1,$2,$3,$4)", [complaintId, complaint.rows[0].status, status, req.user.id]);
    const record = resolution && !existingResolution?.rowCount ? await client.query("INSERT INTO complaint_resolution_records (complaint_id,final_status,safe_resolution_summary,actions_taken,prevention_follow_up_notes,resolver_user_id,resolved_at) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *", [complaintId, status, resolution.summary, resolution.actions, resolution.prevention, req.user.id, updated.rows[0].resolved_at]) : null;
    if (record) await client.query("INSERT INTO complaint_activity_events (complaint_id,event_type,actor_user_id,details) VALUES ($1,'resolution_recorded',$2,$3::jsonb)", [complaintId, req.user.id, JSON.stringify({ resolution_record_id: record.rows[0].id, final_status: status })]);
    await client.query("COMMIT"); res.json({ complaint: updated.rows[0], resolution_record: record && record.rows[0] });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Warden complaint update failed:", error.message);
    res.status(500).json({
      error: "Failed to update complaint",
      ...(process.env.RUN_DB_TESTS === "1" ? { detail: error.message } : {}),
    });
  } finally { client.release(); }
});

app.get(
  "/api/owner/dashboard",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    try {
      await operationalTablesReady;

      const result = await pool.query(
        `
        SELECT
          h.id AS hostel_id,
          h.name AS hostel_name,
          h.city,
          h.hostel_type,
          h.total_rooms,

          COALESCE(h.total_capacity, 0)::int AS total,

          GREATEST(
            COALESCE(h.total_capacity, 0) -
            COALESCE(h.available_capacity, 0),
            0
          )::int AS occupied,

          COALESCE(h.available_capacity, 0)::int AS available,

          (
            SELECT COUNT(DISTINCT b.user_id)::int
            FROM bookings b
            WHERE b.hostel_id = h.id
              AND b.status IN ('pending', 'confirmed')
          ) AS students,

          (
            SELECT COUNT(*)::int
            FROM warden_assignments wa
            WHERE wa.hostel_id = h.id
          ) AS wardens,

          (
            SELECT COUNT(*)::int
            FROM complaints c
            WHERE c.hostel_id = h.id
              AND c.status IN ('open', 'in_progress')
          ) AS complaints,

          (
            SELECT COUNT(*)::int
            FROM bookings b
            WHERE b.hostel_id = h.id
              AND b.status = 'pending'
          ) AS pending_bookings,

          (
            SELECT COUNT(*)::int
            FROM bookings b
            WHERE b.hostel_id = h.id
              AND b.status = 'confirmed'
          ) AS confirmed_bookings,

          (
            SELECT COUNT(*)::int
            FROM bookings b
            WHERE b.hostel_id = h.id
              AND b.status = 'cancelled'
          ) AS cancelled_bookings,

          COALESCE(
            (
              SELECT SUM(p.amount)
              FROM payments p
              INNER JOIN bookings b
                ON b.id = p.booking_id
              WHERE b.hostel_id = h.id
                AND p.status = 'succeeded'
                AND DATE_TRUNC('month', p.created_at)
                    = DATE_TRUNC('month', CURRENT_DATE)
            ),
            0
          )::numeric AS revenue,

          (
            SELECT COUNT(*)::int
            FROM fee_invoices fi
            INNER JOIN student_fee_assignments sfa
              ON sfa.id = fi.assignment_id
            INNER JOIN bookings b
              ON b.id = sfa.booking_id
            WHERE b.hostel_id = h.id
              AND fi.status IN ('pending', 'overdue')
          ) AS pending_fees

        FROM hostels h

        INNER JOIN hostel_owners ho
          ON ho.hostel_id = h.id

        WHERE ho.owner_user_id = $1

        ORDER BY h.id
        `,
        [req.user.id]
      );

      res.json({
        success: true,
        hostels: result.rows
      });

    } catch (error) {
      console.error(
        "OWNER DASHBOARD FAILED:",
        error
      );

      res.status(500).json({
        error: "Failed to fetch owner dashboard",
        detail: error.message
      });
    }
  }
);
// ============================================================================
// OWNER REPORTS / ANALYTICS
// Real data from PostgreSQL
// ============================================================================
app.get(
  "/api/owner/reports",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    try {
      await operationalTablesReady;

      // ----------------------------------------------------------
      // 1. Get all hostels belonging to logged-in owner
      // ----------------------------------------------------------
      const hostelResult = await pool.query(
        `
        SELECT
          h.id AS hostel_id,
          h.name AS hostel_name,
          h.city,
          h.hostel_type,
          COALESCE(h.total_capacity, 0)::int AS total,
          COALESCE(h.available_capacity, 0)::int AS available,
          GREATEST(
            COALESCE(h.total_capacity, 0) -
            COALESCE(h.available_capacity, 0),
            0
          )::int AS occupied
        FROM hostels h
        INNER JOIN hostel_owners ho
          ON ho.hostel_id = h.id
        WHERE ho.owner_user_id = $1
        ORDER BY h.id
        `,
        [req.user.id]
      );

      const reports = [];

      // ----------------------------------------------------------
      // 2. Build real report for each owned hostel
      // ----------------------------------------------------------
      for (const hostel of hostelResult.rows) {
        const hostelId = hostel.hostel_id;

        // Current students
        const studentsResult = await pool.query(
          `
          SELECT COUNT(DISTINCT b.user_id)::int AS count
          FROM bookings b
          WHERE b.hostel_id = $1
            AND b.status IN ('pending', 'confirmed')
          `,
          [hostelId]
        );

        // Wardens
        const wardensResult = await pool.query(
          `
          SELECT COUNT(*)::int AS count
          FROM warden_assignments wa
          WHERE wa.hostel_id = $1
          `,
          [hostelId]
        );

        // Complaints summary
        const complaintsResult = await pool.query(
          `
          SELECT
            COUNT(*)::int AS total,
            COUNT(*) FILTER (
              WHERE status IN ('open', 'in_progress')
            )::int AS active,
            COUNT(*) FILTER (
              WHERE status IN ('resolved', 'closed')
            )::int AS resolved
          FROM complaints
          WHERE hostel_id = $1
          `,
          [hostelId]
        );

        // Both verified student receipts and warden-recorded cash payments are
        // valid collections. Keep them together so the owner report always
        // matches the fee register used by wardens.
        const revenueResult = await pool.query(
          `
          SELECT
            COALESCE(SUM(payment.amount), 0)::numeric AS revenue
          FROM (
            SELECT a.amount
            FROM fee_receipt_allocations a
            JOIN fee_receipt_submissions r ON r.id=a.receipt_submission_id
            WHERE r.hostel_id = $1
            UNION ALL
            SELECT c.amount
            FROM warden_cash_fee_payments c
            WHERE c.hostel_id = $1 AND c.voided_at IS NULL
          ) payment
          `,
          [hostelId]
        );

        const feeSummaryResult = await pool.query(
          `SELECT
             COALESCE(SUM(i.amount), 0)::numeric AS invoiced,
             COALESCE(SUM(paid.amount), 0)::numeric AS collected,
             COALESCE(SUM(GREATEST(i.amount - COALESCE(paid.amount, 0), 0)), 0)::numeric AS pending
           FROM fee_invoices i
           JOIN student_fee_assignments a ON a.id=i.assignment_id
           JOIN hostel_fee_schedules s ON s.id=a.schedule_id
           LEFT JOIN LATERAL (
             SELECT COALESCE(SUM(amount), 0)::numeric AS amount FROM (
               SELECT amount FROM fee_receipt_allocations WHERE invoice_id=i.id
               UNION ALL
               SELECT amount FROM warden_cash_fee_payments WHERE invoice_id=i.id AND voided_at IS NULL
             ) allocations
           ) paid ON true
           WHERE s.hostel_id=$1 AND i.status <> 'cancelled'`,
          [hostelId],
        );

        const registeredStudentsResult = await pool.query(
          `SELECT DISTINCT ON (b.user_id)
             b.user_id AS student_id,u.name,u.email,u.phone,b.status AS booking_status,
             COALESCE(r.floor_label,'Ground Floor') AS floor_label,r.room_number,b.bed_number,
             COALESCE(fee.total_due,0)::numeric AS total_fee,
             COALESCE(fee.total_paid,0)::numeric AS paid_fee,
             GREATEST(COALESCE(fee.total_due,0)-COALESCE(fee.total_paid,0),0)::numeric AS remaining_fee
           FROM bookings b
           JOIN users u ON u.id=b.user_id
           LEFT JOIN rooms r ON r.id=b.room_id
           LEFT JOIN LATERAL (
             SELECT SUM(i.amount)::numeric AS total_due,
                    SUM(COALESCE(paid.amount,0))::numeric AS total_paid
             FROM student_fee_assignments a
             JOIN hostel_fee_schedules s ON s.id=a.schedule_id
             JOIN fee_invoices i ON i.assignment_id=a.id
             LEFT JOIN LATERAL (
               SELECT SUM(amount)::numeric AS amount FROM (
                 SELECT amount FROM fee_receipt_allocations WHERE invoice_id=i.id
                 UNION ALL
                 SELECT amount FROM warden_cash_fee_payments WHERE invoice_id=i.id AND voided_at IS NULL
               ) allocations
             ) paid ON true
             WHERE a.booking_id=b.id AND s.hostel_id=$1 AND i.status <> 'cancelled'
           ) fee ON true
           WHERE b.hostel_id=$1 AND b.status IN ('pending','confirmed')
           ORDER BY b.user_id,b.updated_at DESC,b.id DESC`,
          [hostelId],
        );

        const recentComplaintsResult = await pool.query(
          `SELECT c.id,c.category,c.severity,c.status,c.description,c.created_at,
                  u.name AS student_name,COALESCE(r.floor_label,'Ground Floor') AS floor_label,
                  r.room_number,c.bed_number
             FROM complaints c
             JOIN users u ON u.id=c.user_id
             LEFT JOIN rooms r ON r.id=c.room_id
            WHERE c.hostel_id=$1
            ORDER BY c.created_at DESC
            LIMIT 100`,
          [hostelId],
        );

        const roomPerformanceResult = await pool.query(
          `SELECT r.id,r.room_number,COALESCE(r.floor_label,'Ground Floor') AS floor_label,
                  r.capacity,r.available_capacity,r.monthly_rent,
                  GREATEST(r.capacity-r.available_capacity,0)::int AS occupied
             FROM rooms r
            WHERE r.hostel_id=$1
            ORDER BY r.floor_label,r.room_number`,
          [hostelId],
        );

        // ----------------------------------------------------------
        // Last 6 months REAL revenue
        // ----------------------------------------------------------
        const monthlyRevenueResult = await pool.query(
          `
          WITH months AS (
            SELECT generate_series(
              date_trunc('month', CURRENT_DATE) - INTERVAL '5 months',
              date_trunc('month', CURRENT_DATE),
              INTERVAL '1 month'
            ) AS month_start
          )
          SELECT
            TO_CHAR(m.month_start, 'Mon') AS month,
            COALESCE(
              SUM(a.amount),
              0
            )::numeric AS value
          FROM months m
          LEFT JOIN fee_receipt_submissions r ON r.hostel_id=$1
          LEFT JOIN fee_receipt_allocations a
            ON a.receipt_submission_id=r.id
           AND a.allocated_at >= m.month_start
           AND a.allocated_at < m.month_start + INTERVAL '1 month'
          GROUP BY m.month_start
          ORDER BY m.month_start
          `,
          [hostelId]
        );

        // ----------------------------------------------------------
        // Last 6 months REAL student growth
        // Cumulative number of students booked up to each month.
        // ----------------------------------------------------------
        const studentGrowthResult = await pool.query(
          `
          WITH months AS (
            SELECT generate_series(
              date_trunc('month', CURRENT_DATE) - INTERVAL '5 months',
              date_trunc('month', CURRENT_DATE),
              INTERVAL '1 month'
            ) AS month_start
          )
          SELECT
            TO_CHAR(m.month_start, 'Mon') AS month,
            COUNT(DISTINCT b.user_id)::int AS value
          FROM months m
          LEFT JOIN bookings b
            ON b.hostel_id = $1
           AND b.created_at < m.month_start + INTERVAL '1 month'
           AND b.status <> 'cancelled'
          GROUP BY m.month_start
          ORDER BY m.month_start
          `,
          [hostelId]
        );

        const complaintSummary =
          complaintsResult.rows[0] || {
            total: 0,
            active: 0,
            resolved: 0,
          };

        reports.push({
          hostel_id: hostel.hostel_id,
          hostel_name: hostel.hostel_name,
          city: hostel.city,
          hostel_type: hostel.hostel_type,

          total: Number(hostel.total || 0),
          occupied: Number(hostel.occupied || 0),
          available: Number(hostel.available || 0),

          students: Number(
            studentsResult.rows[0]?.count || 0
          ),

          wardens: Number(
            wardensResult.rows[0]?.count || 0
          ),

          complaints: Number(
            complaintSummary.active || 0
          ),

          total_complaints: Number(
            complaintSummary.total || 0
          ),

          resolved_complaints: Number(
            complaintSummary.resolved || 0
          ),

          revenue: Number(
            revenueResult.rows[0]?.revenue || 0
          ),

          fees_collected: Number(feeSummaryResult.rows[0]?.collected || 0),
          fees_pending: Number(feeSummaryResult.rows[0]?.pending || 0),
          registered_students: registeredStudentsResult.rows.map((student) => ({
            ...student,
            total_fee: Number(student.total_fee || 0),
            paid_fee: Number(student.paid_fee || 0),
            remaining_fee: Number(student.remaining_fee || 0),
          })),
          recent_complaints: recentComplaintsResult.rows,
          room_performance: roomPerformanceResult.rows.map((room) => ({
            ...room,
            capacity: Number(room.capacity || 0),
            available_capacity: Number(room.available_capacity || 0),
            occupied: Number(room.occupied || 0),
            monthly_rent: Number(room.monthly_rent || 0),
          })),

          monthly_revenue:
            monthlyRevenueResult.rows.map((row) => ({
              month: row.month,
              value: Number(row.value || 0),
            })),

          student_growth:
            studentGrowthResult.rows.map((row) => ({
              month: row.month,
              value: Number(row.value || 0),
            })),
        });
      }

      res.json({
        success: true,
        reports,
      });
    } catch (error) {
      console.error(
        "OWNER REPORTS FAILED:",
        error
      );

      res.status(500).json({
        error: "Failed to fetch owner reports",
        detail: error.message,
      });
    }
  }
);
// ============================================================================
// GET OWNER WARDENS
// Returns only wardens belonging to hostels owned by the logged-in owner.
// ============================================================================
false && app.get(
  "/api/owner/wardens",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    try {
      await operationalTablesReady;

      const result = await pool.query(
        `
        SELECT
          u.id,
          u.name,
          u.email,
          u.phone,

          wa.hostel_id,
          wa.assigned_at,

          h.name AS hostel_name,

          wp.cnic,
          wp.city,
          COALESCE(wp.shift, 'Morning') AS shift,

          COALESCE(
            wp.assigned_floors,
            '[]'::jsonb
          ) AS floors,

          COALESCE(
            wp.permissions,
            '{}'::jsonb
          ) AS permissions,

          COALESCE(
            wp.account_status,
            'Active'
          ) AS account_status

        FROM warden_assignments wa

        JOIN hostel_owners ho
          ON ho.hostel_id = wa.hostel_id

        JOIN users u
          ON u.id = wa.warden_user_id

        JOIN hostels h
          ON h.id = wa.hostel_id

        LEFT JOIN warden_profiles wp
          ON wp.warden_user_id = u.id

        WHERE ho.owner_user_id = $1

        ORDER BY wa.assigned_at DESC
        `,
        [req.user.id]
      );

      res.json(result.rows);
    } catch (error) {
      console.error(
        "Get owner wardens failed:",
        error
      );

      res.status(500).json({
        error: "Failed to fetch wardens"
      });
    }
  }
);


// ============================================================================
// CREATE WARDEN ACCOUNT + ASSIGN TO OWNER'S HOSTEL
//
// Owner supplies Warden details.
// Backend:
//   1. verifies hostel ownership
//   2. verifies email is unused
//   3. generates temporary password
//   4. hashes password
//   5. creates users row with role = warden
//   6. creates warden profile
//   7. assigns Warden to hostel
//   8. returns temporary password ONCE to owner
// ============================================================================
false && app.post(
  "/api/owner/wardens",
  requireAuth,
  requireRole("owner"),
  async (req, res) => {
    const client = await pool.connect();

    try {
      await operationalTablesReady;

      const hostelId = Number(req.body.hostel_id);

      const name = String(
        req.body.name || ""
      ).trim();

      const email = String(
        req.body.email || ""
      )
        .trim()
        .toLowerCase();

      const phone = String(
        req.body.phone || ""
      ).trim();

      const cnic = String(
        req.body.cnic || ""
      ).trim();

      const city = String(
        req.body.city || ""
      ).trim();

      const shift = String(
        req.body.shift || "Morning"
      ).trim();

      const floors = Array.isArray(req.body.floors)
        ? req.body.floors
        : [];

      const permissions =
        req.body.permissions &&
        typeof req.body.permissions === "object" &&
        !Array.isArray(req.body.permissions)
          ? req.body.permissions
          : {};

      // ----------------------------------------------------------------------
      // BASIC VALIDATION
      // ----------------------------------------------------------------------

      if (!isValidInt(hostelId)) {
        return res.status(400).json({
          error: "A valid hostel_id is required"
        });
      }

      if (!name) {
        return res.status(400).json({
          error: "Warden name is required"
        });
      }

      if (!email) {
        return res.status(400).json({
          error: "Warden email is required"
        });
      }

      const emailPattern =
        /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

      if (!emailPattern.test(email)) {
        return res.status(400).json({
          error: "Please provide a valid email address"
        });
      }

      if (!phone) {
        return res.status(400).json({
          error: "Warden phone number is required"
        });
      }

      if (!cnic) {
        return res.status(400).json({
          error: "Warden CNIC is required"
        });
      }

      if (!city) {
        return res.status(400).json({
          error: "Warden city is required"
        });
      }

      if (
        ![
          "Morning",
          "Evening",
          "Night"
        ].includes(shift)
      ) {
        return res.status(400).json({
          error: "Invalid Warden shift"
        });
      }

      // ----------------------------------------------------------------------
      // SECURITY:
      // Logged-in Owner must actually own the requested hostel.
      // ----------------------------------------------------------------------

      const ownsHostel =
        await ownerHasHostel(
          client,
          req.user.id,
          hostelId
        );

      if (!ownsHostel) {
        return res.status(403).json({
          error:
            "You are not allowed to add a Warden to this hostel"
        });
      }

      await client.query("BEGIN");

      // ----------------------------------------------------------------------
      // EMAIL MUST BE UNIQUE
      // ----------------------------------------------------------------------

      const existingUser =
        await client.query(
          `
          SELECT
            id,
            name,
            email,
            role
          FROM users
          WHERE LOWER(email) = LOWER($1)
          LIMIT 1
          `,
          [email]
        );

      if (existingUser.rowCount > 0) {
        await client.query("ROLLBACK");

        return res.status(409).json({
          error:
            "An account with this email already exists. Please use another email address."
        });
      }

      // ----------------------------------------------------------------------
      // GENERATE TEMPORARY PASSWORD
      //
      // Example:
      // SB@7F31C9A2x
      //
      // We return this plain password ONLY once.
      // Database receives only password_hash.
      // ----------------------------------------------------------------------

      const randomPart =
        crypto
          .randomBytes(6)
          .toString("hex")
          .toUpperCase();

      const temporaryPassword =
        `SB@${randomPart}x`;

      const passwordHash =
        await hashPassword(
          temporaryPassword
        );

      // ----------------------------------------------------------------------
      // CREATE WARDEN USER
      // ----------------------------------------------------------------------

      const userResult =
        await client.query(
          `
          INSERT INTO users (
            name,
            email,
            phone,
            password_hash,
            role
          )
          VALUES (
            $1,
            $2,
            $3,
            $4,
            'warden'
          )

          RETURNING
            id,
            name,
            email,
            phone,
            role
          `,
          [
            name,
            email,
            phone,
            passwordHash
          ]
        );

      const warden =
        userResult.rows[0];

      // ----------------------------------------------------------------------
      // CREATE WARDEN PROFILE
      // ----------------------------------------------------------------------

      await client.query(
        `
        INSERT INTO warden_profiles (
          warden_user_id,
          cnic,
          city,
          shift,
          assigned_floors,
          permissions,
          account_status
        )
        VALUES (
          $1,
          $2,
          $3,
          $4,
          $5::jsonb,
          $6::jsonb,
          'Active'
        )
        `,
        [
          warden.id,
          cnic,
          city,
          shift,
          JSON.stringify(floors),
          JSON.stringify(permissions)
        ]
      );

      // ----------------------------------------------------------------------
      // ASSIGN WARDEN TO THE OWNER'S HOSTEL
      // ----------------------------------------------------------------------

      const assignmentResult =
        await client.query(
          `
          INSERT INTO warden_assignments (
            warden_user_id,
            hostel_id,
            assigned_by_user_id
          )
          VALUES (
            $1,
            $2,
            $3
          )

          RETURNING
            warden_user_id,
            hostel_id,
            assigned_by_user_id,
            assigned_at
          `,
          [
            warden.id,
            hostelId,
            req.user.id
          ]
        );

      const hostelResult =
        await client.query(
          `
          SELECT
            id,
            name,
            city
          FROM hostels
          WHERE id = $1
          `,
          [hostelId]
        );

      await client.query("COMMIT");

      // ----------------------------------------------------------------------
      // IMPORTANT:
      // temporaryPassword is intentionally NOT saved as plain text anywhere.
      // Owner must copy/share it now.
      // ----------------------------------------------------------------------

      res.status(201).json({
        success: true,

        message:
          "Warden account created successfully",

        warden: {
          ...warden,

          cnic,
          city,
          shift,
          floors,
          permissions,

          account_status: "Active"
        },

        hostel:
          hostelResult.rows[0],

        assignment:
          assignmentResult.rows[0],

        temp_password:
          temporaryPassword
      });
    } catch (error) {
      try {
        await client.query("ROLLBACK");
      } catch (_) {
        // Ignore rollback errors.
      }

      console.error(
        "Create Warden failed:",
        error
      );

      res.status(500).json({
        error:
          "Failed to create Warden account"
      });
    } finally {
      client.release();
    }
  }
);


// GET /api/warden/dashboard
app.get("/api/warden/dashboard", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await operationalTablesReady;

    const result = await pool.query(
      `SELECT
         h.id,
         h.name,
         h.city,
         h.total_capacity,
         h.available_capacity,
         COUNT(DISTINCT b.user_id) FILTER (
           WHERE b.status = 'confirmed'
         )::int AS confirmed_students,
         COUNT(DISTINCT c.id) FILTER (
           WHERE c.status NOT IN ('resolved', 'closed')
         )::int AS pending_complaints,
         COALESCE(
           ROUND(
             100.0 * COUNT(DISTINCT d.student_user_id) FILTER (
               WHERE d.status = 'present'
             ) / NULLIF(
               COUNT(DISTINCT b.user_id) FILTER (
                 WHERE b.status = 'confirmed'
               ),
               0
             )
           ),
           0
         )::int AS attendance_percentage
       FROM warden_assignments wa
       JOIN hostels h ON h.id = wa.hostel_id
       LEFT JOIN bookings b ON b.hostel_id = h.id
       LEFT JOIN complaints c ON c.hostel_id = h.id
       LEFT JOIN daily_warden_attendance d
         ON d.hostel_id = h.id
        AND d.attendance_date = (NOW() AT TIME ZONE 'Asia/Karachi')::date
       WHERE wa.warden_user_id = $1
       GROUP BY h.id
       ORDER BY h.id
       LIMIT 1`,
      [req.user.id]
    );

    if (result.rowCount === 0) {
      return res.status(404).json({
        error: "No hostel is assigned to this warden"
      });
    }

    const hostel = result.rows[0];

    res.json({
      success: true,
      hostel_id: hostel.id,
      hostel_name: hostel.name,
      city: hostel.city,
      students: hostel.confirmed_students,
      complaints: hostel.pending_complaints,
      attendance: hostel.attendance_percentage,
      available_beds: hostel.available_capacity,
      hostel
    });
  } catch (err) {
    console.error("Warden dashboard fetch failed:", err);
    res.status(500).json({
      error: "Dashboard fetch failed: " + err.message
    });
  }
});

app.get("/api/warden/bookings", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT b.*, u.name AS student_name, u.email AS student_email, h.name AS hostel_name
       FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id
       JOIN users u ON u.id=b.user_id JOIN hostels h ON h.id=b.hostel_id
       WHERE wa.warden_user_id=$1 ORDER BY b.created_at DESC`, [req.user.id]
    );
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch warden bookings" }); }
});

// Students assigned to the logged-in warden's hostel.  This is used by
// operational forms (such as penalty assignment), so the UI never has to
// rely on manually typed names or room numbers.
app.get("/api/warden/students", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT DISTINCT ON (student_id) * FROM (
        SELECT u.id AS student_id, u.name, u.email, u.phone,
               b.id AS booking_id, b.status AS booking_status, b.room_id,
               r.room_number, r.floor_label, b.bed_number,
               COALESCE(d.cnic,ba.student_cnic) AS cnic,
               COALESCE(d.address,ba.parent_city) AS address,
               COALESCE(d.guardian_name,ba.parent_name) AS guardian_name,
               COALESCE(d.guardian_relationship,ba.parent_relation) AS guardian_relationship,
               COALESCE(d.guardian_phone,ba.parent_phone) AS guardian_phone,
               COALESCE(d.guardian_email,ba.parent_email) AS guardian_email,
               COALESCE(d.guardian_address,ba.parent_city) AS guardian_address
          FROM warden_assignments wa
          JOIN bookings b ON b.hostel_id=wa.hostel_id
          JOIN users u ON u.id=b.user_id AND u.role='student'
          LEFT JOIN rooms r ON r.id=b.room_id
          LEFT JOIN warden_student_records d ON d.student_user_id=u.id
          LEFT JOIN booking_applications ba ON ba.booking_id=b.id
         WHERE wa.warden_user_id=$1 AND b.status IN ('pending','confirmed')
        UNION ALL
        SELECT u.id AS student_id, u.name, u.email, u.phone,
               NULL::integer AS booking_id, NULL::varchar AS booking_status,
               NULL::integer AS room_id, NULL::varchar AS room_number, NULL::varchar AS floor_label, NULL::varchar AS bed_number,
               d.cnic, d.address, d.guardian_name, d.guardian_relationship,
               d.guardian_phone, d.guardian_email, d.guardian_address
          FROM warden_student_records d
          JOIN users u ON u.id=d.student_user_id AND u.role='student'
          JOIN warden_assignments wa ON wa.hostel_id=d.hostel_id
         WHERE wa.warden_user_id=$1
      ) students
      ORDER BY student_id, booking_id DESC NULLS LAST`,
      [req.user.id],
    );
    res.json(result.rows.map((row) => ({ ...row, id: row.student_id })));
  } catch (error) {
    console.error("Get warden students failed:", error);
    res.status(500).json({ error: "Failed to fetch warden students" });
  }
});

function wardenStudentPayload(body) {
  const text = (key, limit = 4000) => String(body[key] || '').trim().slice(0, limit);
  return {
    name: text('name', 160), email: text('email', 180).toLowerCase(), phone: text('phone', 30),
    cnic: text('cnic', 20), address: text('address'), guardian_name: text('guardian_name', 160),
    guardian_relationship: text('guardian_relationship', 80), guardian_phone: text('guardian_phone', 30),
    guardian_email: text('guardian_email', 180).toLowerCase(), guardian_address: text('guardian_address'),
  };
}

async function wardenCanManageStudent(wardenId, studentId) {
  const access = await pool.query(
    `SELECT 1 FROM warden_assignments wa
      LEFT JOIN bookings b ON b.hostel_id=wa.hostel_id AND b.user_id=$2 AND b.status IN ('pending','confirmed')
      LEFT JOIN warden_student_records d ON d.hostel_id=wa.hostel_id AND d.student_user_id=$2
     WHERE wa.warden_user_id=$1 AND (b.id IS NOT NULL OR d.student_user_id IS NOT NULL)
     LIMIT 1`,
    [wardenId, studentId],
  );
  return access.rowCount > 0;
}

app.get("/api/warden/students/:id", requireAuth, requireRole("warden"), async (req, res) => {
  const studentId = Number(req.params.id);
  if (!isValidInt(studentId) || !await wardenCanManageStudent(req.user.id, studentId)) return res.status(404).json({ error: 'Student not found' });
  const result = await pool.query(
    `SELECT u.id, u.name, u.email, u.phone, COALESCE(d.cnic,ba.student_cnic) AS cnic,
            COALESCE(d.address,ba.parent_city) AS address, COALESCE(d.guardian_name,ba.parent_name) AS guardian_name,
            COALESCE(d.guardian_relationship,ba.parent_relation) AS guardian_relationship,
            COALESCE(d.guardian_phone,ba.parent_phone) AS guardian_phone,
            COALESCE(d.guardian_email,ba.parent_email) AS guardian_email,
            COALESCE(d.guardian_address,ba.parent_city) AS guardian_address,
            r.room_number
       FROM users u
       LEFT JOIN warden_student_records d ON d.student_user_id=u.id
       LEFT JOIN LATERAL (
         SELECT room_id FROM bookings WHERE user_id=u.id AND status IN ('pending','confirmed') ORDER BY id DESC LIMIT 1
       ) b ON TRUE
       LEFT JOIN rooms r ON r.id=b.room_id
       LEFT JOIN booking_applications ba ON ba.booking_id=(SELECT id FROM bookings WHERE user_id=u.id AND status IN ('pending','confirmed') ORDER BY id DESC LIMIT 1)
      WHERE u.id=$1`, [studentId],
  );
  res.json(result.rows[0]);
});

app.post("/api/warden/students", requireAuth, requireRole("warden"), async (req, res) => {
  const data = wardenStudentPayload(req.body);
  if (!data.name || !data.email) return res.status(400).json({ error: 'Student full name and email are required' });
  const hostel = await pool.query('SELECT hostel_id FROM warden_assignments WHERE warden_user_id=$1 LIMIT 1', [req.user.id]);
  if (!hostel.rowCount) return res.status(409).json({ error: 'No hostel is assigned to this warden' });
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const exists = await client.query('SELECT id FROM users WHERE email=$1', [data.email]);
    if (exists.rowCount) { await client.query('ROLLBACK'); return res.status(409).json({ error: 'A student with this email already exists' }); }
    const temporaryPassword = `SB@${crypto.randomBytes(6).toString('hex').toUpperCase()}x`;
    const student = await client.query(
      `INSERT INTO users (name,email,phone,password_hash,role) VALUES ($1,$2,$3,$4,'student') RETURNING id,name,email,phone`,
      [data.name, data.email, data.phone || null, await hashPassword(temporaryPassword)],
    );
    await client.query(
      `INSERT INTO warden_student_records
       (student_user_id,hostel_id,cnic,address,guardian_name,guardian_relationship,guardian_phone,guardian_email,guardian_address,created_by_warden_id)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`,
      [student.rows[0].id, hostel.rows[0].hostel_id, data.cnic || null, data.address || null, data.guardian_name || null, data.guardian_relationship || null, data.guardian_phone || null, data.guardian_email || null, data.guardian_address || null, req.user.id],
    );
    await client.query('COMMIT');
    res.status(201).json({ student: student.rows[0], temporary_password: temporaryPassword });
  } catch (error) { await client.query('ROLLBACK'); res.status(500).json({ error: 'Failed to create student record' }); }
  finally { client.release(); }
});

app.put("/api/warden/students/:id", requireAuth, requireRole("warden"), async (req, res) => {
  const studentId = Number(req.params.id), data = wardenStudentPayload(req.body);
  if (!isValidInt(studentId) || !data.name || !data.email || !await wardenCanManageStudent(req.user.id, studentId)) return res.status(400).json({ error: 'Valid student name and email are required' });
  const hostel = await pool.query('SELECT hostel_id FROM warden_assignments WHERE warden_user_id=$1 LIMIT 1', [req.user.id]);
  try {
    await pool.query('UPDATE users SET name=$1,email=$2,phone=$3 WHERE id=$4 AND role=\'student\'', [data.name, data.email, data.phone || null, studentId]);
    await pool.query(
      `INSERT INTO warden_student_records
       (student_user_id,hostel_id,cnic,address,guardian_name,guardian_relationship,guardian_phone,guardian_email,guardian_address,created_by_warden_id)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
       ON CONFLICT (student_user_id) DO UPDATE SET cnic=EXCLUDED.cnic,address=EXCLUDED.address,guardian_name=EXCLUDED.guardian_name,guardian_relationship=EXCLUDED.guardian_relationship,guardian_phone=EXCLUDED.guardian_phone,guardian_email=EXCLUDED.guardian_email,guardian_address=EXCLUDED.guardian_address,updated_at=NOW()`,
      [studentId, hostel.rows[0].hostel_id, data.cnic || null, data.address || null, data.guardian_name || null, data.guardian_relationship || null, data.guardian_phone || null, data.guardian_email || null, data.guardian_address || null, req.user.id],
    );
    res.json({ success: true });
  } catch (error) { if (error.code === '23505') return res.status(409).json({ error: 'This email is already in use' }); res.status(500).json({ error: 'Failed to update student record' }); }
});

app.get("/api/warden/room-allocations", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT b.id AS booking_id, u.id AS student_id, u.name AS student_name,
              r.id AS room_id, r.room_number, COALESCE(r.floor_label, 'Ground Floor') AS floor_label,
              r.capacity, r.available_capacity, b.bed_number
       FROM bookings b
       JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id
       JOIN users u ON u.id=b.user_id
       JOIN rooms r ON r.id=b.room_id
       WHERE wa.warden_user_id=$1 AND b.status='confirmed'
       ORDER BY r.floor_label, r.room_number, b.bed_number`,
      [req.user.id],
    );
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch room allocations" });
  }
});

// Complete room map for the Room & Beds management screen. Unlike the legacy
// allocation list, this also returns rooms with no allocated students.
app.get("/api/warden/rooms", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT r.id, r.room_number, COALESCE(r.floor_label, 'Ground Floor') AS floor_label,
              r.capacity,
              COALESCE(json_agg(json_build_object(
                'name', u.name, 'booking_id', b.id, 'bed_number', b.bed_number
              ) ORDER BY b.bed_number) FILTER (WHERE b.id IS NOT NULL), '[]'::json) AS students
         FROM rooms r
         JOIN warden_assignments wa ON wa.hostel_id=r.hostel_id
         LEFT JOIN bookings b ON b.room_id=r.id AND b.status='confirmed'
         LEFT JOIN users u ON u.id=b.user_id
        WHERE wa.warden_user_id=$1
        GROUP BY r.id, r.room_number, r.floor_label, r.capacity
        ORDER BY r.floor_label, r.room_number`,
      [req.user.id],
    );
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch warden rooms' });
  }
});

app.patch("/api/warden/bookings/:id", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const bookingId = Number(req.params.id); const nextStatus = normalizeBookingStatus(req.body.status);
    if (!isValidInt(bookingId) || !isBookingStatus(nextStatus)) return res.status(400).json({ error: "A valid booking id and status are required" });
    await client.query("BEGIN");
    const booking = await client.query(
      "SELECT b.* FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id WHERE b.id=$1 AND wa.warden_user_id=$2 FOR UPDATE",
      [bookingId, req.user.id]
    );
    if (booking.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    const currentStatus = normalizeBookingStatus(booking.rows[0].status);
    if (currentStatus !== "pending" || !["confirmed","rejected"].includes(nextStatus)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Only pending bookings may be confirmed or rejected" }); }
    if (nextStatus === "confirmed") {
      const missing = await client.query(`SELECT r.id,r.document_code,r.label FROM booking_document_requirements r
        WHERE r.hostel_id=$1 AND r.active=true AND r.required=true AND NOT EXISTS (
          SELECT 1 FROM booking_documents d WHERE d.booking_id=$2 AND d.requirement_id=r.id AND d.status='accepted' AND d.deleted_at IS NULL
        )`, [booking.rows[0].hostel_id, bookingId]);
      if (missing.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Required booking documents are not accepted", missing_requirements: missing.rows }); }
    }
    if (isCapacityHoldingStatus(currentStatus) && !isCapacityHoldingStatus(nextStatus)) { await releaseHostelCapacity(client, booking.rows[0].hostel_id); const hold=await client.query("UPDATE booking_inventory_holds SET released_at=NOW() WHERE booking_id=$1 AND released_at IS NULL AND converted_at IS NULL RETURNING room_id",[bookingId]); if(hold.rowCount)await client.query("UPDATE rooms SET available_capacity=LEAST(capacity,available_capacity+1) WHERE id=$1",[hold.rows[0].room_id]); }
    if (!isCapacityHoldingStatus(currentStatus) && isCapacityHoldingStatus(nextStatus)) {
      const hostel = await client.query("SELECT available_capacity FROM hostels WHERE id=$1 FOR UPDATE", [booking.rows[0].hostel_id]);
      if (Number(hostel.rows[0].available_capacity) <= 0) { await client.query("ROLLBACK"); return res.status(409).json({ error: "No beds available" }); }
      await client.query("UPDATE hostels SET available_capacity=available_capacity-1 WHERE id=$1", [booking.rows[0].hostel_id]);
    }
    const updated = await client.query("UPDATE bookings SET status=$1,expires_at=NULL,updated_at=NOW() WHERE id=$2 RETURNING *", [nextStatus, bookingId]);
    await client.query("INSERT INTO booking_status_history (booking_id,previous_status,next_status,actor_user_id,reason) VALUES ($1,$2,$3,$4,$5)", [bookingId,currentStatus,nextStatus,req.user.id,String(req.body.reason||'warden decision').slice(0,2000)]);
    await createNotification(client, booking.rows[0].user_id, "booking_status", "Booking status updated", `Your booking is now ${nextStatus}.`, { booking_id: bookingId, status: nextStatus }, "booking_updates");
    await client.query("COMMIT");
    res.json({ booking: updated.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to update booking" }); } finally { client.release(); }
});

app.post("/api/warden/complaints", requireAuth, requireRole("warden"), async (req, res) => {
  const studentId = Number(req.body.student_id);
  const description = String(req.body.description || '').trim();
  if (!isValidInt(studentId) || description.length < 3 || description.length > 4000) return res.status(400).json({ error: 'Student and a complaint description of 3 to 4000 characters are required' });
  try {
    const booking = await pool.query(
      `SELECT b.hostel_id,b.room_id FROM bookings b
        JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id
       WHERE b.user_id=$1 AND wa.warden_user_id=$2 AND b.status IN ('pending','confirmed')
       ORDER BY b.id DESC LIMIT 1`,
      [studentId, req.user.id],
    );
    if (!booking.rowCount) return res.status(404).json({ error: 'Student is not assigned to your hostel' });
    const aiResult = await callAiComplaintCategorizer(description);
    if (!aiResult?.success || !aiResult.category) {
      return res.status(503).json({ error: 'The trained complaint AI is unavailable. Start the Complaint AI service and try again.' });
    }
    const category = String(aiResult.category).trim().slice(0, 80);
    const severity = priorityToSeverity(aiResult.priority || 'Medium');
    const confidence = Number.isFinite(Number(aiResult.confidence))
      ? Math.max(0, Math.min(1, Number(aiResult.confidence)))
      : null;
    const saved = await pool.query(
      `INSERT INTO complaints (user_id,hostel_id,room_id,category,severity,status,description,idempotency_key,triage_source,triage_confidence)
       VALUES ($1,$2,$3,$4,$5,'open',$6,$7,'ai',$8) RETURNING *`,
      [studentId, booking.rows[0].hostel_id, booking.rows[0].room_id, category, severity, description, `warden-ai-${req.user.id}-${studentId}-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`, confidence],
    );
    res.status(201).json({ complaint: saved.rows[0], triage: { source: 'ai', category, subcategory: aiResult.subcategory || 'General', priority: aiResult.priority || 'Medium', confidence, suggestion: aiResult.suggestion || null, model_version: aiResult.model_version || null } });
  } catch (error) { res.status(500).json({ error: 'Failed to save complaint' }); }
});

app.post("/api/warden/complaint-analysis", requireAuth, requireRole("warden"), async (req, res) => {
  const description = String(req.body.description || '').trim();
  if (description.length < 3 || description.length > 4000) {
    return res.status(400).json({ error: 'Complaint description must be 3 to 4000 characters.' });
  }
  const aiResult = await callAiComplaintCategorizer(description);
  if (!aiResult?.success || !aiResult.category) {
    return res.status(503).json({ error: 'The trained complaint AI is unavailable. Start the Complaint AI service and try again.' });
  }
  const confidence = Number.isFinite(Number(aiResult.confidence))
    ? Math.max(0, Math.min(1, Number(aiResult.confidence)))
    : null;
  res.json({
    source: 'trained_model',
    model_version: aiResult.model_version || null,
    category: String(aiResult.category),
    subcategory: String(aiResult.subcategory || 'General'),
    priority: String(aiResult.priority || 'Medium'),
    confidence,
    suggestion: String(aiResult.suggestion || 'Review the complaint and respond appropriately.'),
    all_suggestions: Array.isArray(aiResult.all_suggestions) ? aiResult.all_suggestions : [],
    top3_categories: Array.isArray(aiResult.top3_categories) ? aiResult.top3_categories : [],
  });
});

app.get("/api/warden/complaints", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query(
      `SELECT c.*, u.name AS student_name, u.email AS student_email, h.name AS hostel_name,
              r.room_number, COALESCE(r.floor_label, 'Ground Floor') AS floor_label,
              (
                SELECT b.bed_number
                  FROM bookings b
                 WHERE b.user_id=c.user_id AND b.room_id=c.room_id
                   AND b.status IN ('pending','confirmed')
                 ORDER BY CASE b.status WHEN 'confirmed' THEN 0 ELSE 1 END,
                          b.updated_at DESC, b.id DESC
                 LIMIT 1
              ) AS bed_number
       FROM complaints c JOIN warden_assignments wa ON wa.hostel_id=c.hostel_id
       JOIN users u ON u.id=c.user_id JOIN hostels h ON h.id=c.hostel_id
       LEFT JOIN rooms r ON r.id=c.room_id
       WHERE wa.warden_user_id=$1 ORDER BY c.created_at DESC`, [req.user.id]
    );
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch warden complaints" }); }
});

app.patch("/api/warden/complaints/:id", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const complaintId = Number(req.params.id); const status = req.body.status;
    if (!isValidInt(complaintId) || !isValidComplaintStatus(status)) return res.status(400).json({ error: "A valid complaint id and status are required" });
    await client.query("BEGIN");
    const complaint = await client.query(
      "SELECT c.* FROM complaints c JOIN warden_assignments wa ON wa.hostel_id=c.hostel_id WHERE c.id=$1 AND wa.warden_user_id=$2 FOR UPDATE",
      [complaintId, req.user.id]
    );
    if (complaint.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Complaint not found" }); }
    if (!canTransitionComplaintStatus(complaint.rows[0].status, status)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Invalid complaint status transition" }); }
    if (status === "closed") { await client.query("ROLLBACK"); return res.status(400).json({ error: "Students accept resolutions; the durable job closes unchallenged resolutions" }); }
    const finalizing = status === "resolved";
    const existingResolution = finalizing ? await client.query("SELECT id FROM complaint_resolution_records WHERE complaint_id=$1", [complaintId]) : null;
    const resolutionFieldsSupplied = ["safe_resolution_summary", "actions_taken", "prevention_follow_up_notes"].some((key) => Object.hasOwn(req.body, key));
    const resolution = finalizing ? complaintResolutionInput(req.body) : null;
    if (finalizing && !resolution) { await client.query("ROLLBACK"); return res.status(400).json({ error: "Resolution requires safe_resolution_summary and actions_taken (each up to 4000 characters), with optional prevention_follow_up_notes" }); }
    const updated = await client.query("UPDATE complaints SET status=$1::varchar, assigned_to=$2, resolved_at=CASE WHEN $1::varchar='resolved' THEN NOW() ELSE NULL END, resolution_due_at=CASE WHEN $1::varchar='resolved' THEN NOW()+INTERVAL '72 hours' ELSE NULL END WHERE id=$3 RETURNING *", [status, req.user.id, complaintId]);
    await client.query("INSERT INTO complaint_status_history (complaint_id, previous_status, next_status, changed_by_user_id) VALUES ($1,$2,$3,$4)", [complaintId, complaint.rows[0].status, status, req.user.id]);
    const record = resolution && !existingResolution?.rowCount ? await client.query("INSERT INTO complaint_resolution_records (complaint_id,final_status,safe_resolution_summary,actions_taken,prevention_follow_up_notes,resolver_user_id,resolved_at) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *", [complaintId, status, resolution.summary, resolution.actions, resolution.prevention, req.user.id, updated.rows[0].resolved_at]) : null;
    const revision = resolution ? await client.query("INSERT INTO complaint_resolution_revisions (complaint_id,safe_resolution_summary,actions_taken,prevention_follow_up_notes,resolver_user_id) VALUES ($1,$2,$3,$4,$5) RETURNING *", [complaintId,resolution.summary,resolution.actions,resolution.prevention,req.user.id]) : null;
    if (record) await client.query("INSERT INTO complaint_activity_events (complaint_id,event_type,actor_user_id,details) VALUES ($1,'resolution_recorded',$2,$3::jsonb)", [complaintId, req.user.id, JSON.stringify({ resolution_record_id: record.rows[0].id, final_status: status })]);
    await createNotification(client, complaint.rows[0].user_id, "complaint_status", "Complaint status updated", `Your complaint is now ${status}.`, { complaint_id: complaintId, status }, "complaint_updates");
    await client.query("COMMIT");
    res.json({ complaint: updated.rows[0], resolution_record: record && record.rows[0], resolution_revision: revision && revision.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); console.error("Warden complaint update failed:", error.message); res.status(500).json({ error: "Failed to update complaint" }); } finally { client.release(); }
});

app.get("/api/complaints/:id/resolution", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const complaintId = Number(req.params.id);
    if (!isValidInt(complaintId)) return res.status(400).json({ error: "Invalid complaint id" });
    const result = req.user.role === "student"
      ? await pool.query(`SELECT r.id,r.complaint_id,r.final_status,r.safe_resolution_summary,r.resolved_at FROM complaint_resolution_records r JOIN complaints c ON c.id=r.complaint_id WHERE r.complaint_id=$1 AND c.user_id=$2 AND c.status IN ('resolved','closed')`, [complaintId, req.user.id])
      : await pool.query(`SELECT r.* FROM complaint_resolution_records r JOIN complaints c ON c.id=r.complaint_id WHERE r.complaint_id=$1 AND c.status IN ('resolved','closed') AND (($2='owner' AND EXISTS (SELECT 1 FROM hostel_owners ho WHERE ho.hostel_id=c.hostel_id AND ho.owner_user_id=$3)) OR ($2='warden' AND EXISTS (SELECT 1 FROM warden_assignments wa WHERE wa.hostel_id=c.hostel_id AND wa.warden_user_id=$3)))`, [complaintId, req.user.role, req.user.id]);
    if (!result.rowCount) return res.status(404).json({ error: "Resolution record not found" });
    res.json({ resolution: result.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to fetch resolution record" }); }
});

app.post("/api/warden/complaints/:id/triage", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const complaintId = Number(req.params.id), priority = String(req.body.priority || ""), slaDueAt = req.body.sla_due_at, note = String(req.body.note || "").trim();
    if (!isValidInt(complaintId) || !["low", "medium", "high", "urgent"].includes(priority) || !slaDueAt || Number.isNaN(Date.parse(slaDueAt)) || Date.parse(slaDueAt) <= Date.now() || !note || note.length > 2000) return res.status(400).json({ error: "Valid priority, future sla_due_at, and note up to 2000 characters are required" });
    await client.query("BEGIN");
    const complaint = await client.query("SELECT c.* FROM complaints c JOIN warden_assignments wa ON wa.hostel_id=c.hostel_id WHERE c.id=$1 AND wa.warden_user_id=$2 FOR UPDATE", [complaintId, req.user.id]);
    if (!complaint.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Complaint not found" }); }
    const updated = await client.query("UPDATE complaints SET priority=$1,sla_due_at=$2,assigned_to=$3 WHERE id=$4 RETURNING *", [priority, slaDueAt, req.user.id, complaintId]);
    await client.query("INSERT INTO complaint_triage_history (complaint_id,previous_priority,next_priority,previous_sla_due_at,next_sla_due_at,note,triaged_by_user_id) VALUES ($1,$2,$3,$4,$5,$6,$7)", [complaintId, complaint.rows[0].priority, priority, complaint.rows[0].sla_due_at, slaDueAt, note, req.user.id]);
    await createNotification(client, complaint.rows[0].user_id, "complaint_triaged", "Complaint priority updated", `Your complaint has been prioritized as ${priority}.`, { complaint_id: complaintId, priority, sla_due_at: slaDueAt }, "complaint_updates");
    await client.query("COMMIT"); res.json({ complaint: updated.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to triage complaint" }); } finally { client.release(); }
});

async function staffCanManageHostel(client, user, hostelId) {
  if (user.role === "owner") return ownerHasHostel(client, user.id, hostelId);
  if (user.role === "warden") return wardenHasHostel(client, user.id, hostelId);
  return false;
}

app.post("/api/announcements", requireAuth, requireRole("owner", "warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const hostelId = Number(req.body.hostel_id); const title = String(req.body.title || "").trim();
    const body = String(req.body.body || "").trim(); const audience = req.body.audience || "residents";
    if (!isValidInt(hostelId) || !title || !body || !["residents", "booked", "favorited"].includes(audience)) return res.status(400).json({ error: "hostel_id, title, body, and a valid audience are required" });
    if (!await staffCanManageHostel(client, req.user, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    await client.query("BEGIN");
    const announcement = await client.query("INSERT INTO announcements (hostel_id,author_user_id,title,body,audience) VALUES ($1,$2,$3,$4,$5) RETURNING *", [hostelId, req.user.id, title, body, audience]);
    const includeBookings = audience !== "favorited";
    const includeFavorites = audience !== "booked";
    await client.query(
      `WITH recipients AS (
        SELECT DISTINCT user_id FROM bookings WHERE hostel_id=$1 AND status IN ('pending','confirmed') AND $2
        UNION
        SELECT DISTINCT user_id FROM favorites WHERE hostel_id=$1 AND $3
        UNION
        SELECT DISTINCT l.parent_user_id FROM bookings b
        JOIN parent_student_links l ON l.student_user_id=b.user_id AND l.status='active'
        WHERE b.hostel_id=$1 AND b.status IN ('pending','confirmed') AND $2
      )
      INSERT INTO notifications (user_id,announcement_id,type,title,body,data)
      SELECT r.user_id,$4,'announcement',$5,$6,jsonb_build_object('hostel_id',$1)
      FROM recipients r LEFT JOIN notification_preferences np ON np.user_id=r.user_id
      WHERE COALESCE(np.announcements,true)`,
      [hostelId, includeBookings, includeFavorites, announcement.rows[0].id, title, body]
    );
    await client.query("COMMIT");
    res.status(201).json({ announcement: announcement.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to post announcement" }); } finally { client.release(); }
});

app.get("/api/announcements", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    if (req.user.role === "student") {
      const result = await pool.query(
        `SELECT DISTINCT a.* FROM announcements a
         LEFT JOIN bookings b ON b.hostel_id=a.hostel_id AND b.user_id=$1 AND b.status IN ('pending','confirmed')
         LEFT JOIN favorites f ON f.hostel_id=a.hostel_id AND f.user_id=$1
         WHERE (a.audience IN ('residents','booked') AND b.user_id IS NOT NULL) OR (a.audience IN ('residents','favorited') AND f.user_id IS NOT NULL)
         ORDER BY a.created_at DESC`, [req.user.id]
      );
      return res.json(result.rows);
    }
    if (req.user.role === "parent") {
      const result = await pool.query(
        `SELECT DISTINCT a.* FROM announcements a
         JOIN bookings b ON b.hostel_id=a.hostel_id AND b.status IN ('pending','confirmed')
         JOIN parent_student_links l ON l.student_user_id=b.user_id AND l.parent_user_id=$1 AND l.status='active'
         WHERE a.audience IN ('residents','booked') ORDER BY a.created_at DESC`, [req.user.id]
      );
      return res.json(result.rows);
    }
    const result = await pool.query(
      `SELECT DISTINCT a.* FROM announcements a LEFT JOIN hostel_owners ho ON ho.hostel_id=a.hostel_id LEFT JOIN warden_assignments wa ON wa.hostel_id=a.hostel_id
       WHERE (ho.owner_user_id=$1 OR wa.warden_user_id=$1) ORDER BY a.created_at DESC`, [req.user.id]
    );
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch announcements" }); }
});

app.get("/api/notifications", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const result = await pool.query("SELECT * FROM notifications WHERE user_id=$1 ORDER BY created_at DESC", [req.user.id]);
    res.json(result.rows);
  } catch (error) { res.status(500).json({ error: "Failed to fetch notifications" }); }
});

app.patch("/api/notifications/:id/read", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const notificationId = Number(req.params.id);
    if (!isValidInt(notificationId)) return res.status(400).json({ error: "Invalid notification id" });
    const result = await pool.query("UPDATE notifications SET read_at=COALESCE(read_at,NOW()) WHERE id=$1 AND user_id=$2 RETURNING *", [notificationId, req.user.id]);
    if (result.rowCount === 0) return res.status(404).json({ error: "Notification not found" });
    res.json({ notification: result.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to mark notification as read" }); }
});

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  ROOM MANAGEMENT ROUTES
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// GET /api/hostels/:hostel_id/rooms
app.get("/api/hostels/:hostel_id/rooms", requireAuth, requireRole("owner", "warden"), async (req, res) => {
  try {
    await operationalTablesReady;
    const hostelId = parseInt(req.params.hostel_id, 10);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    if (!await staffCanManageHostel(pool, req.user, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    
    const hostelResult = await pool.query("SELECT id FROM hostels WHERE id=$1", [hostelId]);
    if (hostelResult.rows.length === 0) return res.status(404).json({ error: "Hostel not found" });
    
    const roomsResult = await pool.query(
      `SELECT r.id, r.hostel_id, r.floor_label, r.room_number, r.capacity,
              r.available_capacity, r.room_type, r.monthly_rent, r.amenities,
              r.notes, r.created_at,
              COALESCE(
                json_agg(
                  json_build_object('id', rb.id, 'bed_number', rb.bed_number,
                                    'status', rb.status)
                  ORDER BY rb.bed_number
                ) FILTER (WHERE rb.id IS NOT NULL),
                '[]'::json
              ) AS beds
       FROM rooms r
       LEFT JOIN room_beds rb ON rb.room_id = r.id
       WHERE r.hostel_id=$1
       GROUP BY r.id
       ORDER BY r.floor_label ASC, r.room_number ASC`,
      [hostelId]
    );
    res.json(roomsResult.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch rooms: " + err.message }); }
});

// Exact room inventory is released only after the owner-issued hostel
// credential has been activated for the student's pending request.
app.get("/api/student/hostels/:hostel_id/rooms", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    await ensureHostelAccessCredentialTable();
    const hostelId = parseInt(req.params.hostel_id, 10);
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "Invalid hostel id" });
    const hostel = await pool.query("SELECT id FROM hostels WHERE id=$1", [hostelId]);
    if (!hostel.rowCount) return res.status(404).json({ error: "Hostel not found" });
    const access = await pool.query(
      `SELECT 1 FROM hostel_access_credentials c
         JOIN bookings b ON b.id=c.booking_id
        WHERE c.student_user_id=$1 AND c.hostel_id=$2
          AND c.activated_at IS NOT NULL AND b.status='pending'
        LIMIT 1`,
      [req.user.id, hostelId]
    );
    if (!access.rowCount) return res.status(403).json({ error: "Activate the owner-issued hostel credential before selecting a room" });
    const rooms = await pool.query(
      `SELECT r.id,r.hostel_id,r.floor_label,r.room_number,r.capacity,r.available_capacity,
              r.room_type,r.monthly_rent,r.amenities,
              COALESCE(json_agg(json_build_object('id',rb.id,'bed_number',rb.bed_number,'status',rb.status)
                ORDER BY rb.bed_number) FILTER (WHERE rb.id IS NOT NULL),'[]'::json) AS beds
       FROM rooms r LEFT JOIN room_beds rb ON rb.room_id=r.id
       WHERE r.hostel_id=$1 AND r.available_capacity>0
       GROUP BY r.id ORDER BY r.floor_label ASC,r.room_number ASC`,
      [hostelId]
    );
    res.json(rooms.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch available rooms" });
  }
});

// A credential-gated ranking of real, currently available rooms.  This is
// deliberately available only after the owner has issued hostel access and
// the student has activated it for this exact pending booking.
app.get("/api/student/bookings/:id/room-matches", requireAuth, requireRole("student"), async (req, res) => {
  try {
    await operationalTablesReady;
    await ensureHostelAccessCredentialTable();
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });

    const booking = await pool.query(
      `SELECT b.id,b.hostel_id,b.requested_room_type,p.budget_max,p.study_preference
         FROM bookings b
         JOIN hostel_access_credentials c ON c.booking_id=b.id
         LEFT JOIN student_profiles p ON p.user_id=b.user_id
        WHERE b.id=$1 AND b.user_id=$2 AND b.status='pending'
          AND c.activated_at IS NOT NULL
        LIMIT 1`,
      [bookingId, req.user.id],
    );
    if (!booking.rowCount) {
      return res.status(403).json({ error: "Activate your owner-issued hostel credential before using room matching" });
    }

    const roomsResult = await pool.query(
      `SELECT r.id,r.hostel_id,r.floor_label,r.room_number,r.capacity,r.available_capacity,
              r.room_type,r.monthly_rent,r.amenities,
              COALESCE(json_agg(json_build_object('id',rb.id,'bed_number',rb.bed_number,'status',rb.status)
                ORDER BY rb.bed_number) FILTER (WHERE rb.id IS NOT NULL),'[]'::json) AS beds
         FROM rooms r
         LEFT JOIN room_beds rb ON rb.room_id=r.id
        WHERE r.hostel_id=$1 AND r.available_capacity>0
        GROUP BY r.id
        ORDER BY r.floor_label ASC,r.room_number ASC`,
      [booking.rows[0].hostel_id],
    );

    const rows = roomsResult.rows;
    const highestRent = Math.max(1, ...rows.map((room) => Number(room.monthly_rent || 0)));
    const budget = Number(booking.rows[0].budget_max);
    const hasBudget = Number.isFinite(budget) && budget > 0;
    const prefersQuiet = Number(booking.rows[0].study_preference || 0.5) >= 0.6;
    const requestedType = String(booking.rows[0].requested_room_type || "").toLowerCase();

    const matches = rows.map((room) => {
      const capacity = Math.max(1, Number(room.capacity || 1));
      const available = Math.max(0, Number(room.available_capacity || 0));
      const rent = Math.max(0, Number(room.monthly_rent || 0));
      const availabilityScore = available / capacity;
      const budgetScore = hasBudget
        ? (rent <= budget ? 1 : Math.max(0, budget / Math.max(rent, 1)))
        : 1 - Math.min(rent / highestRent, 1) * 0.35;
      const privacyScore = prefersQuiet
        ? 1 - Math.min(Math.max(capacity - 1, 0) / 5, 1)
        : Math.min(capacity / 5, 1);
      const amenities = Array.isArray(room.amenities) ? room.amenities : [];
      const amenitiesScore = Math.min(amenities.length / 4, 1);
      const typeScore = requestedType && String(room.room_type || "").toLowerCase() === requestedType ? 1 : 0.5;
      const score = 0.35 * availabilityScore + 0.25 * budgetScore + 0.20 * privacyScore + 0.10 * amenitiesScore + 0.10 * typeScore;
      const reasons = [
        `${available} bed${available === 1 ? "" : "s"} currently available`,
        hasBudget && rent <= budget ? "Within your saved monthly budget" : `PKR ${rent.toLocaleString()} monthly rent`,
        prefersQuiet && capacity <= 2 ? "Smaller room suits your quiet-study preference" : `${capacity}-seater room`,
      ];
      return {
        ...room,
        match_score: Math.round(score * 100),
        match_reasons: reasons,
      };
    }).sort((left, right) => right.match_score - left.match_score || Number(right.available_capacity) - Number(left.available_capacity));

    res.json({
      model: hasBudget || booking.rows[0].study_preference != null ? "room_match_v1_personalized" : "room_match_v1_inventory",
      matches,
    });
  } catch (error) {
    console.error("ROOM MATCHING FAILED:", error);
    res.status(500).json({ error: "Failed to generate room matches" });
  }
});

// Final step after hostel-access activation: the student chooses one real
// available floor/room/bed and the enquiry becomes a confirmed booking.
app.post("/api/student/bookings/:id/select-bed", requireAuth, requireRole("student"), async (req, res) => {
  const client = await pool.connect();
  try {
    await ensureHostelAccessCredentialTable();
    const bookingId = Number(req.params.id);
    const roomId = Number(req.body?.room_id);
    const bedNumber = String(req.body?.bed_number || "").trim();
    if (!isValidInt(bookingId) || !isValidInt(roomId) || !bedNumber) {
      return res.status(400).json({ error: "Booking, room and bed are required" });
    }
    await client.query("BEGIN");
    const booking = await client.query(
      `SELECT b.id,b.hostel_id,b.status
         FROM bookings b
         JOIN hostel_access_credentials c ON c.booking_id=b.id
        WHERE b.id=$1 AND b.user_id=$2 AND b.status='pending'
          AND c.activated_at IS NOT NULL
        FOR UPDATE`,
      [bookingId, req.user.id]
    );
    if (!booking.rowCount) { await client.query("ROLLBACK"); return res.status(403).json({ error: "Activate your hostel credential before choosing a bed" }); }
    const room = await client.query(
      "SELECT id,hostel_id,available_capacity FROM rooms WHERE id=$1 FOR UPDATE",
      [roomId]
    );
    if (!room.rowCount || Number(room.rows[0].hostel_id) !== Number(booking.rows[0].hostel_id)) {
      await client.query("ROLLBACK"); return res.status(400).json({ error: "This room does not belong to your requested hostel" });
    }
    if (Number(room.rows[0].available_capacity) < 1) { await client.query("ROLLBACK"); return res.status(409).json({ error: "This room is full" }); }
    const bed = await client.query(
      "SELECT id,status FROM room_beds WHERE room_id=$1 AND bed_number=$2 FOR UPDATE",
      [roomId, bedNumber]
    );
    if (!bed.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Bed not found" }); }
    const used = await client.query(
      "SELECT 1 FROM bookings WHERE room_id=$1 AND bed_number=$2 AND status='confirmed' LIMIT 1",
      [roomId, bedNumber]
    );
    if (used.rowCount || bed.rows[0].status === "occupied") { await client.query("ROLLBACK"); return res.status(409).json({ error: "This bed is no longer available" }); }
    const hostel = await client.query(
      "UPDATE hostels SET available_capacity=available_capacity-1 WHERE id=$1 AND available_capacity>0 RETURNING id",
      [booking.rows[0].hostel_id]
    );
    if (!hostel.rowCount) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Hostel is full" }); }
    await client.query("UPDATE rooms SET available_capacity=available_capacity-1 WHERE id=$1", [roomId]);
    await client.query("UPDATE room_beds SET status='occupied' WHERE id=$1", [bed.rows[0].id]);
    const updated = await client.query(
      "UPDATE bookings SET room_id=$1,bed_number=$2,status='confirmed',updated_at=NOW() WHERE id=$3 RETURNING *",
      [roomId, bedNumber, bookingId]
    );
    await client.query(
      "INSERT INTO booking_status_history (booking_id,previous_status,next_status,actor_user_id,reason) VALUES ($1,'pending','confirmed',$2,'student selected floor, room and bed')",
      [bookingId, req.user.id]
    );
    await client.query("COMMIT");
    res.json({ success: true, booking: updated.rows[0] });
  } catch (error) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to confirm room selection", detail: error.message });
  } finally { client.release(); }
});

// POST /api/rooms
app.post("/api/rooms", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const {
  hostel_id,
  room_number,
  floor_label,
  capacity,
  room_type,
  monthly_rent,
  bed_numbers,
  amenities,
  notes,
} = req.body;
    const hostelId = parseInt(hostel_id, 10);
    const capacityInt = parseInt(capacity, 10);
    const monthlyRentInt = parseInt(monthly_rent, 10);
    
    if (!isValidInt(hostelId)) return res.status(400).json({ error: "hostel_id must be a positive integer" });
    if (!room_number) return res.status(400).json({ error: "room_number is required" });
    if (!isValidInt(capacityInt) || capacityInt < 1 || capacityInt > 6) return res.status(400).json({ error: "capacity must be between 1 and 6" });
    if (!Number.isInteger(monthlyRentInt) || monthlyRentInt < 0) {
  return res.status(400).json({
    error: "monthly_rent must be a non-negative integer",
  });
}
    
    if (!await ownerHasHostel(pool, req.user.id, hostelId)) return res.status(404).json({ error: "Hostel not found" });
    
    const bedNumbers = Array.isArray(bed_numbers) && bed_numbers.length
      ? bed_numbers.map((bed) => String(bed || "").trim())
      : Array.from({ length: capacityInt }, (_, index) => String(index + 1));
    if (bedNumbers.length !== capacityInt || bedNumbers.some((bed) => !bed) ||
        new Set(bedNumbers.map((bed) => bed.toLowerCase())).size !== bedNumbers.length) {
      return res.status(400).json({ error: "Provide exactly one unique bed number for each bed" });
    }
    const roomAmenities = Array.isArray(amenities)
      ? [...new Set(amenities.map((item) => String(item || "").trim()).filter(Boolean))]
      : [];

    const client = await pool.connect();
    try {
      await client.query("BEGIN");
    const roomResult = await client.query(
      `INSERT INTO rooms
  (
    hostel_id,
    room_number,
    floor_label,
    capacity,
    available_capacity,
    room_type,
    monthly_rent,
    amenities,
    notes
  )
  VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
  RETURNING *`,
[
  hostelId,
  room_number,
  typeof floor_label === "string" && floor_label.trim() ? floor_label.trim() : "Ground Floor",
  capacityInt,
  capacityInt,
  room_type || "shared",
  monthlyRentInt,
  JSON.stringify(roomAmenities),
  typeof notes === "string" && notes.trim() ? notes.trim() : null,
]
    );
    const room = roomResult.rows[0];
    for (const bedNumber of bedNumbers) {
      await client.query(
        "INSERT INTO room_beds(room_id, bed_number) VALUES($1,$2)",
        [room.id, bedNumber],
      );
    }
    await client.query("COMMIT");
    room.beds = bedNumbers.map((bed_number) => ({ bed_number, status: "available" }));
    res.status(201).json({ message: "Room created", room });
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  } catch (err) { res.status(500).json({ error: "Failed to create room: " + err.message }); }
});

// PATCH /api/rooms/:id
app.patch("/api/rooms/:id", requireAuth, requireRole("owner"), async (req, res) => {
  try {
    await operationalTablesReady;
    const roomId = parseInt(req.params.id, 10);
    if (!isValidInt(roomId)) return res.status(400).json({ error: "Invalid room id" });
    
    const {
  room_number,
  floor_label,
  capacity,
  available_capacity,
  room_type,
  monthly_rent,
  amenities,
  notes,
} = req.body;
    
    const roomResult = await pool.query("SELECT id, hostel_id, capacity, available_capacity FROM rooms WHERE id=$1", [roomId]);
    if (roomResult.rows.length === 0) return res.status(404).json({ error: "Room not found" });
    if (!await ownerHasHostel(pool, req.user.id, roomResult.rows[0].hostel_id)) return res.status(404).json({ error: "Room not found" });
    
    let updates = [];
    let params = [];
    let paramCount = 1;
    
    if (room_number != null) {
      updates.push(`room_number=$${paramCount}`);
      params.push(room_number);
      paramCount++;
    }
    if (floor_label != null) {
      updates.push(`floor_label=$${paramCount}`);
      params.push(String(floor_label).trim() || "Ground Floor");
      paramCount++;
    }
    if (capacity != null) {
      const capacityInt = parseInt(capacity, 10);
      if (!isValidInt(capacityInt)) return res.status(400).json({ error: "capacity must be a positive integer" });
      const occupied=Number(roomResult.rows[0].capacity)-Number(roomResult.rows[0].available_capacity);
      if (capacityInt < occupied) return res.status(409).json({ error: "capacity cannot be below occupied or reserved beds" });
      updates.push(`capacity=$${paramCount}`);
      params.push(capacityInt);
      paramCount++;
      if(available_capacity==null){updates.push(`available_capacity=$${paramCount}`);params.push(capacityInt-occupied);paramCount++;}
    }
    if (available_capacity != null) {
      const availInt = parseInt(available_capacity, 10);
      const resultingCapacity = capacity == null ? Number(roomResult.rows[0].capacity) : parseInt(capacity, 10);
      if (availInt < 0 || availInt > resultingCapacity) return res.status(400).json({ error: "available_capacity must be between zero and capacity" });
      updates.push(`available_capacity=$${paramCount}`);
      params.push(availInt);
      paramCount++;
    }
    if (room_type != null) {
      updates.push(`room_type=$${paramCount}`);
      params.push(room_type);
      paramCount++;
    }
    if (monthly_rent != null) {
  const monthlyRentInt = parseInt(monthly_rent, 10);

  if (!Number.isInteger(monthlyRentInt) || monthlyRentInt < 0) {
    return res.status(400).json({
      error: "monthly_rent must be a non-negative integer",
    });
  }
  updates.push(`monthly_rent=$${paramCount}`);
  params.push(monthlyRentInt);
  paramCount++;
}
    if (amenities != null) {
      if (!Array.isArray(amenities)) return res.status(400).json({ error: "amenities must be a list" });
      const roomAmenities = [...new Set(amenities.map((item) => String(item || "").trim()).filter(Boolean))];
      updates.push(`amenities=$${paramCount}`);
      params.push(JSON.stringify(roomAmenities));
      paramCount++;
    }
    if (notes != null) {
      updates.push(`notes=$${paramCount}`);
      params.push(String(notes).trim() || null);
      paramCount++;
    }
    
    if (updates.length === 0) return res.status(400).json({ error: "No fields to update" });
    
    params.push(roomId);
    const updateQuery = `UPDATE rooms SET ${updates.join(", ")} WHERE id=$${paramCount} RETURNING *`;
    const updatedResult = await pool.query(updateQuery, params);
    res.json({ message: "Room updated âœ…", room: updatedResult.rows[0] });
  } catch (err) { res.status(500).json({ error: "Failed to update room: " + err.message }); }
});

// POST /api/bookings/:id/assign-room (assign a specific room to a booking)
app.post("/api/bookings/:id/assign-room", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = parseInt(req.params.id, 10);
    const roomId = parseInt(req.body.room_id, 10);
    const requestedBedNumber = req.body.bed_number == null ? null : String(req.body.bed_number).trim();
    
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    if (!isValidInt(roomId)) return res.status(400).json({ error: "room_id must be a positive integer" });
    
    await client.query("BEGIN");
    
    const bookingResult = await client.query(
      "SELECT id, hostel_id, room_id, status FROM bookings WHERE id=$1 FOR UPDATE",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Booking not found" });
    }
    
    const booking = bookingResult.rows[0];
    if (!await wardenHasHostel(client, req.user.id, booking.hostel_id)) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Booking not found" });
    }
    
    const roomResult = await client.query(
      "SELECT id, hostel_id, capacity, available_capacity FROM rooms WHERE id=$1 FOR UPDATE",
      [roomId]
    );
    if (roomResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Room not found" });
    }
    
    const room = roomResult.rows[0];
    const holdResult=await client.query("SELECT * FROM booking_inventory_holds WHERE booking_id=$1 AND released_at IS NULL AND converted_at IS NULL FOR UPDATE",[bookingId]);
    const heldRoomId=holdResult.rowCount?Number(holdResult.rows[0].room_id):null;
    if (Number(room.hostel_id) !== Number(booking.hostel_id)) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "Room does not belong to the booking's hostel" });
    }
    
    if (Number(booking.room_id) === roomId) {
      await client.query("ROLLBACK");
      return res.json({ message: "Room is already assigned to this booking" });
    }

    if (Number(room.available_capacity) <= 0 && heldRoomId !== roomId) {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "Room has no available capacity" });
    }
    
    const bookingStatus = normalizeBookingStatus(booking.status);
    if (bookingStatus !== "confirmed") {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "Only confirmed bookings can be assigned a room" });
    }
    
    const oldRoomId = booking.room_id;
    let bedNumber = requestedBedNumber;
    const usedBeds = await client.query("SELECT bed_number FROM bookings WHERE room_id=$1 AND status='confirmed' AND id<>$2 AND bed_number IS NOT NULL", [roomId, bookingId]);
    const used = new Set(usedBeds.rows.map((row) => String(row.bed_number)));
    if (bedNumber && used.has(bedNumber)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Bed is already assigned" }); }
    if (!bedNumber) {
      for (let candidate = 1; candidate <= Number(room.capacity); candidate += 1) if (!used.has(String(candidate))) { bedNumber = String(candidate); break; }
    }
    if (!bedNumber) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Room has no unassigned bed" }); }
    
    if (oldRoomId != null) {
      await client.query(
        "UPDATE rooms SET available_capacity = LEAST(available_capacity + 1, capacity) WHERE id=$1",
        [oldRoomId]
      );
    }
    
    if(heldRoomId!==roomId){
      if(heldRoomId!=null)await client.query("UPDATE rooms SET available_capacity=LEAST(capacity,available_capacity+1) WHERE id=$1",[heldRoomId]);
      await client.query("UPDATE rooms SET available_capacity = available_capacity - 1 WHERE id=$1",[roomId]);
    }
    if(holdResult.rowCount)await client.query("UPDATE booking_inventory_holds SET converted_at=NOW() WHERE booking_id=$1",[bookingId]);
    
    const updatedBooking = await client.query(
      "UPDATE bookings SET room_id=$1,bed_number=$2,updated_at=NOW() WHERE id=$3 RETURNING *",
      [roomId, bedNumber, bookingId]
    );
    await client.query(
      "INSERT INTO room_assignment_history (booking_id,previous_room_id,next_room_id,assigned_by_user_id) VALUES ($1,$2,$3,$4)",
      [bookingId, oldRoomId, roomId, req.user.id]
    );
    
    await client.query("COMMIT");
    res.json({ message: "Room assigned to booking âœ…", booking: updatedBooking.rows[0] });
  } catch (err) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to assign room: " + err.message });
  } finally {
    client.release();
  }
});

// Removes a confirmed student's current room/bed allocation and returns the
// released bed to the room inventory.
app.delete("/api/bookings/:id/room-allocation", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    await client.query("BEGIN");
    const booking = await client.query(
      "SELECT b.* FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id WHERE b.id=$1 AND wa.warden_user_id=$2 FOR UPDATE",
      [bookingId, req.user.id],
    );
    if (!booking.rowCount) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    const current = booking.rows[0];
    if (current.room_id == null) { await client.query("ROLLBACK"); return res.status(409).json({ error: "This student has no room allocation" }); }
    await client.query("UPDATE rooms SET available_capacity=LEAST(capacity, available_capacity+1) WHERE id=$1", [current.room_id]);
    const updated = await client.query("UPDATE bookings SET room_id=NULL, bed_number=NULL, updated_at=NOW() WHERE id=$1 RETURNING *", [bookingId]);
    await client.query("COMMIT");
    res.json({ message: "Room and bed deallocated", booking: updated.rows[0] });
  } catch (error) {
    await client.query("ROLLBACK");
    res.status(500).json({ error: "Failed to deallocate room: " + error.message });
  } finally { client.release(); }
});

// GET /api/bookings/:id/available-rooms
app.get("/api/bookings/:id/available-rooms", requireAuth, requireRole("warden"), async (req, res) => {
  try {
    const bookingId = parseInt(req.params.id, 10);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    
    const bookingResult = await pool.query(
      "SELECT id, hostel_id FROM bookings WHERE id=$1",
      [bookingId]
    );
    if (bookingResult.rows.length === 0) return res.status(404).json({ error: "Booking not found" });
    
    const booking = bookingResult.rows[0];
    if (!await wardenHasHostel(pool, req.user.id, booking.hostel_id)) return res.status(404).json({ error: "Booking not found" });
    const roomsResult = await pool.query(
      "SELECT id, room_number, capacity, available_capacity, room_type FROM rooms WHERE hostel_id=$1 AND available_capacity > 0 ORDER BY room_number ASC",
      [booking.hostel_id]
    );
    
    res.json(roomsResult.rows);
  } catch (err) { res.status(500).json({ error: "Failed to fetch available rooms: " + err.message }); }
});

app.post("/api/warden/bookings/:id/check-in", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    await client.query("BEGIN");
    const booking = await client.query("SELECT b.* FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id WHERE b.id=$1 AND wa.warden_user_id=$2 FOR UPDATE", [bookingId, req.user.id]);
    if (booking.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    if (normalizeBookingStatus(booking.rows[0].status) !== "confirmed" || booking.rows[0].room_id == null) { await client.query("ROLLBACK"); return res.status(409).json({ error: "A confirmed booking with an assigned room is required" }); }
    const stay = await client.query("INSERT INTO booking_stays (booking_id,checked_in_at,checked_in_by_user_id) VALUES ($1,NOW(),$2) ON CONFLICT (booking_id) DO UPDATE SET checked_in_at=COALESCE(booking_stays.checked_in_at,EXCLUDED.checked_in_at), checked_in_by_user_id=COALESCE(booking_stays.checked_in_by_user_id,EXCLUDED.checked_in_by_user_id) WHERE booking_stays.checked_out_at IS NULL RETURNING *", [bookingId, req.user.id]);
    if (stay.rowCount === 0) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Stay has already been checked out" }); }
    await client.query("COMMIT");
    res.json({ stay: stay.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to check in booking" }); } finally { client.release(); }
});

app.post("/api/warden/bookings/:id/check-out", requireAuth, requireRole("warden"), async (req, res) => {
  const client = await pool.connect();
  try {
    await operationalTablesReady;
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    await client.query("BEGIN");
    const booking = await client.query("SELECT b.* FROM bookings b JOIN warden_assignments wa ON wa.hostel_id=b.hostel_id WHERE b.id=$1 AND wa.warden_user_id=$2 FOR UPDATE", [bookingId, req.user.id]);
    if (booking.rowCount === 0) { await client.query("ROLLBACK"); return res.status(404).json({ error: "Booking not found" }); }
    const stay = await client.query("SELECT * FROM booking_stays WHERE booking_id=$1 FOR UPDATE", [bookingId]);
    if (stay.rowCount === 0 || stay.rows[0].checked_in_at == null || stay.rows[0].checked_out_at != null) { await client.query("ROLLBACK"); return res.status(409).json({ error: "An active checked-in stay is required" }); }
    if (!await releaseHostelCapacity(client, booking.rows[0].hostel_id)) { await client.query("ROLLBACK"); return res.status(409).json({ error: "Hostel availability is already fully restored" }); }
    await client.query("UPDATE rooms SET available_capacity=LEAST(available_capacity+1,capacity) WHERE id=$1", [booking.rows[0].room_id]);
    const updatedStay = await client.query("UPDATE booking_stays SET checked_out_at=NOW(), checked_out_by_user_id=$1 WHERE booking_id=$2 RETURNING *", [req.user.id, bookingId]);
    await client.query("UPDATE bookings SET status='completed' WHERE id=$1", [bookingId]);
    await createNotification(client, booking.rows[0].user_id, "booking_status", "Move-out recorded", "Your hostel stay has been checked out.", { booking_id: bookingId, status: "completed" }, "booking_updates");
    await client.query("COMMIT");
    res.json({ stay: updatedStay.rows[0] });
  } catch (error) { await client.query("ROLLBACK"); res.status(500).json({ error: "Failed to check out booking" }); } finally { client.release(); }
});

app.get("/api/bookings/:id/stay", requireAuth, async (req, res) => {
  try {
    await operationalTablesReady;
    const bookingId = Number(req.params.id);
    if (!isValidInt(bookingId)) return res.status(400).json({ error: "Invalid booking id" });
    const result = await pool.query("SELECT bs.* FROM booking_stays bs JOIN bookings b ON b.id=bs.booking_id WHERE bs.booking_id=$1 AND b.user_id=$2", [bookingId, req.user.id]);
    if (result.rowCount === 0) return res.status(404).json({ error: "Stay not found" });
    res.json({ stay: result.rows[0] });
  } catch (error) { res.status(500).json({ error: "Failed to fetch stay" }); }
});

app.post("/api/payments/intents", requireAuth, requireRole("student"), (_req, res) => {
  res.status(410).json({ error: "In-app payment processing is disabled; submit a payment receipt instead" });
});

app.get("/api/payments", requireAuth, requireRole("student"), (_req, res) => {
  res.status(410).json({ error: "In-app payments are disabled; use /api/fee-receipts for receipt history" });
});

const PORT = process.env.PORT || 5000;
if (require.main === module) {
  const server = app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
  server.on("error", (error) => {
    console.error("Server error:", error.message);
  });
  server.on("close", () => {
    console.warn("Server listener closed.");
  });
  let shuttingDown = false;
  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.once(signal, () => {
      if (shuttingDown) return;
      shuttingDown = true;
      console.warn(`Received ${signal}; shutting down server.`);
      server.close(async () => {
        await pool.end();
        process.exit(0);
      });
      setTimeout(() => process.exit(1), 10000).unref();
    });
  }
}

module.exports = { app, pool };
