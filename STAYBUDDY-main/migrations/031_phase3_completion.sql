BEGIN;

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS account_status VARCHAR(20) NOT NULL DEFAULT 'active';
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_account_status_check;
ALTER TABLE users ADD CONSTRAINT users_account_status_check
  CHECK (account_status IN ('active','disabled'));

CREATE TABLE IF NOT EXISTS auth_sessions (
  id UUID PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  last_seen_at TIMESTAMPTZ,
  user_agent TEXT,
  ip_address INET,
  CHECK (expires_at > issued_at)
);
CREATE INDEX IF NOT EXISTS auth_sessions_user_active_idx
  ON auth_sessions (user_id, expires_at DESC) WHERE revoked_at IS NULL;

ALTER TABLE hostels
  ADD COLUMN IF NOT EXISTS publication_status VARCHAR(20) NOT NULL DEFAULT 'draft',
  ADD COLUMN IF NOT EXISTS facilities JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS rules JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS mess_info JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS media JSONB NOT NULL DEFAULT '[]'::jsonb;
UPDATE hostels SET publication_status=CASE WHEN verified THEN 'published' ELSE 'draft' END
WHERE publication_status='draft';
ALTER TABLE hostels DROP CONSTRAINT IF EXISTS hostels_publication_status_check;
ALTER TABLE hostels ADD CONSTRAINT hostels_publication_status_check
  CHECK (publication_status IN ('draft','submitted','published','suspended'));

ALTER TABLE rooms ADD COLUMN IF NOT EXISTS monthly_rent INTEGER;
ALTER TABLE rooms DROP CONSTRAINT IF EXISTS rooms_monthly_rent_check;
ALTER TABLE rooms ADD CONSTRAINT rooms_monthly_rent_check
  CHECK (monthly_rent IS NULL OR monthly_rent >= 0);

ALTER TABLE bookings
  ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(160),
  ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS requested_room_type VARCHAR(40),
  ADD COLUMN IF NOT EXISTS bed_number VARCHAR(40),
  ADD COLUMN IF NOT EXISTS price_snapshot NUMERIC(12,2),
  ADD COLUMN IF NOT EXISTS currency CHAR(3) NOT NULL DEFAULT 'PKR',
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE bookings DROP CONSTRAINT IF EXISTS bookings_status_check;
ALTER TABLE bookings ADD CONSTRAINT bookings_status_check
  CHECK (status IN ('pending','confirmed','rejected','cancelled','expired','completed'));
ALTER TABLE bookings DROP CONSTRAINT IF EXISTS bookings_price_snapshot_check;
ALTER TABLE bookings ADD CONSTRAINT bookings_price_snapshot_check
  CHECK (price_snapshot IS NULL OR price_snapshot >= 0);
ALTER TABLE bookings DROP CONSTRAINT IF EXISTS bookings_currency_check;
ALTER TABLE bookings ADD CONSTRAINT bookings_currency_check CHECK (currency='PKR');
CREATE UNIQUE INDEX IF NOT EXISTS bookings_student_idempotency_idx
  ON bookings (user_id,idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS bookings_active_room_bed_idx
  ON bookings (room_id,bed_number)
  WHERE room_id IS NOT NULL AND bed_number IS NOT NULL AND status='confirmed';

CREATE TABLE IF NOT EXISTS booking_status_history (
  id BIGSERIAL PRIMARY KEY,
  booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  previous_status VARCHAR(20),
  next_status VARCHAR(20) NOT NULL,
  actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS booking_status_history_booking_idx
  ON booking_status_history (booking_id,created_at DESC);

CREATE TABLE IF NOT EXISTS booking_inventory_holds (
  booking_id INTEGER PRIMARY KEY REFERENCES bookings(id) ON DELETE CASCADE,
  room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE RESTRICT,
  expires_at TIMESTAMPTZ NOT NULL,
  released_at TIMESTAMPTZ,
  converted_at TIMESTAMPTZ,
  CHECK (NOT (released_at IS NOT NULL AND converted_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS booking_inventory_holds_active_idx ON booking_inventory_holds(room_id,expires_at) WHERE released_at IS NULL AND converted_at IS NULL;

ALTER TABLE fee_receipt_submissions
  ADD COLUMN IF NOT EXISTS supersedes_receipt_id BIGINT REFERENCES fee_receipt_submissions(id) ON DELETE SET NULL;
CREATE TABLE IF NOT EXISTS fee_receipt_allocations (
  id BIGSERIAL PRIMARY KEY,
  receipt_submission_id BIGINT NOT NULL UNIQUE REFERENCES fee_receipt_submissions(id) ON DELETE RESTRICT,
  invoice_id BIGINT NOT NULL REFERENCES fee_invoices(id) ON DELETE RESTRICT,
  amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
  allocated_by_user_id INTEGER NOT NULL REFERENCES users(id),
  allocated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS fee_receipt_allocations_invoice_idx
  ON fee_receipt_allocations (invoice_id,allocated_at DESC);

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(160),
  ADD COLUMN IF NOT EXISTS triage_source VARCHAR(20) NOT NULL DEFAULT 'fallback',
  ADD COLUMN IF NOT EXISTS triage_confidence NUMERIC(5,4),
  ADD COLUMN IF NOT EXISTS triage_model_version VARCHAR(80),
  ADD COLUMN IF NOT EXISTS resolution_due_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS reopened_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS reopen_reason TEXT,
  ADD COLUMN IF NOT EXISTS reopen_count INTEGER NOT NULL DEFAULT 0;
CREATE UNIQUE INDEX IF NOT EXISTS complaints_student_idempotency_idx
  ON complaints (user_id,idempotency_key) WHERE idempotency_key IS NOT NULL;
ALTER TABLE complaints DROP CONSTRAINT IF EXISTS complaints_reopen_count_check;
ALTER TABLE complaints ADD CONSTRAINT complaints_reopen_count_check CHECK (reopen_count BETWEEN 0 AND 1);
ALTER TABLE complaint_status_history ALTER COLUMN changed_by_user_id DROP NOT NULL;

CREATE TABLE IF NOT EXISTS complaint_resolution_revisions (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  safe_resolution_summary TEXT NOT NULL CHECK (char_length(trim(safe_resolution_summary)) BETWEEN 1 AND 4000),
  actions_taken TEXT NOT NULL CHECK (char_length(trim(actions_taken)) BETWEEN 1 AND 4000),
  prevention_follow_up_notes TEXT,
  resolver_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS complaint_resolution_revisions_complaint_idx
  ON complaint_resolution_revisions (complaint_id,created_at DESC);

CREATE TABLE IF NOT EXISTS chat_sessions (
  id UUID PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title VARCHAR(160),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role VARCHAR(20) NOT NULL CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL CHECK (char_length(trim(content)) BETWEEN 1 AND 4000),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id,created_at DESC);

ALTER TABLE student_profiles
  ADD COLUMN IF NOT EXISTS roommate_profile_sharing BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE attendance_events DROP CONSTRAINT IF EXISTS attendance_events_idempotency_key_key;
DROP INDEX IF EXISTS attendance_events_idempotency_key_key;
CREATE UNIQUE INDEX IF NOT EXISTS attendance_events_scoped_idempotency_idx
  ON attendance_events (hostel_id,source,idempotency_key);

ALTER TABLE operational_jobs DROP CONSTRAINT IF EXISTS operational_jobs_job_type_check;
ALTER TABLE operational_jobs ADD CONSTRAINT operational_jobs_job_type_check CHECK (job_type IN (
  'emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check',
  'fee_invoice_generation','fee_reminder_check','complaint_sla_check','complaint_volume_check',
  'location_retention_check','booking_expiry_check','complaint_auto_close','attendance_curfew_check'
));

COMMIT;
