BEGIN;

ALTER TABLE hostel_fee_schedules
  ADD COLUMN IF NOT EXISTS payment_option VARCHAR(20) NOT NULL DEFAULT 'full_only';
ALTER TABLE hostel_fee_schedules DROP CONSTRAINT IF EXISTS hostel_fee_schedules_payment_option_check;
ALTER TABLE hostel_fee_schedules ADD CONSTRAINT hostel_fee_schedules_payment_option_check
  CHECK (payment_option IN ('full_only','installments_only','both'));
ALTER TABLE hostel_fee_schedules
  ADD COLUMN IF NOT EXISTS minimum_installment_amount NUMERIC(12,2);
ALTER TABLE hostel_fee_schedules DROP CONSTRAINT IF EXISTS hostel_fee_schedules_minimum_installment_amount_check;
ALTER TABLE hostel_fee_schedules ADD CONSTRAINT hostel_fee_schedules_minimum_installment_amount_check
  CHECK (minimum_installment_amount IS NULL OR minimum_installment_amount > 0);

ALTER TABLE operational_jobs DROP CONSTRAINT IF EXISTS operational_jobs_job_type_check;
ALTER TABLE operational_jobs ADD CONSTRAINT operational_jobs_job_type_check
  CHECK (job_type IN ('emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check','fee_invoice_generation','fee_reminder_check','complaint_sla_check','complaint_volume_check'));

CREATE TABLE IF NOT EXISTS hostel_payment_destinations (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  account_holder_name VARCHAR(180) NOT NULL,
  institution_name VARCHAR(180) NOT NULL,
  account_reference VARCHAR(180) NOT NULL,
  instructions TEXT,
  status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','revoked')),
  submitted_by_user_id INTEGER NOT NULL REFERENCES users(id),
  approved_by_user_id INTEGER REFERENCES users(id),
  approved_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK ((status <> 'approved') OR (approved_by_user_id IS NOT NULL AND approved_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS fee_receipt_submissions (
  id BIGSERIAL PRIMARY KEY,
  invoice_id BIGINT NOT NULL REFERENCES fee_invoices(id) ON DELETE CASCADE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  submitted_by_user_id INTEGER NOT NULL REFERENCES users(id),
  payment_destination_id BIGINT REFERENCES hostel_payment_destinations(id) ON DELETE SET NULL,
  amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
  currency CHAR(3) NOT NULL DEFAULT 'PKR' CHECK (currency = 'PKR'),
  payment_reference VARCHAR(180) NOT NULL,
  paid_at TIMESTAMPTZ NOT NULL,
  storage_provider VARCHAR(40) NOT NULL,
  storage_key TEXT NOT NULL UNIQUE,
  checksum CHAR(64) NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  content_type VARCHAR(120) NOT NULL,
  byte_size BIGINT NOT NULL CHECK (byte_size > 0 AND byte_size <= 5242880),
  status VARCHAR(30) NOT NULL DEFAULT 'submitted' CHECK (status IN ('submitted','verified','rejected','needs_clarification')),
  reviewed_by_user_id INTEGER REFERENCES users(id),
  reviewed_at TIMESTAMPTZ,
  review_note TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK ((status = 'submitted' AND reviewed_by_user_id IS NULL AND reviewed_at IS NULL) OR (status <> 'submitted' AND reviewed_by_user_id IS NOT NULL AND reviewed_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS fee_receipt_audit_events (
  id BIGSERIAL PRIMARY KEY,
  receipt_submission_id BIGINT NOT NULL REFERENCES fee_receipt_submissions(id) ON DELETE CASCADE,
  event_type VARCHAR(40) NOT NULL CHECK (event_type IN ('submitted','verified','rejected','needs_clarification')),
  actor_user_id INTEGER NOT NULL REFERENCES users(id),
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fee_invoice_reminder_events (
  id BIGSERIAL PRIMARY KEY,
  invoice_id BIGINT NOT NULL REFERENCES fee_invoices(id) ON DELETE CASCADE,
  reminder_code VARCHAR(30) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (invoice_id, reminder_code)
);

CREATE INDEX IF NOT EXISTS hostel_payment_destinations_hostel_status_idx ON hostel_payment_destinations (hostel_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS fee_receipt_submissions_queue_idx ON fee_receipt_submissions (hostel_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS fee_receipt_submissions_student_idx ON fee_receipt_submissions (student_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS fee_receipt_audit_events_receipt_idx ON fee_receipt_audit_events (receipt_submission_id, created_at DESC);

COMMIT;
