BEGIN;

ALTER TABLE hostel_attendance_settings
  ADD COLUMN IF NOT EXISTS earliest_exit_time TIME NOT NULL DEFAULT '07:00',
  ADD COLUMN IF NOT EXISTS grace_minutes INTEGER NOT NULL DEFAULT 10 CHECK (grace_minutes BETWEEN 0 AND 120),
  ADD COLUMN IF NOT EXISTS escalation_minutes INTEGER NOT NULL DEFAULT 30 CHECK (escalation_minutes BETWEEN 1 AND 240),
  ADD COLUMN IF NOT EXISTS consecutive_warning_days INTEGER NOT NULL DEFAULT 3 CHECK (consecutive_warning_days BETWEEN 2 AND 30),
  ADD COLUMN IF NOT EXISTS attendance_fine_amount NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (attendance_fine_amount >= 0);

ALTER TABLE attendance_exceptions DROP CONSTRAINT IF EXISTS attendance_exceptions_exception_type_check;
ALTER TABLE attendance_exceptions ADD CONSTRAINT attendance_exceptions_exception_type_check
  CHECK (exception_type IN ('early_exit','late_exit','late_return','absent_at_curfew','unusual_pattern'));

CREATE TABLE hostel_gate_qr_codes (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  gate_label VARCHAR(120) NOT NULL,
  code_hash CHAR(64) NOT NULL UNIQUE,
  active BOOLEAN NOT NULL DEFAULT true,
  created_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  revoked_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX hostel_gate_qr_one_active_label_idx
  ON hostel_gate_qr_codes(hostel_id,gate_label) WHERE active;

CREATE TABLE attendance_gate_requests (
  id UUID PRIMARY KEY,
  gate_code_id BIGINT NOT NULL REFERENCES hostel_gate_qr_codes(id),
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  direction VARCHAR(3) NOT NULL CHECK (direction IN ('in','out')),
  idempotency_key VARCHAR(160) NOT NULL,
  submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  latitude NUMERIC(10,8),
  longitude NUMERIC(11,8),
  accuracy_m NUMERIC(10,2),
  status VARCHAR(12) NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','approved','rejected','expired')),
  reviewed_by_user_id INTEGER REFERENCES users(id),
  reviewed_at TIMESTAMPTZ,
  review_reason TEXT,
  attendance_event_id BIGINT REFERENCES attendance_events(id),
  UNIQUE(student_user_id,idempotency_key),
  CHECK (expires_at > submitted_at),
  CHECK ((latitude IS NULL AND longitude IS NULL AND accuracy_m IS NULL) OR
    (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180 AND accuracy_m BETWEEN 0 AND 100000))
);
CREATE UNIQUE INDEX attendance_gate_one_pending_idx
  ON attendance_gate_requests(hostel_id,student_user_id) WHERE status='pending';
CREATE INDEX attendance_gate_queue_idx ON attendance_gate_requests(hostel_id,status,submitted_at DESC);

CREATE TABLE attendance_disciplinary_actions (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  exception_id BIGINT NOT NULL REFERENCES attendance_exceptions(id),
  action_type VARCHAR(10) NOT NULL CHECK (action_type IN ('warning','fine')),
  consecutive_days INTEGER NOT NULL CHECK (consecutive_days > 0),
  amount NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (amount >= 0),
  status VARCHAR(12) NOT NULL DEFAULT 'issued' CHECK (status IN ('issued','paid','waived')),
  issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  changed_by_user_id INTEGER REFERENCES users(id),
  changed_at TIMESTAMPTZ,
  change_reason TEXT,
  payment_reference VARCHAR(180),
  UNIQUE(exception_id,action_type),
  CHECK (action_type='fine' OR amount=0)
);
CREATE INDEX attendance_disciplinary_student_idx
  ON attendance_disciplinary_actions(student_user_id,issued_at DESC);

CREATE TABLE attendance_curfew_escalations (
  exception_id BIGINT PRIMARY KEY REFERENCES attendance_exceptions(id) ON DELETE CASCADE,
  escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
