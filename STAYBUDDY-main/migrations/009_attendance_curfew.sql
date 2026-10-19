BEGIN;

ALTER TABLE notification_preferences
  ADD COLUMN IF NOT EXISTS attendance_updates BOOLEAN NOT NULL DEFAULT true;

CREATE TABLE IF NOT EXISTS hostel_attendance_settings (
  hostel_id INTEGER PRIMARY KEY REFERENCES hostels(id) ON DELETE CASCADE,
  curfew_time TIME NOT NULL DEFAULT '22:00:00',
  timezone VARCHAR(64) NOT NULL DEFAULT 'Asia/Karachi',
  enabled BOOLEAN NOT NULL DEFAULT true,
  updated_by_user_id INTEGER REFERENCES users(id),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attendance_integration_credentials (
  id SERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  label VARCHAR(120) NOT NULL,
  credential_hash CHAR(64) NOT NULL UNIQUE,
  active BOOLEAN NOT NULL DEFAULT true,
  created_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  revoked_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS resident_attendance_identifiers (
  id SERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  external_identifier VARCHAR(160) NOT NULL,
  active BOOLEAN NOT NULL DEFAULT true,
  enrolled_by_user_id INTEGER NOT NULL REFERENCES users(id),
  enrolled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  revoked_at TIMESTAMPTZ,
  UNIQUE (hostel_id, external_identifier)
);

CREATE TABLE IF NOT EXISTS attendance_events (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  direction VARCHAR(3) NOT NULL CHECK (direction IN ('in', 'out')),
  occurred_at TIMESTAMPTZ NOT NULL,
  source VARCHAR(20) NOT NULL CHECK (source IN ('biometric', 'rfid', 'manual')),
  entry_point VARCHAR(120),
  external_event_id VARCHAR(160),
  idempotency_key VARCHAR(160) NOT NULL UNIQUE,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_by_user_id INTEGER REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (hostel_id, source, external_event_id)
);

CREATE TABLE IF NOT EXISTS attendance_corrections (
  id BIGSERIAL PRIMARY KEY,
  attendance_event_id BIGINT NOT NULL REFERENCES attendance_events(id) ON DELETE CASCADE,
  corrected_direction VARCHAR(3) NOT NULL CHECK (corrected_direction IN ('in', 'out')),
  corrected_occurred_at TIMESTAMPTZ NOT NULL,
  reason TEXT NOT NULL,
  corrected_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attendance_exceptions (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  policy_date DATE NOT NULL,
  exception_type VARCHAR(30) NOT NULL CHECK (exception_type IN ('late_exit', 'late_return', 'absent_at_curfew')),
  status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  resolved_by_user_id INTEGER REFERENCES users(id),
  resolved_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (hostel_id, student_user_id, policy_date, exception_type)
);

CREATE INDEX IF NOT EXISTS attendance_events_hostel_time_idx ON attendance_events (hostel_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS attendance_events_student_time_idx ON attendance_events (student_user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS resident_attendance_identifiers_student_idx ON resident_attendance_identifiers (student_user_id, hostel_id) WHERE active;
CREATE INDEX IF NOT EXISTS attendance_exceptions_hostel_open_idx ON attendance_exceptions (hostel_id, policy_date DESC) WHERE status='open';

COMMIT;
