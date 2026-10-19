BEGIN;

CREATE TABLE IF NOT EXISTS location_sharing_sessions (
  id BIGSERIAL PRIMARY KEY,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  revoked_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  CHECK (expires_at > started_at)
);
CREATE UNIQUE INDEX IF NOT EXISTS location_sharing_one_active_session
  ON location_sharing_sessions (student_user_id) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS location_sharing_updates (
  id BIGSERIAL PRIMARY KEY,
  session_id BIGINT NOT NULL REFERENCES location_sharing_sessions(id) ON DELETE CASCADE,
  latitude NUMERIC(10,8) NOT NULL CHECK (latitude BETWEEN -90 AND 90),
  longitude NUMERIC(11,8) NOT NULL CHECK (longitude BETWEEN -180 AND 180),
  accuracy_m NUMERIC(10,2) CHECK (accuracy_m IS NULL OR accuracy_m >= 0),
  client_event_id VARCHAR(160) NOT NULL,
  reported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (session_id, client_event_id)
);
CREATE INDEX IF NOT EXISTS location_sharing_updates_session_time_idx ON location_sharing_updates (session_id, reported_at DESC);

CREATE TABLE IF NOT EXISTS location_sharing_audit_events (
  id BIGSERIAL PRIMARY KEY,
  session_id BIGINT REFERENCES location_sharing_sessions(id) ON DELETE SET NULL,
  event_type VARCHAR(40) NOT NULL CHECK (event_type IN ('consent_granted','location_updated','viewed','revoked','expired')),
  actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS location_sharing_audit_events_session_idx ON location_sharing_audit_events (session_id, created_at DESC);

ALTER TABLE operational_jobs DROP CONSTRAINT IF EXISTS operational_jobs_job_type_check;
ALTER TABLE operational_jobs ADD CONSTRAINT operational_jobs_job_type_check CHECK (job_type IN ('emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check','fee_invoice_generation','fee_reminder_check','complaint_sla_check','complaint_volume_check','location_retention_check'));

COMMIT;
