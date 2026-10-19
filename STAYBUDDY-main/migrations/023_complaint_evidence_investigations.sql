BEGIN;

CREATE TABLE IF NOT EXISTS complaint_evidence (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  storage_provider VARCHAR(40) NOT NULL,
  storage_key TEXT NOT NULL UNIQUE,
  checksum CHAR(64) NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  content_type VARCHAR(120) NOT NULL,
  byte_size BIGINT NOT NULL CHECK (byte_size > 0 AND byte_size <= 5242880),
  uploaded_by_user_id INTEGER NOT NULL REFERENCES users(id),
  deleted_at TIMESTAMPTZ,
  deleted_by_user_id INTEGER REFERENCES users(id),
  retention_until TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS complaint_investigations (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  findings TEXT NOT NULL,
  root_cause TEXT,
  immediate_actions TEXT,
  preventive_measures TEXT,
  created_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS complaint_owner_escalations (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE,
  reason TEXT NOT NULL,
  urgency VARCHAR(20) NOT NULL CHECK (urgency IN ('high','urgent')),
  idempotency_key VARCHAR(160) NOT NULL UNIQUE,
  escalated_by_user_id INTEGER NOT NULL REFERENCES users(id),
  escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS complaint_activity_events (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  event_type VARCHAR(50) NOT NULL,
  actor_user_id INTEGER NOT NULL REFERENCES users(id),
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS complaint_evidence_active_idx ON complaint_evidence (complaint_id, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS complaint_investigations_complaint_idx ON complaint_investigations (complaint_id, created_at DESC);
CREATE INDEX IF NOT EXISTS complaint_activity_events_complaint_idx ON complaint_activity_events (complaint_id, created_at DESC);

COMMIT;
