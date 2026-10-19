BEGIN;

CREATE TABLE IF NOT EXISTS hostel_verification_checklist_items (
  id BIGSERIAL PRIMARY KEY,
  application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE,
  item_code VARCHAR(80) NOT NULL,
  label VARCHAR(240) NOT NULL,
  required BOOLEAN NOT NULL DEFAULT true,
  status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','passed','failed','waived')),
  review_note TEXT,
  reviewed_by_user_id INTEGER REFERENCES users(id),
  reviewed_at TIMESTAMPTZ,
  created_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (application_id, item_code)
);

CREATE TABLE IF NOT EXISTS hostel_verification_checklist_events (
  id BIGSERIAL PRIMARY KEY,
  checklist_item_id BIGINT NOT NULL REFERENCES hostel_verification_checklist_items(id) ON DELETE CASCADE,
  event_type VARCHAR(30) NOT NULL CHECK (event_type IN ('created','reviewed')),
  actor_user_id INTEGER REFERENCES users(id),
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hostel_verification_information_requests (
  id BIGSERIAL PRIMARY KEY,
  application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE,
  details TEXT NOT NULL,
  due_at TIMESTAMPTZ NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','responded','expired','cancelled')),
  requested_by_user_id INTEGER NOT NULL REFERENCES users(id),
  requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  responded_by_user_id INTEGER REFERENCES users(id),
  responded_at TIMESTAMPTZ,
  response_note TEXT
);

CREATE TABLE IF NOT EXISTS hostel_approval_condition_evidence (
  id BIGSERIAL PRIMARY KEY,
  condition_id BIGINT NOT NULL REFERENCES hostel_approval_conditions(id) ON DELETE CASCADE,
  document_id BIGINT NOT NULL REFERENCES hostel_registration_documents(id) ON DELETE RESTRICT,
  submitted_by_user_id INTEGER NOT NULL REFERENCES users(id),
  note TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (condition_id, document_id)
);

CREATE INDEX IF NOT EXISTS hostel_verification_checklist_application_idx ON hostel_verification_checklist_items (application_id, status);
CREATE INDEX IF NOT EXISTS hostel_verification_information_requests_application_idx ON hostel_verification_information_requests (application_id, status, due_at);
CREATE INDEX IF NOT EXISTS hostel_approval_condition_evidence_condition_idx ON hostel_approval_condition_evidence (condition_id, created_at);

COMMIT;
