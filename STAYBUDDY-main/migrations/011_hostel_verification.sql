BEGIN;

CREATE TABLE IF NOT EXISTS hostel_registration_applications (
  id BIGSERIAL PRIMARY KEY,
  owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  hostel_id INTEGER REFERENCES hostels(id) ON DELETE SET NULL,
  status VARCHAR(40) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','submitted','more_information_requested','inspection_scheduled','provisional','conditionally_approved','approved','rejected','withdrawn')),
  reapplication_allowed BOOLEAN NOT NULL DEFAULT true,
  submission_notes TEXT,
  submitted_at TIMESTAMPTZ, decided_at TIMESTAMPTZ, decided_by_user_id INTEGER REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS hostel_registration_documents (
  id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE,
  document_type VARCHAR(80) NOT NULL, storage_provider VARCHAR(40) NOT NULL DEFAULT 'local', storage_key TEXT NOT NULL,
  checksum VARCHAR(128), verification_status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (verification_status IN ('pending','accepted','rejected')),
  submitted_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), reviewed_at TIMESTAMPTZ, reviewed_by_user_id INTEGER REFERENCES users(id), review_note TEXT
);
CREATE TABLE IF NOT EXISTS hostel_verification_events (
  id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE,
  event_type VARCHAR(40) NOT NULL, actor_user_id INTEGER REFERENCES users(id), details JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS hostel_inspections (
  id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE,
  priority VARCHAR(20) NOT NULL CHECK (priority IN ('high','standard','low')), assigned_to_user_id INTEGER REFERENCES users(id), scheduled_for TIMESTAMPTZ NOT NULL,
  scope JSONB NOT NULL DEFAULT '[]'::jsonb, findings TEXT, outcome VARCHAR(20) CHECK (outcome IN ('passed','failed','follow_up_required')),
  completed_at TIMESTAMPTZ, created_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS hostel_approval_conditions (
  id BIGSERIAL PRIMARY KEY, application_id BIGINT NOT NULL REFERENCES hostel_registration_applications(id) ON DELETE CASCADE,
  requirement TEXT NOT NULL, due_at TIMESTAMPTZ NOT NULL, status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','completed','overdue','waived')),
  completed_at TIMESTAMPTZ, completed_by_user_id INTEGER REFERENCES users(id), evidence_storage_key TEXT, created_by_user_id INTEGER NOT NULL REFERENCES users(id), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS hostel_registration_owner_idx ON hostel_registration_applications (owner_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS hostel_registration_status_idx ON hostel_registration_applications (status, created_at DESC);
CREATE INDEX IF NOT EXISTS hostel_verification_events_application_idx ON hostel_verification_events (application_id, created_at);

COMMIT;
