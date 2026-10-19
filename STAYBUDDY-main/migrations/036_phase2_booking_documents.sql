BEGIN;

CREATE TABLE IF NOT EXISTS booking_document_requirements (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  document_code VARCHAR(80) NOT NULL,
  label VARCHAR(180) NOT NULL,
  required BOOLEAN NOT NULL DEFAULT true,
  active BOOLEAN NOT NULL DEFAULT true,
  created_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (hostel_id,document_code)
);

CREATE TABLE IF NOT EXISTS booking_documents (
  id BIGSERIAL PRIMARY KEY,
  booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  requirement_id BIGINT REFERENCES booking_document_requirements(id) ON DELETE RESTRICT,
  document_code VARCHAR(80) NOT NULL,
  storage_provider VARCHAR(40) NOT NULL,
  storage_key TEXT NOT NULL UNIQUE,
  checksum CHAR(64) NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  content_type VARCHAR(120) NOT NULL,
  byte_size INTEGER NOT NULL CHECK (byte_size BETWEEN 1 AND 5242880),
  status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','accepted','rejected')),
  review_note TEXT,
  submitted_by_user_id INTEGER NOT NULL REFERENCES users(id),
  reviewed_by_user_id INTEGER REFERENCES users(id),
  reviewed_at TIMESTAMPTZ,
  deleted_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS booking_documents_booking_idx ON booking_documents (booking_id,created_at DESC) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS booking_document_events (
  id BIGSERIAL PRIMARY KEY,
  booking_document_id BIGINT NOT NULL REFERENCES booking_documents(id) ON DELETE CASCADE,
  event_type VARCHAR(20) NOT NULL CHECK (event_type IN ('uploaded','accepted','rejected','deleted')),
  actor_user_id INTEGER NOT NULL REFERENCES users(id),
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
