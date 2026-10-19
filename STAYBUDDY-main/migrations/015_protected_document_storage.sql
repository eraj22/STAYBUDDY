BEGIN;

ALTER TABLE hostel_registration_documents
  ADD COLUMN IF NOT EXISTS original_filename VARCHAR(255),
  ADD COLUMN IF NOT EXISTS content_type VARCHAR(120),
  ADD COLUMN IF NOT EXISTS byte_size BIGINT,
  ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS deleted_by_user_id INTEGER REFERENCES users(id),
  ADD COLUMN IF NOT EXISTS retention_until TIMESTAMPTZ;

ALTER TABLE hostel_registration_documents
  DROP CONSTRAINT IF EXISTS hostel_registration_documents_byte_size_check;
ALTER TABLE hostel_registration_documents
  ADD CONSTRAINT hostel_registration_documents_byte_size_check CHECK (byte_size IS NULL OR byte_size > 0);

CREATE INDEX IF NOT EXISTS hostel_registration_documents_active_application_idx
  ON hostel_registration_documents (application_id, created_at DESC) WHERE deleted_at IS NULL;

COMMIT;
