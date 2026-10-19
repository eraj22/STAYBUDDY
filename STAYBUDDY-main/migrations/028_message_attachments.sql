BEGIN;

CREATE TABLE IF NOT EXISTS conversation_message_attachments (
  id BIGSERIAL PRIMARY KEY,
  message_id BIGINT NOT NULL REFERENCES conversation_messages(id) ON DELETE CASCADE,
  uploaded_by_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  storage_provider VARCHAR(40) NOT NULL,
  storage_key TEXT NOT NULL UNIQUE,
  checksum CHAR(64) NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  content_type VARCHAR(100) NOT NULL CHECK (content_type IN ('application/pdf','image/jpeg','image/png')),
  byte_size BIGINT NOT NULL CHECK (byte_size BETWEEN 1 AND 5242880),
  scan_status VARCHAR(30) NOT NULL DEFAULT 'accepted' CHECK (scan_status IN ('accepted','rejected')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conversation_message_attachments_message_idx
  ON conversation_message_attachments (message_id, created_at);

CREATE TABLE IF NOT EXISTS conversation_message_attachment_audit_events (
  id BIGSERIAL PRIMARY KEY,
  attachment_id BIGINT NOT NULL REFERENCES conversation_message_attachments(id) ON DELETE CASCADE,
  event_type VARCHAR(40) NOT NULL CHECK (event_type IN ('uploaded','downloaded')),
  actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conversation_message_attachment_audit_events_attachment_idx
  ON conversation_message_attachment_audit_events (attachment_id, created_at);

COMMIT;
