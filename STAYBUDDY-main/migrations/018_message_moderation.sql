BEGIN;

CREATE TABLE IF NOT EXISTS conversation_message_reports (
  id BIGSERIAL PRIMARY KEY,
  message_id BIGINT NOT NULL REFERENCES conversation_messages(id) ON DELETE CASCADE,
  reporter_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  reason VARCHAR(30) NOT NULL CHECK (reason IN ('harassment','spam','safety','other')),
  details TEXT,
  status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','reviewed','dismissed')),
  reviewed_by_user_id INTEGER REFERENCES users(id),
  reviewed_at TIMESTAMPTZ,
  review_note TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (message_id, reporter_user_id)
);

CREATE INDEX IF NOT EXISTS conversation_message_reports_status_idx ON conversation_message_reports (status, created_at DESC);
CREATE INDEX IF NOT EXISTS conversation_message_reports_message_idx ON conversation_message_reports (message_id, created_at DESC);

COMMIT;
