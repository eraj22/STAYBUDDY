BEGIN;

CREATE TABLE IF NOT EXISTS conversation_staff_handovers (
  id BIGSERIAL PRIMARY KEY,
  conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  previous_staff_user_id INTEGER NOT NULL REFERENCES users(id),
  next_staff_user_id INTEGER NOT NULL REFERENCES users(id),
  handed_over_by_user_id INTEGER NOT NULL REFERENCES users(id),
  reason TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (previous_staff_user_id <> next_staff_user_id)
);

CREATE INDEX IF NOT EXISTS conversation_staff_handovers_conversation_idx ON conversation_staff_handovers (conversation_id, created_at DESC);

COMMIT;
