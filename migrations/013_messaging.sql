BEGIN;

CREATE TABLE IF NOT EXISTS conversations (
  id BIGSERIAL PRIMARY KEY,
  conversation_type VARCHAR(30) NOT NULL CHECK (conversation_type IN ('parent_student','parent_staff')),
  parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  hostel_id INTEGER REFERENCES hostels(id) ON DELETE SET NULL,
  staff_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK ((conversation_type='parent_student' AND hostel_id IS NULL AND staff_user_id IS NULL) OR (conversation_type='parent_staff' AND hostel_id IS NOT NULL AND staff_user_id IS NOT NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS conversations_unique_participants
  ON conversations (conversation_type, parent_user_id, student_user_id, COALESCE(hostel_id, 0), COALESCE(staff_user_id, 0));

CREATE TABLE IF NOT EXISTS conversation_messages (
  id BIGSERIAL PRIMARY KEY,
  conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  sender_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  body TEXT NOT NULL CHECK (char_length(trim(body)) BETWEEN 1 AND 4000),
  delivered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  read_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conversations_parent_idx ON conversations (parent_user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS conversations_student_idx ON conversations (student_user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS conversation_messages_conversation_idx ON conversation_messages (conversation_id, created_at);

COMMIT;
