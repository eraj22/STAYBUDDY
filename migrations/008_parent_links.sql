BEGIN;

ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users ADD CONSTRAINT users_role_check
  CHECK (role IN ('student', 'parent', 'owner', 'warden', 'admin'));

CREATE TABLE IF NOT EXISTS parent_student_links (
  id SERIAL PRIMARY KEY,
  parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  invited_by_student_id INTEGER NOT NULL REFERENCES users(id),
  invitation_token_hash CHAR(64) NOT NULL UNIQUE,
  status VARCHAR(20) NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'active', 'revoked')),
  expires_at TIMESTAMPTZ NOT NULL,
  accepted_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (parent_user_id <> student_user_id),
  CHECK ((status <> 'active') OR accepted_at IS NOT NULL),
  CHECK ((status <> 'revoked') OR revoked_at IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS parent_student_links_active_or_pending_unique
  ON parent_student_links (parent_user_id, student_user_id)
  WHERE status IN ('pending', 'active');
CREATE INDEX IF NOT EXISTS parent_student_links_parent_active_idx
  ON parent_student_links (parent_user_id, student_user_id)
  WHERE status = 'active';
CREATE INDEX IF NOT EXISTS parent_student_links_student_active_idx
  ON parent_student_links (student_user_id, parent_user_id)
  WHERE status = 'active';

COMMIT;
