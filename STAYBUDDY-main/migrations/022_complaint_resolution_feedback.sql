BEGIN;

CREATE TABLE IF NOT EXISTS complaint_resolution_feedback (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id),
  rating SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
  comment TEXT,
  submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (comment IS NULL OR char_length(comment) <= 2000)
);

CREATE INDEX IF NOT EXISTS complaint_resolution_feedback_student_idx
  ON complaint_resolution_feedback (student_user_id, submitted_at DESC);

COMMIT;
