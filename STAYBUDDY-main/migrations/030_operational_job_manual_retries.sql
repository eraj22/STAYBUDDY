BEGIN;

CREATE TABLE IF NOT EXISTS operational_job_manual_retries (
  id BIGSERIAL PRIMARY KEY,
  job_id BIGINT NOT NULL REFERENCES operational_jobs(id) ON DELETE CASCADE,
  retried_by_user_id INTEGER NOT NULL REFERENCES users(id),
  reason TEXT NOT NULL CHECK (char_length(trim(reason)) BETWEEN 1 AND 2000),
  previous_attempt_count INTEGER NOT NULL CHECK (previous_attempt_count >= 0),
  previous_max_attempts INTEGER NOT NULL CHECK (previous_max_attempts BETWEEN 1 AND 20),
  retry_max_attempts INTEGER NOT NULL CHECK (retry_max_attempts BETWEEN 1 AND 20),
  retried_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (retry_max_attempts = previous_max_attempts + 1)
);

CREATE INDEX IF NOT EXISTS operational_job_manual_retries_job_idx
  ON operational_job_manual_retries (job_id, retried_at DESC);

COMMIT;
