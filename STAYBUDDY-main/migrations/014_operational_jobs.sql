BEGIN;

CREATE TABLE IF NOT EXISTS operational_jobs (
  id BIGSERIAL PRIMARY KEY,
  job_type VARCHAR(60) NOT NULL CHECK (job_type IN ('emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check')),
  dedupe_key VARCHAR(220) NOT NULL UNIQUE,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  status VARCHAR(20) NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','processing','succeeded','failed','dead')),
  run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  max_attempts INTEGER NOT NULL DEFAULT 5 CHECK (max_attempts BETWEEN 1 AND 20),
  locked_at TIMESTAMPTZ,
  locked_by VARCHAR(120),
  last_error TEXT,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS operational_job_attempts (
  id BIGSERIAL PRIMARY KEY,
  job_id BIGINT NOT NULL REFERENCES operational_jobs(id) ON DELETE CASCADE,
  attempt_number INTEGER NOT NULL,
  outcome VARCHAR(20) NOT NULL CHECK (outcome IN ('succeeded','failed','dead')),
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (job_id, attempt_number)
);

ALTER TABLE fee_reminder_deliveries
  ADD COLUMN IF NOT EXISTS provider VARCHAR(60),
  ADD COLUMN IF NOT EXISTS provider_reference VARCHAR(160),
  ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS error_message TEXT,
  ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS operational_jobs_due_idx
  ON operational_jobs (run_at, id) WHERE status='queued';
CREATE INDEX IF NOT EXISTS operational_jobs_type_status_idx
  ON operational_jobs (job_type, status, created_at DESC);
CREATE INDEX IF NOT EXISTS operational_job_attempts_job_idx
  ON operational_job_attempts (job_id, created_at DESC);

COMMIT;
