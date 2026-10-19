BEGIN;
CREATE TABLE IF NOT EXISTS complaint_sla_escalations (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE,
  escalated_by_worker VARCHAR(120) NOT NULL,
  escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  details JSONB NOT NULL DEFAULT '{}'::jsonb
);
ALTER TABLE operational_jobs DROP CONSTRAINT IF EXISTS operational_jobs_job_type_check;
ALTER TABLE operational_jobs ADD CONSTRAINT operational_jobs_job_type_check CHECK (job_type IN ('emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check','fee_invoice_generation','complaint_sla_check'));
CREATE INDEX IF NOT EXISTS complaints_sla_due_idx ON complaints (sla_due_at) WHERE status NOT IN ('resolved','closed') AND escalated_at IS NULL;
COMMIT;
