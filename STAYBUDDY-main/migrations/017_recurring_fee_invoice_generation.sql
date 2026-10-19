BEGIN;

ALTER TABLE operational_jobs DROP CONSTRAINT IF EXISTS operational_jobs_job_type_check;
ALTER TABLE operational_jobs ADD CONSTRAINT operational_jobs_job_type_check
  CHECK (job_type IN ('emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check','fee_invoice_generation'));

CREATE INDEX IF NOT EXISTS student_fee_assignments_generation_idx
  ON student_fee_assignments (schedule_id, active, starts_on, ends_on);

COMMIT;
