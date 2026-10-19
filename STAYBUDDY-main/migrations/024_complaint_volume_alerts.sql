BEGIN;

CREATE TABLE IF NOT EXISTS hostel_complaint_alert_rules (
  hostel_id INTEGER PRIMARY KEY REFERENCES hostels(id) ON DELETE CASCADE,
  open_complaint_threshold INTEGER NOT NULL CHECK (open_complaint_threshold BETWEEN 1 AND 1000),
  cooldown_minutes INTEGER NOT NULL DEFAULT 1440 CHECK (cooldown_minutes BETWEEN 15 AND 10080),
  enabled BOOLEAN NOT NULL DEFAULT true,
  updated_by_user_id INTEGER NOT NULL REFERENCES users(id),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS complaint_volume_alerts (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  threshold_count INTEGER NOT NULL,
  observed_open_count INTEGER NOT NULL,
  triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  triggered_by_worker VARCHAR(120) NOT NULL
);

ALTER TABLE operational_jobs DROP CONSTRAINT IF EXISTS operational_jobs_job_type_check;
ALTER TABLE operational_jobs ADD CONSTRAINT operational_jobs_job_type_check CHECK (job_type IN ('emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check','fee_invoice_generation','complaint_sla_check','complaint_volume_check'));
CREATE INDEX IF NOT EXISTS complaint_volume_alerts_hostel_time_idx ON complaint_volume_alerts (hostel_id, triggered_at DESC);

COMMIT;
