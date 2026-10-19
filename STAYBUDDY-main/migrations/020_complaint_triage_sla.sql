BEGIN;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS priority VARCHAR(20) NOT NULL DEFAULT 'medium',
  ADD COLUMN IF NOT EXISTS sla_due_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS escalated_at TIMESTAMPTZ;

ALTER TABLE complaints DROP CONSTRAINT IF EXISTS complaints_priority_check;
ALTER TABLE complaints ADD CONSTRAINT complaints_priority_check CHECK (priority IN ('low','medium','high','urgent'));

CREATE TABLE IF NOT EXISTS complaint_triage_history (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  previous_priority VARCHAR(20),
  next_priority VARCHAR(20) NOT NULL,
  previous_sla_due_at TIMESTAMPTZ,
  next_sla_due_at TIMESTAMPTZ,
  note TEXT NOT NULL,
  triaged_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS complaints_priority_sla_idx ON complaints (hostel_id, priority, sla_due_at) WHERE status NOT IN ('resolved','closed');
CREATE INDEX IF NOT EXISTS complaint_triage_history_complaint_idx ON complaint_triage_history (complaint_id, created_at DESC);

COMMIT;
