BEGIN;

-- A parent link permits history access, but student approval is required for
-- automatic movement and safety notifications. Consent belongs to one link,
-- so a revoked link cannot silently regain consent if the users link again.
CREATE TABLE IF NOT EXISTS student_attendance_notification_permissions (
  parent_student_link_id INTEGER PRIMARY KEY REFERENCES parent_student_links(id) ON DELETE CASCADE,
  allowed BOOLEAN NOT NULL DEFAULT false,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE attendance_qr_scan_attempts DROP CONSTRAINT IF EXISTS attendance_qr_scan_attempts_outcome_check;
ALTER TABLE attendance_qr_scan_attempts ADD CONSTRAINT attendance_qr_scan_attempts_outcome_check
  CHECK (outcome IN ('accepted','invalid_token','expired','already_used','wrong_hostel',
    'inactive_stay','duplicate_direction','out_of_order','idempotency_conflict'));

COMMIT;
