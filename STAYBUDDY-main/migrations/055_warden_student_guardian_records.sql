BEGIN;

-- Operational profile information entered by a warden for a student who is
-- staying (or will stay) in their assigned hostel.  Login credentials remain
-- in users; this table holds the hostel-facing information and guardian data.
CREATE TABLE IF NOT EXISTS warden_student_records (
  student_user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  cnic VARCHAR(20),
  address TEXT,
  guardian_name VARCHAR(160),
  guardian_relationship VARCHAR(80),
  guardian_phone VARCHAR(30),
  guardian_email VARCHAR(180),
  guardian_address TEXT,
  created_by_warden_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS warden_student_records_hostel_idx
  ON warden_student_records (hostel_id);

COMMIT;
