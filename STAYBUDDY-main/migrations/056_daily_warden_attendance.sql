BEGIN;

CREATE TABLE IF NOT EXISTS daily_warden_attendance (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  attendance_date DATE NOT NULL,
  status VARCHAR(10) NOT NULL CHECK (status IN ('present','absent')),
  marked_by_warden_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(hostel_id, student_user_id, attendance_date)
);

CREATE INDEX IF NOT EXISTS daily_warden_attendance_hostel_date_idx
  ON daily_warden_attendance(hostel_id, attendance_date);

COMMIT;
