BEGIN;

ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users ADD CONSTRAINT users_role_check
  CHECK (role IN ('student', 'owner', 'warden', 'admin'));

CREATE TABLE IF NOT EXISTS student_profiles (
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  university VARCHAR(150),
  department VARCHAR(150),
  gender VARCHAR(30),
  budget_max NUMERIC(12, 2) CHECK (budget_max IS NULL OR budget_max >= 0),
  max_distance_km NUMERIC(6, 2) CHECK (max_distance_km IS NULL OR max_distance_km >= 0),
  study_preference NUMERIC(3, 2) CHECK (study_preference IS NULL OR (study_preference >= 0 AND study_preference <= 1)),
  food_preference VARCHAR(30),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS notification_preferences (
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  booking_updates BOOLEAN NOT NULL DEFAULT true,
  complaint_updates BOOLEAN NOT NULL DEFAULT true,
  announcements BOOLEAN NOT NULL DEFAULT true,
  email_enabled BOOLEAN NOT NULL DEFAULT true,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hostel_owners (
  owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (owner_user_id, hostel_id),
  UNIQUE (hostel_id)
);

CREATE TABLE IF NOT EXISTS warden_assignments (
  warden_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  assigned_by_user_id INTEGER NOT NULL REFERENCES users(id),
  assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (warden_user_id, hostel_id)
);

CREATE TABLE IF NOT EXISTS complaint_status_history (
  id SERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  previous_status VARCHAR(30),
  next_status VARCHAR(30) NOT NULL,
  changed_by_user_id INTEGER NOT NULL REFERENCES users(id),
  changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS hostel_owners_owner_idx ON hostel_owners (owner_user_id);
CREATE INDEX IF NOT EXISTS warden_assignments_hostel_idx ON warden_assignments (hostel_id);
CREATE INDEX IF NOT EXISTS complaint_status_history_complaint_idx ON complaint_status_history (complaint_id, changed_at DESC);

COMMIT;