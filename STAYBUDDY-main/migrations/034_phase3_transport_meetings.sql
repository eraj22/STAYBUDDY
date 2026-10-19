BEGIN;

ALTER TABLE notification_preferences
  ADD COLUMN IF NOT EXISTS transport_updates BOOLEAN NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS communication_updates BOOLEAN NOT NULL DEFAULT true;

CREATE TABLE IF NOT EXISTS hostel_transport_services (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  name VARCHAR(160) NOT NULL,
  mode VARCHAR(30) NOT NULL CHECK (mode IN ('shuttle','bus','van','partner','information_only')),
  vehicle_number VARCHAR(80),
  driver_name VARCHAR(160),
  driver_phone VARCHAR(40),
  operating_days JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(operating_days)='array'),
  safety_features JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(safety_features)='array'),
  monthly_cost NUMERIC(12,2) CHECK (monthly_cost IS NULL OR monthly_cost >= 0),
  currency CHAR(3) NOT NULL DEFAULT 'PKR' CHECK (currency='PKR'),
  active BOOLEAN NOT NULL DEFAULT true,
  created_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS hostel_transport_services_hostel_idx ON hostel_transport_services (hostel_id,active);

CREATE TABLE IF NOT EXISTS hostel_transport_stops (
  id BIGSERIAL PRIMARY KEY,
  service_id BIGINT NOT NULL REFERENCES hostel_transport_services(id) ON DELETE CASCADE,
  name VARCHAR(160) NOT NULL,
  latitude NUMERIC(10,8) CHECK (latitude IS NULL OR latitude BETWEEN -90 AND 90),
  longitude NUMERIC(11,8) CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180),
  stop_order INTEGER NOT NULL CHECK (stop_order >= 0),
  scheduled_time TIME,
  UNIQUE (service_id,stop_order),
  CHECK ((latitude IS NULL AND longitude IS NULL) OR (latitude IS NOT NULL AND longitude IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS student_transport_assignments (
  id BIGSERIAL PRIMARY KEY,
  service_id BIGINT NOT NULL REFERENCES hostel_transport_services(id) ON DELETE RESTRICT,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  pickup_stop_id BIGINT REFERENCES hostel_transport_stops(id) ON DELETE SET NULL,
  dropoff_stop_id BIGINT REFERENCES hostel_transport_stops(id) ON DELETE SET NULL,
  active BOOLEAN NOT NULL DEFAULT true,
  assigned_by_user_id INTEGER NOT NULL REFERENCES users(id),
  assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS student_transport_one_active_service
  ON student_transport_assignments (service_id,student_user_id) WHERE active;
CREATE INDEX IF NOT EXISTS student_transport_assignments_student_idx
  ON student_transport_assignments (student_user_id,active);

CREATE TABLE IF NOT EXISTS transport_trip_events (
  id BIGSERIAL PRIMARY KEY,
  assignment_id BIGINT NOT NULL REFERENCES student_transport_assignments(id) ON DELETE CASCADE,
  event_type VARCHAR(30) NOT NULL CHECK (event_type IN ('departed','picked_up','arrived','dropped_off','delayed','cancelled')),
  occurred_at TIMESTAMPTZ NOT NULL,
  note TEXT,
  idempotency_key VARCHAR(160) NOT NULL,
  created_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (assignment_id,idempotency_key)
);
CREATE INDEX IF NOT EXISTS transport_trip_events_assignment_time_idx
  ON transport_trip_events (assignment_id,occurred_at DESC);

CREATE TABLE IF NOT EXISTS parent_staff_meeting_requests (
  id BIGSERIAL PRIMARY KEY,
  parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  requested_staff_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  scheduled_for TIMESTAMPTZ NOT NULL,
  duration_minutes INTEGER NOT NULL CHECK (duration_minutes BETWEEN 15 AND 120),
  mode VARCHAR(20) NOT NULL CHECK (mode IN ('in_person','voice')),
  topic VARCHAR(240) NOT NULL,
  details TEXT,
  status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','accepted','declined','cancelled','completed','rescheduled')),
  response_note TEXT,
  responded_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  responded_at TIMESTAMPTZ,
  cancelled_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS parent_staff_meetings_parent_idx ON parent_staff_meeting_requests (parent_user_id,created_at DESC);
CREATE INDEX IF NOT EXISTS parent_staff_meetings_hostel_status_idx ON parent_staff_meeting_requests (hostel_id,status,scheduled_for);

CREATE TABLE IF NOT EXISTS parent_staff_meeting_events (
  id BIGSERIAL PRIMARY KEY,
  meeting_id BIGINT NOT NULL REFERENCES parent_staff_meeting_requests(id) ON DELETE CASCADE,
  event_type VARCHAR(30) NOT NULL CHECK (event_type IN ('requested','accepted','declined','cancelled','completed','rescheduled')),
  actor_user_id INTEGER NOT NULL REFERENCES users(id),
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
