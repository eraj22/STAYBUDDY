BEGIN;

ALTER TABLE notification_preferences
  ADD COLUMN IF NOT EXISTS emergency_alerts BOOLEAN NOT NULL DEFAULT true;

CREATE TABLE IF NOT EXISTS emergency_contacts (
  id SERIAL PRIMARY KEY,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR(120) NOT NULL,
  relationship VARCHAR(80) NOT NULL,
  phone VARCHAR(30),
  email VARCHAR(180),
  priority SMALLINT NOT NULL DEFAULT 1 CHECK (priority > 0),
  verified_at TIMESTAMPTZ,
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (phone IS NOT NULL OR email IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS emergency_incidents (
  id BIGSERIAL PRIMARY KEY,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE RESTRICT,
  booking_id INTEGER REFERENCES bookings(id) ON DELETE SET NULL,
  room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL,
  alert_type VARCHAR(30) NOT NULL CHECK (alert_type IN ('medical','safety','fire','accident','general')),
  description TEXT,
  status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active','acknowledged','escalated','cancelled','resolved')),
  latitude NUMERIC(10,8), longitude NUMERIC(11,8), location_accuracy_m NUMERIC(10,2),
  idempotency_key VARCHAR(160) NOT NULL UNIQUE,
  cancelled_by_user_id INTEGER REFERENCES users(id), cancelled_at TIMESTAMPTZ, cancellation_reason TEXT,
  resolved_by_user_id INTEGER REFERENCES users(id), resolved_at TIMESTAMPTZ, resolution_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK ((latitude IS NULL AND longitude IS NULL) OR (latitude IS NOT NULL AND longitude IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS emergency_timeline_events (
  id BIGSERIAL PRIMARY KEY,
  incident_id BIGINT NOT NULL REFERENCES emergency_incidents(id) ON DELETE CASCADE,
  event_type VARCHAR(30) NOT NULL CHECK (event_type IN ('created','acknowledged','escalated','cancelled','resolved','note')),
  actor_user_id INTEGER REFERENCES users(id),
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS emergency_delivery_attempts (
  id BIGSERIAL PRIMARY KEY,
  incident_id BIGINT NOT NULL REFERENCES emergency_incidents(id) ON DELETE CASCADE,
  recipient_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  contact_id INTEGER REFERENCES emergency_contacts(id) ON DELETE SET NULL,
  channel VARCHAR(20) NOT NULL CHECK (channel IN ('in_app','push','sms','email','call')),
  status VARCHAR(20) NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','delivered','failed')),
  provider VARCHAR(60), provider_reference VARCHAR(160), attempt_count INTEGER NOT NULL DEFAULT 0,
  error_message TEXT, next_retry_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (recipient_user_id IS NOT NULL OR contact_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS emergency_incidents_hostel_status_idx ON emergency_incidents (hostel_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS emergency_incidents_student_created_idx ON emergency_incidents (student_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS emergency_timeline_incident_idx ON emergency_timeline_events (incident_id, created_at);
CREATE INDEX IF NOT EXISTS emergency_delivery_pending_idx ON emergency_delivery_attempts (status, next_retry_at) WHERE status='queued';

COMMIT;
