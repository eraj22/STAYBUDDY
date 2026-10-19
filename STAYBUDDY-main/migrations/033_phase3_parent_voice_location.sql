BEGIN;

ALTER TABLE notification_preferences
  ADD COLUMN IF NOT EXISTS location_updates BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE location_sharing_sessions
  ADD COLUMN IF NOT EXISTS history_enabled BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS geofencing_enabled BOOLEAN NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS location_geofences (
  id BIGSERIAL PRIMARY KEY,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR(120) NOT NULL,
  latitude NUMERIC(10,8) NOT NULL CHECK (latitude BETWEEN -90 AND 90),
  longitude NUMERIC(11,8) NOT NULL CHECK (longitude BETWEEN -180 AND 180),
  radius_m INTEGER NOT NULL CHECK (radius_m BETWEEN 50 AND 5000),
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  disabled_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS location_geofences_parent_student_idx
  ON location_geofences (parent_user_id, student_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS location_geofences_student_active_idx
  ON location_geofences (student_user_id) WHERE active;

CREATE TABLE IF NOT EXISTS location_geofence_states (
  geofence_id BIGINT NOT NULL REFERENCES location_geofences(id) ON DELETE CASCADE,
  session_id BIGINT NOT NULL REFERENCES location_sharing_sessions(id) ON DELETE CASCADE,
  last_inside BOOLEAN NOT NULL,
  last_evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_transition_at TIMESTAMPTZ,
  PRIMARY KEY (geofence_id, session_id)
);

CREATE TABLE IF NOT EXISTS location_geofence_events (
  id BIGSERIAL PRIMARY KEY,
  geofence_id BIGINT NOT NULL REFERENCES location_geofences(id) ON DELETE CASCADE,
  session_id BIGINT NOT NULL REFERENCES location_sharing_sessions(id) ON DELETE CASCADE,
  location_update_id BIGINT NOT NULL REFERENCES location_sharing_updates(id) ON DELETE CASCADE,
  event_type VARCHAR(10) NOT NULL CHECK (event_type IN ('entered','exited')),
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (geofence_id, location_update_id)
);
CREATE INDEX IF NOT EXISTS location_geofence_events_parent_time_idx
  ON location_geofence_events (geofence_id, occurred_at DESC);

CREATE TABLE IF NOT EXISTS announcement_acknowledgements (
  announcement_id INTEGER NOT NULL REFERENCES announcements(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  student_user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (announcement_id, user_id)
);

COMMIT;
