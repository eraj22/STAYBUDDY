BEGIN;

-- The polygon is an ordered ring of latitude/longitude vertices.  Keeping it
-- separate from the hostel's map pin lets attendance use the actual building,
-- gate and entrance area rather than a broad circular radius.
CREATE TABLE IF NOT EXISTS hostel_attendance_geofences (
  hostel_id INTEGER PRIMARY KEY REFERENCES hostels(id) ON DELETE CASCADE,
  boundary JSONB NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT true,
  updated_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (jsonb_typeof(boundary) = 'array'),
  CHECK (jsonb_array_length(boundary) >= 3)
);

COMMIT;
