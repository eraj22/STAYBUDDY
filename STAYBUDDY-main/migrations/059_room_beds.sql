BEGIN;

-- A room can have its own named beds (for example A, B, C or 1, 2, 3).
-- This makes the owner-provided room/bed plan the source of truth for
-- allocation and future attendance records.
CREATE TABLE IF NOT EXISTS room_beds (
  id BIGSERIAL PRIMARY KEY,
  room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
  bed_number VARCHAR(40) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'available'
    CHECK (status IN ('available', 'occupied', 'inactive')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(room_id, bed_number)
);

CREATE INDEX IF NOT EXISTS room_beds_room_id_idx ON room_beds(room_id);

COMMIT;
