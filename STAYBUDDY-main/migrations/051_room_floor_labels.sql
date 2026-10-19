BEGIN;

-- Rooms belong to a physical hostel floor. Existing rooms are safely kept on
-- Ground Floor until an owner assigns their actual location.
ALTER TABLE rooms
  ADD COLUMN IF NOT EXISTS floor_label VARCHAR(80) NOT NULL DEFAULT 'Ground Floor';

CREATE INDEX IF NOT EXISTS rooms_hostel_floor_room_idx
  ON rooms (hostel_id, floor_label, room_number);

COMMIT;
