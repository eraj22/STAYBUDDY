BEGIN;

ALTER TABLE complaints
  ADD COLUMN IF NOT EXISTS room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS complaints_room_active_idx
  ON complaints (room_id,created_at DESC)
  WHERE room_id IS NOT NULL AND status IN ('open','in_progress','resolved');

COMMIT;
