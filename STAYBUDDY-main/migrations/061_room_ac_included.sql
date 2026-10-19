BEGIN;

-- Hostel-wide amenities belong on the hostel profile. A room varies only by
-- whether air conditioning is included in its rent.
ALTER TABLE rooms
  ADD COLUMN IF NOT EXISTS ac_included BOOLEAN NOT NULL DEFAULT false;

COMMIT;
