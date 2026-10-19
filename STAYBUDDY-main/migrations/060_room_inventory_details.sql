BEGIN;

-- Owner-managed room inventory details. These are deliberately stored on the
-- room, not in a booking/assignment table, because an owner defines them
-- before any student is allocated.
ALTER TABLE rooms
  ADD COLUMN IF NOT EXISTS amenities JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS notes TEXT;

COMMIT;
