BEGIN;

ALTER TABLE hostels ADD COLUMN IF NOT EXISTS source_available_capacity INTEGER;

UPDATE hostels h
SET source_available_capacity = LEAST(
  h.total_capacity,
  h.available_capacity + COALESCE((
    SELECT COUNT(*)::int
    FROM bookings b
    WHERE b.hostel_id = h.id
      AND b.status IN ('pending', 'confirmed')
  ), 0)
)
WHERE h.source = 'ml_catalog'
  AND h.source_available_capacity IS NULL;

ALTER TABLE hostels DROP CONSTRAINT IF EXISTS hostels_source_available_capacity_check;
ALTER TABLE hostels ADD CONSTRAINT hostels_source_available_capacity_check
  CHECK (
    source_available_capacity IS NULL
    OR source_available_capacity BETWEEN 0 AND total_capacity
  );

COMMIT;