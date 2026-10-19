-- Opt-in sample listing data for the two original Islamabad hostel records.
-- The owner confirmed these two hostels are verified. Prices below are examples
-- for chatbot testing, not independently verified advertised rates.
-- Run with: Get-Content scripts/seed_verified_chatbot_hostels.sql -Raw | docker compose exec -T postgres psql -X -v ON_ERROR_STOP=1 -U postgres -d staybuddy
BEGIN;

DO $$
BEGIN
  IF (SELECT COUNT(*) FROM hostels
      WHERE (id = 1 AND name = 'Islamia Girls Hostel' AND city = 'Islamabad' AND source = 'manual')
         OR (id = 2 AND name = 'Fatima Girls Hostel' AND city = 'Islamabad' AND source = 'manual')) <> 2 THEN
    RAISE EXCEPTION 'Expected original Islamia and Fatima manual hostel records; no changes applied';
  END IF;
  IF EXISTS (SELECT 1 FROM hostels WHERE id IN (1, 2)
             AND publication_status NOT IN ('draft', 'published')) THEN
    RAISE EXCEPTION 'A hostel is submitted or suspended; no changes applied';
  END IF;
END $$;

UPDATE hostels
SET verified = true,
    publication_status = 'published',
    hostel_type = 'girls',
    university = 'FAST University Islamabad (FAST-NUCES)',
    amenities = CASE WHEN amenities = '[]'::jsonb THEN '["wifi", "mess"]'::jsonb ELSE amenities END,
    description = CASE WHEN description LIKE '%Sample room prices for chatbot testing.%'
                       THEN description ELSE description || ' Sample room prices for chatbot testing.' END
WHERE id = 1;

UPDATE hostels
SET verified = true,
    publication_status = 'published',
    hostel_type = 'girls',
    amenities = CASE WHEN amenities = '[]'::jsonb THEN '["laundry", "study room"]'::jsonb ELSE amenities END,
    description = CASE WHEN description LIKE '%Sample room prices for chatbot testing.%'
                       THEN description ELSE description || ' Sample room prices for chatbot testing.' END
WHERE id = 2;

UPDATE rooms
SET monthly_rent = COALESCE(monthly_rent,
  CASE
    WHEN hostel_id = 1 AND lower(room_type) = 'shared' THEN 12000
    WHEN hostel_id = 1 AND lower(room_type) = 'double' THEN 17000
    WHEN hostel_id = 1 AND lower(room_type) = 'single' THEN 22000
    WHEN hostel_id = 2 AND lower(room_type) = 'shared' THEN 11000
    WHEN hostel_id = 2 AND lower(room_type) = 'single' THEN 19000
  END)
WHERE hostel_id IN (1, 2);

INSERT INTO rooms (hostel_id, room_number, room_type, capacity, available_capacity, monthly_rent)
SELECT 1, '301', 'single', 1, 1, 22000
WHERE NOT EXISTS (SELECT 1 FROM rooms WHERE hostel_id = 1 AND lower(room_type) = 'single')
  AND NOT EXISTS (SELECT 1 FROM rooms WHERE hostel_id = 1 AND room_number = '301');

INSERT INTO rooms (hostel_id, room_number, room_type, capacity, available_capacity, monthly_rent)
SELECT 2, '201', 'single', 1, 1, 19000
WHERE NOT EXISTS (SELECT 1 FROM rooms WHERE hostel_id = 2 AND lower(room_type) = 'single')
  AND NOT EXISTS (SELECT 1 FROM rooms WHERE hostel_id = 2 AND room_number = '201');

UPDATE hostels AS h
SET single_room_price = (SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id = h.id AND lower(room_type) = 'single'),
    double_room_price = (SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id = h.id AND lower(room_type) = 'double'),
    dorm_room_price = (SELECT MIN(monthly_rent) FROM rooms WHERE hostel_id = h.id AND lower(room_type) = 'shared'),
    room_types_available = (SELECT COALESCE(jsonb_agg(DISTINCT lower(room_type)), '[]'::jsonb)
                            FROM rooms WHERE hostel_id = h.id),
    total_rooms = (SELECT COUNT(*) FROM rooms WHERE hostel_id = h.id)
WHERE h.id IN (1, 2);

COMMIT;
