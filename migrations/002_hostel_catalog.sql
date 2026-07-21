BEGIN;

ALTER TABLE hostels ADD COLUMN IF NOT EXISTS external_id VARCHAR(20);
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS hostel_type VARCHAR(20);
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS area VARCHAR(100);
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS verified BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS year_established INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS distance_from_fast_km NUMERIC(6, 2);
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS price_tier VARCHAR(20);
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS single_room_price INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS double_room_price INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS dorm_room_price INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS electricity_included BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS electricity_bill_est INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS meal_included BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS food_type VARCHAR(20);
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS food_rating NUMERIC(3, 1);
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS room_types_available JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS total_rooms INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS amenities JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS internet_speed_mbps INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS noise_level INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS curfew_hour INTEGER;
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS study_environment_score NUMERIC(3, 2);
ALTER TABLE hostels ADD COLUMN IF NOT EXISTS source VARCHAR(30) NOT NULL DEFAULT 'manual';

CREATE UNIQUE INDEX IF NOT EXISTS hostels_external_id_unique
  ON hostels (external_id)
  WHERE external_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS hostels_city_index ON hostels (city);
CREATE INDEX IF NOT EXISTS hostels_university_index ON hostels (university);
CREATE INDEX IF NOT EXISTS hostels_single_room_price_index ON hostels (single_room_price);
CREATE INDEX IF NOT EXISTS hostels_hostel_type_index ON hostels (hostel_type);
CREATE INDEX IF NOT EXISTS hostels_amenities_gin_index ON hostels USING GIN (amenities);

COMMIT;