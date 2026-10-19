BEGIN;

ALTER TABLE notification_preferences
  ADD COLUMN IF NOT EXISTS availability_updates BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE favorites
  ADD COLUMN IF NOT EXISTS note VARCHAR(500),
  ADD COLUMN IF NOT EXISTS price_snapshot NUMERIC(12,2),
  ADD COLUMN IF NOT EXISTS availability_snapshot INTEGER,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE TABLE IF NOT EXISTS favorite_lists (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR(120) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id,name)
);

CREATE TABLE IF NOT EXISTS favorite_list_items (
  list_id BIGINT NOT NULL REFERENCES favorite_lists(id) ON DELETE CASCADE,
  favorite_id INTEGER NOT NULL REFERENCES favorites(id) ON DELETE CASCADE,
  added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (list_id,favorite_id)
);

CREATE TABLE IF NOT EXISTS favorite_share_links (
  id UUID PRIMARY KEY,
  list_id BIGINT NOT NULL REFERENCES favorite_lists(id) ON DELETE CASCADE,
  token_hash CHAR(64) NOT NULL UNIQUE,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS favorite_share_links_list_idx ON favorite_share_links (list_id,created_at DESC);

CREATE TABLE IF NOT EXISTS favorite_share_comments (
  id BIGSERIAL PRIMARY KEY,
  share_id UUID NOT NULL REFERENCES favorite_share_links(id) ON DELETE CASCADE,
  favorite_id INTEGER NOT NULL REFERENCES favorites(id) ON DELETE CASCADE,
  author_name VARCHAR(120) NOT NULL,
  comment VARCHAR(1000) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hostel_availability_alerts (
  id BIGSERIAL PRIMARY KEY,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  university VARCHAR(180),
  city VARCHAR(120),
  hostel_type VARCHAR(60),
  max_price NUMERIC(12,2) CHECK (max_price IS NULL OR max_price >= 0),
  radius_km NUMERIC(6,2) CHECK (radius_km IS NULL OR radius_km > 0),
  latitude NUMERIC(10,8) CHECK (latitude IS NULL OR latitude BETWEEN -90 AND 90),
  longitude NUMERIC(11,8) CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180),
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  disabled_at TIMESTAMPTZ,
  CHECK (university IS NOT NULL OR city IS NOT NULL OR (latitude IS NOT NULL AND longitude IS NOT NULL)),
  CHECK ((latitude IS NULL AND longitude IS NULL AND radius_km IS NULL) OR (latitude IS NOT NULL AND longitude IS NOT NULL AND radius_km IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS hostel_availability_alerts_active_idx ON hostel_availability_alerts (active,student_user_id);

CREATE TABLE IF NOT EXISTS hostel_availability_alert_matches (
  alert_id BIGINT NOT NULL REFERENCES hostel_availability_alerts(id) ON DELETE CASCADE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  notified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (alert_id,hostel_id)
);

CREATE TABLE IF NOT EXISTS hostel_nearby_places (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  category VARCHAR(30) NOT NULL CHECK (category IN ('transport','market','hospital','restaurant','education','other')),
  name VARCHAR(180) NOT NULL,
  latitude NUMERIC(10,8) NOT NULL CHECK (latitude BETWEEN -90 AND 90),
  longitude NUMERIC(11,8) NOT NULL CHECK (longitude BETWEEN -180 AND 180),
  distance_m INTEGER CHECK (distance_m IS NULL OR distance_m >= 0),
  source VARCHAR(30) NOT NULL DEFAULT 'hostel_staff' CHECK (source IN ('hostel_staff','catalog_import')),
  active BOOLEAN NOT NULL DEFAULT true,
  created_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS hostel_nearby_places_hostel_category_idx ON hostel_nearby_places (hostel_id,category) WHERE active;

CREATE TABLE IF NOT EXISTS hostel_search_history (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  source VARCHAR(20) NOT NULL CHECK (source IN ('text','voice')),
  query_text VARCHAR(1000),
  filters JSONB NOT NULL DEFAULT '{}'::jsonb,
  result_count INTEGER CHECK (result_count IS NULL OR result_count >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS hostel_search_history_user_time_idx ON hostel_search_history (user_id,created_at DESC);

ALTER TABLE operational_jobs DROP CONSTRAINT IF EXISTS operational_jobs_job_type_check;
ALTER TABLE operational_jobs ADD CONSTRAINT operational_jobs_job_type_check CHECK (job_type IN (
  'emergency_delivery','fee_reminder_delivery','condition_deadline_check','invoice_overdue_check',
  'fee_invoice_generation','fee_reminder_check','complaint_sla_check','complaint_volume_check',
  'location_retention_check','booking_expiry_check','complaint_auto_close','attendance_curfew_check',
  'availability_alert_check'
));

COMMIT;
