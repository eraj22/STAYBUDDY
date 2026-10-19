BEGIN;

-- Warden profile data used by the owner/staff lifecycle belongs in the
-- migration-managed schema.  Older builds created this table from index.js,
-- which meant a clean Phase 3 database could start without it.
CREATE TABLE IF NOT EXISTS warden_profiles (
  warden_user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  cnic VARCHAR(20),
  city VARCHAR(120),
  shift VARCHAR(30) NOT NULL DEFAULT 'Morning'
    CHECK (shift IN ('Morning','Evening','Night')),
  assigned_floors JSONB NOT NULL DEFAULT '[]'::jsonb,
  permissions JSONB NOT NULL DEFAULT '{}'::jsonb,
  account_status VARCHAR(20) NOT NULL DEFAULT 'Active'
    CHECK (account_status IN ('Active','Disabled')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (jsonb_typeof(assigned_floors) = 'array'),
  CHECK (jsonb_typeof(permissions) = 'object')
);

CREATE INDEX IF NOT EXISTS warden_profiles_status_idx
  ON warden_profiles (account_status);

-- The historical schema dump predates the room-number uniqueness constraint
-- present in migration 001.  Because CREATE TABLE IF NOT EXISTS does not add
-- omitted constraints to an existing table, enforce it explicitly here.
CREATE UNIQUE INDEX IF NOT EXISTS rooms_hostel_room_number_unique_idx
  ON rooms (hostel_id,room_number);

COMMIT;
