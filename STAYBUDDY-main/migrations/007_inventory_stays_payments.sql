BEGIN;

CREATE TABLE IF NOT EXISTS room_assignment_history (
  id SERIAL PRIMARY KEY,
  booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  previous_room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL,
  next_room_id INTEGER NOT NULL REFERENCES rooms(id),
  assigned_by_user_id INTEGER NOT NULL REFERENCES users(id),
  assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS booking_stays (
  booking_id INTEGER PRIMARY KEY REFERENCES bookings(id) ON DELETE CASCADE,
  checked_in_at TIMESTAMPTZ,
  checked_in_by_user_id INTEGER REFERENCES users(id),
  checked_out_at TIMESTAMPTZ,
  checked_out_by_user_id INTEGER REFERENCES users(id),
  CHECK (checked_out_at IS NULL OR checked_in_at IS NOT NULL),
  CHECK (checked_out_at IS NULL OR checked_out_at >= checked_in_at)
);

CREATE TABLE IF NOT EXISTS payments (
  id SERIAL PRIMARY KEY,
  booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id),
  provider VARCHAR(30) NOT NULL,
  provider_payment_id VARCHAR(255) UNIQUE,
  amount NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
  currency CHAR(3) NOT NULL DEFAULT 'PKR',
  status VARCHAR(20) NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'succeeded', 'failed', 'cancelled', 'refunded')),
  receipt_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS room_assignment_history_booking_idx
  ON room_assignment_history (booking_id, assigned_at DESC);
CREATE INDEX IF NOT EXISTS payments_user_created_idx
  ON payments (user_id, created_at DESC);

COMMIT;