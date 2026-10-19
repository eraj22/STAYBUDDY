BEGIN;

ALTER TABLE notification_preferences
  ADD COLUMN IF NOT EXISTS fee_reminders BOOLEAN NOT NULL DEFAULT true;

CREATE TABLE IF NOT EXISTS hostel_fee_schedules (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  title VARCHAR(120) NOT NULL,
  amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
  currency CHAR(3) NOT NULL DEFAULT 'PKR',
  due_day SMALLINT NOT NULL CHECK (due_day BETWEEN 1 AND 28),
  active BOOLEAN NOT NULL DEFAULT true,
  created_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS student_fee_assignments (
  id BIGSERIAL PRIMARY KEY,
  schedule_id BIGINT NOT NULL REFERENCES hostel_fee_schedules(id) ON DELETE CASCADE,
  booking_id INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  amount_override NUMERIC(12,2) CHECK (amount_override IS NULL OR amount_override > 0),
  starts_on DATE NOT NULL, ends_on DATE,
  active BOOLEAN NOT NULL DEFAULT true,
  assigned_by_user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (ends_on IS NULL OR ends_on >= starts_on),
  UNIQUE (schedule_id, booking_id)
);

CREATE TABLE IF NOT EXISTS fee_invoices (
  id BIGSERIAL PRIMARY KEY,
  assignment_id BIGINT NOT NULL REFERENCES student_fee_assignments(id) ON DELETE CASCADE,
  billing_month DATE NOT NULL,
  due_date DATE NOT NULL,
  amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
  currency CHAR(3) NOT NULL DEFAULT 'PKR',
  status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','paid','overdue','cancelled')),
  paid_payment_id INTEGER REFERENCES payments(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (assignment_id, billing_month)
);

CREATE TABLE IF NOT EXISTS fee_reminder_deliveries (
  id BIGSERIAL PRIMARY KEY,
  invoice_id BIGINT NOT NULL REFERENCES fee_invoices(id) ON DELETE CASCADE,
  recipient_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  channel VARCHAR(20) NOT NULL DEFAULT 'in_app' CHECK (channel IN ('in_app','email','sms')),
  status VARCHAR(20) NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','delivered','failed')),
  scheduled_for TIMESTAMPTZ NOT NULL,
  delivered_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS hostel_fee_schedules_hostel_idx ON hostel_fee_schedules (hostel_id, active);
CREATE INDEX IF NOT EXISTS fee_invoices_status_due_idx ON fee_invoices (status, due_date);
CREATE INDEX IF NOT EXISTS fee_reminders_delivery_idx ON fee_reminder_deliveries (status, scheduled_for);

COMMIT;
