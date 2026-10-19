BEGIN;

CREATE TABLE IF NOT EXISTS warden_cash_fee_payments (
  id BIGSERIAL PRIMARY KEY,
  invoice_id BIGINT NOT NULL REFERENCES fee_invoices(id) ON DELETE RESTRICT,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
  received_on DATE NOT NULL,
  note VARCHAR(1000),
  received_by_warden_id INTEGER NOT NULL REFERENCES users(id),
  voided_at TIMESTAMPTZ,
  voided_by_user_id INTEGER REFERENCES users(id),
  void_reason VARCHAR(1000),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK ((voided_at IS NULL AND voided_by_user_id IS NULL AND void_reason IS NULL)
      OR (voided_at IS NOT NULL AND voided_by_user_id IS NOT NULL AND void_reason IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS warden_cash_fee_payments_invoice_idx
  ON warden_cash_fee_payments (invoice_id, received_on DESC)
  WHERE voided_at IS NULL;

CREATE INDEX IF NOT EXISTS warden_cash_fee_payments_hostel_student_idx
  ON warden_cash_fee_payments (hostel_id, student_user_id, received_on DESC)
  WHERE voided_at IS NULL;

COMMIT;
