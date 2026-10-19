BEGIN;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM warden_assignments
    GROUP BY warden_user_id HAVING COUNT(*) > 1
  ) THEN
    RAISE EXCEPTION 'Resolve wardens assigned to multiple hostels before applying migration 046';
  END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS warden_assignments_one_hostel_per_warden_idx
  ON warden_assignments (warden_user_id);

COMMIT;
