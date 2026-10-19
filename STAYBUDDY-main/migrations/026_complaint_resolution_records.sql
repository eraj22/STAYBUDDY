BEGIN;

CREATE TABLE IF NOT EXISTS complaint_resolution_records (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE,
  final_status VARCHAR(20) NOT NULL CHECK (final_status IN ('resolved','closed')),
  safe_resolution_summary TEXT NOT NULL CHECK (char_length(trim(safe_resolution_summary)) BETWEEN 1 AND 4000),
  actions_taken TEXT NOT NULL CHECK (char_length(trim(actions_taken)) BETWEEN 1 AND 4000),
  prevention_follow_up_notes TEXT CHECK (prevention_follow_up_notes IS NULL OR char_length(prevention_follow_up_notes) <= 4000),
  resolver_user_id INTEGER NOT NULL REFERENCES users(id),
  resolved_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS complaint_resolution_records_resolver_idx
  ON complaint_resolution_records (resolver_user_id, resolved_at DESC);

COMMIT;
