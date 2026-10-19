BEGIN;

CREATE TABLE IF NOT EXISTS complaint_related_links (
  id BIGSERIAL PRIMARY KEY,
  source_complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id) ON DELETE CASCADE,
  primary_complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  reason TEXT NOT NULL,
  idempotency_key VARCHAR(160) NOT NULL UNIQUE,
  linked_by_user_id INTEGER NOT NULL REFERENCES users(id),
  linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (source_complaint_id <> primary_complaint_id)
);

CREATE INDEX IF NOT EXISTS complaint_related_links_primary_idx ON complaint_related_links (primary_complaint_id, linked_at DESC);

COMMIT;
