BEGIN;

-- Old scans cannot be approved after the switch to immediate recording.
UPDATE attendance_gate_requests SET status='expired' WHERE status='pending';
ALTER TABLE attendance_gate_requests DROP CONSTRAINT IF EXISTS attendance_gate_requests_status_check;
ALTER TABLE attendance_gate_requests ADD CONSTRAINT attendance_gate_requests_status_check
  CHECK (status IN ('pending','approved','rejected','expired','recorded'));

CREATE TABLE attendance_discipline_appeals (
  id BIGSERIAL PRIMARY KEY,
  action_id BIGINT NOT NULL REFERENCES attendance_disciplinary_actions(id),
  student_user_id INTEGER NOT NULL REFERENCES users(id),
  hostel_id INTEGER NOT NULL REFERENCES hostels(id),
  reason TEXT NOT NULL CHECK (length(trim(reason)) BETWEEN 10 AND 1000),
  status VARCHAR(10) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
  requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  reviewed_by_user_id INTEGER REFERENCES users(id),
  reviewed_at TIMESTAMPTZ,
  review_reason TEXT
);
CREATE UNIQUE INDEX attendance_discipline_one_pending_appeal_idx
  ON attendance_discipline_appeals(action_id) WHERE status='pending';
CREATE INDEX attendance_discipline_appeals_hostel_idx
  ON attendance_discipline_appeals(hostel_id,status,requested_at DESC);

-- The previous guide described a review queue; update the assistant's active source.
UPDATE chatbot_knowledge_chunks SET content = CASE chunk_index
  WHEN 0 THEN 'A hostel warden can print a hostel-specific gate QR. A student with a confirmed booking and active checked-in stay scans it on their own phone, signs in and selects IN or OUT. A valid scan records attendance immediately at server time; the warden does not approve routine scans. A permanent QR can be photographed or shared, and optional phone GPS is unverified, so the code alone cannot prove physical presence. The older student-generated one-time QR scanner integration remains available.'
  WHEN 1 THEN 'Each hostel can configure earliest exit, return curfew, grace and escalation minutes. An early exit, late return after grace, late-night exit or student still recorded outside at curfew creates an exception and alerts. The third consecutive late-return day creates a warning; a further consecutive day creates a fine only when a nonzero PKR amount is configured. A student with an issued warning or fine can request its removal; the assigned warden approves or rejects the appeal with a reason. Attendance fines are separate from rent invoices. A gate event does not change accommodation check-in or check-out.'
  ELSE content END
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='posted-gate-qr-attendance') AND chunk_index IN (0,1);
UPDATE chatbot_knowledge_chunks SET content = 'A student scanning the assigned hostel posted gate QR records IN or OUT attendance immediately after sign-in and resident checks. No warden approval is needed for a routine scan. The warden may review scan history, attendance and pending appeals against issued warnings or fines; approving an appeal waives the action. The chatbot may summarize active checked-in residents as inside, outside or unknown from recorded events. Phone GPS alone does not prove presence. Curfew, grace, warning and fine rules come from the assigned hostel settings.'
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='warden-chat-attendance') AND chunk_index=0;

COMMIT;
