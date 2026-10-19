BEGIN;

CREATE TABLE IF NOT EXISTS chatbot_knowledge_documents (
  id BIGSERIAL PRIMARY KEY,
  slug VARCHAR(120) NOT NULL UNIQUE,
  title VARCHAR(240) NOT NULL,
  category VARCHAR(80) NOT NULL,
  audience TEXT[] NOT NULL DEFAULT ARRAY['all']::text[],
  source_kind VARCHAR(30) NOT NULL DEFAULT 'platform_policy'
    CHECK (source_kind IN ('platform_policy','hostel_policy','help_article','faq')),
  source_reference VARCHAR(500),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
  active BOOLEAN NOT NULL DEFAULT true,
  reviewed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  created_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (audience <@ ARRAY['all','student','parent']::text[])
);

CREATE TABLE IF NOT EXISTS chatbot_knowledge_chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id BIGINT NOT NULL REFERENCES chatbot_knowledge_documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
  content TEXT NOT NULL CHECK (char_length(trim(content)) BETWEEN 20 AND 8000),
  search_vector TSVECTOR GENERATED ALWAYS AS
    (to_tsvector('simple', coalesce(content,''))) STORED,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (document_id,chunk_index)
);
CREATE INDEX IF NOT EXISTS chatbot_knowledge_chunks_search_idx
  ON chatbot_knowledge_chunks USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS chatbot_knowledge_documents_scope_idx
  ON chatbot_knowledge_documents (active,category,reviewed_at DESC);

CREATE TABLE IF NOT EXISTS chatbot_retrieval_events (
  id BIGSERIAL PRIMARY KEY,
  session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
  user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  query_text TEXT NOT NULL,
  detected_intent VARCHAR(80),
  language VARCHAR(20),
  retrieved_chunk_ids BIGINT[] NOT NULL DEFAULT '{}'::bigint[],
  live_context_types TEXT[] NOT NULL DEFAULT '{}'::text[],
  provider VARCHAR(40) NOT NULL,
  grounded BOOLEAN NOT NULL DEFAULT true,
  confidence NUMERIC(5,4),
  escalation_recommended BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chatbot_retrieval_events_user_time_idx
  ON chatbot_retrieval_events (user_id,created_at DESC);

CREATE TABLE IF NOT EXISTS chatbot_message_feedback (
  assistant_message_id BIGINT PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  helpful BOOLEAN NOT NULL,
  reason VARCHAR(1000),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chatbot_support_escalations (
  id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  reason VARCHAR(1000) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'open'
    CHECK (status IN ('open','in_progress','resolved','closed')),
  conversation_snapshot JSONB NOT NULL,
  assigned_to_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  resolution_note TEXT,
  resolved_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chatbot_support_escalations_status_idx
  ON chatbot_support_escalations (status,created_at DESC);

INSERT INTO chatbot_knowledge_documents (slug,title,category,audience,source_kind,source_reference)
VALUES
  ('booking-process','Booking process','booking',ARRAY['all'],'help_article','HMS Core UC-STU-013 / UC-PAR-005'),
  ('booking-documents','Booking documents','booking',ARRAY['all'],'faq','StayBuddy booking document workflow'),
  ('fees-and-receipts','Fees and receipt verification','fees',ARRAY['all'],'platform_policy','StayBuddy fee workflow'),
  ('booking-cancellation','Booking cancellation','booking',ARRAY['all'],'platform_policy','StayBuddy booking lifecycle'),
  ('complaints','Complaint support','complaints',ARRAY['all'],'help_article','StayBuddy complaint workflow'),
  ('attendance','Attendance and QR tracking','attendance',ARRAY['all'],'help_article','HMS Core W-2 / P-2'),
  ('parent-access','Parent access and consent','parents',ARRAY['parent'],'platform_policy','StayBuddy parent-link policy'),
  ('human-support','Human support escalation','support',ARRAY['all'],'help_article','HMS Core chatbot escalation flow')
ON CONFLICT (slug) DO UPDATE SET
  title=EXCLUDED.title,category=EXCLUDED.category,audience=EXCLUDED.audience,
  source_kind=EXCLUDED.source_kind,source_reference=EXCLUDED.source_reference,
  active=true,reviewed_at=NOW(),updated_at=NOW();

INSERT INTO chatbot_knowledge_chunks (document_id,chunk_index,content)
SELECT d.id,v.chunk_index,v.content
FROM chatbot_knowledge_documents d
JOIN (VALUES
  ('booking-process',0,'To book a hostel, search published hostels, review rooms and availability, choose dates and an available room, submit the booking request, upload every document required by that hostel, and wait for authorized hostel staff to confirm it. A pending booking holds inventory for a limited time.'),
  ('booking-documents',0,'Each hostel can define required booking documents. Students upload PDF, JPEG, or PNG files privately against their own booking. Hostel staff review each document as accepted or rejected. A booking cannot be confirmed until every active required document has an accepted submission.'),
  ('fees-and-receipts',0,'StayBuddy records fees in PKR. Payments happen outside StayBuddy using the hostel approved payment destination. A student or linked parent uploads a PDF, JPEG, or PNG receipt. Only an assigned warden can verify the receipt, and only verified allocations reduce the invoice balance. StayBuddy does not charge cards or process online payments.'),
  ('booking-cancellation',0,'A student can cancel an eligible own booking through the booking cancellation endpoint. Capacity is released exactly once and the status history is audited. Refund amounts are not calculated by StayBuddy; consult the hostel terms and staff because this backend uses external payments and receipt verification.'),
  ('complaints',0,'A checked-in student can submit a complaint for the current hostel. The system classifies it, warns about similar active complaints in the same room, tracks status and SLA, accepts private evidence, and supports one reopen after resolution. Students can view only their own complaint details and safe resolution records.'),
  ('attendance',0,'Hostel attendance records verified IN and OUT events from biometric, RFID, QR, or authorized manual entry. Linked parents can view the timeline and receive enabled alerts. QR codes are short-lived, one-time tokens and must be scanned by an authorized hostel gate integration. Wardens can correct events with a reason; corrections remain audited.'),
  ('parent-access',0,'A parent can view a child only through an active parent-student link accepted through the invitation flow. Revoking the link immediately removes future access. Exact location requires a separate active student-controlled consent session. Attendance, fee, transport, announcement, and communication access remains relationship scoped.'),
  ('human-support',0,'If the assistant cannot answer from current knowledge or authorized account data, it should say so and offer human support. Escalation saves the conversation context for an administrator or support operator. Never invent policy, payment, booking, attendance, or complaint facts.')
) AS v(slug,chunk_index,content) ON v.slug=d.slug
ON CONFLICT (document_id,chunk_index) DO UPDATE SET content=EXCLUDED.content;

ALTER TABLE attendance_events DROP CONSTRAINT IF EXISTS attendance_events_source_check;
ALTER TABLE attendance_events ADD CONSTRAINT attendance_events_source_check
  CHECK (source IN ('biometric','rfid','qr','manual'));

ALTER TABLE attendance_exceptions DROP CONSTRAINT IF EXISTS attendance_exceptions_exception_type_check;
ALTER TABLE attendance_exceptions ADD CONSTRAINT attendance_exceptions_exception_type_check
  CHECK (exception_type IN ('late_exit','late_return','absent_at_curfew','unusual_pattern'));

CREATE TABLE IF NOT EXISTS attendance_qr_tokens (
  id UUID PRIMARY KEY,
  token_hash CHAR(64) NOT NULL UNIQUE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at TIMESTAMPTZ NOT NULL,
  used_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (expires_at > created_at)
);
CREATE INDEX IF NOT EXISTS attendance_qr_tokens_active_idx
  ON attendance_qr_tokens (student_user_id,hostel_id,expires_at)
  WHERE used_at IS NULL AND revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS attendance_qr_scan_attempts (
  id BIGSERIAL PRIMARY KEY,
  token_id UUID REFERENCES attendance_qr_tokens(id) ON DELETE SET NULL,
  hostel_id INTEGER REFERENCES hostels(id) ON DELETE SET NULL,
  integration_credential_id INTEGER REFERENCES attendance_integration_credentials(id) ON DELETE SET NULL,
  direction VARCHAR(3) CHECK (direction IN ('in','out')),
  entry_point VARCHAR(120),
  outcome VARCHAR(30) NOT NULL CHECK (outcome IN ('accepted','invalid_token','expired','already_used','wrong_hostel','inactive_stay','duplicate_direction')),
  reason TEXT,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  remote_address INET
);
CREATE INDEX IF NOT EXISTS attendance_qr_scan_attempts_hostel_time_idx
  ON attendance_qr_scan_attempts (hostel_id,occurred_at DESC);

CREATE TABLE IF NOT EXISTS parent_attendance_preferences (
  parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  notify_entry BOOLEAN NOT NULL DEFAULT true,
  notify_exit BOOLEAN NOT NULL DEFAULT true,
  notify_late BOOLEAN NOT NULL DEFAULT true,
  notify_curfew BOOLEAN NOT NULL DEFAULT true,
  notify_unusual BOOLEAN NOT NULL DEFAULT true,
  daily_summary BOOLEAN NOT NULL DEFAULT false,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (parent_user_id,student_user_id)
);

CREATE TABLE IF NOT EXISTS attendance_return_watches (
  id BIGSERIAL PRIMARY KEY,
  parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  fulfilled_by_event_id BIGINT REFERENCES attendance_events(id) ON DELETE SET NULL,
  fulfilled_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS attendance_return_watches_one_active_idx
  ON attendance_return_watches (parent_user_id,student_user_id,hostel_id) WHERE active;

CREATE TABLE IF NOT EXISTS attendance_exception_acknowledgements (
  exception_id BIGINT NOT NULL REFERENCES attendance_exceptions(id) ON DELETE CASCADE,
  parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  note VARCHAR(1000),
  acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (exception_id,parent_user_id)
);

CREATE TABLE IF NOT EXISTS attendance_audit_events (
  id BIGSERIAL PRIMARY KEY,
  hostel_id INTEGER NOT NULL REFERENCES hostels(id) ON DELETE CASCADE,
  student_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  attendance_event_id BIGINT REFERENCES attendance_events(id) ON DELETE SET NULL,
  event_type VARCHAR(40) NOT NULL,
  actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS attendance_audit_events_hostel_time_idx
  ON attendance_audit_events (hostel_id,created_at DESC);

COMMIT;
