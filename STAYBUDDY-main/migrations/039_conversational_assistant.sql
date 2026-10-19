BEGIN;
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS request_id VARCHAR(120);
CREATE UNIQUE INDEX IF NOT EXISTS chat_messages_retry_idx ON chat_messages(session_id,request_id,role) WHERE request_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS chatbot_discrepancies (
  id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  message_id BIGINT REFERENCES chat_messages(id) ON DELETE SET NULL,
  description TEXT NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK(status IN ('open','resolved')),
  resolution_note TEXT,
  reviewed_by_user_id INTEGER REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), resolved_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS chatbot_knowledge_revisions (
  id BIGSERIAL PRIMARY KEY,
  document_id BIGINT NOT NULL REFERENCES chatbot_knowledge_documents(id),
  version INTEGER NOT NULL,
  snapshot JSONB NOT NULL,
  actor_user_id INTEGER REFERENCES users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS chatbot_complaint_comments (
  id BIGSERIAL PRIMARY KEY,
  complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id),
  content TEXT NOT NULL CHECK(char_length(content) BETWEEN 1 AND 2000),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO chatbot_knowledge_documents(slug,title,category,audience,source_kind,source_reference)
VALUES
 ('app-reminders','Fee reminders and notification settings','app',ARRAY['all'],'help_article','StayBuddy notification preferences and fee reminder endpoints'),
 ('app-location','Location sharing and privacy','app',ARRAY['all'],'help_article','StayBuddy consented location sharing'),
 ('hostel-deposits','Deposits and hostel rules','policies',ARRAY['all'],'faq','Hostel-specific published terms'),
 ('app-support','Account help and support','app',ARRAY['all'],'help_article','StayBuddy account and chat support endpoints')
ON CONFLICT(slug) DO NOTHING;
INSERT INTO chatbot_knowledge_chunks(document_id,chunk_index,content)
SELECT d.id,0,v.content FROM chatbot_knowledge_documents d JOIN (VALUES
 ('app-reminders','Fee reminders depend on scheduled invoices and notification preferences. Review the current invoice due date and your notification settings. Actual email, SMS or push delivery depends on configured delivery providers; an in-app record does not prove external delivery. Parents can review fee records only for actively linked children.'),
 ('app-location','Location sharing is controlled by the student and requires active consent. A linked parent may view only the authorized live sharing session. Attendance is a recorded gate entry or exit, not proof of the student current GPS location. Revoking location consent stops subsequent location access.'),
 ('hostel-deposits','A security deposit generally covers obligations specified by a hostel. The exact amount, deductions, notice period and refund timing depend on that hostel terms. StayBuddy does not have a universal deposit refund percentage or timeline. Check the published hostel rules and consult hostel staff when the terms do not answer the question.'),
 ('app-support','For account problems, describe the error without sharing passwords, authentication tokens or payment credentials. The app supports authentication and password reset. In a saved support conversation you can request human support, report an inaccurate answer, and read the support agent replies in the same history. Voice questions are processed from their text transcript; the chatbot does not record audio.')
) AS v(slug,content) ON v.slug=d.slug ON CONFLICT(document_id,chunk_index) DO NOTHING;
COMMIT;
