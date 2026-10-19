BEGIN;

ALTER TABLE chatbot_knowledge_documents DROP CONSTRAINT chatbot_knowledge_documents_audience_check;
ALTER TABLE chatbot_knowledge_documents ADD CONSTRAINT chatbot_knowledge_documents_audience_check
  CHECK (audience <@ ARRAY['all','student','parent','warden']::text[]);

INSERT INTO chatbot_knowledge_documents (slug,title,category,audience,source_kind,source_reference)
VALUES
  ('warden-chat-assignment','Warden hostel assignment and chatbot access','app',ARRAY['warden']::text[],'help_article','docs/rag/WARDEN_CHATBOT.md#assignment-and-records'),
  ('warden-chat-attendance','Warden gate attendance workflow','attendance',ARRAY['warden']::text[],'help_article','docs/rag/WARDEN_CHATBOT.md#attendance'),
  ('warden-chat-bookings','Warden booking and room workflow','booking',ARRAY['warden']::text[],'help_article','docs/rag/WARDEN_CHATBOT.md#bookings-and-rooms'),
  ('warden-chat-complaints','Warden complaint workflow','complaints',ARRAY['warden']::text[],'help_article','docs/rag/WARDEN_CHATBOT.md#complaints'),
  ('warden-chat-fees','Warden fee receipt review','fees',ARRAY['warden']::text[],'help_article','docs/rag/WARDEN_CHATBOT.md#fees-and-receipts')
ON CONFLICT (slug) DO NOTHING;

INSERT INTO chatbot_knowledge_chunks (document_id,chunk_index,content)
SELECT d.id,0,v.content FROM chatbot_knowledge_documents d JOIN (VALUES
  ('warden-chat-assignment','A warden has one assigned hostel. Chat may show its name, city, rules and attendance settings from the current authorized assignment. A different hostel ID in a question does not grant access. Chat can explain and summarize but cannot change operational records by itself.'),
  ('warden-chat-attendance','A student scanning the assigned hostel posted gate QR creates a pending request, not attendance. The warden verifies physical presence and approves within the request lifetime before an IN or OUT event is recorded. The chatbot may summarize active checked-in residents as inside, outside or unknown from gate records and count pending scans. Phone GPS alone does not prove presence. Curfew, grace, warning and fine rules come from that hostel settings.'),
  ('warden-chat-bookings','A warden may review recent bookings for the assigned hostel, including status, student name, dates and room number when assigned. A pending booking is not a confirmed stay. Required documents must be accepted before staff confirmation. Room assignment, check-in and check-out require separate authorized operations; chat alone does not change a booking.'),
  ('warden-chat-complaints','A warden may review recent complaints for the assigned hostel, including category, severity, priority and status. Triage, investigation, escalation and resolution are separate recorded steps. The chatbot does not expose private evidence files or change complaint status. Related complaints do not automatically close one another.'),
  ('warden-chat-fees','A warden may review recent receipt submissions for the assigned hostel. A submitted receipt is not verified payment. Only authorized verification and allocation change the invoice balance. The chatbot cannot approve a receipt through conversation and must not expose receipt files or payment credentials.')
) AS v(slug,content) ON v.slug=d.slug
ON CONFLICT (document_id,chunk_index) DO NOTHING;

COMMIT;
