BEGIN;

INSERT INTO chatbot_knowledge_documents (slug,title,category,audience,source_kind,source_reference)
VALUES ('qr-attendance-permission-and-security','QR attendance permission and scanner security','attendance',ARRAY['all']::text[],'help_article','docs/rag/KNOWLEDGE_CORRECTIONS.md#qr-attendance-permission-and-security')
ON CONFLICT (slug) DO NOTHING;

INSERT INTO chatbot_knowledge_chunks (document_id,chunk_index,content)
SELECT d.id,v.chunk_index,v.content FROM chatbot_knowledge_documents d
JOIN (VALUES
  (0,'An active parent link permits a parent to view the child''s recorded attendance history. Automatic entry, exit, curfew, correction and return alerts additionally require the student to allow attendance notifications for that specific active parent link. Parent alert preferences and the account''s attendance notification setting can further limit delivery. Revoking student permission stops future automatic alerts and cancels active return watches.'),
  (1,'A checked-in student can generate a one-time QR value for the hostel gate. An authorized scanner must submit it within its short lifetime. A repeated scan or a scan for a different direction, gate or token with the same idempotency key is rejected. The verified server receive time is the attendance event time; a scanner-reported clock must be close to server time. The hostel owner can revoke a scanner credential if a device or key is lost. Students can review their own corrected attendance timeline, while wardens and linked parents use their separately scoped views.')
) AS v(chunk_index,content) ON true
WHERE d.slug='qr-attendance-permission-and-security'
ON CONFLICT (document_id,chunk_index) DO NOTHING;

COMMIT;
