BEGIN;

INSERT INTO chatbot_knowledge_documents
  (slug,title,category,audience,source_kind,source_reference)
VALUES
  ('posted-gate-qr-attendance','Printed hostel gate QR attendance','attendance',ARRAY['all']::text[],
    'help_article','docs/QR_ATTENDANCE_WORKFLOW.md')
ON CONFLICT (slug) DO UPDATE SET reviewed_at=NOW(),updated_at=NOW(),active=true;

INSERT INTO chatbot_knowledge_chunks(document_id,chunk_index,content)
SELECT d.id,v.chunk_index,v.content FROM chatbot_knowledge_documents d
JOIN (VALUES
  (0,'A hostel warden can print a hostel-specific gate QR. A student with a confirmed booking and active checked-in stay scans it on their own phone, signs in and requests IN or OUT. This request is not yet attendance. An assigned warden must verify the student physically at the gate and approve within five minutes before the backend records the server-timestamped event. A photo of a permanent QR or phone GPS does not prove gate presence. The older student-generated one-time QR scanner integration also remains available.'),
  (1,'Each hostel can configure earliest exit, return curfew, grace and escalation minutes. An early exit, late return after grace, late-night exit, or student still recorded outside at curfew creates an attendance exception and relevant alerts. A third consecutive late-return day creates a warning; a further consecutive day creates a fine only when the hostel configured a nonzero PKR amount. Wardens can waive an outstanding fine with a reason. Attendance fines are tracked separately from rent invoices. A gate event does not change accommodation check-in or check-out.'),
  (2,'Linked parents can view attendance history. Automatic parent alerts additionally require the student to allow notifications for that parent link, the parent attendance preference and account notification setting. Warden corrections preserve the original event and are audited. Phone coordinates are optional single-scan evidence, not verified continuous location. Push, SMS and email delivery depend on configured notification providers.')
) AS v(chunk_index,content) ON true
WHERE d.slug='posted-gate-qr-attendance'
ON CONFLICT(document_id,chunk_index) DO UPDATE SET content=EXCLUDED.content;

COMMIT;
