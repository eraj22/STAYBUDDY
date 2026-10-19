BEGIN;

UPDATE chatbot_knowledge_chunks
SET content = 'A guest can ask public questions and search published accommodation through the public chatbot. Saved AI conversations are available to students, parents, wardens and owners. Account-specific answers use the caller''s current authorized records: students see their own records, parents see active linked-child records, wardens see only their assigned hostel and owners see only properties they own. The assistant can offer a small set of permitted actions. Each action needs an explicit confirmation in the client and its normal backend API repeats the role and resource checks before making a change. Administrators have separate chat-support and knowledge-management tools.'
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='manual-kb001')
  AND chunk_index=2;

UPDATE chatbot_knowledge_documents
SET reviewed_at=NOW(),updated_at=NOW()
WHERE slug='manual-kb001';

COMMIT;
