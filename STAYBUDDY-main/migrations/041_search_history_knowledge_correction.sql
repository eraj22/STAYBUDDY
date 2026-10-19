BEGIN;
INSERT INTO chatbot_knowledge_documents (slug,title,category,audience,source_kind,source_reference)
VALUES ('search-history-and-favorites','Search history, saved searches, and favorites','app',ARRAY['all']::text[],'help_article','docs/rag/KNOWLEDGE_CORRECTIONS.md#search-history-and-favorites')
ON CONFLICT (slug) DO NOTHING;
INSERT INTO chatbot_knowledge_chunks (document_id,chunk_index,content)
SELECT d.id,0,'StayBuddy stores a student''s hostel search history with previous query text and filters. It separately supports favorite hostels and named favorite lists. A favorite or previous search does not hold a room or create a booking. The reviewed backend does not establish a separate named saved-search feature that automatically reruns stored filters. Explain search history and favorites without promising a saved-search button, page, alert, or automatic replay; exact frontend navigation is not verified.'
FROM chatbot_knowledge_documents d WHERE d.slug='search-history-and-favorites'
ON CONFLICT (document_id,chunk_index) DO NOTHING;
COMMIT;
