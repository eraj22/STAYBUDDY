BEGIN;

UPDATE chatbot_knowledge_chunks SET content = 'Each hostel can configure earliest exit, return curfew, grace and escalation minutes. An early exit, late return after grace, late-night exit or student still recorded outside at curfew creates an exception and alerts. Each consecutive late-return day creates a warning up to the configured warning-day limit, three by default; the next consecutive late day creates a fine only when a nonzero PKR amount is configured. A student may appeal a fine as soon as it is issued, or appeal warnings after receiving the configured number of warnings. The assigned warden approves or rejects each appeal with a reason. Approval waives the action but preserves the attendance event. Attendance fines are separate from rent invoices. A gate event does not change accommodation check-in or check-out.'
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='posted-gate-qr-attendance') AND chunk_index=1;

COMMIT;
