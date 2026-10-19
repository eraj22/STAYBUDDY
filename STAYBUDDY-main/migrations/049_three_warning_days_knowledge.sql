BEGIN;

UPDATE chatbot_knowledge_chunks SET content = 'Each hostel can configure earliest exit, return curfew, grace and escalation minutes. An early exit, late return after grace, late-night exit or student still recorded outside at curfew creates an exception and alerts. Each consecutive late-return day creates a warning up to the configured warning-day limit, three by default; the next consecutive late day creates a fine only when a nonzero PKR amount is configured. A student with an issued warning or fine can request its removal; the assigned warden approves or rejects the appeal with a reason. Attendance fines are separate from rent invoices. A gate event does not change accommodation check-in or check-out.'
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='posted-gate-qr-attendance') AND chunk_index=1;

COMMIT;
