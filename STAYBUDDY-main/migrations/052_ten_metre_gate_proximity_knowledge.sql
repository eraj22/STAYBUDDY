BEGIN;

UPDATE chatbot_knowledge_chunks SET content = 'A hostel warden prints a hostel-specific gate QR. A student with a confirmed booking and active checked-in stay scans it on their own phone, signs in, allows fresh location access, and selects IN or OUT. The backend records attendance immediately only if the phone reports GPS accuracy of 10 metres or better and a measured position within 10 metres of the registered hostel gate coordinate. Missing hostel coordinates, denied or stale location, inaccurate GPS, and distant scans do not record attendance. Routine scans do not need warden approval. The older staff-operated one-time QR scanner remains a separate integration; phone GPS can be spoofed and cannot prove physical gate crossing.'
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='posted-gate-qr-attendance') AND chunk_index=0;

UPDATE chatbot_knowledge_chunks SET content = 'A student scanning the assigned hostel posted gate QR can record IN or OUT attendance immediately after sign-in, resident checks and a fresh phone-location check with reported accuracy of 10 metres or better and a measured position within 10 metres of the hostel gate coordinate. No warden approval is needed for a routine scan. Missing or inaccurate GPS, unconfigured hostel coordinates or a distant scan records no event. The warden can review scan history, attendance and student appeals against issued warnings or fines; approving an appeal waives the action. The chatbot may summarize active checked-in residents as inside, outside or unknown from recorded events. Phone GPS can be spoofed and does not prove physical gate crossing. Curfew and discipline rules come from the assigned hostel settings.'
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='warden-chat-attendance') AND chunk_index=0;

UPDATE chatbot_knowledge_chunks SET content = 'Attendance records gate IN and OUT events. The backend accepts a student phone scan of the printed hostel gate QR, a separate staff-operated one-time QR scanner, biometric, RFID and authorized manual events. For a printed gate QR, the signed-in resident must submit a fresh phone location with reported accuracy of 10 metres or better and a measured position within 10 metres of the registered hostel coordinate; a distant, inaccurate or unavailable location cannot create an event. These events do not establish a person''s continuous GPS position or prove gate crossing against GPS spoofing.'
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='manual-kb018') AND chunk_index=0;

UPDATE chatbot_knowledge_chunks SET content = 'Explain that attendance requires an authorized gate event or staff correction. For the printed hostel gate QR, a checked-in student scans with their phone at their hostel, signs in, allows a fresh location with reported accuracy of 10 metres or better and selects IN or OUT; a measured position beyond 10 metres, missing or inaccurate GPS, or unconfigured hostel coordinates records no event. The separate student-generated one-time QR can be presented to an authorized gate scanner. Chat text cannot stand in for a valid scan, and phone GPS can be spoofed.'
WHERE document_id=(SELECT id FROM chatbot_knowledge_documents WHERE slug='manual-kb035') AND chunk_index=11;

UPDATE chatbot_knowledge_documents SET reviewed_at=NOW(),updated_at=NOW()
WHERE slug IN ('posted-gate-qr-attendance','warden-chat-attendance','manual-kb018','manual-kb035');

COMMIT;
