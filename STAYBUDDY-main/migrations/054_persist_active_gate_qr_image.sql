-- Lets an authorized warden re-open and download the current printable QR.
-- Replaced/revoked gate rows are never returned with this image by the API.
ALTER TABLE hostel_gate_qr_codes ADD COLUMN IF NOT EXISTS qr_image TEXT;
