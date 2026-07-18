# StayBuddy Backend AI Handoff

Last updated: 2026-07-18

This document describes the current backend state, the work already completed, operational rules that must be preserved, how to validate changes, and the recommended continuation order. It is intended to be handed directly to another AI agent or backend developer.

For the complete current workspace tree, file purposes, and active-versus-prototype boundaries, read `../DIRECTORY_ARCHITECTURE.md` alongside this document. For the ordered plan to complete the student journey, read `../STUDENT_USE_CASE_HANDOFF.md`.

## Scope And Repository Layout

The current backend root is `STAYBUDDY-main/`.

- `index.js`: Node.js/Express operational API. Uses PostgreSQL on port 5000 by default.
- `migrations/`: additive PostgreSQL migrations. Migrations `001`, `002`, and `003` have been applied to the active local database.
- `scripts/import_hostel_catalog.js`: repeatable importer from the ML hostel CSV to PostgreSQL.
- `ml-notebooks/app_api.py`: FastAPI hybrid recommendation service on port 8000.
- `ml-notebooks/ml_ai_complaint/app_api.py`: FastAPI complaint categorizer on port 8001.
- `ml-notebooks/data/hostels.csv`: current catalog source used for import and ML recommendations.
- `ml-notebooks/models/`: saved recommender artifacts.
- `test/catalog_importer.test.js`: database-free Node test coverage for the catalog importer.
- `recommender.js`: lightweight Node interaction-based recommendation helpers used by the older Node recommendation endpoint.

The Flutter application lives outside this backend folder. Current work has intentionally focused on backend behavior. Do not assume Flutter screens are connected to the latest Node endpoints.

## Implemented Backend Features

The Node API already includes operational endpoints for:

- Hostel discovery, details, comparison, location-based search, and filtering.
- Users, owners, OTP/login flows, authentication, and a basic warden dashboard.
- Bookings, booking status updates, cancellation, room assignment, and available-room lookup.
- Room CRUD and favorites.
- Reviews and complaints. Complaint creation can call the Python complaint categorizer.
- A lightweight Node recommendation endpoint at `POST /api/recommendations`.
- A catalog-backed ML recommendation endpoint at `POST /api/recommendations/ml`.
- A basic Node chat endpoint.

Recent role-scoped work adds student profile and notification preferences, owner-owned hostel registration/profile management, booking and complaint queues, occupancy summaries, explicit warden-to-hostel assignments, assignment-scoped warden booking/complaint operations, hostel announcements, in-app notification delivery, owner/warden-scoped room inventory, warden-only room assignment for confirmed bookings, check-in/check-out move workflows, and a server-verified Stripe payment intent/webhook flow. Public `/api/auth/register` creates **student** accounts only; owner, warden, and admin roles must be provisioned through a controlled administrative process. Do not restore client-selected privileged roles.

**Known client gap:** `lib/api.dart` still has an `Api().registerOwner(...)` helper that calls `POST /api/auth/register` with `role: 'owner'`. The backend now rejects any non-student role on that public route with `403`. Do not "fix" this by re-allowing client-selected roles; instead, route owner account creation through a controlled admin/staff provisioning flow before wiring that client method to any UI.

Owner data access is always established by `hostel_owners`; warden access is always established by `warden_assignments`. Do not accept a client-provided hostel ID as authorization. Owner and warden complaint status changes append to `complaint_status_history`, while preserving the original student report. Room capacity edits, reassignment releases, and check-out releases are always bounded by each room's declared `capacity`, and hostel capacity releases are always bounded by `source_available_capacity` (catalog rows) or `total_capacity` (manual/owner rows) via `releaseHostelCapacity`. A confirmed booking's room can only be assigned by a warden explicitly assigned to that hostel, and only `confirmed` bookings are eligible; every assignment is recorded in `room_assignment_history`. Check-in requires a confirmed booking with an assigned room; check-out is idempotent-safe (returns `409` on repeat) and marks the booking `completed`.

Announcements are always scoped to the hostel an owner or assigned warden controls (`staffCanManageHostel`), and delivered only to the hostel's active bookers and/or favoriters, filtered again by each recipient's `notification_preferences.announcements`. Booking-status and complaint-status changes create notifications through the shared `createNotification` helper, which is a no-op when the recipient has disabled that preference category. Stripe payment intents are only created for a `confirmed` booking owned by the requesting student and require `STRIPE_SECRET_KEY`; without it, `POST /api/payments/intents` returns `503` rather than silently succeeding. A payment only transitions out of `pending` through the signature-verified `POST /api/payments/webhook/stripe` route — never from client-supplied status.

The primary implementation file is `index.js`. Preserve the existing endpoint contracts unless there is an explicit reason to version or migrate them.

## PostgreSQL Catalog State

### Current live data

- The operational `hostels` table contains 75 imported ML catalog hostels plus 1 legacy/manual hostel record: 76 total at the last validation.
- Imported catalog rows have stable IDs such as `HST-001` through `HST-075` in `hostels.external_id`.
- PostgreSQL numeric `hostels.id` remains the operational foreign-key key for bookings, reviews, favorites, rooms, and complaints.
- The external ML ID is only a bridge between the ML dataset/model and operational database rows.
- The catalog source has valid coordinates for all 75 imported records.

### Important: do not rerun the old schema dump

`staybuddy_schema.sql` is an old PostgreSQL dump and is not a safe migration path for the current database. It does not encode the catalog migration history. Do not rerun it against the active database.

Apply the incremental migration files instead, in this order:

1. `migrations/001_rooms_locations_favorites.sql`
2. `migrations/002_hostel_catalog.sql`
3. `migrations/003_catalog_capacity_baseline.sql`
4. `migrations/004_password_reset_tokens.sql`
5. `migrations/005_profiles_staff_ownership.sql`
6. `migrations/006_announcements_notifications.sql`
7. `migrations/007_inventory_stays_payments.sql`

All are written to be additive/idempotent and have already been applied locally.

### Migration 005: profiles and operational role ownership

`005_profiles_staff_ownership.sql` creates `student_profiles`, `notification_preferences`, `hostel_owners`, `warden_assignments`, and `complaint_status_history`; it also adds `owner` to the permitted `users.role` values. The API initializes these additive tables on startup for local development and disposable regression databases, but production deployments must apply the migration explicitly.

### Migration 006: announcements and notifications

`006_announcements_notifications.sql` creates `announcements` (hostel-scoped, authored by an owner or assigned warden) and `notifications` (per-user, optionally linked to an announcement, with `read_at` state). Delivery always checks `notification_preferences` before inserting a row.

### Migration 007: authoritative inventory, stays, and payments

`007_inventory_stays_payments.sql` creates `room_assignment_history` (audit trail for every room assignment), `booking_stays` (one row per booking, enforcing `checked_out_at >= checked_in_at` and requiring a check-in before a checkout), and `payments` (provider-agnostic payment/receipt ledger with a `pending -> succeeded/failed/cancelled/refunded` status lifecycle). Only the Stripe webhook route may transition a payment out of `pending`.

Relevant protected endpoints:

- `GET`/`PUT /api/student/profile`
- `GET`/`PUT /api/notification-preferences`
- `GET`/`POST /api/owner/hostels`, `PATCH /api/owner/hostels/:id`
- `GET /api/owner/bookings`, `PATCH /api/owner/bookings/:id`
- `GET`/`PATCH /api/owner/complaints`
- `GET /api/owner/dashboard`
- `GET`/`POST /api/owner/wardens`

The disposable owner authorization regression is `test/owner_authorization.integration.test.js` and runs through `node scripts/run_db_integration_tests.js` with the existing student/capacity tests.

### Warden, announcement, notification, room, stay, and payment endpoints (use cases 22-29)

- `GET`/`PATCH /api/warden/bookings`, `/api/warden/bookings/:id` \u2014 scoped via `warden_assignments`; sends a `booking_status` notification on change.
- `GET`/`PATCH /api/warden/complaints`, `/api/warden/complaints/:id` \u2014 scoped via `warden_assignments`; appends to `complaint_status_history`.
- `POST /api/warden/bookings/:id/check-in`, `POST /api/warden/bookings/:id/check-out` \u2014 require a confirmed booking with an assigned room; check-out releases hostel/room capacity exactly once and marks the booking `completed`.
- `GET /api/bookings/:id/available-rooms`, `POST /api/bookings/:id/assign-room` \u2014 warden-scoped; assignment requires a `confirmed` booking and records `room_assignment_history`.
- `GET /api/bookings/:id/stay` \u2014 student-owned stay lookup.
- `GET`/`POST /api/hostels/:hostel_id/rooms`, `POST /api/rooms`, `PATCH /api/rooms/:id` \u2014 owner-scoped via `hostel_owners`; capacity edits are bounded by room capacity and current occupancy.
- `POST`/`GET /api/announcements` \u2014 owner/warden authors scoped via `staffCanManageHostel`; students see only announcements matching their active bookings or favorites, gated by `notification_preferences.announcements`.
- `GET /api/notifications`, `PATCH /api/notifications/:id/read` \u2014 user-scoped notification inbox and read-state.
- `POST /api/payments/intents` (student-only, requires `STRIPE_SECRET_KEY`), `GET /api/payments` (student-only), `POST /api/payments/webhook/stripe` (HMAC-signature verified, the only route allowed to move a payment out of `pending`).

The disposable regression covering all of the above is `test/warden_notifications.integration.test.js`, run through the same `node scripts/run_db_integration_tests.js` entrypoint. It proves warden-to-warden isolation across bookings/complaints/room-assignment, a full check-in/check-out cycle with exactly-once capacity restoration, and preference-gated announcement delivery to a student's favorited hostel.

**Payments (use case 29) is now covered by `test/payments.integration.test.js`.** `index.js` reads `STRIPE_API_BASE` (default `https://api.stripe.com`) so the test can point `POST /api/payments/intents` at a local HTTP server that mimics Stripe's `payment_intents` response, avoiding any need for real Stripe credentials in CI. The test proves: a non-`confirmed` booking cannot get a payment intent (`409`); a confirmed booking gets a `pending` payment row and `client_secret`; another student cannot see the payment (`GET /api/payments` is user-scoped); a forged `stripe-signature` webhook is rejected (`400`) and never changes payment status; and a correctly HMAC-signed webhook settles the payment to `succeeded` with the receipt URL. **This only proves StayBuddy's own webhook/authorization contract** \u2014 it does not validate against Stripe's real API surface. Before treating use case 29 as production-ready, configure real `STRIPE_SECRET_KEY`/`STRIPE_WEBHOOK_SECRET` values and perform one live test-mode round trip (create a real payment intent, complete it with a Stripe test card, and confirm the real webhook delivery settles the `payments` row).

### Migration 001: rooms, locations, favorites

`001_rooms_locations_favorites.sql` adds hostel university and coordinates, creates `rooms`, adds `bookings.room_id`, and creates `favorites`.

### Migration 002: ML-compatible hostel catalog

`002_hostel_catalog.sql` adds catalog fields to `hostels`, including:

- `external_id`, `hostel_type`, `area`, `verified`, and `source`
- Pricing fields: `single_room_price`, `double_room_price`, `dorm_room_price`, `price_tier`
- Food, electricity, internet, noise, curfew, and study-environment fields
- `room_types_available` and `amenities` as JSONB
- Catalog `total_rooms`, coordinates, and discovery metadata

It also creates an indexed unique constraint for non-null `external_id`, plus indexes for city, university, price, hostel type, and amenities.

### Migration 003: supplier availability baseline

`003_catalog_capacity_baseline.sql` adds `hostels.source_available_capacity`.

This field matters because CSV `available_rooms` is a supplier snapshot, not simply `total_capacity - local bookings`. For example, an imported hostel can have 126 total beds but only 2 supplier-reported available beds.

The operational invariant for catalog rows is:

```text
available_capacity = source_available_capacity - count(pending and confirmed bookings)
```

The baseline is checked to remain between zero and `total_capacity`.

Do not replace this formula with `total_capacity - bookings`; that would incorrectly reopen supplier-occupied beds.

## Catalog Importer

File: `scripts/import_hostel_catalog.js`

Source: `ml-notebooks/data/hostels.csv`

The importer is transactional and idempotent:

1. Parses quoted CSV values without an extra npm dependency.
2. Validates required fields, numeric values, boolean flags, capacities, and JSON arrays.
3. Matches an existing row by `external_id`, with a legacy-safe fallback for an older record with matching name/city and no external ID.
4. Counts active local bookings (`pending` and `confirmed`).
5. Sets `source_available_capacity` from CSV `available_rooms`.
6. Sets live `available_capacity` to source availability minus active local bookings.
7. Inserts or updates the catalog record in the same transaction.

The importer intentionally fails if active bookings exceed the imported source availability. Do not weaken that error; it exposes a real data conflict that requires intervention.

Run it from the backend root:

```powershell
node scripts/import_hostel_catalog.js
```

Last validated behavior:

- First import: 74 catalog rows inserted and 1 legacy matching row updated.
- Re-import: 0 inserted and 75 updated.
- Capacity invariant violations after import: 0.

## Booking And Availability Behavior

`index.js` treats `pending` and `confirmed` bookings as capacity-holding states. `cancelled` and `completed` do not hold capacity.

### Implemented lifecycle behavior

- `POST /api/bookings`: locks the hostel row, checks availability, creates the booking, and decrements hostel availability for a capacity-holding booking.
- `PUT /api/bookings/:id`: safely adjusts capacity when a booking moves between holding and non-holding statuses.
- `POST /api/bookings/:id/cancel`: restores capacity only for a booking that currently holds capacity.
- `POST /api/bookings/:id/assign-room`: manages room-level capacity for a specific assigned room.

The helper `releaseHostelCapacity` restores hostel capacity with an atomic upper bound:

- For ML catalog rows, the ceiling is `source_available_capacity`.
- For manual rows, the ceiling is `total_capacity`.

This prevents cancellation/status updates from restoring more capacity than the supplier says is available.

### Room-model limitation

The ML catalog CSV has aggregate room types/counts, but not real physical room inventory. Do not generate fake rooms from aggregate data without a product/data decision. Catalog booking is intentionally supported at the hostel capacity level.

Existing physical room records are meaningful only for manually managed hostels. Their `rooms.available_capacity` is maintained separately when a booking is assigned to a room.

### Validation already performed

For catalog hostel `HST-001`, an API-level create/cancel test verified:

```text
13 available -> 12 while pending -> 13 after cancellation
```

The same transition was verified for updating a booking from `pending` to `cancelled`. Both temporary test bookings were deleted. Final database checks reported zero capacity-invariant violations and no test-booking residue.

## Discovery And Catalog APIs

The Node hostel response now exposes rich catalog fields, including external ID, catalog pricing, type, amenities, coordinates, capacity, food, and other discovery metadata.

The filter endpoint supports parameterized criteria such as:

```text
GET /api/hostels/search/filter?max_price=15000&amenities=WiFi,Hot%20Water&hostel_type=Girls&verified=true
```

Supported filter concepts include price, hostel type, verified status, all-required amenities, capacity, city, university, and free-text search.

Previously validated results included:

- 76 records from `GET /api/hostels`.
- 32 catalog records at or below a sample price threshold of 15000.
- 22 catalog records containing both `WiFi` and `Hot Water`.
- Actual HTTP filtering returned matching catalog hostels and catalog fields.

## Recommendation Architecture

There are two separate recommendation paths. Do not conflate them.

### 1. Node interaction-based recommender

Endpoint: `POST /api/recommendations`

This is a lightweight Node implementation that uses local interactions, ratings, availability, and inferred city preference. Helpers are in `recommender.js`.

It expects an operational numeric `student_id` and does not call the Python ML model.

### 2. Hybrid ML recommender with operational hydration

Python service: `ml-notebooks/app_api.py`

- Runs on `http://127.0.0.1:8000`.
- `GET /health` reports model/data readiness.
- `POST /recommend` receives preference fields and returns ranked ML hostel IDs (`HST-*`) plus hybrid/content/collaborative scores.
- It loads 75 hostels, 200 student profiles, and trained model artifacts.

Node bridge endpoint: `POST /api/recommendations/ml`

Flow:

1. Validates `top_k` is an integer from 1 through 10.
2. Calls FastAPI `/recommend` using `AI_RECOMMENDER_URL` if set, otherwise `http://127.0.0.1:8000/recommend`.
3. Uses a five-second timeout.
4. Requests extra ML candidates to tolerate unavailable operational catalog entries.
5. Finds returned ML IDs through `hostels.external_id`.
6. Returns only catalog rows with `available_capacity > 0`.
7. Preserves ML ranking fields as `model_score`, `content_score`, `collaborative_score`, `student_type`, and `alpha_used`.

Example request:

```json
{
  "gender": "Female",
  "department": "Computer Science",
  "budget_max": 20000,
  "max_distance_km": 3.0,
  "study_preference": 0.6,
  "food_preference": "Both",
  "room_type": "Single",
  "price_sensitivity": 0.6,
  "comfort_preference": 0.5,
  "noise_tolerance": 0.3,
  "curfew_flexibility": 0.5,
  "needs_transport": false,
  "must_have": ["WiFi", "Hot Water"],
  "top_k": 3
}
```

The verified end-to-end path returned a FastAPI-ranked `HST-014`, hydrated it to PostgreSQL hostel ID 15, included amenities and positive capacity, and returned three recommendations.

### FastAPI chatbot startup improvement

`app_api.py` used to load the optional chatbot model before starting Uvicorn. That delayed recommendation startup. Chatbot initialization is now lazy: the recommendation API can start after recommender artifacts load, and chatbot assets are loaded only when `/chat` is first called.

## ML Service Caveats

- The recommender artifacts were saved with scikit-learn 1.6.1. The current local environment loaded them with scikit-learn 1.8.0 and emitted compatibility warnings, although the tested request succeeded.
- **Resolved (use case 30).** `ml-notebooks/requirements.txt` and `ml-notebooks/ml_ai_complaint/requirements.txt` now pin exact versions (`fastapi==0.139.2`, `uvicorn==0.51.0`, `pydantic==2.13.4`, `numpy==2.5.1`, `pandas==3.0.3`/`scipy==1.18.0`, `scikit-learn==1.9.0`, `joblib==1.5.3`) matching the versions actually installed in the project `.venv`. Install with `pip install -r requirements.txt` from each service's directory before running `app_api.py`. Both docstrings were updated to reference the requirements file instead of an ad hoc `pip install fastapi uvicorn`.
- `app_api.py` can install FastAPI dependencies as a fallback, but package setup should be explicit rather than relying on startup installation.
- The room matcher under `ml-notebooks/room_matcher/` is an independent Flask prototype with hardcoded/in-memory room data and its own resident dataset. It is not connected to PostgreSQL and can conflict with Node because it also uses port 5000 by default.

## Automated Tests And Validation

Node has no external testing framework. The project now uses the built-in Node test runner.

Test file: `test/catalog_importer.test.js`

Coverage:

- Real CSV parses into exactly 75 valid and unique records.
- Quoted fields and JSON arrays are parsed correctly.
- Importer capacity calculation subtracts active bookings.
- Importer rejects overbooking.

Run tests directly from the backend root:

```powershell
node --test
```

Last result: 3 passing, 0 failed.

`package.json` contains `"test": "node --test"`, but this machine's npm installation is broken before scripts start because it cannot locate `npm-prefix.js`. Until npm is repaired, use direct Node commands.

**Resolved npm dependency gap (use case 30):** `index.js` requires `nodemailer`, but `STAYBUDDY-main/package.json` never declared it — it only worked locally because Node's module resolution walked up to the workspace-root `node_modules/nodemailer` (declared in the root `package.json`). A standalone deploy of `STAYBUDDY-main/` alone would have failed with `Cannot find module 'nodemailer'`. `nodemailer` is now declared in both `STAYBUDDY-main/package.json` and `STAYBUDDY-main/package-lock.json`, with its exact resolved version/integrity recorded. On a machine with working npm, run `npm ci` from `STAYBUDDY-main/` for a reproducible standalone install. The local npm executable remains broken (`npm-prefix.js` missing), so this repository could not run that final install command here.

### Isolated booking integration test

The dedicated runner `scripts/run_db_integration_tests.js` now provides a real API and database test without writing to the active `DB_NAME` database.

It creates `staybuddy_test`, restores a disposable full copy of the configured source database through `pg_dump` and `psql`, launches `index.js` on port 5002 (each test file launches its own additional server instance on its own port), and runs, serialized via `--test-concurrency=1`: `test/booking_capacity.integration.test.js`, `test/owner_authorization.integration.test.js`, `test/warden_notifications.integration.test.js`, and `test/payments.integration.test.js`. The test database is dropped in cleanup.

The runner refuses to use any `TEST_DB_NAME` that does not begin with `staybuddy_test` or that equals `DB_NAME`. The normal `node --test` command skips this database test. Run it explicitly with:

```powershell
node scripts/run_db_integration_tests.js
```

The initial verified run passed one integration test. It exercised both `POST /api/bookings/:id/cancel` and `PUT /api/bookings/:id` with `status: cancelled`, confirming each restored catalog availability after a pending booking.

Latest full run: **5 passed, 0 failed** (booking lifecycle, student isolation, owner isolation, Stripe payment intents/webhook, warden/room/stay/notification regression).

Useful validation commands:

```powershell
node --check index.js
node --check scripts/import_hostel_catalog.js
node --test
```

For database validation, compare catalog availability with the baseline minus active bookings. Do not run destructive schema-reset commands against the active database.

## Environment And Dependencies

Node dependencies declared in `package.json`:

- `express`
- `cors`
- `dotenv`
- `pg`

Important discrepancy: `index.js` requires `nodemailer`, but `package.json` does not declare it. The local runtime has been able to start, but this is a deployment/setup defect and should be corrected when npm is usable.

Keep secrets out of documentation and commits. `.env.example` currently contains real-looking email credentials and should be sanitized before sharing or deploying this project.

Typical local services:

```text
Node operational API:          http://127.0.0.1:5000
Python recommender:            http://127.0.0.1:8000
Python complaint categorizer:  http://127.0.0.1:8001
PostgreSQL:                    configured through .env
```

## Frontend Integration Status

The first student-facing Flutter operational slice is now connected to Node.

- `lib/config.dart` uses port 5000 across web, Android emulator, and local desktop/iOS targets.
- `lib/services/ai_recommendation_service.dart` calls `POST /api/recommendations/ml`; the client no longer calls FastAPI directly. `AiRecommendation` accepts the Node bridge's hydrated catalog field names and ML score fields.
- `lib/api.dart` maps student discovery, favorites, bookings, reviews, and complaints to active Node endpoints. It obtains the numeric `user.id` from `AuthStore` and sends it only where the current backend contract requires it.
- `GET /api/bookings?user_id=<id>` now filters in PostgreSQL and returns hostel name, location, and catalog monthly price alongside booking fields. The Flutter My Bookings screen uses this scoped endpoint.
- The Flutter booking screen now calls `POST /api/bookings` and displays the returned database booking ID. It submits a pending hostel-capacity booking and surfaces API errors, including the capacity conflict response.
- The Flutter My Bookings screen loads live booking data and uses `POST /api/bookings/:id/cancel`, then reloads the list. It contains no fixture bookings.
- The Flutter hostel detail page now loads and persists favorites, opens the live capacity-request booking screen, and exposes review and complaint submission actions. It reloads hostel detail after a review so the live review list updates.
- `GET /api/complaints?user_id=<id>` now filters in PostgreSQL. The Flutter API client uses this scoped query instead of downloading all complaints and filtering in memory.

Important current limitation:

- The booking UI still has a visual room-type preference and price preview inherited from the prototype. Catalog booking remains hostel-capacity only because the CSV lacks authoritative physical room inventory. The request does not assign a room or persist the preference. Do not represent that UI choice as a room reservation until an authoritative room model exists.
- The chatbot and room-matcher screens remain prototypes with separate contracts. Do not connect them to this vertical slice without reconciling their data models.
- **Authorization hardening is now done for the student routes.** Login/register issue an HMAC-SHA256-signed token (`AUTH_SECRET` in `.env`); `requireAuth` middleware verifies it and derives `req.user.id`. `GET/POST /api/bookings`, `PUT /api/bookings/:id`, `POST /api/bookings/:id/cancel`, `POST/DELETE/GET /api/favorites*`, `POST /api/reviews`, and `GET/POST /api/complaints` now require a valid token and enforce that the acting user matches the token identity (403 on mismatch); any client-supplied `user_id` in these routes is ignored in favor of the token. See `STAYBUDDY-main/test/booking_capacity.integration.test.js` (`a student cannot read, cancel, or forge another student's data`) for the isolation proof.
- **Password verification is now enabled.** `passwords.js` uses Node `crypto.scrypt` with a random salt; login uses a timing-safe comparison, registration stores only the derived hash, and auth responses omit `password_hash`. The disposable integration suite proves that plaintext is not stored, a correct password succeeds, and an incorrect password receives `401`. The existing two local seed accounts still contain unrecoverable `dummyhash*` values; reset each only by setting `NEW_PASSWORD` directly in the terminal and running `node scripts/reset_legacy_password.js --email user@example.com` from `STAYBUDDY-main/`.
- **Password recovery is now enabled.** `POST /api/auth/password-reset/request` always returns the same response for known and unknown accounts, stores only a SHA-256 reset-token digest in `password_reset_tokens`, invalidates prior unused tokens, and emails a 15-minute reset link. `POST /api/auth/password-reset/confirm` validates the token transactionally, replaces the password hash, and invalidates all remaining reset tokens for that user. `PASSWORD_RESET_URL` in `.env` controls the link base; the Flutter `PasswordResetScreen` accepts either the token or the complete link. The isolated test runner enables test-only token delivery and verifies invalid-token rejection, old-password rejection, new-password login, and single-use behavior.

The complete two-student regression is now automated and terminal-only. `node scripts/run_db_integration_tests.js` clones the active database into a disposable `staybuddy_test` database, registers a second disposable student, and proves booking capacity restoration plus token-derived isolation of bookings, favorites, reviews, and complaints. It also verifies that a submitted review reappears in the operational hostel-detail response. The test passed on 2026-07-18.

## Recommended Continuation Order

1. Extend the isolated PostgreSQL integration suite with catalog re-import behavior and `POST /api/recommendations/ml` cases. Tests must continue to use only a disposable `staybuddy_test*` database.
2. Add a Python requirements/lock file and pin the scikit-learn version compatible with the saved artifacts. Then verify `/health` and `/recommend` in a clean environment.
3. Repair the local npm installation and add the missing `nodemailer` dependency to `package.json` and its lock file. Do not attempt further npm installs until the npm path problem is fixed.
4. Decide whether catalog hostels should remain aggregate-capacity bookable or receive real room inventory from an authoritative room source. Do not synthesize room records from CSV aggregates without that decision.
5. Replace the booking screen's inherited fictional room-type/pricing preview with catalog-backed aggregate-capacity information, or add an authoritative room inventory product model before enabling room selection.
6. Continue client integration with favorites, reviews, complaints, and verified owner/warden workflows. The API-client mappings exist, but their screens require individual runtime validation.

## Non-Negotiable Rules For The Next Agent

- Do not rerun `staybuddy_schema.sql` on the active database.
- Preserve numeric PostgreSQL IDs for relational foreign keys; use `external_id` only to map ML catalog IDs.
- Preserve the catalog availability invariant based on `source_available_capacity`, not total capacity.
- Keep catalog importer operations transactional and idempotent.
- Never let an import overwrite local booking consumption.
- Keep the existing Node recommendation route separate from the ML bridge unless explicitly replacing/versioning it.
- Do not make the room matcher prototype operational without reconciling its independent data model with PostgreSQL.
- Test changes narrowly after each edit, then run a final sanity check before reporting completion.