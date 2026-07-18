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

All are written to be additive/idempotent and have already been applied locally.

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
- Add and pin a Python dependency file before deployment/retraining so the ML runtime is reproducible.
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

### Isolated booking integration test

The dedicated runner `scripts/run_db_integration_tests.js` now provides a real API and database test without writing to the active `DB_NAME` database.

It creates `staybuddy_test`, restores a disposable full copy of the configured source database through `pg_dump` and `psql`, launches `index.js` on port 5002, runs `test/booking_capacity.integration.test.js`, and drops the test database in cleanup.

The runner refuses to use any `TEST_DB_NAME` that does not begin with `staybuddy_test` or that equals `DB_NAME`. The normal `node --test` command skips this database test. Run it explicitly with:

```powershell
node scripts/run_db_integration_tests.js
```

The initial verified run passed one integration test. It exercised both `POST /api/bookings/:id/cancel` and `PUT /api/bookings/:id` with `status: cancelled`, confirming each restored catalog availability after a pending booking.

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
- Remaining gap: `/api/auth/login` still accepts any password for a known email (an explicit, commented dev-mode shortcut). Token forgery is fixed, but real password verification is not yet enforced — required before real deployment.

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