# StayBuddy Student Use-Case Handoff

Last updated: 2026-07-18

Give this document, `DIRECTORY_ARCHITECTURE.md`, and `STAYBUDDY-main/AI_HANDOFF.md` to the next AI agent. This is the ordered implementation plan for completing the **student** experience from sign-in through booking and post-booking actions.

## Mission

Finish one trustworthy student workflow before expanding owner, warden, chatbot, or room-matcher prototypes:

```text
Register or sign in
  -> Discover live hostels
  -> View a live hostel detail
  -> Save/remove favorite
  -> Receive catalog-backed recommendations
  -> Submit an aggregate-capacity booking
  -> View and cancel own bookings
  -> Submit a review or complaint for a relevant hostel
```

Do not present prototype behavior, synthetic availability, fake bookings, or a room preference as a confirmed reservation.

## Required Reading And Entry Points

| File | Why it matters |
|---|---|
| `DIRECTORY_ARCHITECTURE.md` | Current workspace tree and active versus prototype boundaries. |
| `STAYBUDDY-main/AI_HANDOFF.md` | Backend constraints, capacity invariant, ML integration, validation commands. |
| `lib/api.dart` | Flutter's current Node API client and auth storage. |
| `STAYBUDDY-main/index.js` | Active Express/PostgreSQL API. |
| `lib/screens/hostels_screen.dart` | Live discovery list. |
| `lib/screens/hostel_detail_screen.dart` | Hostel detail, favorite, review, and booking entry point. |
| `lib/screens/booking_screen.dart` | Current live booking creation with inherited prototype room UI. |
| `lib/screens/my_bookings_screen.dart` | Live, user-scoped booking list and cancellation. |
| `lib/screens/ai_recommendation_screen.dart` | Live recommendation preference and result UI. |

## Current Working Baseline

These are already implemented and should be preserved:

- Flutter talks to the Node API on port `5000` through `AppConfig.baseUrl`.
- Hostel discovery uses `GET /api/hostels`, hostel details use `GET /api/hostels/:id`, and catalog fields are available in the response.
- `AiRecommendationService` uses `POST /api/recommendations/ml`; Node calls FastAPI on port `8000`, hydrates ML IDs from PostgreSQL, and omits unavailable hostels.
- `Api.createBooking` calls `POST /api/bookings` with the authenticated stored user's numeric ID, dates, and `pending` status.
- `MyBookingsScreen` uses `GET /api/bookings?user_id=<id>` and `POST /api/bookings/:id/cancel`.
- Catalog capacity is authoritative at the hostel level. Pending and confirmed bookings consume one available capacity; cancellation releases it only up to the supplier availability baseline.
- `Api` has initial methods for favorites, reviews, and complaints, but their screens still need end-to-end verification and repair.

The following student client work was completed after this plan was created:

- Hostel detail loads and persists favorite state through the Node favorites API.
- Hostel detail opens the live booking screen directly and disables booking when the displayed catalog capacity is zero.
- The booking screen fetches live hostel capacity and uses a single aggregate capacity request instead of fabricated room inventory.
- Hostel detail exposes live review and complaint submission dialogs.
- Complaint listing is now scoped in PostgreSQL through `GET /api/complaints?user_id=<id>`.
- **Phase 5 is complete.** Login/register now issue an HMAC-SHA256-signed token (`AUTH_SECRET` in `.env`) instead of the old forgeable `staybuddy-token-{id}-{role}` string. A `requireAuth` middleware validates the signature and attaches `req.user = { id, role }`. `GET/POST /api/bookings`, `POST /api/bookings/:id/cancel`, `PUT /api/bookings/:id`, `POST/DELETE/GET /api/favorites`, `POST /api/reviews`, and `GET/POST /api/complaints` now require a valid bearer token and derive the acting user from that token rather than trusting the client-supplied `user_id`. Cross-user ownership is enforced (403 on mismatch). A new integration test, `a student cannot read, cancel, or forge another student's data` in `STAYBUDDY-main/test/booking_capacity.integration.test.js`, proves a second user cannot list, cancel, or read a first user's bookings/favorites, and that a spoofed `user_id` in the request body is ignored in favor of the token identity.

**Secure sign-in is complete for new and reset accounts.** Passwords are salted and derived with Node's `crypto.scrypt`; login performs a timing-safe comparison and returns `401` for incorrect credentials. Password hashes are never returned by the API. The two legacy seed accounts still contain unrecoverable `dummyhash*` values, so reset each local account by setting `NEW_PASSWORD` directly in the terminal and running `node scripts/reset_legacy_password.js --email user@example.com` from `STAYBUDDY-main/`. Do not place the password in source code, `.env.example`, or a committed command history.

## Rules That Cannot Be Broken

1. Use numeric PostgreSQL `hostels.id` for Flutter, bookings, reviews, favorites, and complaints. `external_id` such as `HST-001` is only for the ML-to-database bridge.
2. Preserve this catalog capacity invariant:

   ```text
   available_capacity = source_available_capacity
                        - count(pending and confirmed bookings)
   ```

3. Do not derive catalog availability from total capacity, and do not generate physical rooms from CSV aggregates.
4. A selected room type in the current Flutter booking prototype is not an actual room assignment. Do not claim it is one or send it as one.
5. Keep operational client requests behind Node. Do not make Flutter call the recommender FastAPI service directly.
6. Never run `STAYBUDDY-main/staybuddy_schema.sql` against the active database.
7. Treat `ai_room_search_screen.dart` and chatbot flows as separate prototypes until their data contracts are reconciled.
8. Do not expand into owner/warden workflows until every stage below has a focused validation result.

## Ordered Implementation Plan

### Phase 0: Establish A Repeatable Local Baseline

**Goal:** ensure the next agent can test actual behavior before changing code.

1. Start PostgreSQL using the existing local configuration.
2. Start the Node API from `STAYBUDDY-main/` on port `5000`.
3. Start the recommender FastAPI service from `STAYBUDDY-main/ml-notebooks/` on port `8000` when testing recommendations.
4. Run the narrow existing checks before editing:

   ```powershell
   Set-Location D:\staybuddy\STAYBUDDY-main
   node --check index.js
   node --test
   node scripts/run_db_integration_tests.js
   ```

5. Run Flutter analysis only for files being changed. Existing repository lint information, especially `withOpacity` deprecation messages, is not a new functional failure. Resolve compile errors introduced by the current task.

**Done when:** API startup and the isolated database booking lifecycle test both pass.

### Phase 1: Make The Booking UI Honest

**Priority:** highest. The create/cancel API flow works, but the UI still displays fabricated room availability and prices.

**Files:**

- `lib/screens/booking_screen.dart`
- likely `lib/screens/hostel_detail_screen.dart`
- possibly `lib/screens/booking_choice_screen.dart` and `lib/api.dart`

**Work:**

1. Read the real hostel detail response fields: `available_capacity`, `single_room_price`, `double_room_price`, `dorm_room_price`, `room_types_available`, and `amenities`.
2. Pass only real catalog fields needed by the booking screen, or reload the detail before confirmation.
3. Replace hard-coded room cards, fake bed counts, and fixed rent values with an aggregate booking presentation:
   - show live hostel-wide available capacity;
   - show the catalog's published price fields only when present;
   - label a booking as a capacity request, not a particular bed or physical room;
   - show an unavailable state before confirmation if capacity is zero.
4. Keep the backend transaction check. The screen must still handle a `409 No beds available` response because another student may book first.
5. Remove or clearly relabel non-persisted personal-detail and special-request controls if the backend does not store them. Do not imply that the server saved information it does not accept.
6. Ensure check-out remains after check-in and derive it from the selected duration with calendar-safe date arithmetic.

**Acceptance checks:**

- A hostel with `available_capacity = 0` cannot be submitted from the UI.
- A successful booking shows the actual returned booking ID and appears in My Bookings after navigation or refresh.
- The UI never says a specific room was reserved for a catalog hostel.
- The isolated DB integration test still proves capacity decreases for pending booking and returns on cancellation.

### Phase 2: Complete Favorite Behavior

**Files:**

- `lib/screens/hostel_detail_screen.dart`
- any favorites/list screen under `lib/screens/`
- `lib/api.dart` only if actual response shapes show a mismatch

**Active API contract:**

```text
GET    /api/favorites/:user_id
POST   /api/favorites                 { user_id, hostel_id }
DELETE /api/favorites/:user_id/:hostel_id
```

**Work:**

1. On hostel detail load, fetch or maintain the student's favorites so the favorite icon reflects server state.
2. Call add/remove endpoints from the favorite control. Disable duplicate actions while a request is in progress.
3. Interpret `409` from add as already favorited and recover by refreshing state instead of crashing.
4. Render the saved-hostel list using live favorite rows. The backend returns hostel fields with `hostel_id`; normalize that shape deliberately rather than assuming `id`.
5. Handle signed-out state by sending the user to login or showing a clear sign-in prompt.

**Acceptance checks:**

- Add favorite, restart/reload, and see it retained.
- Remove favorite, reload, and confirm it is absent.
- Another user's favorite list does not appear in the current student UI.

### Phase 3: Complete Reviews

**Files:**

- review submission/detail screen(s) under `lib/screens/`
- `lib/api.dart`
- `STAYBUDDY-main/index.js` only when a documented response gap blocks the UI

**Active API contract:**

```text
GET  /api/hostels/:id                includes reviews
POST /api/reviews
{
  "user_id": 1,
  "hostel_id": 15,
  "overall_rating": 4.5,
  "cleanliness": 4.0,
  "facilities": 4.0,
  "management": 4.5,
  "text_review": "Optional feedback"
}
```

**Work:**

1. Replace static review displays or placeholder submit actions with live detail reviews.
2. Validate rating bounds and non-empty required fields before sending.
3. Refresh the hostel detail after a successful review so the result appears immediately.
4. Do not add a second review endpoint or legacy `/api/student/reviews` alias.
5. Decide, based on existing database/API behavior, whether a student may review without a completed stay. Document the chosen rule; do not silently invent enforcement.

**Acceptance checks:**

- A submitted review is returned by the hostel detail route after refresh.
- Invalid rating input is rejected in the UI before the request.
- API failure leaves the form usable and exposes a readable error.

### Phase 4: Complete Complaints

**Files:**

- complaint screens under `lib/screens/`
- `lib/api.dart`
- `STAYBUDDY-main/index.js` if a student-scoped listing route is required

**Current API contract:**

```text
GET  /api/complaints
POST /api/complaints
{
  "user_id": 1,
  "hostel_id": 15,
  "category": "Cleanliness",
  "severity": "medium",
  "description": "..."
}
```

**Work:**

1. Connect complaint creation to `Api.fileComplaint`.
2. Use the categorizer response only if the existing Node route returns it; otherwise present the chosen category as user input rather than pretending AI classified it.
3. Make the student complaint list live and show server status.
4. Do not fetch every user's complaints into the client. Prefer a server-filtered `GET /api/complaints?user_id=<id>` route, implemented with a parameterized SQL query, then update `Api.getComplaints` to use it.
5. Link a complaint to a real selected hostel and validate category, severity, and description.

**Acceptance checks:**

- Created complaint reappears after reload with its real status.
- The list contains only the signed-in student's complaints.
- Invalid or incomplete fields produce a useful error without losing the draft.

### Phase 5: Harden Student Authorization Before Multi-User Testing

**Status: complete.**

**Files:**

- `STAYBUDDY-main/index.js`
- `STAYBUDDY-main/test/booking_capacity.integration.test.js`
- `STAYBUDDY-main/.env`, `.env.example` (new `AUTH_SECRET`)

**What was done:**

1. Added `signAuthToken`/`verifyAuthToken` (HMAC-SHA256, `crypto` module, timing-safe comparison) and a `requireAuth` Express middleware in `index.js`, replacing the forgeable `staybuddy-token-{id}-{role}` string.
2. `/api/auth/login` and `/api/auth/register` now issue signed tokens; `/api/auth/me` uses `requireAuth`.
3. `requireAuth` is now applied to `GET/POST /api/bookings`, `PUT /api/bookings/:id`, `POST /api/bookings/:id/cancel`, `POST/DELETE/GET /api/favorites*`, `POST /api/reviews`, and `GET/POST /api/complaints`. Each derives the acting `user_id` from `req.user.id` (the verified token), not the request body/query, and returns `403` on any path/body `user_id` mismatch.
4. Flutter already sent `Authorization: Bearer <token>` on every one of these calls (`lib/api.dart`), so no client changes were required for this phase.
5. Added an integration test proving isolation: unauthenticated/forged tokens are rejected (401), a spoofed body `user_id` is ignored in favor of token identity, and Student B cannot list, cancel, or read Student A's bookings/favorites (403).

**Password verification update:** `/api/auth/register` now stores a salted `crypto.scrypt` password hash, `/api/auth/login` verifies it with a timing-safe comparison, and auth responses omit `password_hash`. Existing local `dummyhash*` seed accounts must be explicitly reset with `NEW_PASSWORD` and `node scripts/reset_legacy_password.js --email user@example.com`; their old values cannot and should not be treated as valid passwords.

**Acceptance checks (all passing via `node scripts/run_db_integration_tests.js`):**

- A request without a valid bearer token is rejected.
- Student A cannot list, cancel, favorite, review, or complain as Student B.
- Student A's valid booking and cancellation behavior still preserves the capacity invariant.
- New authorization cases run against a disposable `staybuddy_test*` database.


## Recommended Validation Pattern

After every substantive edit:

1. Run the narrowest Flutter analyzer command for touched Dart files, or `node --check` for touched Node files.
2. Run the closest behavioral test. For booking/capacity changes, use:

   ```powershell
   Set-Location D:\staybuddy\STAYBUDDY-main
   node scripts/run_db_integration_tests.js
   ```

3. For a changed API response, test the live local endpoint with an authenticated client or a small focused integration test. Do not use the active production-like database for destructive tests.
4. At the end of each completed phase, update `STAYBUDDY-main/AI_HANDOFF.md`. Update `DIRECTORY_ARCHITECTURE.md` only when file architecture changes.

## Ordered Product Use-Case Roadmap

This is the implementation order for a coherent StayBuddy product. Complete the first ten use cases today; the next twenty are intentionally sequenced to build on the data, roles, and authorization already in place. Never start a later item by bypassing the ownership, capacity, or source-of-truth rules above.

### First 10 Use Cases: Complete Today

| # | Use case | Primary actor | Definition of done |
|---|---|---|---|
| 1 | Student registration and secure sign-in | Student | Register hashes the password; login verifies it; invalid password returns `401`; signed token is stored and `/api/auth/me` returns the current user. |
| 2 | Student hostel discovery | Student | Live hostel list supports city, price, amenity, and nearby filters using catalog fields. |
| 3 | Student hostel detail and comparison | Student | A student can inspect live price/capacity/amenities/reviews and compare two valid hostels. |
| 4 | Student personalized recommendations | Student | The Node-to-FastAPI bridge returns hydrated, currently available catalog results and has a terminal-tested health check. |
| 5 | Student favorites | Student | Add, list, and remove favorites persist across reload; a student can access only their own favorites. |
| 6 | Student capacity booking | Student | Student creates a pending aggregate-capacity booking; capacity reduces atomically and `409` is handled when unavailable. |
| 7 | Student booking management | Student | Student lists and cancels only their own bookings; cancellation restores capacity no higher than the supplier baseline. |
| 8 | Student review submission | Student | Student submits a bounded rating and optional text; the review appears on the correct hostel detail. |
| 9 | Student complaint submission and tracking | Student | Student files a complaint tied to a real hostel and sees only their own complaint statuses. |
| 10 | Two-student authorization regression | Student | Disposable DB test proves forged tokens fail, body `user_id` spoofing fails, and Student B cannot view or change Student A's records. |

**Current state:** Use cases 1 through 11 are implemented and terminal-verified. Use case 11 provides `POST /api/auth/password-reset/request` and `POST /api/auth/password-reset/confirm`, stores only hashed single-use tokens with a 15-minute expiry, sends a configurable reset link, and has a Flutter reset screen linked from student login. Before production deployment, set `PASSWORD_RESET_URL` to the deployed app's reset-link target and reset or replace legacy accounts that still have `dummyhash*` values.

### Next 20 Use Cases: Implement In This Order

| # | Use case | Primary actor | Why it follows now |
|---|---|---|---|
| 11 | Password reset and account recovery | Student/Owner/Warden | **Complete.** A generic request endpoint prevents account enumeration; hashed, 15-minute, single-use tokens update passwords through the Flutter recovery screen. Configure `PASSWORD_RESET_URL` for the deployed client. |
| 12 | Student profile and preferences | Student | Stores budget, university, gender, and commute preferences used by discovery and recommendations. |
| 13 | Student notification preferences | Student | Defines which booking, complaint, and announcement events a student wants to receive before notifications are sent. |
| 14 | Owner onboarding and hostel claim | Owner | Owner creates a verified profile and can claim/create only their own hostel records. |
| 15 | Owner hostel profile management | Owner | Owner edits descriptive and price fields for owned hostels; imported capacity baselines remain protected. |
| 16 | Owner booking queue | Owner | Owner sees pending bookings only for hostels they own, never another owner's data. |
| 17 | Owner booking decision | Owner | Owner confirms or rejects a pending booking; rejection releases capacity exactly once. |
| 18 | Owner complaint inbox | Owner | Owner sees complaints only for owned hostels and can update operational status without changing the student's report. |
| 19 | Owner occupancy dashboard | Owner | Owner sees current pending/confirmed/cancelled counts and live capacity for owned hostels. |
| 20 | Owner authorization regression | Owner | Disposable DB tests prove Owner A cannot read or modify Owner B's hostels, bookings, or complaints. |
| 21 | Warden account assignment | Admin/Owner | **Complete.** `POST /api/owner/wardens` assigns an existing warden account to an owner's hostel via `warden_assignments`; access is always established by this table, never a client-supplied hostel ID. |
| 22 | Warden operational booking view | Warden | **Complete.** `GET`/`PATCH /api/warden/bookings/:id` are scoped through `warden_assignments`; a warden outside the assignment receives `404`, not another warden's data. |
| 23 | Warden complaint triage | Warden | **Complete.** `PATCH /api/warden/complaints/:id` updates status/assignment and appends to `complaint_status_history` for an auditable trail. |
| 24 | Hostel announcements | Owner/Warden | **Complete.** `POST`/`GET /api/announcements` let owners or assigned wardens publish to a hostel; students only see announcements tied to their own confirmed/pending bookings or favorites. |
| 25 | Student notification delivery | Student | **Complete.** Booking-status, complaint-status, and announcement events insert into `notifications` only when the recipient's `notification_preferences` allow that category; `GET`/`PATCH /api/notifications/:id/read` expose read/unread state. |
| 26 | Authoritative room inventory onboarding | Owner/Admin | **Complete.** `POST`/`PATCH /api/rooms` are owner-scoped through `hostel_owners`; capacity edits cannot exceed declared room capacity. Catalog CSV aggregate capacity is still not used to synthesize rooms. |
| 27 | Room assignment after confirmation | Warden | **Complete.** `POST /api/bookings/:id/assign-room` requires a warden assignment to the booking's hostel and a `confirmed` booking; every assignment is recorded in `room_assignment_history`. |
| 28 | Student move-in and move-out workflow | Student/Warden | **Complete.** `POST /api/warden/bookings/:id/check-in` requires a confirmed booking with an assigned room; `check-out` restores hostel and room capacity exactly once, marks the booking `completed`, and notifies the student. `GET /api/bookings/:id/stay` lets the student see their own stay record. |
| 29 | Payments and receipts | Student/Owner | **Complete (mock-verified).** `POST /api/payments/intents` creates a server-side Stripe payment intent for a confirmed booking and stores a `pending` row in `payments`; `POST /api/payments/webhook/stripe` verifies the provider signature (HMAC-SHA256, 300-second tolerance, `crypto.timingSafeEqual`) before marking a payment `succeeded`/`failed` and storing the receipt URL. No booking is ever marked paid from client input — only the signature-verified webhook can settle a payment. `STRIPE_API_BASE` is an env override (default `https://api.stripe.com`) that lets `test/payments.integration.test.js` point the intent-creation call at a local mock HTTP server, so the full create-intent + forged-webhook-rejected + verified-webhook-settles flow is now proven by the disposable regression suite without needing real Stripe credentials. **Still required before production use:** set real `STRIPE_SECRET_KEY`/`STRIPE_WEBHOOK_SECRET` values and run one live test-mode round trip against the actual Stripe API and dashboard-configured webhook endpoint — the mock only proves StayBuddy's own contract logic, not Stripe's real behavior. |
| 30 | Production readiness and deployment | Operations | **Complete for the concrete, code-level items.** `STAYBUDDY-main/package.json` now declares `nodemailer` (it was previously only resolved by hoisting from the workspace-root `node_modules`, so a standalone deploy of `STAYBUDDY-main/` would have failed with `Cannot find module 'nodemailer'`). `STAYBUDDY-main/ml-notebooks/requirements.txt` and `STAYBUDDY-main/ml-notebooks/ml_ai_complaint/requirements.txt` now pin exact versions for both FastAPI services. `.env.example` documents the new `STRIPE_SECRET_KEY`/`STRIPE_WEBHOOK_SECRET` variables. Secrets were already protected: `.env`/`.env.*` are gitignored at both the root and `STAYBUDDY-main/` level, and only `.env.example` (with placeholder values) is tracked. **Still open (operational, not code):** provisioning managed PostgreSQL backups/point-in-time-recovery, application monitoring/alerting, running the migrations (`STAYBUDDY-main/migrations/001` through `007`) explicitly against a staging database before production cut-over (the app's `ensureOperationalTables()` auto-create path is for local/disposable-test convenience only and must not be relied on in production), and a role-based (student/owner/warden) manual or scripted end-to-end pass in a real staging environment. |

## Handoff: What A New Agent Should Do Next

Every use case in the original 1-30 roadmap now has a working, test-verified implementation at the API layer. The remaining work for a new agent, in priority order, is:

1. **Stripe live validation (use case 29 hardening).** Configure real Stripe test-mode keys, create a test-mode webhook endpoint in the Stripe dashboard pointed at a tunneled/staging `POST /api/payments/webhook/stripe`, and confirm one real `payment_intent.succeeded` round trip updates a `payments` row exactly as the mock test does.
2. **Deployment operations (use case 30 hardening).** Apply migrations 001-007 explicitly against a staging database (do not depend on `ensureOperationalTables()`), configure automated PostgreSQL backups, and wire up basic uptime/error monitoring for the Node API and both FastAPI services.
3. **Flutter UI wiring.** The backend and `lib/api.dart` client now expose every use case 22-29 endpoint (warden bookings/complaints, check-in/check-out, room assignment, announcements, notifications, payment intents), but no Flutter screens consume them yet. Build warden/owner dashboards and a student payment/notification UI against the existing client methods.
4. **Known client gap:** `Api().registerOwner(...)` in `lib/api.dart` still targets the now-student-only public `/api/auth/register` route and will receive `403` if called. Do not re-open client-selected roles to fix this — route owner/warden account creation through an admin-controlled flow instead.
5. **Staging end-to-end pass.** Once a staging environment exists, run a manual or scripted pass through all three roles (student, owner, warden) against real (non-disposable) data before declaring the product launch-ready.

## Delivery Guardrails

- Use cases 14 through 20 must not begin until use case 1 has password verification and the student regression remains green.
- Use cases 21 through 25 require the owner authorization regression from use case 20. **Verified**: the disposable warden regression (`test/warden_notifications.integration.test.js`) proves a warden outside an assignment cannot read or mutate another warden's bookings or complaints, and that notification/announcement delivery respects saved preferences.
- Use cases 26 through 28 require an authoritative room source; catalog CSV aggregate capacity is not room inventory. **Verified**: the same disposable regression exercises owner room creation, warden-only assignment to a confirmed booking, check-in, and exactly-once capacity release on check-out for both the hostel and the room.
- Use case 29 requires a provider-specific payment design and server-side verification before any booking is marked paid. **Verified against a mock Stripe server** by `test/payments.integration.test.js`; a real test-mode Stripe round trip is still required before this is production-trustworthy.
- Every new role-specific action needs both a focused API test and a disposable multi-user authorization regression.

## Completion Definition

The student use case is complete when a signed-in student can discover live inventory, receive hydrated recommendations, favorite hostels, create and cancel their own capacity-safe booking, submit and see reviews/complaints, and cannot access another student's data. All modifications must preserve the catalog availability invariant and have focused automated or live validation recorded in `STAYBUDDY-main/AI_HANDOFF.md`.

All 30 roadmap use cases now have this level of validation at the API layer (`node scripts/run_db_integration_tests.js` runs 5 disposable-database regressions covering student, owner, warden, room/stay, and payment flows). The product is not yet launch-ready: see "Handoff: What A New Agent Should Do Next" above for the remaining Flutter UI and deployment-operations work.
