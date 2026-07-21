# StayBuddy Backend (Phase 1)

## Requirements
- Node.js (LTS)
- Docker
- PostgreSQL (via Docker)

## 1) Run PostgreSQL (Docker)
```bash
docker run --name staybuddy-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=staybuddy \
  -p 5433:5432 \
  -d postgres:16

## Tests

The catalog importer tests use Node's built-in test runner and do not connect to PostgreSQL or modify catalog data:

```bash
node --test test/catalog_importer.test.js
```

They validate CSV parsing, catalog-record normalization, unique hostel IDs, and the availability rule used during import:

```text
live availability = source availability - pending/confirmed bookings
```

`package.json` also exposes this command as `npm test`. If it fails before running the tests, repair the local npm installation and run the direct `node --test` command above in the meantime.

### Booking Integration Test

The booking lifecycle integration test creates a disposable database named `staybuddy_test`, restores a temporary copy of the configured source database, starts the API on port `5002`, then drops the test database after completion. It never writes to the configured `DB_NAME` database.

```powershell
node scripts/run_db_integration_tests.js
```

Set `TEST_DB_NAME` only to a name beginning with `staybuddy_test`; the runner rejects every other test database name as a safety guard.
