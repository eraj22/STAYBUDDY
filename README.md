# StayBuddy

StayBuddy is a Flutter hostel-discovery application with a Node/PostgreSQL operational API and Python recommendation services.

## Workspace Layout

- `lib/`: Flutter client.
- `STAYBUDDY-main/`: active Node/Express API, PostgreSQL migrations, integration tests, and ML services.
- `assets/data/hostels_featured.csv`: catalog import source.
- `STUDENT_USE_CASE_HANDOFF.md`: prioritized product use-case roadmap and implementation status.

## Local Validation

```powershell
Set-Location D:\staybuddy\STAYBUDDY-main
node --check index.js
node --test
node scripts/run_db_integration_tests.js
```

Read `STUDENT_USE_CASE_HANDOFF.md` and `STAYBUDDY-main/AI_HANDOFF.md` before changing operational workflows.
