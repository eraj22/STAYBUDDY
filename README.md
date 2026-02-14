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
