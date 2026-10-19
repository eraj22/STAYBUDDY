# StayBuddy

StayBuddy is a hostel management system with Flutter portals for students, parents, wardens, and owners, plus a Node.js/PostgreSQL backend.

## Included features

- Hostel discovery, bookings, rooms, documents, fees, receipts, complaints, announcements, emergency support, and parent services.
- QR gate attendance with geofencing, curfew rules, warnings, fines, and attendance appeals.
- Role-aware RAG AI assistant using Groq or Ollama.
- Hostel recommendations, owner operations, and warden management.

## Run locally

Start Docker services from the backend folder:

```powershell
cd STAYBUDDY-main
docker compose up --build -d
```

Check the API:

```powershell
Invoke-RestMethod "http://127.0.0.1:5000/api/health"
```

Start Flutter from the repository root:

```powershell
flutter pub get
flutter run -d chrome --dart-define=API_BASE_URL=http://127.0.0.1:5000
```

Create `STAYBUDDY-main/.env` locally before running Docker. Do not commit credentials such as database passwords, Groq API keys, or notification-provider secrets.