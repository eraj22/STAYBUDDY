# StayBuddy Current Directory Architecture

Last updated: 2026-07-18

This is the current on-disk structure of the `D:\staybuddy` workspace. Give this file together with `STAYBUDDY-main/AI_HANDOFF.md` and `STUDENT_USE_CASE_HANDOFF.md` to another AI agent. This document explains where code and data live; the handoffs explain the implemented backend behavior, constraints, and next student workflow work.

## Read This First

This workspace contains two application layers:

1. The workspace root is a Flutter mobile/desktop/web application.
2. `STAYBUDDY-main/` is a nested Node.js, PostgreSQL, and Python ML backend workspace.

They are not yet fully integrated. The active operational backend is inside `STAYBUDDY-main/`, while Flutter has older direct-Python and placeholder API paths. Do not assume a source file is active just because it exists.

```text
D:\staybuddy
|-- Flutter client application and platform runners
|-- Firebase configuration and static assets
|-- Documentation and extracted report material
`-- STAYBUDDY-main\
    |-- Active Node.js/PostgreSQL operational backend
    |-- Python recommendation and complaint ML services
    |-- Migrations, import scripts, and automated tests
    `-- Earlier standalone prototypes and training material
```

## Important Entry Points

| Location | Purpose | Current status |
|---|---|---|
| `lib/main.dart` | Flutter application entry point. | Flutter client entry point. |
| `lib/api.dart` | Flutter HTTP/API helper surface. | Needs contract alignment with the active Node API. |
| `STAYBUDDY-main/index.js` | Express operational API and PostgreSQL access. | Active backend entry point, default port `5000`. |
| `STAYBUDDY-main/ml-notebooks/app_api.py` | FastAPI hybrid hostel recommendation service. | Active ML service, port `8000`. |
| `STAYBUDDY-main/ml-notebooks/ml_ai_complaint/app_api.py` | FastAPI complaint categorizer. | Active AI sidecar, port `8001`. |
| `STAYBUDDY-main/scripts/import_hostel_catalog.js` | Imports the ML hostel catalog into PostgreSQL. | Active maintenance command. |
| `STAYBUDDY-main/AI_HANDOFF.md` | Backend state, invariants, routes, and continuation instructions. | Required reading before backend changes. |
| `STUDENT_USE_CASE_HANDOFF.md` | Ordered implementation and validation plan for the complete student journey. | Required reading before continuing student workflows. |

## Root Flutter Application

```text
.
|-- android/
|-- assets/
|-- build/
|-- docs/
|-- ios/
|-- lib/
|-- linux/
|-- macos/
|-- md/
|-- test/
|-- web/
|-- windows/
|-- analysis_options.yaml
|-- firebase.json
|-- package.json
|-- pubspec.yaml
|-- README.md
|-- staybuddy_student.iml
`-- STAYBUDDY-main/
```

### Root configuration and metadata

- `pubspec.yaml`: Flutter package manifest, SDK constraints, dependencies, assets, and fonts.
- `analysis_options.yaml`: Dart/Flutter static-analysis configuration.
- `firebase.json`: Firebase project/deployment configuration. It is a file, not a `firebase/` directory.
- `package.json`: root JavaScript metadata, separate from the backend `STAYBUDDY-main/package.json`; do not confuse the two.
- `README.md`: top-level project readme.
- `staybuddy_student.iml`: IntelliJ/Android Studio project metadata.

### `lib/`: Flutter application source

```text
lib/
|-- main.dart
|-- api.dart
|-- config.dart
|-- firebase_options.dart
|-- routes.dart
|-- theme.dart
|-- model/
|-- screens/
|-- services/
|-- theme/
`-- widgets/
```

- `main.dart`: application startup, initialization, and root widget setup.
- `api.dart`: HTTP request helper(s) and client-facing API access.
- `config.dart`: Flutter runtime configuration, including service-base configuration used by client code.
- `firebase_options.dart`: generated Firebase configuration for supported platforms.
- `routes.dart`: named-route/navigation definitions.
- `theme.dart`: global Flutter theme configuration.

#### `lib/model/`

- `ai_recommendation.dart`: Dart data model for recommendation results returned to the client.

#### `lib/services/`

- `ai_recommendation_service.dart`: calls recommendation-related service endpoints.
- `location_service.dart`: device-location access and coordinate handling.

#### `lib/theme/`

- `app_sizes.dart`: shared visual dimensions and spacing values.
- `responsive.dart`: responsive layout helpers.

#### `lib/widgets/`

- `app_button.dart`: shared button component.
- `glass_card.dart`, `soft_card.dart`: reusable decorative/content-card components.
- `responsive_shell.dart`: responsive page-layout shell.
- `three_d_container.dart`, `three_d_flip_card.dart`: reusable 3D-style visual components.
- `video_background.dart`, `video_scaffold.dart`: video-backed page/layout components.

#### `lib/screens/`

This folder contains Flutter page widgets. It currently includes student discovery, booking, AI, owner, and warden UI flows.

- Student/discovery screens: `welcome_screen.dart`, `splash_screen.dart`, `role_screen.dart`, `login_screen.dart`, `signup_screen.dart`, `discover_hostel_home_page.dart`, `hostels_screen.dart`, `hostel_discovery_screen.dart`, `hostel_dataset_discovery.dart`, `hostel_detail_screen.dart`, `location_access_screen.dart`.
- Recommendation and AI screens: `ai_recommendation_screen.dart`, `ai_room_search_screen.dart`, `chatbot_screen.dart`, `room_matcher_service.dart`.
- Booking screens: `booking_choice_screen.dart`, `booking_screen.dart`, `my_bookings_screen.dart`, `student_booking_info_screen.dart`.
- Owner onboarding and management screens: `owner_login_screen.dart`, `owner_otp_screen.dart`, `owner_info_screen.dart`, `owner_registered_screen.dart`, `owner_dashboard_screen.dart`, `owner_hostel_info_screen.dart`, `owner_hostel_map_screen.dart`, `owner_hostel_profile_screen.dart`, `owner_hostel_rooms_screen.dart`, `owner_hostel_success_screen.dart`, `owner_manage_wardens_screen.dart`, `owner_step_facilities.dart`, `owner_step_images.dart`, `owner_steps_mess_safety_rules.dart`.
- Warden screens: `warden_login_screen.dart`, `warden_dashboard_screen.dart`, `warden_use_cases.dart`.

Important integration note: some of these screens predate the active Node API. In particular, direct calls to Python port `8000`, old `/api/student/*` paths, and simulated booking behavior need deliberate migration. Do not change routes blindly to make an old screen work.

### Flutter assets, tests, and platform folders

- `assets/data/hostels_featured.csv`: Flutter-bundled derived catalog dataset. It is useful for client/demo views but is not the operational PostgreSQL source of truth.
- `test/widget_test.dart`: existing Flutter widget test entry point.
- `android/`, `ios/`, `linux/`, `macos/`, `web/`, `windows/`: Flutter platform runners and platform-specific build configuration. Most nested runner/generated files are framework-managed and should only be edited for deliberate platform changes.
- `web/index.html`, `web/manifest.json`, and `web/icons/`: Flutter web bootstrap, manifest, and icons.
- `build/`: generated Flutter build output. Do not edit or treat it as source.

### Project documentation folders

- `docs/`: original project reports, preliminary guide, implementation-plan PDFs, and role/use-case documents. These are reference material, not runtime code.
- `md/`: Markdown/text extractions of reports, plans, and use-case documents. Some are noisy document conversions; use the original PDFs when formatting/context matters.

## `STAYBUDDY-main/`: Backend And ML Workspace

```text
STAYBUDDY-main/
|-- ai-complaint-system/
|-- migrations/
|-- ml-notebooks/
|-- scripts/
|-- test/
|-- .env.example
|-- AI_HANDOFF.md
|-- README.md
|-- index.js
|-- package.json
|-- recommender.js
`-- staybuddy_schema.sql
```

### Active Node.js backend files

- `index.js`: Express server, PostgreSQL pool, operational REST routes, booking lifecycle/capacity logic, favorites, rooms, reviews, complaints, owner flows, and the ML recommendation bridge. Default port is `5000`.
- `recommender.js`: lightweight Node interaction-based recommender helpers. This supports the legacy Node endpoint `POST /api/recommendations`; it is separate from the Python hybrid recommender.
- `package.json`: backend CommonJS package manifest and scripts. `node --test` is the safe regular test command; local `npm` is currently broken on this machine.
- `.env.example`: environment-variable template for Node/PostgreSQL/email configuration. It currently includes real-looking email credentials and must be sanitized/rotated before sharing.
- `README.md`: backend run and test instructions.
- `AI_HANDOFF.md`: authoritative backend continuation document; read this before editing backend behavior.
- `staybuddy_schema.sql`: legacy PostgreSQL dump/seed. It is not a migration and must not be rerun against the active database.

### `migrations/`: additive PostgreSQL schema changes

These migrations have been applied to the active local database in numeric order.

- `001_rooms_locations_favorites.sql`: adds hostel university/coordinates, creates `rooms`, attaches optional `bookings.room_id`, and creates `favorites`.
- `002_hostel_catalog.sql`: adds ML-compatible catalog fields to `hostels`, including `external_id`, prices, amenities, metadata, source information, and indexes.
- `003_catalog_capacity_baseline.sql`: adds `source_available_capacity`, which preserves supplier availability independently from local bookings.

Do not replace these files with a new schema dump. New database changes should be additive migrations with the next numeric prefix.

### `scripts/`: backend maintenance and integration tooling

- `import_hostel_catalog.js`: transactional, idempotent importer from `ml-notebooks/data/hostels.csv` to PostgreSQL. It maps ML `HST-*` IDs into `hostels.external_id` and preserves active-booking availability.
- `run_db_integration_tests.js`: creates an isolated disposable `staybuddy_test*` database, restores a source copy, runs database-backed API tests, then drops the test database. It must never target the live database.

### `test/`: Node test suite

- `catalog_importer.test.js`: database-free native Node tests for CSV parsing, catalog normalization, unique IDs, and capacity calculations.
- `booking_capacity.integration.test.js`: real Express/PostgreSQL booking lifecycle test. It is skipped by default and runs only through the guarded database test runner.

### `ml-notebooks/`: Python ML, data, artifacts, and prototypes

```text
ml-notebooks/
|-- app.py
|-- app_api.py
|-- chatbot.py
|-- generate_dataset.py
|-- retrain.py
|-- training_data.json
|-- label_encoder.pkl
|-- data/
|-- intent_model/
|-- models/
|-- notebooks/
|-- ml_ai_complaint/
`-- room_matcher/
```

- `app_api.py`: active FastAPI hybrid recommendation service. It loads catalog/student/interaction data and saved recommender artifacts, exposes `/health`, `/recommend`, and chatbot-related API behavior on port `8000`.
- `app.py`: Streamlit presentation/demo UI for the recommendation system. It is not the operational Node API.
- `chatbot.py`: chatbot pipeline with intent classification, entity extraction, context tracking, optional Ollama response wrapping, and template fallbacks. The recommender API initializes it lazily so it does not block recommendation startup.
- `generate_dataset.py`: deterministic synthetic-data generator for the ML catalog, student profiles, and interactions.
- `retrain.py`: retraining helper for model artifacts; use only with a controlled Python dependency environment.
- `training_data.json`: chatbot intent training examples.
- `label_encoder.pkl`: serialized label encoder used by the intent model.
- `intent_model/`: local serialized DistilBERT intent-classifier files. These are model artifacts, not source code.
- `notebooks/StayBuddy_Recommendation_Engine.ipynb`: notebook documenting EDA, content-based filtering, collaborative filtering, hybrid evaluation, and ML demonstration work.
- `__pycache__/`: Python bytecode cache; generated and not source.

#### `ml-notebooks/data/`: ML source datasets

- `hostels.csv`: 75-hostel ML catalog source. This is the current backend importer input and uses IDs such as `HST-001`.
- `students.csv`: 200 synthetic student preference profiles used by recommender training/inference.
- `interactions.csv`: synthetic interaction history used by collaborative/hybrid recommendation logic.

#### `ml-notebooks/models/`: saved recommender artifacts

- `hostel_feature_matrix.npy`, `cb_scaler.pkl`, `cb_feature_cols.json`, `cb_metrics.json`, `content_based_analysis.png`: content-based feature matrix, scaler/configuration, metrics, and chart.
- `interaction_matrix.csv`, `predicted_matrix.csv`, `svd_model.pkl`, `U_student_factors.npy`, `Vt_hostel_factors.npy`, `cf_k_tuning.json`, `cf_metrics.json`, `collaborative_filtering_analysis.png`: collaborative-filtering input/output artifacts, model factors, metrics, tuning, and chart.
- `cold_start_models.pkl`: clustering/scaling artifacts used for cold-start recommendation behavior.
- `hybrid_config.json`, `hybrid_metrics.json`, `hybrid_model_analysis.png`: hybrid weighting configuration, evaluation metrics, and chart.
- `hostels_featured.csv`: ML-derived featured catalog output; distinct from both the raw `data/hostels.csv` and Flutter asset copy.

The artifacts were created with scikit-learn `1.6.1`; the current local Python runtime warned when loading them under `1.8.0`. Pin compatible dependencies before deployment or retraining.

#### `ml-notebooks/ml_ai_complaint/`: active complaint classifier sidecar

- `app_api.py`: FastAPI service on port `8001`; categorizes complaint text and exposes categorization, batch, pattern, and health endpoints.
- `complaint_model.pkl`, `complaint_word_vec.pkl`, `complaint_char_vec.pkl`, `complaint_label_enc.pkl`, `complaint_priority_model.pkl`, `complaint_priority_vec.pkl`: serialized complaint category/priority model artifacts and vectorizers.
- `complaint_model_meta.json`: model metadata, categories, metrics, and response suggestions.
- Training/supporting data and scripts in this directory: only use them to retrain or inspect the classifier; Node only needs the running API endpoint.

#### `ml-notebooks/room_matcher/`: disconnected room-matching prototype

This is an independent Flask/in-memory roommate and room-matching project. It is not connected to PostgreSQL, uses its own data model, and can conflict with the Node API because its web app defaults to port `5000`.

- Core implementation: `room_matcher.py`, `recommender.py`, `similarity_scorer.py`, `feature_engineering.py`, `data_loader.py`, `config.py`, `utils.py`.
- Prototype application/testing: `web_app.py`, `main.py`, `test_api.py`, `templates/`.
- Prototype datasets/results: `all_residents.csv`, `student_profiles_enhanced.csv`, `job_seekers.csv`, `roommate_matches.csv`.
- Data preparation helpers: `add_professionals.py`, `merge_data.py`, `rebuild_data.py`, `check_data.py`.
- Saved prototype artifacts/configuration: `room_matcher_model.pkl`, `requirements.txt`.

Do not connect this prototype to production flows without reconciling its resident/room assumptions with the operational PostgreSQL schema.

### `ai-complaint-system/`: older separate complaint prototype

This folder is an earlier standalone complaint-management implementation. It duplicates the complaint domain but is not the service called by `index.js`.

- `app.py`: standalone FastAPI complaint dashboard/API using `ComplaintAnalyzer`.
- `complaint_analyzer.py`: rule/keyword-based complaint categorization and suggestions.
- `complaint-system.py`: separate built-in Python HTTP-server version of similar functionality.
- `data/`: standalone prototype complaint CSV data.
- `requirements.txt`: dependencies for this isolated prototype.
- `__pycache__/`: generated Python bytecode.

Use `ml-notebooks/ml_ai_complaint/app_api.py` for the active Node-integrated complaint categorizer. Do not run both systems as though they share state.

## Data And Identifier Boundaries

```text
ml-notebooks/data/hostels.csv
        |  HST-001 ... HST-075
        v
scripts/import_hostel_catalog.js
        v
PostgreSQL hostels.external_id + numeric hostels.id
        |                              |
        |                              `-- bookings, favorites, reviews, rooms, complaints
        v
app_api.py ranks HST-* IDs -> index.js hydrates live PostgreSQL records
```

- PostgreSQL numeric `hostels.id` is the operational relational key.
- ML `HST-*` IDs are stored as `hostels.external_id` solely to map ML results to operational database records.
- For imported catalog rows, live capacity must remain `source_available_capacity - active pending/confirmed bookings`. See `AI_HANDOFF.md` for the complete invariant.

## Generated, Local, And Sensitive Files

- `build/`, Flutter `ephemeral/` folders, generated plugin registrants, `.dart_tool/` if present, and Python `__pycache__/` folders are generated. Do not document or edit every generated child as source.
- `.env` files are local runtime configuration and must not be committed or shared with secrets.
- `google-services.json` and generated Firebase configuration are environment/project configuration; handle them according to your repository's secret policy.
- Model `.pkl`, `.npy`, and large CSV files are data/artifacts. Their schemas and compatibility matter more than their binary internals.

## Safe Starting Points For Another AI

1. Read `STAYBUDDY-main/AI_HANDOFF.md` before changing backend data, APIs, catalog imports, or tests.
2. Read this file before moving files or assuming which application is active.
3. Treat `STAYBUDDY-main/index.js` plus its migrations/scripts/tests as the operational backend path.
4. Treat the room matcher and `ai-complaint-system` as disconnected prototypes unless explicitly consolidating them.
5. Do not rerun `STAYBUDDY-main/staybuddy_schema.sql` on the active database.
6. Do not edit generated Flutter/platform build output.