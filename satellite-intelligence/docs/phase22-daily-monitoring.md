# Phase 22 — Automated Daily Satellite Monitoring

## Scope and result

This phase adds an **opt-in** daily Sentinel-2 L2A monitor. It uses the existing Copernicus Data Space Ecosystem (CDSE) STAC integration for discovery and the existing authenticated CDSE acquisition and approved-model inference adapters for processing. The monitoring job itself never trains a model, uses paid APIs, or invents outputs. Monitoring defaults to disabled; deployment must explicitly configure the AOI, database, and CDSE credentials.

Phase verification results are recorded at the end of this document after the checks run. A mocked acquisition or inference test is not a real catalog query, download, or model result.

## Architecture and lifecycle

1. The FastAPI lifespan starts the in-process scheduler only when monitoring is enabled, persistent database tables are available, and its settings are valid. Shutdown stops the scheduler and waits for active scan work to finish; each upstream HTTP operation already has a bounded timeout.
2. A database singleton row stores the enabled scan state, configured timezone and local daily time, last attempt, last successful catalog scan, next run, and an expiring 60-minute scan lease. Every API worker/replica uses an atomic conditional database update to acquire the lease before scanning. Workers without the lease do no work. Expired leases allow restart recovery.
3. The scheduler checks the persisted due time every 20 seconds. The default schedule is 03:00 in `Asia/Kolkata`. If startup occurs after a persisted run was due, the first eligible worker recovers by running the overdue scan. A manual scan uses the same lease.
4. Each scan queries a configurable rolling date window, AOI, cloud threshold, and at most 10 catalog pages. Repeated STAC items are keyed by stable item ID and do not replace first-seen metadata. Observation time, optional catalog publication/creation time, and first discovery time are separate fields.
5. Only supported Sentinel-2 L2A items with the required 10 m bands and valid CDSE product identifiers are eligible for acquisition. Acquisition downloads B02, B03, B04, and B08 through the existing official CDSE product path, stages a temporary raster, validates raster metadata and pixel coverage, hashes it, and atomically promotes it to the external data directory. Supported SHA-256 STAC checksums are verified by the acquisition adapter when present; output rasters always receive a locally calculated SHA-256. Other STAC checksum encodings are retained as metadata but are not interpreted as SHA-256.
6. Inference calls the existing model-readiness and inference adapters. Only an approved model manifest with a matching artifact checksum can run. If no model is ready, the validated imagery is preserved and the scene remains `model_not_ready`; no prediction artifact or analyzed status is created. The monitor never trains a model.
7. Scene states include `discovered`, `rejected`, `downloading`, `validating`, `analyzing`, `validated`, `model_not_ready`, `analyzed`, and `failed`. Failures store bounded attempts, retry timing, and a sanitized error. Scene output/artifact keys are internal; API responses never disclose absolute filesystem paths.

## Data source and terms

- **Catalog:** CDSE STAC API, `https://stac.dataspace.copernicus.eu/v1/search`, collection `sentinel-2-l2a`. Catalog metadata search is public; there is no paid satellite API.
- **Imagery:** Sentinel-2 Level-2A, acquired only through the project's existing CDSE authenticated OData/S3 acquisition adapter. A CDSE account is required for the configured acquisition path.
- **Terms/attribution:** Sentinel data use is subject to the [Copernicus Data Space terms](https://dataspace.copernicus.eu/terms-and-conditions) and [Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice). The catalog source and collection are stored with each scene. Attribute Copernicus Sentinel data and CDSE when redistributing or presenting source-derived data, consistent with current legal terms.
- **Products:** scene records retain the stable STAC item ID, collection, provider, band asset keys/URLs and available sizes/checksums, acquisition/observation time, optional publication time, cloud-cover metadata, and catalog geometry/bounds. Cloud cover is scene-level metadata, not a pixel cloud mask.
- ESA WorldCover is not used by the daily monitor. The Phase 21 training labels remain weak supervision and are not independent ground truth.

Terms and API behavior can change; review the linked official terms and source metadata before expanding use.

## Configuration

Set variables in the backend environment only. Start from `backend/.env.example`; do not check credentials into files or frontend variables.

| Variable | Default | Meaning |
|---|---|---|
| `SATELLITE_MONITOR_ENABLED` | `false` | Explicit opt-in. No scheduler or acquisitions run when false. |
| `SATELLITE_MONITOR_TIMEZONE` | `Asia/Kolkata` | IANA timezone used for the persistent daily schedule. |
| `SATELLITE_MONITOR_SCHEDULE_TIME` | `03:00` | Local time in strict `HH:MM` 24-hour format. |
| `SATELLITE_MONITOR_AOI` | unset | GeoJSON `Polygon`, `MultiPolygon`, or GeoJSON `Feature` geometry. Must fit `MAX_SATELLITE_ACQUISITION_AREA_KM2`. |
| `SATELLITE_MONITOR_LOOKBACK_DAYS` | `14` | Rolling observation-date search window (1–90 days); historical pilot dates are not hardcoded. |
| `SATELLITE_MONITOR_MAX_CLOUD_COVER` | `20` | Maximum STAC scene-level cloud percentage, 0–100. |
| `SATELLITE_MONITOR_MAX_SCENES` | `3` | Maximum newly discovered/processed scenes per run, 1–10. |
| `SATELLITE_MONITOR_MAX_RETRIES` | `2` | Maximum failed scene attempts (bounded retry/backoff); source-client retries are also bounded. |
| `SATELLITE_MONITOR_MAX_SCENE_BYTES` | `536870912` | Per-scene source/output byte ceiling. |
| `SATELLITE_MONITOR_MAX_STORAGE_BYTES` | `5368709120` | Total recorded imagery plus inference-artifact storage ceiling. The pipeline stops acquiring rather than deleting existing data. |
| `SATELLITE_MONITOR_DATA_DIR` | `%LOCALAPPDATA%\satellite-vision\monitoring` on Windows; `$XDG_DATA_HOME/satellite-vision/monitoring` or `~/.local/share/satellite-vision/monitoring` elsewhere | Local raw/processed scene and artifact files. Must resolve outside the project source tree. |
| `DATABASE_URL` | unset | Existing application metadata database; persistent monitor tables are part of Alembic revision `0005_satellite_monitoring`. |
| `CDSE_USERNAME`, `CDSE_PASSWORD` | unset | Backend-only credentials required for imagery acquisition. Never included in API responses/logs. |

Settings are validated at runtime when monitoring is enabled. The AOI and secrets are not returned by the status API.

## Persistence, storage, and migrations

Revision `0005_satellite_monitoring` adds `satellite_monitor_state` and `monitored_satellite_scenes` with indexes for observation time and scene status. The migration is additive and intentionally has no destructive downgrade. **This phase did not run any migration against production.** The repository's existing Render start command runs `alembic upgrade head` on application startup; therefore an operator must review and apply the new revision through the existing deployment process only after the normal backup/release approvals.

The default data directory is outside the repository. Each scene uses a SHA-256-derived directory name, not a catalog-supplied path. Downloaded source rasters and real inference outputs are retained until an operator manages the configured storage volume. When the byte ceiling is reached, new acquisition is refused; the monitor does not silently delete imagery. Configure a persistent disk/volume if files must survive process or host replacement. The SQL database stores metadata and relative artifact keys, not the imagery bytes.

## API and access control

All monitoring routes require an authenticated application user:

| Method and route | Purpose |
|---|---|
| `GET /api/monitoring/status` | Enabled state, last/next scan times, configured timezone, model readiness, and scene counts. |
| `GET /api/monitoring/scenes?limit=20&offset=0` | Bounded recent scene history (limit 1–100). |
| `GET /api/monitoring/scenes/{scene_id}` | Scene metadata, processing/retry state, model metadata, and real artifact links when present. |
| `POST /api/monitoring/scan` | Queue a manual run; returns 202, 409 if another scan owns the lease, or 503 if prerequisites are absent. |
| `GET /api/monitoring/scenes/{scene_id}/artifacts/{classification.tif\|confidence.tif}` | Authenticated download of a real persisted inference artifact only. |

The existing Supabase request interceptor supplies the user's bearer token. Errors do not return credentials or unrestricted local paths. Any authenticated account can view the deployment-wide monitoring history; this feature does not define a separate administrator role.

## Dashboard behavior

Signed-in users see the Daily satellite monitoring panel with enabled/disabled state, scan schedule and timestamps, source/model readiness, scene counts, scene-level observation/publication/discovery/processing times, bounds, cloud metadata, retries, failures, and artifact download buttons only after actual artifacts exist. A `model_not_ready` state is not labeled as analyzed. The UI polls persisted status every minute; it does not generate imagery or predictions.

## Run locally on Windows

1. Use the project's existing Python environment and set a local PostgreSQL/SQLite `DATABASE_URL` plus the monitor AOI and backend-only CDSE credentials. Do not use production credentials/data for local experiments.
2. From `backend`, apply the schema only to that local development database:

   ```powershell
   & 'E:\project\Hackathon Erode\.venv\Scripts\python.exe' -m alembic -c alembic.ini upgrade head
   ```

3. Set `SATELLITE_MONITOR_ENABLED=true` after the database and other settings are valid, then start the backend using the project's usual Uvicorn command:

   ```powershell
   & 'E:\project\Hackathon Erode\.venv\Scripts\python.exe' -m uvicorn --app-dir . main:app --host 127.0.0.1 --port 8000
   ```

4. Sign in through the frontend and inspect the Monitoring panel or call the authenticated manual-scan endpoint. The scheduler waits for the next persisted due time; it does not immediately fetch data merely because the app starts.

Do not run the migration command against a production URL during development. If no database, credentials, valid AOI, or approved model are configured, the UI/API reports the corresponding disabled/readiness state.

## Deployment and scheduling limits

The scheduler is an in-process FastAPI task coordinated by a persistent database lease. It is not a durable external job runner: it cannot execute while the service is stopped, sleeping, or without a reachable database, and free web-service plans may sleep or restart. A missed run is recovered at startup only if the service starts again and the database/lease are usable. The lease prevents routine concurrent replicas from scanning simultaneously; an interrupted running process may leave its lease until expiration.

For a sleeping host, a free external scheduler such as GitHub Actions `schedule` can be considered, but the existing manual endpoint requires a valid authenticated Supabase bearer session and this repository does not implement a service-to-service/OIDC credential flow. Do not store a long-lived user access token or service-role key in a workflow as a workaround. To automate safely from an external scheduler, first provision a short-lived, least-privilege service identity/token flow or run the monitor on a continuously available trusted host. No external scheduler has been configured or tested in this phase.

## Troubleshooting and recovery

- `disabled`: confirm explicit enablement, valid AOI, database connectivity/schema, CDSE credentials, storage limits/path, schedule timezone, and AOI area.
- Startup logs `monitor.schema_unavailable`: apply Alembic `0005_satellite_monitoring` to the intended database after review; production schema changes are outside this phase execution.
- `running` appears stale: a worker may still own the lease. The lease expires after 60 minutes; do not manually edit the database while a worker may be active.
- `rejected`: inspect the sanitized per-scene error and catalog asset support. Unsupported item IDs/bands are not sent to acquisition.
- `failed`: inspect error and retry timestamps. Attempts are bounded; a validated source remains available for later approved-model inference.
- `model_not_ready`: install/configure an approved compatible model manifest and checksum using the existing model process. The monitor does not download or train weights.
- Storage limit reached: back up and manage the external data volume through an approved retention process; the monitor intentionally does not delete data.
- Catalog or CDSE outages/rate limits: scans record failures and retry within configured bounds; the next scheduled run can recover.

## Local pilot model training

As a separate, explicitly invoked follow-up to this monitoring implementation, the existing CPU training CLI was run against the verified Phase 21 pilot. The daily scheduler does not invoke training. Dataset/source hashes, all 48 NPZ sample hashes, chip contracts, class summaries, region-to-split assignment, and raster alignment were checked before training. The three 512 x 512 source/label pairs use EPSG:32643, 10 m pixels, and B02/B03/B04/B08 imagery aligned to single-band WorldCover labels. Every pair had 262,144 valid joint pixels; the prepared splits each contain 16 chips/262,144 valid pixels.

The train region was `tamilnadu-sathyamangalam`, validation was `tamilnadu-mettur`, and held-out test was `tamilnadu-erode`. The manifest's projected windows have a recalculated minimum edge-to-edge gap of 44,879.28 m (required: 10,000 m), so this is a geographic holdout by region/window, not a random chip split. All acquisitions share the date `2021-12-24`; Erode (test) and Sathyamangalam (train) also cite the same Sentinel-2 product ID `S2B_43PGN_20211224_0_L2A`. Their windows do not overlap, but sharing the acquisition/product means this is not an acquisition-independent evaluation.

Reproducible command, run from `backend` with the configured project environment:

```powershell
& 'E:\project\Hackathon Erode\.venv\Scripts\python.exe' -m geoai.train_landcover `
  'E:\satellite-vision-data\phase21-pilot-20261011-final\prepared' `
  'E:\satellite-vision-data\trained-models\phase22-pilot-20261011' `
  --seed 42 --max-train-pixels 50000 --max-pixels-per-class 5000 `
  --max-iterations 100 --learning-rate 0.1 --max-leaf-nodes 15 `
  --min-samples-leaf 20 --l2-regularization 1.0
```

The run used 22,321 selected training pixels from 262,144 train-region valid pixels, with per-class selected counts 10: 5,000; 20: 265; 30: 5,000; 40: 5,000; 50: 5,000; 60: 788; 80: 1,268. Only these seven classes occurred in the train region. Classes 90, 70, 95, and 100 had no train support; class 90 occurs in the held-out test region without being represented in training.

Measured metrics from the saved report are **agreement with ESA WorldCover 2021 v200 weak labels**, not independent scientific accuracy:

| Split | Valid pixels | Overall agreement | Mean IoU (present classes) |
|---|---:|---:|---:|
| Validation (`tamilnadu-mettur`) | 262,144 | 0.5518 | 0.2587 |
| Held-out test (`tamilnadu-erode`) | 262,144 | 0.5869 | 0.1542 |

Held-out test per-class weak-label metrics:

| WorldCover code | Class | Support | Precision | Recall | F1 | IoU |
|---:|---|---:|---:|---:|---:|---:|
| 10 | Tree cover | 32,055 | 0.9640 | 0.1346 | 0.2363 | 0.1340 |
| 20 | Shrubland | 56 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30 | Grassland | 10,231 | 0.0971 | 0.1035 | 0.1002 | 0.0527 |
| 40 | Cropland | 15,691 | 0.0729 | 0.0054 | 0.0101 | 0.0051 |
| 50 | Built-up | 202,773 | 0.7920 | 0.7278 | 0.7586 | 0.6110 |
| 60 | Bare/sparse vegetation | 854 | 0.0088 | 0.5878 | 0.0174 | 0.0088 |
| 80 | Permanent water | 431 | 0.4946 | 0.7425 | 0.5937 | 0.4222 |
| 90 | Herbaceous wetland | 53 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 70, 95, 100 | No held-out support | 0 each | — | — | — | — |

The low held-out mean IoU and the single-region training split do not justify scientific, operational, or production use. The artifact is outside Git at `E:\satellite-vision-data\trained-models\phase22-pilot-20261011\sentinel2-landcover.joblib`, SHA-256 `18b36c762d31851fc698d4b5a3fa4e3d2e0cdb01f1eb99c979eed37a4ccf47f3`. Its manifest has `deployment_approved: false`. The local model registry reports `model_not_ready`; an explicit inference attempt raises `ModelNotReadyError` before creating outputs. Approval was not changed or bypassed. The report, confusion matrix, and manifest are beside the artifact; no model artifact is in the repository.

WorldCover is weak supervision derived from a map product and is not independent ground truth. The available pilot contains only three nearby Tamil Nadu windows from one acquisition date, shares a Sentinel product ID across two spatially distinct split regions, has severe class imbalance, and has very little support for rare classes. An independent reference dataset and additional dates/regions are required before any scientific or deployment assessment.

## Verification record

Commands are run from the project directories, with the existing virtual environment and installed packages. No real scene acquisition or model inference is initiated by the tests.

| Check | Result |
|---|---|
| Focused backend monitor and API tests | `python -m pytest -q tests/test_satellite_monitoring.py tests/test_monitoring_api.py`: 17 passed; one Starlette/httpx deprecation warning. The final full run below also covers these tests after the signature cleanup. |
| Full backend test suite | `python -m pytest -q`: 247 passed, 4 warnings (Starlette/httpx deprecation, two Alembic path-separator deprecations, and one Rasterio not-georeferenced fixture warning). |
| Python compilation and affected backend imports | `python -m compileall -q .` and imports of `main`, API routes, persistence models, monitoring, catalog, and acquisition completed successfully. |
| Dependency consistency | `python -m pip check`: no broken requirements found. |
| Alembic history and heads | Revision history is a single chain through `0005_satellite_monitoring`, which is the sole head. The migration was applied only to a newly created disposable SQLite database under `%TEMP%`; no production database was configured or modified. |
| Full frontend tests | `npm test -- --pool=threads --maxWorkers=1`: 13 files and 75 tests passed. |
| Frontend production build | `npm run build -- --outDir <unique-temp-directory>` succeeded (815 modules); the external temporary build directory was removed afterward. |
| Bounded live catalog query | CDSE STAC `sentinel-2-l2a`, 14-day window, one-scene limit and the monitor's 20% cloud threshold: completed without provider error; 1 result, `S2B_MSIL2A_20260929T050649_N0513_R019_T43PHN_20260929T091314`, observed 2026-09-29 at 05:06:49 UTC, cloud cover 15.73%, no next page. No assets were downloaded by the monitor/audit. |
| Pilot data validation and training | Pilot validation passed: 3 regions, 48 chips, 786,432 valid pixels total, checksums/alignment/class codes/nodata verified. Deterministic CPU training completed; see the metrics and limitations above. |
| Trained model loading / inference safeguard | Manifest and artifact SHA-256 matched. Registry reported `model_not_ready` because deployment approval is false; live inference was blocked with no output files created. Existing inference tests exercise the approved-model prediction/confidence path using test fixtures. |
| Local HTTP smoke test | Backend `/api/health` returned 200; `/api/ready` returned 503 (`configuration` and `storage` unavailable; local disposable DB/schema passed). `/api/auth/status` returned 200 with Supabase auth unconfigured; protected monitoring status returned 401 without a valid session. Vite returned 200 and rendered sign-in, but dashboard/monitoring could not be opened without configured Supabase authentication. CORS was enabled only for the alternate local test origin. |
| Pylance diagnostics | The latest workspace diagnostic reported 228 diagnostics across 25 Python files: 65 severity-1 errors and 163 severity-4 hints. Targeted checks found no diagnostics in `monitoring.py`, `satellite/acquisition.py`, `train_landcover.py`, or `inference.py`. Raster windows use `Window.from_slices` and AOI reprojection uses the typed coordinate-array callback pattern already used in monitoring. Two remaining diagnostics do not match current on-disk source: the `training_data.py` report points at line 227's `index_path` assignment while saying “Expected 0 positional arguments,” and the `acquire_pilot.py` report rejects `geographic_independence_evidence` although that keyword is present in the current `prepare_training_dataset` signature and is accepted by the completed training run. They remain unresolved editor diagnostics; other workspace errors are concentrated in test files and other project modules. |
| Diff and artifact checks | `git diff --check` passed. No tracked raster or model-weight artifacts were found. Secret-pattern scan matches were limited to test fixtures/environment lookups; no credential value was added to Phase 22 production code or documentation. |
| Real imagery download during Phase 22 | Not performed. Phase 21 pilot Sentinel-2/WorldCover rasters existed outside Git and were checksum/alignment validated for the training run. |
| Approved model availability and real inference | The newly trained pilot model is intentionally unapproved; production inference did not run. |
| Git/production actions | No files staged, committed, pushed, or deployed. No production migration or data change was made. |

The final code review corrected monitoring-module typing issues by explicitly narrowing optional timestamps/AOI values, using typed `RETURNING` results for lease updates, and transforming AOI coordinate arrays through a typed Shapely callback. The unused `manual` lease argument was removed after confirming all resolved call sites. Focused Pylance diagnostics and signature-compatibility checks passed for the changed functions.

## Final operational status and limitations

- Scheduling is **not operational in the current deployment as verified here**: this phase was not deployed, the feature defaults to `SATELLITE_MONITOR_ENABLED=false`, and the required `0005_satellite_monitoring` schema migration was not applied by this audit. The authoritative production environment values were not inspected. A service that sleeps or is stopped cannot run the in-process scheduler.
- The public catalog query reached CDSE successfully and returned one matching scene for the bounded test AOI/date window. The query was capped at one item and did not download any assets; it therefore does not demonstrate imagery acquisition or validation.
- The tests used local fixtures/mocks for scene acquisition, scheduler coordination, and model-readiness behavior. The live catalog match was not downloaded, so no new monitor acquisition was performed.
- A CPU model was trained from the existing Phase 21 pilot, but its `deployment_approved` flag remains false. No production inference occurred, and this is not an independently validated land-cover model.
- Workspace Pylance is not clean: 65 severity-1 and 163 severity-4 diagnostics remain across 25 files. The two reports in the training-data path described above could not be reconciled with current source through the available Pylance document state.

## Local startup and immediate next step

The audit started the API at `http://127.0.0.1:8012` and Vite at `http://127.0.0.1:5178`, using the disposable SQLite DB and external model directory listed above. The services responded during the audit. Ports 8000 and 5173 were already occupied and were left untouched. To repeat with the same local settings, start separate PowerShell windows:

```powershell
# Backend, from satellite-intelligence\backend
$env:DATABASE_URL = 'sqlite:///C:/Users/T.sakthivel/AppData/Local/Temp/satvision-phase22-localdb-9038a496bf444d768740295b9de465ae/monitor.sqlite'
$env:SATELLITE_MONITOR_ENABLED = 'false'
$env:AI_MODEL_DIR = 'E:\satellite-vision-data\trained-models\phase22-pilot-20261011'
$env:AI_LANDCOVER_MODEL_MANIFEST = 'E:\satellite-vision-data\trained-models\phase22-pilot-20261011\sentinel2-landcover.json'
$env:CORS_ORIGINS = 'http://127.0.0.1:5178,http://localhost:5178'
& 'E:\project\Hackathon Erode\.venv\Scripts\python.exe' -m uvicorn main:app --host 127.0.0.1 --port 8012
```

```powershell
# Frontend, from satellite-intelligence\frontend
$env:VITE_API_URL = 'http://127.0.0.1:8012'
npm run dev -- --host 127.0.0.1 --port 5178
```

For normal local development, use the README's `uvicorn main:app --reload` and `npm run dev` commands with their default ports; provide local Supabase auth settings to access authenticated dashboard/monitor routes. Do not copy production credentials into local files.

**Exact next step:** configure a local Supabase Auth project/session and private artifact storage to verify signed-in dashboard and monitoring endpoints end to end; separately collect additional geographically distinct acquisitions and independent reference labels before seeking model review/approval. Keep `SATELLITE_MONITOR_ENABLED=false` and the model unapproved until those prerequisites and deployment review are complete.
