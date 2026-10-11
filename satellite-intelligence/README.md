# Satellite Intelligence & Land Monitoring

Satellite Vision is a GeoTIFF analysis MVP for interpretable spectral indices, heuristic land-cover indicators, historical NDVI comparison, map overlays, and reports. Its free Copernicus Data Space STAC integration discovers Sentinel-2 catalog metadata; it does not download catalog imagery. Normal analysis never substitutes sample data for user inputs. A separate deterministic synthetic demo is explicitly labelled, kept in memory, and has no real-world georeferencing.

## Problem statement

The project addresses the hackathon challenge, “AI/ML based system for deriving value added parameters using satellite imagery,” by turning multispectral imagery into interpretable indicators and visual summaries. The rule-based land-cover baseline is not a trained AI/ML model or ground truth.

## Features

- Local Sentinel-2-style band-token discovery and validation (B02, B03, B04, B08, B11); filenames/tags are not independently authenticated as satellite provenance
- NDVI, McFeeters NDWI, NDBI, heuristic land-cover classes, and NDVI change detection
- Nodata and invalid numeric handling; required-band CRS, transform, and dimension checks
- GeoTIFF outputs preserving projection and transform, plus web overlays and JSON reports
- Metadata, data-quality, visualization, and model-status endpoints for geospatial inspection
- Public Copernicus Data Space Sentinel-2 Level-2A metadata discovery and authenticated selected-scene acquisition, AOI cropping, and GeoTIFF ingestion for supported Sentinel-2A/B/C items
- React dashboard with georeferenced map layers, statistics, charts, careful deterministic insights, and safe result downloads
- FastAPI health and dataset status endpoints; useful empty-dataset state when imagery is absent
- Per-analysis provenance from raster metadata, including explicit unknowns and quality warnings
- Authenticated scene ingestion for GeoTIFF validation, metadata extraction, and bounded RGB/grayscale previews; original imagery remains unchanged
- Imagery scene records use the existing analysis/job database and private artifact-storage workflow; local/demo mode is labelled local-only
- Clearly isolated synthetic calculation demo; no fabricated observation date, location, or model score
- Scene-bound NDVI, McFeeters NDWI, and neutral-label unsupervised K-means with bounded raster-window processing and private result artifacts
- Advanced GIS workspace for saved scene previews and completed analysis results, with visibility, opacity, ordering, fit-to-layer, coordinate inspection, legends, and private raster downloads
- Owner-scoped, paginated analysis history with actual persisted job status, compact provenance/statistics reports, CSV/JSON summaries, and browser print layout

### Evidence status

| Status | Scope |
|---|---|
| Implemented and verified locally — raster analysis | Raster validation, NDVI/NDWI/NDBI, heuristic land-cover baseline, aligned NDVI change, synthetic demo, GeoTIFF scene ingestion/preview, scene-bound NDVI/McFeeters NDWI/K-means, local metadata and artifact persistence, ownership checks, local API/UI tests, and frontend production build. Imagery and scene-analysis fixtures are synthetic test rasters, not satellite observations. |
| Implemented but not fully verified — cloud integration/deployment | Supabase Auth integration, PostgreSQL migrations/metadata, private Storage artifacts, owner-scoped cloud retrieval, and configured production deployment. Source and mock/local tests exist; live cloud credentials and a matching deployed build were unavailable. |
| Implemented and verified locally — catalog discovery and acquisition workflow | Copernicus STAC search is public and paginated. Selected-scene acquisition is implemented with server-side CDSE credentials, four 10 m assets, bounded streaming, strict owner authentication, aligned GeoTIFF conversion, and existing ingestion/persistence. Automated tests use mocked provider calls; live authenticated imagery acquisition and deployment remain unverified. |
| Planned or incomplete | Automatic reprojection/resampling, cloud masking, independently ground-truth-evaluated and approved ML, and durable asynchronous workers. |

Passing software tests does not establish scientific accuracy or prove live deployment/cloud operation. See the [final-submission report](docs/final_submission/Satellite_Vision_Project_Report.docx) and [production verification status](#production-verification-status).

## Architecture

`React dashboard → FastAPI → user-supplied GeoTIFF → bounded validation/preview → private cloud artifacts + PostgreSQL metadata (or explicit local-only mode)`

`LocalDataProvider → pre-downloaded band GeoTIFFs → raster analysis → GeoTIFF / PNG / JSON outputs`

`LocalDataProvider` remains the operational raster-analysis path. The Copernicus Data Space STAC catalog supports public Sentinel-2 Level-2A metadata search through a separate catalog route; it is distinct from the legacy provider search/download endpoints. A selected STAC item can now be acquired through a separate authenticated route; its four 10 m bands are converted to a masked GeoTIFF and passed into the existing owner-scoped imagery ingestion workflow. This does not replace other existing user-supplied GeoTIFF workflows. See the [evidence-based architecture and workflow diagrams](docs/final_submission/Satellite_Vision_Architecture.md).

### Free Sentinel-2 catalog discovery

In **Satellite data mode → Live Satellite**, search the official Copernicus Data Space Ecosystem (CDSE) STAC catalog for Sentinel-2 Level-2A items using a GeoJSON Polygon or MultiPolygon AOI, inclusive start/end dates, and maximum cloud-cover percentage. Search results are capped at 20 items per page and paginated with an opaque catalog cursor. Results include the stable STAC item ID, collection, acquisition time (UTC), cloud percentage, platform, product type/processing level, footprint/bounds when supplied, provider attribution, the official catalog item link, and spectral assets listed by the catalog with resolution and source size when available. Asset listings are metadata only; the backend revalidates asset availability against the official product tree during acquisition. Missing source metadata remains unavailable rather than being inferred.

The backend uses `https://stac.dataspace.copernicus.eu/v1/search` and fixes the collection to `sentinel-2-l2a`. The catalog search is public and requires no CDSE account or credentials; access to the app's API follows its configured Supabase authentication requirements. No catalog credentials or new dependencies are required. AOIs are validated and bounded (maximum 500 coordinates, existing 30-degree extent guard, and 1,000,000 km² area limit), and requests are sent only to the fixed official HTTPS endpoint with bounded timeouts and a 50-item hard result cap.

Selecting a result retains its provider, collection, stable item identifier, catalog reference, and search AOI. **Download and ingest selected scene** requests the complete B02, B03, B04, and B08 JP2 assets at 10 m from the official CDSE OData product tree, combines them on their common grid, masks nodata and pixels outside the searched AOI, and writes the cropped GeoTIFF through the existing owner-scoped ingestion workflow. The source asset downloads are full tile files, not byte-range AOI extracts; their catalog-reported sizes are shown before acquisition. The route re-fetches the fixed official STAC item, validates its ID, collection, product and asset references, verifies each OData file node, enforces official HTTPS hosts, bounded redirects/timeouts and retries, one in-process acquisition at a time, the minimum configured upload/raster/artifact byte cap, `MAX_IMAGERY_PIXELS`, and `MAX_SATELLITE_ACQUISITION_AREA_KM2` (default 1,000 km²). It does not download the complete SAFE product, apply a cloud mask, or claim that source metadata is independently verified.

Acquisition requires an authenticated application user and CDSE account credentials configured only on the backend as `CDSE_USERNAME` and `CDSE_PASSWORD`. The backend uses CDSE's documented password-grant token endpoint; credentials and access tokens are never returned to the browser. Without these backend settings, metadata discovery remains available but acquisition returns a configuration error. `MAX_UPLOAD_BYTES`, `MAX_RASTER_BYTES`, `MAX_ARTIFACT_BYTES`, and `MAX_IMAGERY_PIXELS` bound transfer and raster handling; `MAX_SATELLITE_ACQUISITION_AREA_KM2` bounds the selected AOI. Use the existing GeoTIFF upload workflow for other imagery. Successful acquisitions appear in saved imagery scenes and can be opened in the existing scene-analysis and GIS workspace. Automated provider calls are mocked; live authenticated imagery download and production deployment are not asserted.

**Catalog search is not an imagery download.** A result row is provider metadata only. Imagery is fetched only after an authenticated user selects a supported scene and explicitly starts **Download and ingest selected scene**; success is reported only after all four official assets are downloaded, validated, cropped, and ingested. The latest isolated local check returned one public STAC metadata result, but no imagery assets were downloaded. CDSE credentials are not configured, so real authenticated acquisition remains blocked; with an isolated test user, the API correctly reported the missing backend credentials rather than claiming success. See the [compatibility and verification report](docs/free-data-and-model-compatibility.md) for exact HTTP results and blockers.

#### Configure official CDSE access locally

Create a free account using the [CDSE registration guide](https://documentation.dataspace.copernicus.eu/Registration.html). The backend follows the [official CDSE access-token flow](https://documentation.dataspace.copernicus.eu/APIs/Token.html): it posts the account username and password to the CDSE identity endpoint using the public `cdse-public` OAuth client, then uses the short-lived access token for OData product downloads. STAC discovery remains public. This project does not need a client secret or paid API.

From `satellite-intelligence/backend`, create a private local environment file once, enter the credentials in an editor, and start the backend with Uvicorn's environment-file support:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
uvicorn main:app --env-file .env --host 127.0.0.1 --port 8000 --reload
```

Set only `CDSE_USERNAME` and `CDSE_PASSWORD` in that backend `.env`; leave them empty to keep acquisition unavailable. The repository ignores `.env` files. Never paste credentials into chat, source code, frontend `.env*` files, `VITE_*` variables, logs, or Git. The current password-grant adapter does not submit a changing TOTP code; accounts that require TOTP need an adapter extension before acquisition can authenticate. A failed/missing credential is an acquisition error, not a successful download.

### Opt-in daily monitoring

An authenticated daily monitor can discover a rolling window of Sentinel-2 L2A scenes, acquire and validate the four 10 m input bands, and run only a locally approved checksum-verified land-cover model. It is **disabled by default** and requires a database migration, configured AOI, CDSE backend credentials, and storage outside the source tree. Scheduler execution depends on the API process remaining awake; no scans run while a hosting service is stopped or sleeping. See [Phase 22 daily monitoring](docs/phase22-daily-monitoring.md) for configuration, API details, migration, storage, and deployment limitations.

For a local opt-in run, set `SATELLITE_MONITOR_ENABLED=true`, a GeoJSON `SATELLITE_MONITOR_AOI`, `CDSE_USERNAME`/`CDSE_PASSWORD`, and a persistent local `DATABASE_URL` in the backend `.env`; then run `python -m alembic -c alembic.ini upgrade head` against that local database before starting the API. Leave the setting `false` for ordinary startup. Do not use a production database URL for this local migration.

## Imagery ingestion and open data sources

In the dashboard's **Local Dataset** section, upload one georeferenced `.tif`/`.tiff` scene to inspect it. The API enforces the configured upload/file limits and `MAX_IMAGERY_PIXELS` (default 200,000,000 pixels), requires a CRS and valid sampled preview pixels, and returns its source-derived date (when present), WGS84 extent, CRS, dimensions, resolution, and band descriptions. Multiband scenes with identifiable B04/B03/B02 descriptions get a band-labelled RGB preview; other multiband files use bands 1–3 with their semantics explicitly unverified; single-band files get a grayscale preview. The bounded display preview is reprojected onto an EPSG:4326 grid and includes its exact WGS84 bounds for map placement; the source GeoTIFF is retained unchanged. Cloud masking is not applied, and unknown metadata stays unknown.

The existing ingestion endpoint accepts GeoTIFF only; selected-scene acquisition validates its four Sentinel-2 JP2 assets, crops their combined raster to the submitted AOI, and converts them to GeoTIFF before reusing that endpoint. It does not accept arbitrary Sentinel SAFE/JP2 archives directly. In deployment, imagery uses the existing private Supabase Storage bucket and owner-scoped artifact endpoints when the cloud persistence configuration is active. A no-cloud local/demo configuration is explicitly marked `local-only` and is not durable across ephemeral backend restarts.

### Scene analysis

Open a saved scene in the dashboard's **Inspect a GeoTIFF scene** panel, then choose **Analyze this scene**. The authenticated backend exposes `GET /api/imagery/analysis-types`, `POST /api/imagery/scenes/{scene_id}/analyses`, and `GET /api/imagery/analyses/{analysis_id}`. Scene and result lookups remain owner-scoped, and output downloads use the existing private artifact endpoint; no permanent public URL is created.

- **NDVI** uses `(NIR - Red) / (NIR + Red)` and **McFeeters NDWI** uses `(Green - NIR) / (Green + NIR)`. For Sentinel-2-style inputs the metadata labels `B04`, `B08`, and `B03` are recognized as possible red, near-infrared, and green bands. Metadata labels are not independently verified. If a role is missing or ambiguous, select its source band explicitly. Per-band scale and offset are applied before calculation; nodata, non-finite values, and zero denominators are excluded.
- Index GeoTIFFs preserve the source CRS, transform, dimensions, and nodata locations. Each result includes valid-pixel min/max/mean and an estimated median: histogram-based within [-1, 1] (maximum bin width 0.0004883), with a deterministic bounded sample used when the median lies outside that range. The NDVI legend is illustrative only and is not a universal biome- or sensor-independent classification; the NDWI is the green/NIR water-related formulation, not the Gao NIR/SWIR index.
- **K-means** is a deterministic, standardized `MiniBatchKMeans` baseline over 2–16 explicitly selected feature bands, with 2–10 clusters and a bounded 50,000-pixel deterministic training sample. Pixel windows are processed in 256 × 256 blocks. Output labels remain `Class 1`, `Class 2`, and so on: clusters are spectral groups, not verified land-cover types or evidence of a trained/accurate semantic model.
- Results are georeferenced GeoTIFFs plus a small EPSG:4326 preview, summary statistics/class proportions, and report metadata, stored through the existing analysis/job and artifact persistence flow. Analysis respects `MAX_IMAGERY_PIXELS`, raster/artifact size limits, and `MAX_ANALYSIS_SECONDS`. There is no database schema migration or new runtime dependency for this workflow.

### Optional local AI inference

The **AI land-cover inference** panel is a separate, optional trained-classifier path for an owner-scoped saved scene. `GET /api/imagery/ai-models` reports readiness; `POST /api/imagery/scenes/{scene_id}/ai-inference` runs a ready model; `GET /api/imagery/ai-inference/{analysis_id}` retrieves its owner-scoped result. Inference is synchronous, CPU-based, bounded by the existing analysis slot/deadline and `MAX_AI_INFERENCE_PIXELS` (default 10,000,000), and produces private class-index and maximum-class-probability GeoTIFFs. It does not alter the source image or replace NDVI, NDWI, or K-means.

No pretrained model or weights are included, fetched, or loaded at startup. Without an operator-configured model the registry returns `model_not_ready`, and inference is disabled. Configure `AI_MODEL_DIR` to a private backend directory and `AI_LANDCOVER_MODEL_MANIFEST` to a manifest file inside that directory. `MAX_AI_MODEL_BYTES` (default 268,435,456) limits the model artifact size. The manifest must describe an already-trained scikit-learn-compatible classifier whose fitted `predict_proba` accepts four features, and its band order must be exactly B02, B03, B04, B08:

```json
{
  "name": "Operator-supplied classifier",
  "version": "1.0",
  "license": "The actual model and weights license",
  "source": "The actual model or training source",
  "attribution": "Required model and dataset attribution",
  "artifact": "classifier.joblib",
  "bands": ["B02", "B03", "B04", "B08"],
  "normalization": "divide_by_10000"
}
```

`normalization` must be either `stored_values` or `divide_by_10000`; it is applied exactly as declared. Inputs must be georeferenced GeoTIFFs with unique descriptions/tags identifying each required band. No automatic band resampling, atmospheric correction, or normalization discovery is performed. Class labels are emitted as numeric class indices and reported alongside their model-provided labels. Confidence is the model's maximum class probability, not a calibrated accuracy or scientific validation score. Output grids retain the source dimensions, CRS, transform, and valid-pixel mask.

The runtime library scikit-learn is BSD-3-Clause licensed; that license does not license model weights or training data. Operators are responsible for verifying and documenting the supplied model's actual source, version, license, attribution, Sentinel-2 input compatibility, and training/validation provenance. **Only load trusted artifacts:** `joblib` deserializes pickle-based files and loading an untrusted or tampered artifact can execute code in the backend process. Keep the model directory private and outside frontend/static paths; never accept model files from users or expose them through the API.

The compatible project pilot is a locally trained scikit-learn `HistGradientBoostingClassifier` pixel classifier for B02/B03/B04/B08 reflectance, not an off-the-shelf pretrained model. Its existing artifact is kept outside Git and its manifest is explicitly unapproved, so the registry and dashboard correctly remain `model_not_ready`; no normal API inference can use it yet. It was evaluated only for agreement with weak WorldCover labels and does not establish independent scientific accuracy. See the [free-data and model compatibility report](docs/free-data-and-model-compatibility.md) for the measured pilot limits and remaining approval blockers.

### Advanced GIS workspace

Open **Map layers and scene information** below the imagery upload/analysis controls. Choose a saved scene to add its bounded RGB/grayscale preview, or choose a completed/saved analysis result to add its map preview without rerunning processing. The workspace uses the existing owner-scoped result and artifact APIs; preview requests and raster downloads stay authenticated, and no storage URL is exposed to the browser. Previously stored scene metadata without `preview.bounds_wgs84` is not overlaid using an approximate bounding box; re-ingest that source scene to create an accurately aligned preview.

The React-Leaflet map supports pan/zoom, scale, selected-layer and visible-layer fitting, a neutral world-view reset, layer visibility, opacity, order, metadata/legend inspection, retrying a failed preview, and private source/result GeoTIFF downloads. Bounds use `[west, south, east, north]` in EPSG:4326 and are derived from the actual bounded preview grid; the source raster's CRS and dimensions remain separately identified. Clicking the map displays longitude then latitude in WGS84. Pixel-value sampling is not implemented. Map previews are for display and do not replace georeferenced raster downloads. Removing a layer only removes it from the current in-memory map; it does not delete the saved artifact.

OpenStreetMap is the optional default basemap. The map retains visible raster overlays and controls if tiles fail, but OSM tiles require a network connection and must be used with the provider's [attribution and tile usage policy](https://operations.osmfoundation.org/policies/tiles/). No alternative tile provider or screenshot export is included; cross-origin tile licensing and canvas restrictions make a reliable map-image export inappropriate here. An exported PNG preview would not be a georeferenced raster.

### Analysis history and reports

The **History and reports** panel uses the existing paginated results workflow. `GET /api/results` continues to return completed results by default; the dashboard opts into `GET /api/results?include_incomplete=true` to display stored completed, running, and failed records. The query remains owner-scoped, and report details are fetched from the existing authenticated `GET /api/analyses/{analysis_id}` route only when a report is opened. Artifact manifests and downloads use the existing private, owner-authorized artifact routes.

Filter history by persisted status or analysis name, open a report for recorded source identifiers/dates, analysis parameters, available raster metadata, statistics and metrics, and download available result files. Missing values remain explicitly unknown. Completed results can be sent to the GIS workspace without rerunning analysis. CSV and JSON exports contain a compact allowlisted report summary; CSV fields are quoted and spreadsheet formula-like text is neutralized. Use **Print report** for the browser's print/save-as-PDF workflow. No server-generated PDF, migration, or new report dependency is required. Reports distinguish numerical outputs from interpretation: spectral clusters are not validated land-cover classes, change masks are candidate changes rather than confirmed events, and thresholds are not universal.

### Two-scene change detection

In **Detect candidate NDVI change**, choose a dated baseline and a later comparison scene. `POST /api/imagery/change-detection` validates distinct scene IDs, owner access to both scenes, acquisition-date order, sensor/platform metadata when present, and separate red/NIR mappings for each scene. Sentinel-2-style B04/B08 labels can be suggested, but metadata is unverified; select the actual band numbers when labels are absent or uncertain. Only NDVI difference is currently supported.

The method calculates `(comparison-date NDVI) - (baseline-date NDVI)`. The comparison raster is reprojected/resampled onto the baseline raster grid with bilinear resampling of continuous reflectance bands; output CRS, affine transform, dimensions, and nodata grid follow the baseline. Common finite, non-nodata pixels with non-zero NDVI denominators contribute to statistics. The configurable threshold is in absolute NDVI-difference units, and a pixel is a candidate change when `abs(difference) >= threshold`. Positive values mean relative NDVI increased and negative values mean it decreased; neither result confirms a real-world event.

The response includes input dates and band mappings, alignment and nodata policy, valid/excluded and changed/unchanged pixel counts, difference statistics, warnings, and a candidate-change legend. Area is reported only for supported projected CRS units or unrotated geographic grids (using geodesic row pixel areas); otherwise the response explicitly leaves area unavailable. The authenticated `GET /api/imagery/analyses/{analysis_id}` and existing owner-scoped artifact download route retrieve results. Difference and mask GeoTIFFs and an EPSG:4326 map preview are stored with the existing job/artifact persistence; no migration or new database binary field is used. Scene metadata tags and band labels are not independently authenticated, and thresholds are not universal across sensors, seasons, or biomes.

Verified open-data starting points (source access terms can change; review the provider terms before redistribution):

- **Copernicus Sentinel data:** the Copernicus Data Space Ecosystem states Sentinel data are free, full and open, and requires user registration for EO data access. See the [official terms](https://dataspace.copernicus.eu/terms-and-conditions), [registration instructions](https://documentation.dataspace.copernicus.eu/Registration.html), [Sentinel-2 documentation](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html), [OData API](https://documentation.dataspace.copernicus.eu/APIs/OData.html), and [access-token instructions](https://documentation.dataspace.copernicus.eu/APIs/Token.html). The token documentation requires authorization for OData product downloads and documents the `cdse-public` username/password token grant used by the backend; the account credentials are supplied only through backend environment settings. The [Copernicus S3 documentation](https://documentation.dataspace.copernicus.eu/APIs/S3.html) describes a separate authenticated S3-compatible access method. Generated S3 credentials and account passwords are secrets; never put them in the browser.
- **USGS Landsat Collection 2:** USGS operates a public [LandsatLook STAC API](https://landsatlook.usgs.gov/stac-server/) whose live collection `landsat-c2l2-sr` identifies Collection 2 Level-2 Surface Reflectance and links its license to the [Landsat Data Policy](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/Landsat_Data_Policy.pdf). NASA's [Landsat data access overview](https://science.nasa.gov/mission/landsat/data-overview/) describes search/download tools and USGS commercial-cloud access via AWS. STAC provides searchable scene metadata and asset links; this application does not fetch those assets automatically.

These sources provide actual observations; the application's generated test rasters and synthetic demo are not source scenes or evidence of satellite acquisition.

## Technologies

- Backend: Python 3.11+, FastAPI, Uvicorn, NumPy, Rasterio, PyProj, Shapely, Pillow, SQLAlchemy, Alembic
- Frontend: React, Vite, Tailwind CSS, Axios, React Leaflet, Leaflet, Recharts, Supabase JS
- Tests: pytest, FastAPI TestClient, synthetic arrays and temporary GeoTIFFs

## Installation

Use Python 3.11 or newer and Node.js 20 or newer. From the repository root, enter the project backend:

```powershell
cd satellite-intelligence/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` locally if needed; relative paths resolve from the backend working directory and the defaults point at the included `backend/data` structure. Add CDSE account credentials only when you intend to download selected scenes.

## Dataset structure

Place single-band GeoTIFFs directly in these folders (files are not bundled):

```text
backend/data/
├── current/
│   ├── B02.tif
│   ├── B03.tif
│   ├── B04.tif
│   ├── B08.tif
│   └── B11.tif
├── historical/
│   ├── B02.tif
│   ├── B03.tif
│   ├── B04.tif
│   ├── B08.tif
│   └── B11.tif
└── outputs/
```

Filenames may include other text but must contain a recognizable band token such as `B08`. One file per band is used; if duplicates are found, the first alphabetically is selected. Bands in each analysis must have identical dimensions, CRS, and geotransform. The current and historical rasters must also be aligned for change detection; the application reports mismatches rather than silently warping or resampling scientific measurements.

If current imagery is absent, the dashboard says: “No satellite dataset found. Please add Sentinel-2 GeoTIFF bands to backend/data/current/.” NDVI requires B04 and B08; NDWI requires B03 and B08; NDBI requires B08 and B11; land-cover baseline requires B03, B04, B08, and B11. Historical comparison requires aligned B04 and B08 in both periods.

## Start backend

```powershell
Set-Location .\satellite-intelligence\backend
uvicorn main:app --env-file .env --host 127.0.0.1 --port 8000 --reload
```

The API is available at `http://127.0.0.1:8000`; interactive docs at `/docs`. `--env-file .env` is required for Uvicorn to load the local file. The valid ASGI app notation is `main:app`.

## Deploy the API to Render

The repository-root `render.yaml` defines a Python Web Service that builds from the repository root and explicitly targets the backend in `satellite-intelligence/backend`. It installs `satellite-intelligence/backend/requirements.txt` and starts `main:app` with that backend directory on Uvicorn's import path, listening on `0.0.0.0:$PORT`. Python remains pinned to 3.12.11 for the Render service.

For an existing Render service configured in the Dashboard, set **Root Directory** to blank (the repository root), **Build Command** to `pip install -r satellite-intelligence/backend/requirements.txt`, and **Start Command** to require `DATABASE_URL`, run `python -m alembic -c satellite-intelligence/backend/alembic.ini upgrade head`, then start `uvicorn --app-dir satellite-intelligence/backend main:app --host 0.0.0.0 --port $PORT`. Dashboard settings can override a checked-in Blueprint, so update the Dashboard start command to match the Blueprint; startup now stops instead of silently skipping migrations if `DATABASE_URL` is absent, and Uvicorn starts only after migrations succeed. The backend dependencies have also been installed and exercised locally with Python 3.14.3, but this deployment stays on its existing Python 3.12.11 pin.

`render.yaml` sets the Render `CORS_ORIGINS` value to `https://team-s4-ten.vercel.app`, the deployed frontend origin. If the Render Dashboard overrides Blueprint environment values, set that exact origin there as well (no trailing slash). `SATELLITE_PROVIDER` defaults to `local`; CDSE STAC metadata discovery and selected-scene acquisition use separate catalog routes independently of this legacy provider setting. Render's configured health-check path is `/api/ready`.

## Persistent analysis storage

The app remains usable without cloud services: analyses continue to use the existing local GeoTIFF pipeline and local output directory. This no-auth local mode is a shared, public demo, not a private workspace; only synthetic or otherwise non-sensitive data may be used there. Do not upload private imagery. `DATABASE_URL` or any backend `SUPABASE_*` setting automatically makes authentication mandatory, even if `REQUIRE_AUTH=false`, so persisted records and objects cannot silently become anonymous. PostgreSQL metadata/job persistence is enabled when `DATABASE_URL` is configured; Supabase Storage artifacts require both that database and valid Supabase configuration. `cloud_persistence_active` is true only when PostgreSQL is reachable and the private Storage bucket is available.

Configure these backend environment variables in Render (or copy them from `backend/.env.example` for local development):

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection URL; use `postgresql+psycopg://...` (legacy `postgres://` URLs are normalized). |
| `SUPABASE_URL` | Project URL from Supabase Project Settings → API; use HTTPS. Plain HTTP is allowed only for loopback development. |
| `SUPABASE_ANON_KEY` | Supabase public anon key used by the backend to validate signed-in user sessions. |
| `SUPABASE_SERVICE_ROLE_KEY` | Backend-only Supabase service-role key. Store only in the Render backend environment; never put it in Vercel or a `VITE_*` variable. |
| `SUPABASE_STORAGE_BUCKET` | Private bucket name; defaults to `satellite-analysis-results`. Readiness requires the bucket to exist; the first artifact upload can create a missing bucket as private. An existing public bucket is rejected, not silently changed. |
| `CDSE_USERNAME` | Backend-only username for a free [Copernicus Data Space account](https://documentation.dataspace.copernicus.eu/Registration.html); used by the official `cdse-public` password grant for imagery downloads. |
| `CDSE_PASSWORD` | Backend-only account password used to obtain a short-lived token via CDSE's [official token API](https://documentation.dataspace.copernicus.eu/APIs/Token.html). Never configure it in frontend/Vercel variables or return it to the browser. |
| `REQUIRE_AUTH` | Set to `true` in Render to require a valid Supabase Auth bearer session. Authentication is also mandatory whenever `DATABASE_URL` or any backend `SUPABASE_*` setting is present. Authenticated analyses require PostgreSQL metadata and private Supabase Storage to complete. |
| `MAX_ARTIFACT_BYTES` | Maximum generated artifact size in bytes; defaults to `536870912` (512 MiB) per file. |
| `MAX_UPLOAD_BYTES` | Maximum incoming GeoTIFF size in bytes; defaults to `536870912` (512 MiB). |
| `MAX_RASTER_BYTES` | Maximum source GeoTIFF size read by the analysis pipeline; defaults to `536870912` (512 MiB) per file. |
| `MAX_RASTER_PIXELS` | Maximum width × height for any input raster; defaults to `1000000` pixels. |
| `MAX_INPUT_ARRAY_BYTES` | Maximum combined retained float32 band arrays and validity masks per dataset; defaults to `134217728` (128 MiB). This is an input-array budget, not a cap on total process RSS or temporary algorithm arrays. |
| `MAX_ANALYSIS_SECONDS` | Cooperative synchronous analysis deadline; defaults to `300` seconds (allowed range 1–3600). The deadline is checked between raster reads and processing stages; it cannot preempt an individual native raster/NumPy call already in progress. |
| `MAX_CONCURRENT_ANALYSES` | Synchronous analyses admitted per backend process; defaults to `1` (allowed range 1–8). Excess work receives HTTP 503 with `Retry-After: 1`; this is process-local back-pressure, not a durable queue or cross-instance limit. |
| `DATABASE_POOL_SIZE` | PostgreSQL connections retained per process; defaults to `3`. |
| `DATABASE_MAX_OVERFLOW` | Temporary PostgreSQL connections above the pool size; defaults to `1`. Pool plus overflow is capped at 10 per process. |

When `DATABASE_URL` is present, Render's start command applies additive Alembic migrations before starting Uvicorn. The initial migration creates analysis, processing-job, job-status-history, and artifact-reference tables; later revisions add ownership/artifact metadata, idempotency, and the composite index for paginated history. Migrations preserve existing records; their downgrades intentionally refuse destructive schema removal. Local developers can run `alembic -c alembic.ini upgrade head` from `backend/` after setting `DATABASE_URL`.

Analysis requests keep their existing synchronous API behavior and add a `job_id`. Metadata and job records include input parameters, timestamps, status, errors, and artifact references. `GET /api/analyses/{analysis_id}` and `GET /api/jobs/{job_id}` return the persistent records and job status history. Generated GeoTIFFs, PNG maps/charts, and JSON reports are uploaded to the private bucket; PostgreSQL records the bucket, object path, type, size, timestamp, job/analysis relationship, and metadata. The dashboard loads this manifest from `GET /api/results/{result_id}/artifacts` and downloads only registered artifacts through authenticated backend endpoints. `GET /api/results/{result_id}/signed-download?artifact_name=...` can provide a Supabase signed URL that expires after five minutes; signed URLs are not persisted. Uploads are non-overwriting and identical retries are idempotent. If cloud persistence is configured or authentication is required, the API fails the job rather than report success when required cloud storage is unavailable; generated local outputs are retained for recovery. Authenticated result and artifact access is checked against the verified Supabase user identity. `/api/persistence/status` reports service reachability and only marks cloud persistence active when both PostgreSQL and Supabase Storage are available.

Analysis submissions also accept an optional `Idempotency-Key` header (16–128 safe ASCII letters, digits, `_` or `-`). The dashboard creates the key once per analysis request and preserves it when that request is retried. In PostgreSQL-backed deployments, each key is scoped to its verified user (or to the explicit public-demo scope), so different users do not collide. Replaying a completed request with the same key and parameters returns the stored result without rerunning processing or uploading artifacts again. Reusing a key with different parameters, replaying while the original job is active, or replaying a failed request returns HTTP `409`; use a new key to intentionally start a new request. The database enforces a unique scope/key constraint, and the additive migration leaves legacy records without keys. The no-database local demo does not persist or enforce idempotency keys; do not rely on retry protection there.

### Supabase Auth and authorization

The React client uses `@supabase/supabase-js` for email/password sign-in and sign-up, restores the persisted session, observes sign-in/sign-out/token-refresh events, and attaches the current access token as a bearer credential to API requests. A backend `401` clears the local session and returns the UI to sign-in. Configure the Supabase Auth email provider, the Vercel site URL, and its sign-in redirect allowlist in Supabase. Frontend settings are `VITE_API_URL`, `VITE_SUPABASE_URL`, and `VITE_SUPABASE_ANON_KEY`; the Supabase URL and anon/publishable key are public browser configuration. Never put `DATABASE_URL` or `SUPABASE_SERVICE_ROLE_KEY` in Vercel, a Vite variable, or browser code.

FastAPI validates every protected request by sending its bearer token to Supabase Auth's `GET /auth/v1/user` endpoint with the public anon key. Supabase Auth verifies the token against the project configuration (including signature and token validity/expiry) and returns the user identity; the backend does not trust a user ID supplied in a request. Invalid or expired tokens return `401`, while Auth service/configuration failures return `503`. All API paths are protected when authentication is required, except `GET /api/health`, `GET /api/ready`, and `GET /api/auth/status`; CORS preflight is also permitted. A configured database or any backend Supabase setting forces this protected mode even if `REQUIRE_AUTH` is false.

Backend Supabase endpoints must use HTTPS; plain HTTP is accepted only for localhost or loopback development. Auth fails closed on an invalid/insecure remote URL, and persistence does not send service-role credentials to it.

New analyses and their jobs/artifacts are associated with the verified Supabase user ID. Result, job, artifact-manifest, stream, and signed-URL lookups enforce that owner before data is returned. Existing rows with no owner remain `NULL` and are not assigned to an account; authenticated users cannot see those legacy rows. Uploaded GeoTIFF bands and cached satellite products in authenticated mode are placed under per-user, non-identifying directory namespaces and are not reused across accounts. Users may delete only products in their own cache; no artifact delete endpoint is exposed. Source uploads still reside on the backend filesystem and are ephemeral on Render unless a persistent disk is configured; they are not the durable cloud artifacts.

If no database, Supabase settings, or `REQUIRE_AUTH=true` is configured, the app intentionally runs as a local public demo. Its dataset, uploaded files, and local results are shared by anyone who can reach that API. Keep it limited to synthetic or non-sensitive demo data; do not use it for private imagery or reports.

### Provision PostgreSQL and Supabase Storage

1. Create or select a Supabase project and PostgreSQL database. Configure the backend `DATABASE_URL` with its PostgreSQL connection string. Render-style `postgres://` URLs are supported. The configured Render start command applies additive Alembic migrations before starting the API; the migrations do not drop or rewrite existing application data.
2. In Supabase Storage, create a bucket named `satellite-analysis-results` and leave **Public bucket** disabled. Alternatively, the backend can create a missing bucket as private on its first artifact upload; readiness remains unavailable until the bucket exists. It refuses to use a bucket that is public and does not change an existing bucket's access policy.
3. In Supabase Project Settings → API, obtain the project URL, public anon key, and server-side service-role key. Set `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `SUPABASE_SERVICE_ROLE_KEY` in the Render backend service. Set `SUPABASE_STORAGE_BUCKET` if using a different private bucket. Set `REQUIRE_AUTH=true` in Render. The server-side service-role key is used only by the backend and must never be put in Vercel or any `VITE_*` variable.
4. Set Render `CORS_ORIGINS` to the exact deployed Vercel origin (without a trailing slash); the checked-in Blueprint uses `https://team-s4-ten.vercel.app`. For local development, use only the local frontend origins. Configure the Supabase Auth site URL and redirect allowlist for the deployed Vercel origin.
5. In Vercel, set `VITE_API_URL` to the backend URL (the repository's production frontend build configuration currently points to `https://satellite-vision-dashboard.onrender.com`), and set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` for browser sign-in. These two Vite Supabase variables are public. Never configure `DATABASE_URL` or `SUPABASE_SERVICE_ROLE_KEY` in Vercel.
6. After deployment, inspect `/api/persistence/status`; confirm the database is connected, object storage is available, and `cloud_persistence_active` is `true`. Then submit an analysis with valid GeoTIFF inputs and verify its `job_id`, `GET /api/jobs/{job_id}`, `GET /api/analyses/{analysis_id}`, artifact manifest, and artifact download. Health alone does not verify persistence.

The backend uses the service-role key only for server-side private-bucket operations. Artifact binaries stay in Storage; PostgreSQL stores analysis/job state and artifact metadata. The API streams artifacts only after backend authorization and metadata lookup; private objects are not made public. Local/demo mode remains available when cloud storage is not configured and authentication is disabled. Uploaded source bands continue to use the configured backend `DATA_DIR`; on Render's ephemeral filesystem, source uploads themselves are not durable across instance replacement, so provision persistent source-data storage before relying on uploaded bands across redeployments. `/api/auth/status` reports whether the deployed API requires sign-in.

PostGIS is not required by the current API: it does not run spatial SQL queries. Existing CRS, raster bounds, and processing metadata are preserved in analysis JSON; install a PostGIS extension only if future database-side spatial queries are introduced.

### Production health, logs, and recovery

- `GET /` is public and returns only the API name and health/readiness paths; it does not expose analysis data.
- `GET /api/health` is a public liveness probe. It returns HTTP 200 when the API process can serve requests; it does not claim the database or object store is healthy.
- `GET /api/ready` is a public readiness probe. It checks runtime configuration and, when configured/required, runs a lightweight PostgreSQL query, verifies the mapped analysis/job/artifact tables and columns exist, and checks the configured bucket is reachable and private. It returns HTTP 503 if a required dependency or database schema is unavailable. The response exposes only component status labels, not URLs, credentials, or filesystem paths. The check is read-only: it does not create tables or a missing bucket. Render's Blueprint uses readiness as its health-check path.
- Application logs are structured JSON with UTC timestamp, severity, event, safe route template, status/error type, and request ID. Startup database errors may additionally include PostgreSQL SQLSTATE and validated schema/table/column/constraint identifiers. Recovery failures include only the parameterized SQL statement shape (never bound values). Request bodies, query values, tokens, credentials, connection strings, signed URLs, and stack traces are not logged. `X-Request-ID` is accepted only when it is short and uses a safe character set; otherwise the API generates one and returns it on the response. No external error-tracking service is configured.
- Analysis remains a synchronous HTTP workflow; this deployment does not run a durable background queue. Browser submissions are guarded against duplicate clicks. Supabase Storage requests retry transient network/HTTP 408, 429, and 5xx failures up to three attempts. Artifact writes are non-overwriting and compare content when a retry encounters an existing object. A timed-out browser request may still finish on the server, so inspect recent results/job status before submitting the analysis again. Interrupted queued/running database jobs are marked failed at startup; local outputs are retained when persistence fails.
- PostgreSQL uses SQLAlchemy connection pre-ping and transaction-scoped writes. Additive Alembic migrations preserve existing ownerless records. Keep database backup/retention settings aligned with the actual Supabase plan and verify the current Dashboard backup policy; do not assume database backups also cover Storage objects. Before a migration or cleanup, take an appropriate backup and periodically test restoration of both metadata and needed artifact objects.

For a bounded Storage/metadata consistency report, run from `satellite-intelligence/backend/` with the backend environment configured:

```powershell
cd satellite-intelligence/backend
python -m scripts.check_artifact_consistency --limit 1000
```

The report distinguishes metadata references whose object is missing from objects under `results/` with no metadata row. It is read-only by default and exits non-zero if it finds inconsistencies or the scan is truncated. Review a complete report before any cleanup. If deletion is explicitly approved, the tool requires the exact bucket name and a per-run maximum of at most 100 objects, for example:

```powershell
python -m scripts.check_artifact_consistency --limit 1000 --delete-unreferenced --confirm-bucket satellite-analysis-results --max-delete 10
```

That opt-in only deletes the bounded unreferenced object keys returned by a complete scan under `results/`; it never deletes metadata rows or objects associated with metadata. It does not run automatically at startup or in tests. Database outages, an unavailable bucket, and incomplete scans stop cleanup.

## Start frontend

```powershell
Set-Location .\satellite-intelligence\frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open the Vite URL (normally `http://127.0.0.1:5173`). Set `VITE_API_URL` to override the default development API origin `http://127.0.0.1:8000`. The production build reads `frontend/.env.production`; the deployed Vercel project can also define `VITE_API_URL` as a build-time environment variable. The current production value is `https://satellite-vision-dashboard.onrender.com`. Changing a Vite variable requires rebuilding and redeploying the frontend. The API client accepts either the backend origin or that origin with a trailing `/api`, and adds exactly one `/api` path. Raster rendering is georeferenced from the output transform and CRS. OpenStreetMap tiles provide optional web basemap context; the local raster analysis itself does not depend on satellite APIs. Without internet access, the calculated overlay and local analysis remain available, but the optional basemap may not load.

For authenticated deployments, also set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` from the Supabase project in Vercel. These are the public Auth client settings; never define `SUPABASE_SERVICE_ROLE_KEY` or `DATABASE_URL` in Vercel. The frontend example names are in `frontend/.env.example`.

## Run tests

```powershell
cd satellite-intelligence/backend
pytest -q
```

Tests use synthetic NumPy arrays, temporary GeoTIFFs, SQLite-backed repository tests, and mocked Supabase/Auth transports; no test fixture is represented as real satellite imagery, and mocked storage tests are not live cloud verification. Run frontend checks from `satellite-intelligence/frontend` with `npm test` and `npm run build`. Migration preservation is covered by backend tests; production migrations still require a configured database to be exercised against that deployment.

## Final submission package

Editable report, presentation, demo script, and current architecture diagrams are in [`docs/final_submission/`](docs/final_submission/):

- [Project report (DOCX)](docs/final_submission/Satellite_Vision_Project_Report.docx)
- [Hackathon presentation (PPTX)](docs/final_submission/Satellite_Vision_Hackathon_Presentation.pptx)
- [Demo script](docs/final_submission/Satellite_Vision_Demo_Script.md)
- [Architecture diagrams and verification map](docs/final_submission/Satellite_Vision_Architecture.md)
- [Complete final audit report](docs/final_submission/FINAL_AUDIT_REPORT.md)
- [Exact test and deployment results](docs/final_submission/FINAL_TEST_RESULTS.md)
- [Remaining issues and external actions](docs/final_submission/REMAINING_ISSUES.md)

The Word report and PowerPoint slides distinguish local software verification, mocked cloud tests, and live deployment checks. Their numerical demo chart is generated from the current deterministic synthetic endpoint; no satellite screenshot or real-scene output is presented as genuine data.

To regenerate the DOCX and PPTX, install the optional packaging tools `python-docx` and `python-pptx` in the backend Python environment, then run `python ..\docs\final_submission\generate_submission.py` from `satellite-intelligence/backend`. This helper calls only the local synthetic demo endpoint and is not part of the application runtime requirements.

## Performance limits and local load benchmark

The raster pipeline still uses whole-array NumPy operations rather than windowed output processing. It now selects only bands required by a single-index request, rejects oversized source files and rasters before array reads, applies a combined retained-input-array budget, and admits a bounded number of synchronous analyses per process. Deadline checks mark an over-budget job failed before persistence when they run between pipeline stages; they do not interrupt an in-flight native call. These limits reduce predictable load but are not a measured memory ceiling, durable queue, or cross-instance capacity guarantee. Analysis history is paginated (50 records by default, maximum 100); PostgreSQL reads use `LIMIT`/`OFFSET`, and local history retains only the requested newest page.

Run the bounded synthetic load benchmark from `satellite-intelligence/backend/` in a process without database or Supabase environment configuration:

```powershell
python scripts/benchmark_local.py --raster-size 512 --clients 1,5,10,25 --requests-per-client 1
```

It uses deterministic synthetic 512×512 GeoTIFFs, FastAPI `TestClient`, temporary local files, and no network or cloud resources. Client concurrency is capped at 25, requests per client at 5, and raster side length at 4096. Reported latency includes rejected requests as well as successful ones; inspect `successful_request_*_latency_ms` separately, and treat HTTP 503 as expected back-pressure, not completed analysis throughput. This is an in-process local stress check, not a production HTTP benchmark. Never point it at deployed services.

The unoptimized local baseline on Windows 11 / Python 3.14.3 used the same synthetic 512×512 NDVI request. At 1 / 5 / 10 / 25 clients, all requests returned 200; median/p95 latency was 535.01/535.01, 626.23/659.18, 878.67/916.66, and 2201.30/2242.87 ms. Sampled peak resident memory was 251,760,640 / 373,923,840 / 492,388,352 / 892,571,648 bytes. The final run after selective band reads, per-process concurrency admission, pixel/file/array budgets, and cooperative deadlines produced:

| Clients | HTTP 200 / 503 | Successful median/p95 (ms) | Attempt req/s | Accepted analyses/s | Sampled peak RSS (bytes) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 / 0 | 445.61 / 445.61 | 2.10 | 2.10 | 245,297,152 |
| 5 | 1 / 4 | 384.91 / 384.91 | 11.58 | 2.32 | 245,272,576 |
| 10 | 1 / 9 | 358.39 / 358.39 | 25.40 | 2.54 | 250,494,976 |
| 25 | 1 / 24 | 440.76 / 440.76 | 48.36 | 1.93 | 249,253,888 |

At one client the observed request latency decreased by 89.40 ms (16.7%) from the earlier baseline, but this is a single local run, not a statistically controlled performance guarantee. At 5 / 10 / 25 clients, memory stayed near 245–250 MB while excess requests were rejected (80% / 90% / 96% HTTP 503); these rows therefore demonstrate back-pressure, not successful multi-client capacity. Raw request throughput includes rejected attempts, and error rate counts those intentional 503 responses. A warmed `cProfile` request showed about 0.126 s in PNG encoder calls, 0.087 s loading the two required bands, and 0.10 s in local result persistence; the profile is a single in-process request and does not measure PostgreSQL or Supabase Storage. The Tachyon profiler could not run on Python 3.14.3 (it requires Python 3.15.0b3+); a prior high-volume `sys.monitoring` trace exceeded the capture-size limit.

The production build emits an initial entry bundle of 334.59 kB (93.51 kB gzip) and a framework bundle of 223.34 kB (69.28 kB gzip); together 557.93 kB raw / 162.79 kB gzip. The generated HTML preloads only those entry/framework assets. The chart and map vendors are deferred into dynamic chunks until a result requires them. Compared with the previous build's approximately 313.68 kB gzip across preloaded JavaScript chunks, the new initial JS preload set is approximately 48% smaller. This is a build-artifact comparison, not a browser-rendering or network timing measurement. Database queries were verified through bounded SQLite-backed tests and the additive Alembic migration; PostgreSQL pool behavior, Supabase Storage, satellite-provider latency, deployed Render/Vercel performance, and production load were not measured.

The final-audit rerun on 2026-10-10 used the same 512×512 synthetic raster, one request per client, FastAPI `TestClient`, Windows 11 / Python 3.14.3, and no database or cloud configuration. It reported:

| Clients | HTTP 200 / 503 | All-request median/p95 (ms) | Accepted median/p95 (ms) | Attempt / accepted analyses per second | Sampled peak RSS (bytes) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 / 0 | 631.87 / 631.87 | 631.87 / 631.87 | 1.42 / 1.42 | 239,943,680 |
| 5 | 1 / 4 | 49.98 / 1,774.40 | 1,774.40 / 1,774.40 | 2.56 / 0.51 | 253,022,208 |
| 10 | 1 / 9 | 74.86 / 928.00 | 928.00 / 928.00 | 9.74 / 0.97 | 250,048,512 |
| 25 | 1 / 24 | 154.39 / 232.03 | 1,896.34 / 1,896.34 | 11.83 / 0.47 | 258,654,208 |

This was one repeat run, not a controlled comparison: each level accepted one analysis and intentionally rejected the rest with HTTP 503 (80%, 90%, and 96% rejection at 5, 10, and 25 clients). The wide run-to-run latency variation is evidence not to claim a performance improvement or a production capacity from these figures. Throughput counts include rejected attempts; database and cloud paths were not exercised.

## Scientific methods and validation status

- **NDVI:** `(B08 - B04) / (B08 + B04)`. The vegetation-share default threshold is `0.3`; it is a code parameter, not a universal crop-health measure.
- **NDWI:** McFeeters formulation `(B03 - B08) / (B03 + B08)`. Values above the code default `0.0` are water-related indicators, not verified water extent.
- **NDBI:** `(B11 - B08) / (B11 + B08)`. Values above the code default `0.0` are a built-up indicator, not building detection.
- **Land cover:** Explicit index rules output Agriculture, Vegetation, Water, Built-up, and Bare land. These are heuristic classes with no calibrated probability or accuracy claim. Heuristic thresholds are currently baseline constants in the classifier.
- **Change:** `current NDVI - historical NDVI`. Relative mean change is omitted when the historical mean is zero/undefined. Positive and negative pixel proportions refer to valid aligned pixels, not causal change.

For the land-cover baseline, precedence is water (NDWI > 0.1), built-up indicator (NDBI > 0.05), vegetation (NDVI >= 0.45), agriculture (NDVI >= 0.25), then bare land. The model-status endpoint reports a Random Forest implementation as untrained; the active raster classifier is the rule-based baseline. No labeled training/held-out data is bundled, so supervised accuracy, precision, recall, F1, and confusion-matrix scores are not established.

Temporal history and forecast endpoints use acquisition-date tags on the local rasters only. Untagged inputs are excluded rather than assigned invented dates. These tags are user-provided and unverified; each response reports synthetic/unverified data classification and warnings. The current/historical folder layout provides at most two distinct acquisitions, so the time-ordered forecast baseline evaluation requires five observations and currently returns insufficient data. Even with more tagged inputs, this is not independent ground-truth model validation.

Raster analysis requires aligned dimensions, CRS, and affine transforms. No automatic reprojection, resampling, atmospheric correction, or cloud masking is performed. GeoAI spatial analysis summarizes the full raster extent; AOI clipping is not implemented and requests containing an AOI are rejected rather than silently ignored. Bounds and pixel areas are omitted when CRS metadata cannot support their calculation. For nonnegative input values, normalized-difference indices should fall within [-1, 1]; negative source values can produce values outside that interval, are reported as quality warnings, and are not silently clamped. Area is reported only when the input CRS is projected and its axis units can be converted to metres.

### Reproducible synthetic demo

The dashboard's **Demo mode** uses the deterministic `GET /api/geoai/demo` scenario; it does not need an external satellite provider or a dataset on disk. For a local demonstration:

1. Start the backend and frontend using the commands above, with no cloud credentials required.
2. Open the local dashboard and select **Demo mode**. The UI requests the fixed synthetic scenario separately from user analysis results.
3. Review the returned NDVI, NDWI, NDBI, and rule-based land-cover example. The response labels the data `synthetic`, has no acquisition date or geographic coverage, and reports no real sensor.
4. If the API is temporarily unavailable, restart the backend and retry **Refresh demo**. An unavailable OpenStreetMap basemap does not prevent this in-memory example from running.

The endpoint calculates over a fixed 4x4 in-memory matrix and writes no files, user records, or analysis artifacts. When authentication is enabled, it remains protected by the normal authentication dependency; demo mode does not bypass sign-in or expose private artifacts. In an unauthenticated local setup, the broader API dataset workflow is public and shared, so use synthetic or non-sensitive inputs only. The optional `SATELLITE_PROVIDER=mock` fixture is also tagged synthetic and uses a local-only CRS; it is not a live-provider substitute. Sentinel-2 metadata discovery and selected-scene acquisition use separate CDSE routes; no live authenticated imagery download has been verified.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service status |
| GET | `/api/ready` | Dependency readiness (HTTP 503 when required configuration/database/storage is unavailable) |
| GET | `/api/auth/status`, `/api/persistence/status` | Auth configuration and persistence component status |
| GET | `/api/dataset` | Current/historical band and georeferencing summary |
| POST | `/api/dataset/{period}/upload` | Upload a source band to the configured backend data directory |
| GET | `/api/metadata` | Raster metadata for the current and historical datasets |
| GET | `/api/data-quality` | Data completeness, nodata, CRS, and quality warnings |
| GET | `/api/model/status` | ML-ready land-cover model status and training guardrails |
| GET | `/api/geoai/demo` | Deterministic in-memory synthetic calculation example |
| GET | `/api/geoai/status`, `/api/geoai/risk-indicators` | GeoAI component status and rule-based risk indicators |
| GET | `/api/geoai/history` | Temporal summary only for distinct raster acquisition-date tags |
| GET | `/api/imagery/scenes`, `/api/imagery/scenes/{scene_id}` | List and inspect owned ingested GeoTIFF scenes |
| GET | `/api/imagery/analysis-types` | List supported scene-analysis types and available band roles |
| POST | `/api/imagery/scenes/{scene_id}/analyses` | Run an owned scene's NDVI, McFeeters NDWI, or K-means analysis |
| POST | `/api/imagery/change-detection` | Compare two owned scenes using thresholded NDVI difference |
| GET | `/api/imagery/analyses/{analysis_id}` | Retrieve scene analysis/change-detection status and result |
| GET | `/api/geoai/model-evaluation` | Hold-out metrics only when sufficient metadata-dated observations exist |
| POST | `/api/geoai/spatial-analysis`, `/api/geoai/vegetation-forecast`, `/api/geoai/land-cover-transitions` | Spatial summaries, exploratory forecast, and class transition analysis |
| POST | `/api/satellite/catalog/search` | Public Copernicus Sentinel-2 Level-2A STAC metadata discovery and pagination; application authentication applies when configured |
| POST | `/api/satellite/catalog/acquire` | Authenticated acquisition of four 10 m Sentinel-2 bands into the existing owner-scoped GeoTIFF ingestion workflow; requires backend CDSE credentials |
| POST | `/api/satellite/search`, `/api/satellite/download` | Legacy configured-provider search/download surface; live scene retrieval is incomplete and may return HTTP 501 |
| GET | `/api/visualization/{kind}` | PNG preview for `rgb`, `ndvi`, `ndwi`, `ndbi`, `landcover`, or `change` |
| POST | `/api/analyze/ndvi` | Calculate NDVI |
| POST | `/api/analyze/ndwi` | Calculate NDWI |
| POST | `/api/analyze/ndbi` | Calculate NDBI |
| POST | `/api/analyze/landcover` | Run baseline class rules |
| POST | `/api/analyze/change_detection` | Compare aligned current/historical NDVI |
| POST | `/api/analyze/all` | Attempt each analysis and report unavailable results individually |
| GET | `/api/results` | List paginated completed results by default; `include_incomplete=true` adds owner-scoped running/failed history metadata |
| GET | `/api/results/{result_id}` | Read one JSON summary |
| GET | `/api/analyses/{analysis_id}`, `/api/jobs/{job_id}` | Read analysis and synchronous job lifecycle records |
| GET | `/api/results/{result_id}/artifacts` and registered artifact download routes | List and retrieve registered artifacts, owner-checked when authentication is required |
| GET | `/api/results/{result_id}/download/{file_key}` | Download an available GeoTIFF or JSON report |
| GET | `/api/results/{result_id}/image/{analysis_key}` | Fetch the rendered map overlay |

Result identifiers are server-generated. Download endpoints only expose known output filenames; arbitrary filesystem paths are not accepted. API errors return a readable `detail` message without exposing stack traces.

## Limitations

- Inputs are expected to be pre-downloaded, single-band, co-registered GeoTIFF files. The system rejects misalignment; it does not automatically resample.
- Cloud/shadow masking, atmospheric correction, and radiometric harmonization are not inferred. Inputs should be prepared consistently by the user.
- The rule-based class labels and index thresholds are screening indicators, not validated land-use ground truth.
- Change detection is sensitive to acquisition date, cloud conditions, preprocessing differences, and seasonal variation.
- Raster metadata tags and local filenames do not independently verify a satellite, sensor, acquisition, calibration, or geographic footprint.
- Catalog search returns metadata only. The separate authenticated acquisition route downloads four full selected-scene assets and crops the output to the submitted AOI; it does not retrieve AOI-only byte ranges or complete SAFE products.
- Authenticated CDSE imagery downloads, cloud persistence, and production deployment have not been verified live. Accounts requiring an interactive or changing TOTP code are not supported by the current backend credential flow.
- No labeled model dataset, calibrated risk score, or AOI clipping workflow is available.
- Full rasters are loaded for processing, so available system memory limits the practical image size.
- OpenStreetMap background tiles require internet; data analysis uses local files.

### Production verification status

Live probes on 2026-10-10 confirmed that the configured Vercel page is reachable (HTTP 200), but it serves the older `index-BEv36dup.js` bundle rather than the locally built bundle. Browser inspection showed no visible sign-in control; the page's **Demo mode** button switches to the legacy local-dataset workflow and does not load the current synthetic scenario. The page loaded API resources from `https://satellite-vision-dashboard.onrender.com`. Health and readiness returned HTTP 200 after retrying longer than the initial 25-second timeout; `/api/auth/status`, `/api/persistence/status`, and `/api/geoai/demo` returned 404. Readiness reports configuration only and is not evidence of live database or Storage health.

The Vercel-origin analysis preflight requesting `authorization,content-type,idempotency-key` returned HTTP 400; the live `Access-Control-Allow-Headers` omits `Idempotency-Key`. A harmless empty JSON forecast POST previously returned HTTP 500 without `Access-Control-Allow-Origin`; the same empty request against local source returned HTTP 200 with an `insufficient-data` result. A credential-free `GET /api/results?limit=1` and a request with a malformed test bearer token both returned HTTP 200 with 29-byte responses. Bodies were deliberately discarded and not inspected. This is evidence that the deployed route does not reject missing or malformed credentials, not proof that user data was returned. Treat deployed result access as a potential authorization exposure and do not submit private data until the current authenticated backend is deployed and verified. A legacy `/api/geoai/status` also returns HTTP 200 without authentication; verify its intended exposure.

The frontend configuration and observed API host agree, but the Render Blueprint only declares service name `satellite-vision-api`; the Render Dashboard mapping to that host could not be confirmed. No valid Supabase credentials were available and no GeoTIFF test input was bundled. Therefore real PostgreSQL CRUD, a valid Auth session, private-bucket upload/retrieval, cross-user denial, and a full live analysis/artifact workflow are blocked, not passed. Redeploy matching frontend/backend versions, confirm the Render hostname and Blueprint settings, configure secrets only in the backend, then verify anonymous and malformed tokens are rejected, CORS allows `Idempotency-Key`, and one authorized analysis and private artifact can be retrieved. The public deployment is not currently safe to use with private data or verified as demo-ready/production-ready; the local synthetic demo remains the appropriate presentation path.

## Remaining Stage B work

The selected-scene acquisition path validates, downloads, crops to a bounded AOI, and feeds supported Sentinel-2A/B/C JP2 assets into the existing ingestion workflow. Remaining work includes automated reprojection/resampling, cloud masking, robust handling of account-specific two-factor requirements, and durable asynchronous acquisition jobs. Keep the existing processing behavior and owner-scoped GeoTIFF ingestion contract unchanged.
