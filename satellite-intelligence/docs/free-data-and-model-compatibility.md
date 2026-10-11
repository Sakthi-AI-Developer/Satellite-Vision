# Free satellite data and local model compatibility

Updated: 2026-10-11

## Implemented data and API path

- **Discovery:** public Copernicus Data Space Ecosystem (CDSE) STAC API
  `https://stac.dataspace.copernicus.eu/v1/search`, collection
  `sentinel-2-l2a` (Sentinel-2 Level-2A). The backend searches by AOI, date range,
  and cloud cover, paginates STAC results, and validates asset metadata.
- **Acquisition:** the existing backend adapter obtains an OAuth access token from
  CDSE's identity endpoint, then follows the CDSE OData product/node download API
  to fetch B02, B03, B04, and B08 source assets. It stages temporary files, checks
  the official hosts and product paths, enforces download/pixel limits, validates
  JP2/raster alignment and AOI coverage, writes checksums, and only then promotes
  a cropped GeoTIFF for analysis. CDSE's documented S3 asset references are used
  to validate provenance; the application does not fetch arbitrary catalog URLs.
- **Terms:** Copernicus Sentinel data is free and open under the [Sentinel Data
  Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)
  and [CDSE terms](https://dataspace.copernicus.eu/terms-and-conditions).
  Acquisition through this implementation requires a CDSE account and credentials
  configured only in the backend environment. The search API itself is public.
- **Training labels:** the existing pilot uses ESA WorldCover 2021 v200 labels,
  product DOI [10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221),
  under [CC BY 4.0](https://registry.opendata.aws/esa-worldcover-vito/).
  WorldCover was derived in part from Sentinel-1 and Sentinel-2 and is weak
  supervision, not independent reference truth. The source pilot and preparation
  evidence are in [Phase 21](./phase21-dataset-acquisition.md).

## Existing endpoint surfaces

These are existing FastAPI routes, reused by the React application:

| Method and route | Behavior |
| --- | --- |
| `POST /api/satellite/catalog/search` | CDSE STAC search with AOI/date/cloud filters and cursor pagination; passes through the application's configured user-authentication dependency. |
| `POST /api/satellite/catalog/acquire` | Authenticated, idempotent selected-scene acquisition. Requires backend CDSE credentials and stays within configured AOI/byte limits. |
| `GET /api/imagery/ai-models` | Reports configured local model readiness and Prithvi checkpoint compatibility assessment. |
| `POST /api/imagery/scenes/{scene_id}/ai-inference` | Owner-scoped local inference; refuses any model not marked ready. |
| `GET /api/imagery/ai-inference/{analysis_id}` | Owner-scoped persisted inference result. |
| `GET /api/monitoring/status`, `GET /api/monitoring/scenes`, `GET /api/monitoring/scenes/{scene_id}` | Authenticated persistent daily-monitor state and scene history/detail. |
| `POST /api/monitoring/scan` | Authenticated manual scan; shares the persistent scan lease with scheduled work. |

The existing frontend catalog, analysis, inference, and monitoring panels use
these API methods. The inference panel now displays the Prithvi compatibility
assessment, linking to the upstream checkpoint, without downloading or executing
it.

## IBM-NASA Prithvi checkpoint evaluation

Candidate reviewed: [Prithvi-EO-2.0 tiny-TL on Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL),
revision `335eadc2c45ad5abe7bd307223e1c48c5b60c41b`. The Hugging Face checkpoint
card declares **Apache-2.0**; the upstream [NASA-IMPACT implementation
repository](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) separately carries
an **MIT** license. Checkpoint and code licenses are not interchangeable.

The tiny-TL release is a 5-million-parameter spatiotemporal masked-autoencoder
pretraining checkpoint. Its published configuration specifies 4 frames, 6
channels, 224-pixel inputs and 30 m HLS pretraining data. The checkpoint card's
inference example reconstructs masked HLS sequences; the checkpoint is not a
land-cover segmentation model and has no ready-to-use land-cover prediction head.

Compatibility findings:

| Requirement | Project/checkpoint evidence | Result |
| --- | --- | --- |
| Inference task | The project requests per-pixel land-cover classes. The checkpoint's documented task is masked image reconstruction; a segmentation head and task-specific fine-tuning are needed. | Not compatible as-is. |
| Bands | The pilot and current CDSE acquisition use B02, B03, B04, B08. The HLS checkpoint expects blue, green, red, narrow NIR, SWIR 1 and SWIR 2. Sentinel-2 native B05-B07 are red-edge bands, not equivalent HLS channels. A wavelength-aware mapping to B8A/B11/B12, explicit reflectance scaling, and alignment to the checkpoint's 30 m grid are needed. | Not compatible as-is. |
| Time dimension | The checkpoint expects four frames from the same HLS location sequence. The current pipeline accepts one acquired scene; the pilot has no four-date sequence per region. | Not compatible as-is. |
| Runtime/hardware | The configured Python environment is Python 3.14.3; PyTorch and TerraTorch are not installed. The inspected Windows workstation has 16,869,351,424 bytes of RAM (about 15.71 GiB) and no GPU requirement for the selected scikit-learn model. No Prithvi weights were present in the configured model directory. No framework or weights were installed/downloaded for this evaluation. | Not ready. |
| Supervision | The available pilot contains 48 prepared chips in three regions; WorldCover-derived labels are weak labels, with no independent ground truth. | Not sufficient evidence for a dependable Prithvi land-cover head. |

The 300M and 300M-TL checkpoints were also inspected. Their published configs
retain the six-channel/four-frame HLS input contract and their scale increases
the local runtime burden. They do not resolve the missing land-cover task head,
input harmonization, temporal sequences, or supervision limitations. The tiny-TL
variant is the more plausible CPU experiment if a future, scoped fine-tuning
effort is approved, but it is **not enabled for inference**.

The backend's Prithvi `candidates` entry always reports `model_not_ready` until a
reviewed adapter, compatible inputs, task-trained weights, and verified local
runtime are implemented. It does not load or fetch checkpoints. The existing
four-band scikit-learn model remains subject to its own explicit approval and
checksum checks.

## Selected compatible local model

The compatible candidate is the project's existing locally trained
`scikit-learn HistGradientBoostingClassifier` pixel classifier, not a downloaded
pretrained checkpoint. scikit-learn is BSD-3-Clause licensed
([upstream license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING));
that software license does not grant rights to the separately sourced imagery
or WorldCover labels. The classifier accepts one observation with exactly four
features in B02, B03, B04, B08 order, normalized from uint16 reflectance by
10,000, and emits per-pixel class probabilities. It is CPU-only, uses the
project's already-installed scikit-learn runtime, and its trained artifact is
487,757 bytes, so it fits the inspected Windows workstation without a GPU or a
new deep-learning dependency. It has no spatial context and is not a U-Net or a
Prithvi segmentation head.

The external Phase 21 pilot has three 512 x 512 aligned Sentinel-2/WorldCover
windows and 48 prepared 128 x 128 chips. Training, validation, and test chips
are grouped by separate region IDs; the independently recalculated minimum
projected edge-to-edge gap is 44,879.3 m against the recorded 10,000 m
threshold. This passes the pilot's spatial-overlap check, but is not an
acquisition-independent split: all sites are from 2021-12-24, and the train and
test sites share Sentinel item `S2B_43PGN_20211224_0_L2A`. WorldCover is weak
supervision rather than independent ground truth.

The checked label mapping retains WorldCover codes 10, 20, 30, 40, 50, 60, 70,
80, 90, 95, and 100, with 0 treated as nodata. The pilot contains only codes
10, 20, 30, 40, 50, 60, 80, and 90; code 90 has just 53 pixels overall and
codes 70, 95, and 100 have no support. The saved held-out report records 0.5869
overall agreement and 0.1542 mean IoU across present test classes against
WorldCover weak labels. Those figures, the single date, class scarcity, and no
independent reference labels are insufficient for deployment or scientific
accuracy claims. The model is technically runnable but remains explicitly
unapproved and `model_not_ready`.

The inspected local hardware is a 13th Gen Intel Core i7-13620H Windows
workstation with 16,869,351,424 bytes of RAM (about 15.71 GiB); the model needs
no GPU. The separate Prithvi evaluation above still rules that checkpoint out:
its frame count, channels, grid, and pretraining task do not match this
single-scene four-band classifier contract.

## Verification evidence at update time

- A bounded live query to the public CDSE STAC endpoint completed without a
  provider error for AOI `77.70–77.75 E, 11.30–11.35 N`, a rolling 14-day window,
  a 20% cloud ceiling, and a maximum of three items. It returned one acquisition-
  supported item: `S2B_MSIL2A_20260929T050649_N0513_R019_T43PHN_20260929T091314`,
  observation `2026-09-29T05:06:49.024Z`, cloud cover 15.73%.
- No assets for that newly discovered scene were downloaded or raster-validated.
  Neither `CDSE_USERNAME` nor `CDSE_PASSWORD` is present in the backend/project
  `.env` files or process environment, so authenticated OData acquisition cannot
  run here. No credential values were read into output or changed. The adapter
  uses CDSE's official `cdse-public` password-grant token flow; local setup is
  documented in the project README and [backend example environment](../backend/.env.example).
- The existing Phase 21 pilot imagery/labels are present outside the Git
  repository. All three source image/label SHA-256 values matched the acquisition
  manifest; four uint16 input bands, single-band uint8 labels, exact grid
  alignment, WorldCover code set, all 48 chip checksums, and split grouping were
  rechecked. There are 786,432 valid chip pixels across the three regions.
- The existing 487,757-byte trained artifact SHA-256 matched its manifest; its
  training metadata/index hashes matched the current prepared dataset; bands and
  all 11 class-name mappings matched. The stored `deployment_approved` remains
  `false`, and the model registry reports `model_not_ready`.
- For an offline inference smoke check only, a temporary copy of the trusted
  artifact was used with a temporary approval flag; the stored artifact and
  manifest were not changed. Inference on the held-out Erode pilot GeoTIFF
  completed for 262,144 valid pixels, produced class indices 1–7, and preserved
  the source grid in both class and confidence rasters. Temporary files were
  removed. This was not an API inference, a CDSE acquisition, or independent
  model validation, and no prediction was persisted because the real artifact
  remains unapproved.
- The saved training report records test OA 0.5869 and present-class mIoU 0.1542
  against weak WorldCover labels only. Geographic windows are spatially
  separated, but train and test share the 2021-12-24 date and one Sentinel item;
  those figures are not acquisition-independent or scientific accuracy. No
  new training run or new model metric was generated in this continuation.
- Daily monitoring remains opt-in (`SATELLITE_MONITOR_ENABLED=false` by default).
  Persistent scene deduplication, the singleton scan lease, and timezone-aware
  scheduling are covered by local tests. Monitoring was not enabled, migrated
  against production, scheduled, or deployed by this update.
- A fresh local loopback run used temporary data/model directories and explicitly
  blank CDSE, Supabase, and database settings. On `127.0.0.1:8014`, `/api/health`,
  `/api/ready`, `/api/auth/status`, and `/api/imagery/ai-models` returned HTTP 200;
  readiness was `ok`, authentication was not required, and the configured model
  remained `model_not_ready`. The public catalog route returned HTTP 200 with
  one metadata result for the bounded Erode AOI and 2026-09-28 through 2026-09-30.
  This result was catalog metadata only; no asset was fetched.
- The same loopback server returned HTTP 401 for acquisition without an
  application user session and HTTP 401 for monitoring status without a user
  session. Separately, a local FastAPI HTTP-contract check supplied an isolated
  authenticated-user dependency while leaving CDSE credentials blank; acquisition
  returned HTTP 503 with the explicit `CDSE_USERNAME`/`CDSE_PASSWORD` setup
  message. This did not contact CDSE OData, download imagery, or use real user
  credentials. The test environment kept `SATELLITE_MONITOR_ENABLED=false`.

## Reproducible local checks

For a local Windows PowerShell start, run these in separate terminals from the
repository root. The first command copies the blank template only when a private
backend `.env` does not already exist. Enter CDSE account credentials in that
backend-only file if authenticated imagery acquisition is intended; never place
them in the frontend environment.

```powershell
if (-not (Test-Path .\satellite-intelligence\backend\.env)) { Copy-Item .\satellite-intelligence\backend\.env.example .\satellite-intelligence\backend\.env }
notepad .\satellite-intelligence\backend\.env
Set-Location .\satellite-intelligence\backend
uvicorn main:app --env-file .env --host 127.0.0.1 --port 8000 --reload
```

In a second terminal:

```powershell
Set-Location .\satellite-intelligence\frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Run backend verification from `satellite-intelligence\backend` with the project's
configured Python environment:

```powershell
python -m pytest -q
python -m compileall -q .
python -c "import api.routes, geoai.inference, satellite.catalog, satellite.acquisition, satellite.monitoring"
python -m pip check
```

Run the frontend checks from `satellite-intelligence\frontend`:

```powershell
npm test -- --pool=threads --maxWorkers=1
npm run build
```

For a public discovery probe, use the dashboard's Live Satellite search or call
`search_sentinel2_catalog` with a `SentinelCatalogSearchRequest` containing an AOI,
date range, `max_cloud_cover=20`, and `limit=3`. A returned catalog item is not a
download. Acquisition must be explicitly requested by an authenticated user and
is complete only after the asset download and raster validation succeed.

Latest verification for this update:

| Check | Result |
| --- | --- |
| Backend suite | `python -m pytest -q`: 247 passed, 4 existing warnings. |
| Focused inference/API tests | `python -m pytest -q tests/test_ai_inference.py`: 11 passed, 1 dependency deprecation warning. |
| Frontend suite | Final serial rerun, `npm test -- --pool=threads --maxWorkers=1`: 13 files, 76 tests passed. One earlier run reported seven failures; the standalone dashboard suite and final complete serial rerun passed without frontend changes, so the failures were not reproduced. |
| Production build | `npm run build -- --outDir <temporary-directory>`: succeeded, 815 modules. Temporary output removed. |
| Python checks | `python -m compileall -q .`, imports of affected modules, and `python -m pip check` passed. |
| Targeted Pylance | No diagnostics in changed `geoai/inference.py` or `AIInferencePanel.jsx`. This was not a workspace-wide diagnostics run. |
| Local loopback HTTP | With blank credentials and isolated temporary storage on `127.0.0.1:8014`, `/api/health`, `/api/ready`, `/api/auth/status`, and `/api/imagery/ai-models` returned HTTP 200; readiness was `ok` and the model remained `model_not_ready`. Unauthenticated acquisition and monitoring status returned HTTP 401. |
| Catalog and acquisition contract | `/api/satellite/catalog/search` returned HTTP 200 with one public STAC metadata result. An isolated authenticated-user HTTP-contract check with blank CDSE settings returned HTTP 503 and the explicit missing-credentials message. No imagery assets were downloaded or raster-validated. |
| Migration/security actions | No database migration, production data change, credential change, staging, commit, push, or deployment. |

The latest Phase 22 verification record is in
[Phase 22 verification](./phase22-daily-monitoring.md).

## Exact remaining blockers

- No `CDSE_USERNAME` or `CDSE_PASSWORD` is configured locally. A catalog result
  is only metadata; the authenticated OData acquisition flow did not run, and no
  new satellite imagery was downloaded. Configure a CDSE account in the ignored
  backend `.env` before attempting local acquisition.
- The existing pilot classifier is technically compatible and its artifact and
  inference output were verified, but it is still explicitly unapproved and the
  model registry correctly returns `model_not_ready`. The pilot has only one
  acquisition date, shared Sentinel item across train/test, weak WorldCover
  labels, sparse class support, and no independent reference labels. Do not
  enable API predictions until those validation gaps are addressed and the
  existing approval/checksum contract is deliberately satisfied.
- Persistent daily monitoring remains disabled. It needs a local persistent
  database, the monitoring migration applied only to that local database,
  configured AOI and CDSE credentials, and an external-to-source-tree storage
  directory. No production database migration or scheduled scan was attempted.
- Live cloud authentication/storage and production deployment were not validated;
  no production endpoint or configuration was changed.

Keep daily monitoring disabled and the model unapproved until the relevant local
configuration and independent validation prerequisites are met.
