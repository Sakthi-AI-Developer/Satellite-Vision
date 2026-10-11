# Phase 19 — Reproducible Land-Cover Model Training Pipeline

**Status:** A bounded, offline pixel-wise training/evaluation pipeline is implemented. No real dataset was supplied, so no production checkpoint, real-scene inference, or scientific accuracy result exists. Artifacts emitted by the trainer are marked not approved for inference until an operator explicitly approves one after review.

## 1. Data terms and suitability

### Sentinel-2

The [Copernicus Data Space Ecosystem terms](https://dataspace.copernicus.eu/terms-and-conditions) state that access and use of Copernicus Sentinel data is free, full, and open, governed by the [Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice). The current application input contract remains L2A B02/B03/B04/B08, ordered as listed and scaled by 10,000. The preparation utility records raster checksums and band order; it cannot independently establish the downloaded product's processing level, acquisition provenance, or compliance with use-specific obligations. Keep the scene source/product ID and the applicable legal notice with each real training-data collection, and do not imply ESA/EU endorsement.

### ESA WorldCover

The [AWS Open Data Registry](https://registry.opendata.aws/esa-worldcover-vito/) identifies ESA WorldCover 2021 v200 as a global 10 m, 11-class map and states **CC-BY 4.0**. Its citation points to the [2021 v200 Zenodo record](https://doi.org/10.5281/zenodo.7254221). ESA's [data-access page](https://esa-worldcover.org/en/data-access) provides the product user manual and validation report and describes annual 2020/2021 Sentinel-2 L2A composites. The training metadata carries the product name, DOI, license, and registry access notice. Attribution is retained with the model/report; trained-weight licensing is not independently determined by this code.

WorldCover is **weak supervision**: it is a derived map made using Sentinel-1 and Sentinel-2, not independent reference truth. Its definitions, errors, annual compositing, and temporal mismatch with an individual scene can shape both training and reported agreement. Validation and held-out test metrics produced by this utility are therefore explicitly named **agreement with WorldCover weak labels**. They are not independent scientific accuracy. A separately licensed, geographically appropriate reference dataset is still required for that claim.

### Preparation and geographic leakage

The existing preparation stage aligns the raster grids, preserves CRS/transforms in chip metadata, drops partial edge chips, masks nodata, and groups every observation with the same `region_id` into one deterministic split. Phase 19 adds:

- a required `geographic_independence_evidence` statement before training; it is recorded as an operator attestation;
- per-chip SHA-256 in the index and verification of those hashes before training;
- refusal if any region ID occurs in multiple splits or if any split is empty.

This prevents leakage by repeated/grouped ID and records the asserted split evidence, but does **not** calculate geographic distances, detect adjacent/overlapping sites under different IDs, or verify the operator's evidence. Do not interpret the attestation as machine verification. The operator must define genuinely non-overlapping, buffered geographic sites before preparation; use a separate reference source for an independent final evaluation.

## 2. Implemented model and reproducibility

The new `backend/geoai/train_landcover.py` uses the existing scikit-learn dependency and inference interface. It is a **pixel-wise spectral classifier**, not a convolutional U-Net: each pixel is classified from the four reflectance samples, with no spatial-neighborhood context. It writes the existing class-index and probability outputs through the existing geospatial inference adapter.

| Setting | Implementation |
|---|---|
| Input | Single-scene uint16 Sentinel-2 L2A B02, B03, B04, B08, in that order |
| Preprocessing | Float32 reflectance divided by 10,000; no resampling or atmospheric correction |
| Target | The 11 original WorldCover codes: 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100; 0 is ignore/nodata |
| Class labels | Manifest mapping retains each original WorldCover code and class name; inference emits class index and mapped label separately from probability |
| Estimator/loss | `HistGradientBoostingClassifier(loss="log_loss")`; deterministic seed; early stopping disabled |
| Imbalance | Seeded, capped per-class pixel sampling followed by inverse-frequency sample weights |
| Defaults | Seed 42; max 50,000 training pixels; max 5,000 per class; 100 iterations; learning rate 0.1; 15 leaves; minimum 20 samples per leaf; L2 1.0 |
| Evaluation | Full valid-pixel validation/test samples, streamed in prediction batches; maximum 1,000,000 pixels per evaluation split |
| Report | Per-class support, precision, recall, F1 and IoU; confusion matrix; overall accuracy; mean IoU over classes present in that split |

The 50,000-pixel training cap bounds the feature matrix to about **0.8 MB** of float32 input data (plus labels/estimator working state). The configured maximum is 100,000 rows (about 1.6 MB of feature data); it is not a peak-memory guarantee. Dataset preparation still permits up to 5,000 chips: uncompressed four-band float32 chips plus uint8 labels are approximately 1.3 GiB at 128×128 or 5.3 GiB at 256×256, before archive/index overhead. Preparation and training both read data from local paths; nothing is downloaded automatically.

The private output directory contains:

- `sentinel2-landcover.joblib`: estimator artifact, outside the repository;
- `sentinel2-landcover.json`: version, band order, normalization, class mapping, source-label license/attribution, dataset index/metadata checksums, configuration, training sample counts, artifact SHA-256, and `deployment_approved: false`;
- `training-report.json`: validation and held-out weak-label agreement metrics, region IDs by split, and an explicit warning about interpretation.

After real preparation, the offline training entry point is `python -m geoai.train_landcover <prepared-dataset-directory> <private-model-output-directory>`. Optional flags override the recorded seed and bounded estimator settings. It never downloads input data; the requested output directory must be new and outside the repository.

The inference manifest reader requires and verifies an artifact SHA-256 digest, validates the optional class mapping, and requires `deployment_approved: true`. A newly trained artifact remains `model_not_ready` because its approval flag is false. Setting it true is an explicit local operator action after inspecting the provenance/report and testing the artifact; it does not prove scientific suitability. As with the existing joblib interface, only load artifacts produced by a trusted local training workflow—SHA-256 detects mismatches, not a malicious artifact whose manifest was also replaced.

## 3. Resource feasibility

Measured workstation state during this phase: **16 logical processors**, **15.7 GiB total RAM**, about **3.2 GiB free RAM** at the measurement time, and **190.7 GiB free on E:** (about 20.7 GiB free on C:). scikit-learn, NumPy, Rasterio, and joblib were already available through the project environment/dependency graph. **No package was installed and no dataset or model weight was downloaded.**

A resource-only synthetic CPU run using 50,000 four-feature rows and the default 100-iteration/15-leaf estimator completed fitting in **1.37 seconds** and serialized to **367,186 bytes** in an in-memory buffer. This is a small local configuration benchmark only: it is not a trained project model, does not measure raster I/O, full chip preparation, peak RSS, real-data training time, model quality, or deployment performance. It was not written to disk.

The local cap, established dependencies, and available disk make small offline CPU experiments feasible. Actual peak memory on this busy workstation and end-to-end time over a real dataset remain unmeasured. Render/service CPU and RAM are not established; training is not added to service startup or API requests. Do not raise pixel/chip caps or run a large training batch without measuring the machine's free-memory margin and dataset storage first.

## 4. Verification and honest result

Automated test fixtures create synthetic GeoTIFFs/chips to check validation, fitting, output manifests, metrics shapes, inference compatibility, approval gating, and tamper detection. They are **not satellite observations and are not evidence of model accuracy**. The tests create only temporary model artifacts which are removed with test temporary directories.

No real Sentinel-2 imagery or WorldCover tiles were provided to the training utility. Therefore:

- Real data preparation: **not run**.
- Training on real imagery/labels: **not run**.
- Durable trained checkpoint: **not created**.
- Real-image inference: **not run**.
- Real validation/test metrics: **none**.
- Independent scientific evaluation: **not completed**.

### Authoritative sources

- [Copernicus Data Space terms and conditions](https://dataspace.copernicus.eu/terms-and-conditions) and its linked [Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)
- [ESA WorldCover AWS Open Data Registry](https://registry.opendata.aws/esa-worldcover-vito/)
- [ESA WorldCover 2021 v200 DOI](https://doi.org/10.5281/zenodo.7254221)
- [ESA WorldCover data access, annual composites, user manual and validation report](https://esa-worldcover.org/en/data-access)
- [Copernicus Land Monitoring Service data policy](https://land.copernicus.eu/en/data-policy)

## 5. Next steps

1. Prepare operator-selected, locally stored L2A scenes and aligned WorldCover label tiles; record scene IDs, acquisition dates, legal-notice/source information, and storage estimates.
2. Define and document genuinely buffered, non-overlapping geographic regions before setting the preparation attestation. Review class counts and ensure train/validation/test each contain usable labels.
3. Obtain rights and a separate reference-label source for independent evaluation; freeze its geography before tuning.
4. Run the bounded local training utility only on the approved dataset. Review reported weak-label agreement and the checkpoint digest.
5. Benchmark real-scene inference and peak memory on a real Sentinel-2 raster. Keep inference unapproved until compatibility, resource limits, provenance, and independent evaluation expectations have been reviewed.
