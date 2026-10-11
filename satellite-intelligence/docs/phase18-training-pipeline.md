# Phase 18 — Reproducible Sentinel-2 Land-Cover Training Pipeline

**Decision:** Prepare only local, aligned Sentinel-2/WorldCover training chips. Do not train, provision data, add a neural dependency, or enable a model until independent evaluation labels and the deployment resource budget are approved.

## 1. Dataset suitability and rights

| Candidate | Verified properties | Suitability and unresolved issues |
|---|---|---|
| [ESA WorldCover 2021 v200](https://doi.org/10.5281/zenodo.7254221) | Global 2021 map; 10 m; 11 classes; based on Sentinel-1 and Sentinel-2. The [AWS Open Data Registry](https://registry.opendata.aws/esa-worldcover-vito/) states CC-BY 4.0. The [ESA data-access page](https://esa-worldcover.org/en/data-access) documents 2021 L2A annual composites and their uint16 0–10000 reflectance range, 0 nodata, and 0.0001 scale. | Suitable as versioned **weak supervision**, not independent ground truth. It is itself a derived map and may contain classification errors. An annual composite is not the same observation as a selected single scene; seasonal, temporal, cloud, and mixed-pixel disagreement remain. Users must provide locally prepared pairs on exactly matching CRS, pixel grid, dimensions, and resolution. The training utility does not download, resample, or reproject data. |
| [CORINE Land Cover](https://land.copernicus.eu/en/products/corine-land-cover) | Pan-European inventory with 44 classes; status/change maps are updated every six years, most recently 2018. The [Copernicus Land Monitoring Service data policy](https://land.copernicus.eu/en/data-policy) provides full, open, free access subject to source attribution, disclosure of modifications, and no implication of EU endorsement. | Potentially useful as a **coarse regional comparison**, not independent 10 m pixel truth. Its class taxonomy, mapping scale, and acquisition vintage require a documented crosswalk and aggregation protocol before it can evaluate this model. No such test protocol or AOI-specific label package is provisioned. |
| [LandCoverNet](https://arxiv.org/search/?query=LandCoverNet&searchtype=title) | The Phase 17 review found a paper describing annual 2018 Sentinel-2 L2A imagery at 10 m and labels adjudicated by three annotators; the first release reportedly covers Africa. | Promising for independent evaluation in principle, but the current authoritative distribution, exact version/class mapping, license, access conditions, and geographic coverage for the proposed AOIs have not been verified. Do not use or redistribute it until they are. |
| ESA WorldCover Annual Sentinel-2 Composites | ESA documents 2020 and 2021 composites as yearly median/percentile products derived from Sentinel-1/2; Sentinel-2 composites use L2A, uint16 reflectance, scale 0.0001, and 0 nodata. | Useful context for alignment and preprocessing, but not independent labels. The published registry reports 8.5 TB per year for RGBNIR composites; no bulk download is justified for this phase. |

**License separation:** This scaffold consumes WorldCover labels only. Its label attribution identifies ESA WorldCover 2021 v200 and DOI 10.5281/zenodo.7254221, and records CC-BY-4.0. Sentinel-2 source imagery remains subject to the Copernicus Sentinel data terms and required source notices. No model code, model weights, evaluation labels, or third-party checkpoint is incorporated. Dataset rights must not be inferred from software licenses.

**WorldCover class codes:** 10 tree cover; 20 shrubland; 30 grassland; 40 cropland; 50 built-up; 60 bare/sparse vegetation; 70 snow and ice; 80 permanent water bodies; 90 herbaceous wetland; 95 mangroves; 100 moss and lichen; 0 nodata. These product classes are copied, not reinterpreted or remapped.

## 2. Proposed training design (not implemented)

- **Task:** One-scene, per-pixel semantic segmentation. One output class code per valid pixel; confidence/probability is a separate optional product and must never be confused with class codes.
- **Architecture candidate:** Compact U-Net, four input channels, encoder widths 32/64/128/256 with a 512-channel bottleneck, skip-connected decoder, and 11 logits. This is a proposal, not an existing or trained model. A rough parameter estimate is about 7.7 million (about 31 MB of FP32 weights); it is not a measured checkpoint size.
- **Inputs:** B02, B03, B04, B08 in that exact order, single-date Sentinel-2 L2A uint16 surface reflectance, divided by 10,000 to float32. Keep original CRS, pixel grid, and 10 m sampling. The implemented preparation utility verifies four uint16 bands and their descriptions, but cannot prove source product provenance or correct atmospheric processing.
- **Labels and nodata:** WorldCover codes above; ignore code 0 in loss and metrics. Raster mask/nodata pixels are excluded, and invalid chip input values are zeroed in the prepared arrays. Use class-weighted or focal/weighted cross-entropy only after inspecting training-region class distribution; do not hide absent or rare classes.
- **Chips:** Default 128×128; maximum 256×256. Full chips only; incomplete edge strips are dropped. Fix and record seed, preprocessing, label version, source-raster SHA-256, class counts, and chip georeferencing.
- **Split:** Assign whole, non-overlapping geographic blocks to train/validation/test before chip extraction. Keep all dates, adjacent chips, and repeat observations from a site in one partition. The current utility deterministically groups by caller-provided `region_id`, but does **not** prove those IDs are spatially disjoint; the operator must supply non-overlapping region IDs and document AOI boundaries. A 70/15/15 target is appropriate only when there are enough independent regions and each split has valid pixels.
- **Evaluation:** Freeze an independently licensed and spatially separated test set before tuning. Report support and class imbalance, per-class precision/recall/F1, confusion matrix, overall accuracy, per-class IoU and mean IoU, valid-pixel coverage, and uncertainty. Add boundary metrics only if boundary quality becomes a product objective. WorldCover training labels cannot serve as independent test truth. No metrics are available in this phase.
- **Reproducibility:** Version source IDs and checksums, label/data licenses and attribution, dates, AOIs, projection/transform, band list, scale, nodata policy, chip size, split assignment, seed, class mapping, dependency lock/version, model configuration, and final checkpoint digest in a future manifest. A trained checkpoint must be written outside source control and pass format, size, checksum, and inference compatibility validation before readiness can change from `model_not_ready`.

## 3. Resource and deployment gate

The local machine was observed in the prior inspection as 16 logical CPUs, 15.7 GiB RAM, and 190.7 GiB free on E:. Those figures describe this workstation, not Render. PyTorch is absent from the selected environment. The effective production CPU, memory, disk, model-startup budget, and neural-inference latency have not been established.

The proposed ~31 MB FP32 parameter payload is only an architecture estimate. Training additionally requires gradients, optimizer state, and feature activations; actual peak memory and time depend on batch size, chip size, hardware, and implementation. A 128×128 float32 four-band input is 256 KiB before labels and intermediate arrays; the utility permits up to 5,000 samples, so uncompressed prepared arrays can approach about 1.3 GiB at the default chip size or about 5.2 GiB at 256×256 (before metadata/filesystem overhead). These are upper-bound estimates, not benchmark results. The utility reads one chip at a time and caps chip size and sample count, but callers must check free space before preparing a dataset.

Before adding a training runtime or model to the service:

1. Obtain the actual deployment memory/CPU/disk allocation and confirm a model checkpoint can fit the configured 256 MiB artifact ceiling.
2. Benchmark CPU inference and peak RSS at the app's actual raster size and 300-second analysis limit; use the existing 10-million-pixel AI cap as a hard ceiling, not a performance guarantee.
3. Decide whether training is external/offline or supported on a provisioned machine. Do not train in web-service startup or ordinary tests.
4. Verify the independent evaluation-data license and geographic suitability. If suitable labels cannot be cleared, commission or manually curate AOI-scoped reference polygons/pixels with written use rights.

No dataset, weights, or heavy dependencies were downloaded or installed, and no runtime/deployment configuration was changed.

## 4. Implemented preparation scaffold

`backend/geoai/training_data.py` provides a local-only `prepare_training_dataset` helper. It:

- requires at least three distinct `region_id`s and ISO `YYYY-MM-DD` dates in 2021;
- accepts GeoTIFF pairs only, verifies B02/B03/B04/B08 descriptions and uint16 imagery, single-band uint8/uint16 labels, and exact CRS/dimension/transform alignment;
- enforces chip-size and sample-count limits, validates WorldCover codes, masks raster nodata and class 0, scales reflectance by 10,000, and stores invalid features as zero;
- assigns all pairs with the same region ID to the same deterministic split;
- writes compressed NumPy chips, a JSONL index with CRS/window transform and source SHA-256 values, and an attributed dataset manifest;
- refuses output within the repository or an existing destination and removes its temporary staging directory on failure.

This is **data preparation only**: it does not validate geospatial adjacency between region IDs, check atmospheric-correction provenance, reproject/resample, balance classes, train a model, create a checkpoint, or enable inference readiness. Inputs are local and caller-supplied. Synthetic GeoTIFFs in tests verify mechanics only; they are not training data or evidence of accuracy.

## 5. Verification and status

The repository was already dirty before this phase: branch `main`, existing tracked modifications and untracked Phase 11–17 files were present. No pre-existing files were staged, committed, pushed, deployed, deleted, or intentionally rewritten in this phase.

The focused tests initially exposed a test expectation that incorrectly treated every chip's upper-left pixel as nodata. The test was corrected to assert nodata only in the chip containing the source nodata pixel; the preparation logic was also tightened to zero masked features, validate calendar dates, and construct raster windows with explicit keyword arguments. The focused suite passed (6 tests).

| Check | Result |
|---|---|
| `python -m pytest -q tests/test_training_data.py` | **6 passed** |
| `python -m pytest -q` (backend) | **210 passed**, 4 dependency/fixture warnings |
| Python compilation (`python -m compileall -q .`) | **Passed** |
| Backend imports (`main`, inference, training-data and satellite modules) | **Passed** |
| `python -m pip check` | **Passed**, no broken requirements |
| `python -m alembic -c alembic.ini history` | **Passed**, inspected through head `0004_analysis_history_index`; no migration was applied |
| Frontend `npm test -- --maxWorkers=1` | **12 files / 72 tests passed** |
| Frontend production build with `--outDir <unique temp> --emptyOutDir` | **Passed** (814 modules); temporary output removed |
| Pylance syntax checks for both changed Python files | **No syntax errors** |
| Pylance semantic diagnostics | **Not cleanly verifiable.** It continued returning one `reportCallIssue` (“Expected 0 positional arguments”) at a range that maps to the old positional `Window` call after the source was changed to keyword arguments. Pylance hover results likewise reflected the old call buffer. The updated source passes runtime tests; the stale editor diagnostic is not represented as resolved. |
| `git diff --check` and trailing-whitespace scan of new files | **Passed** |

### Authoritative references

- [ESA WorldCover AWS Open Data Registry](https://registry.opendata.aws/esa-worldcover-vito/) — global scope, 2020/2021, 10 m, 11 classes, CC-BY 4.0.
- [ESA WorldCover 2021 v200 Zenodo record](https://doi.org/10.5281/zenodo.7254221) — product version, date, class count, source description, product user manual/validation report links.
- [ESA WorldCover data access and composites](https://esa-worldcover.org/en/data-access) — annual composites, Sentinel-2 L2A basis, band values/scale/nodata and data access.
- [Copernicus Land Monitoring Service data policy](https://land.copernicus.eu/en/data-policy) and [CORINE Land Cover product](https://land.copernicus.eu/en/products/corine-land-cover) — reuse conditions and product scope.
- [LandCoverNet paper search](https://arxiv.org/search/?query=LandCoverNet&searchtype=title) — leads only; distribution and license remain unverified.

## 6. Completion state

| Category | Status |
|---|---|
| Research and design | **Documented.** WorldCover is chosen only as a weak-supervision candidate; independent evaluation data and deployment resources remain blockers. |
| Data/training code | **Partially implemented.** Local alignment-checked chip preparation exists; no training loop or model adapter was added. |
| Training completed | **No.** |
| Trained checkpoint provisioned and verified | **No.** |
| Real-scene model inference | **No.** |
| Independent scientific evaluation | **No.** No performance results or accuracy claims are made. |
