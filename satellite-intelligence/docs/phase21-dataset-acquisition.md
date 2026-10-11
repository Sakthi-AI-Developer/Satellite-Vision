# Phase 21 — Real Dataset Acquisition and Validation

**Outcome: a small real-data pilot was acquired, aligned, checksummed, and prepared. No model training or inference was performed.**

## Sources, terms, and attribution

| Data | Authoritative terms/product evidence | Access used |
|---|---|---|
| Sentinel-2 L2A | [Copernicus Data Space terms](https://dataspace.copernicus.eu/terms-and-conditions) and the [Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) describe Sentinel data as free, full, and open. The [AWS Sentinel-2 L2A COG registry](https://registry.opendata.aws/sentinel-2-l2a-cogs/) documents the public COG distribution. | Public Element 84 [Earth Search STAC collection](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a) metadata and public AWS COG byte-range reads. No Copernicus credentials or paid API were used. |
| ESA WorldCover labels | [ESA data access](https://esa-worldcover.org/en/data-access), [AWS Open Data Registry](https://registry.opendata.aws/esa-worldcover-vito/), and [DOI 10.5281/zenodo.7254221](https://zenodo.org/records/7254221) identify WorldCover 2021 v200 as the 10 m, 11-class product, under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). | Public AWS COG byte-range reads of map tile `ESA_WorldCover_10m_2021_v200_N09E075_Map.tif`. |

Attribution recorded in the prepared dataset is: **ESA WorldCover 2021 v200 (CC-BY-4.0), DOI: 10.5281/zenodo.7254221; accessed via the AWS Open Data Registry.** The Earth Search collection reports a normalized `proprietary` license value; that value was not treated as the authoritative Sentinel data license. The Copernicus legal notice and AWS source registry were used instead.

Direct Copernicus Earth-observation access requires registration; the public AWS COG and Earth Search assets used here required no credentials. CC-BY-4.0 permits reuse, including commercial reuse, with attribution and other license conditions; the recorded attribution is retained with the data. The Sentinel legal notice governs Sentinel data use.

WorldCover labels are weak supervision, not independent ground truth. They are an annual map product, are not temporally identical to the December Sentinel observations, and may contain classification errors. No independent evaluation reference data was acquired.

## Acquisition and resource result

- **Real data acquired:** yes. The command opened public Cloud-Optimized GeoTIFFs using byte-range access, read three bounded windows, and saved aligned image/label crops. Complete source COGs were **not** downloaded. The actual aggregate network bytes for GDAL's range reads were not instrumented.
- **Output directory:** `E:\satellite-vision-data\phase21-pilot-20261011-final`, outside the Git repository and not present before acquisition.
- **On-disk output:** 10,697,701 bytes across 57 files (six georeferenced source windows, 48 prepared chips, and three indexes/manifests).
- **Resource check before acquisition:** E: had 204,744,798,208 bytes free (about 190.7 GiB); memory was 15.71 GiB total with 2.56 GiB free. Peak process memory was not measured. No packages were installed.
- Full Sentinel source objects listed by the STAC asset headers total about 1.71 GB for the eight selected band COGs; the WorldCover source object is 75,590,338 bytes. Their full-object sizes are not the amount downloaded. Public one-byte range probes confirmed the object lengths and multipart ETags.

The reproducible acquisition command, run from `backend`, was:

```powershell
& 'E:\project\Hackathon Erode\.venv\Scripts\python.exe' -m geoai.acquire_pilot `
  --output-dir 'E:\satellite-vision-data\phase21-pilot-20261011-final' `
  --chip-size 128
```

The output path must be new. A repeat run must use a different, nonexistent output directory; the command refuses to overwrite an existing path or write within the repository.

## Acquired scenes and regions

Both Sentinel items are Sentinel-2B L2A scenes from 2021-12-24. The catalog-wide cloud percentages below are STAC scene metadata, not measurements over the individual crops and not a cloud mask.

| STAC item / product | Scene date (UTC) | STAC cloud cover | Pilot region(s) |
|---|---|---:|---|
| `S2B_43PGN_20211224_0_L2A` / `S2B_MSIL2A_20211224T051219_N0301_R019_T43PGN_20211224T075301.SAFE` | 2021-12-24 05:25:44 | 0.69016% | Erode; Sathyamangalam |
| `S2B_43PGP_20211224_0_L2A` / `S2B_MSIL2A_20211224T051219_N0301_R019_T43PGP_20211224T075301.SAFE` | 2021-12-24 05:25:29 | 0.586601% | Mettur |

Each region is a 512 × 512 pixel, 10 m window (5.12 km square), using the inference/training band order **B02, B03, B04, B08**. The three centers are Erode (77.72°E, 11.34°N), Mettur (77.80°E, 11.79°N), and Sathyamangalam (77.24°E, 11.51°N). Their image windows are EPSG:32643; WorldCover was reprojected with nearest-neighbor resampling onto each exact Sentinel grid. STAC declares the source bands as `uint16`, nodata 0, scale 0.0001, offset 0, and 10 m spatial resolution. Those source scale/offset values are validated and persisted on the crop rasters; the existing preparation then normalizes raw DN values by 10,000.

The WorldCover tile covers 9–12°N and 75–78°E. Each prepared image has four `uint16` bands, 10 m pixels, and nodata 0. Each label raster is a single `uint8` band with nodata 0. CRS, dimensions, and transforms were verified identical for every image/label pair. The 0.69% and 0.59% cloud figures are scene-level metadata only; no per-pixel cloud mask was applied.

## Labels, chips, and geographic splits

The class code mapping used by the existing training preparation is:

| Code | Class | Pixels across the three crops |
|---:|---|---:|
| 10 | Tree cover | 277,562 |
| 20 | Shrubland | 1,004 |
| 30 | Grassland | 69,200 |
| 40 | Cropland | 89,371 |
| 50 | Built-up | 313,239 |
| 60 | Bare or sparse vegetation | 6,288 |
| 70 | Snow and ice | 0 |
| 80 | Permanent water bodies | 29,715 |
| 90 | Herbaceous wetland | 53 |
| 95 | Mangroves | 0 |
| 100 | Moss and lichen | 0 |

The 786,432 label pixels comprise three complete 512 × 512 windows. Each window had **0 image-nodata pixels and 0 label-nodata pixels**; all 262,144 pixels per region had a nonzero image and label. The preparation generated 48 chips of 128 × 128 pixels (16 per region), rejected **0** chips, and recorded class counts and SHA-256 hashes per sample. Codes absent from these small locations remain in the class mapping with zero counts; this pilot does not represent all classes.

| Geographic region | Split | Prepared chips |
|---|---|---:|
| Sathyamangalam | train | 16 |
| Mettur | validation | 16 |
| Erode | test | 16 |

The minimum measured edge-to-edge distance between the fixed image windows is **44,879.3 m**, exceeding the configured 10,000 m minimum. All chips from one window stay in one deterministic split; split membership is not based on random pixels or filenames. This supports geographic separation of this pilot's windows. It does **not** make the labels independent ground truth. Also, Erode and Sathyamangalam use the same `43PGN` source item, so the held-out test region is not scene-independent; cloud, orbit, and sensor-domain effects can be shared.

## Checksums and provenance

The acquisition manifest at `E:\satellite-vision-data\phase21-pilot-20261011-final\acquisition-manifest.json` records:

- The STAC item IDs, product URIs, timestamps, cloud percentages, per-band public asset URLs, full-object sizes, and ETags.
- The WorldCover product/version, source tile and URL, license, attribution, class codes, and nodata code.
- Region coordinates, image-window bounds, pixel grid, nodata counts, per-region class distributions, chip split counts, rejected-chip reasons, and the minimum measured region gap.
- SHA-256 checksums for all six persisted source-window GeoTIFFs. The prepared `dataset-index.jsonl` records image, label, and sample SHA-256 values for each chip.

The remote S3 ETags are recorded as storage validators only; these objects have multipart ETags, which are **not** treated as cryptographic SHA-256 checksums. Full source-object SHA-256 checksums were not computed because that would require reading the complete large COGs. The six locally persisted crop hashes are:

| Region | Sentinel crop SHA-256 | WorldCover crop SHA-256 |
|---|---|---|
| Erode | `2548bb5cc3238d24e8b589e75bf42134e8091500a27c7154e00a3db0db39b195` | `da5e727215b8a2fb608deebcbe2cee96441b11a3aa28f811add0669703a63416` |
| Mettur | `69dfd43b8b968e0d3ed6c7d59a5902e13d3a19b3fb1f7106f256054c172302c4` | `2c4d76dca1021eb7fba152e4239686b47d2fa1cd9663d26bba1f3353b4db5708` |
| Sathyamangalam | `e9d6626c02018df669829d5034f238223c1cf60b1a7b37914857191e6a067167` | `09c5f1dd48192edeb2f13621f2d9ada6fe86c2459c37043ce6ca49e3fea03395` |

## Implementation and tests

At phase start, Git was on `main` with no staged changes and numerous pre-existing modified and untracked files from prior project work. Those files were left intact. Phase 21 added only the acquisition module, its focused tests, and this report; it did not edit prior project code or configuration.

Added:

- `backend/geoai/acquire_pilot.py`: bounded, fixed-region public-STAC/COG acquisition; validates item identity, date, bands, asset host, ranged object metadata, grids, classes, and region separation; writes checksummed aligned crops and a provenance manifest; passes the rasters to the existing offline `prepare_training_dataset` contract. It requires a new output path outside the repository and never trains or downloads complete source objects.
- `backend/tests/test_acquire_pilot.py`: local-only tests for STAC metadata, invalid dates/assets, range metadata, SHA-256, spatial window gaps, WorldCover codes/nodata, and manifest safety. Existing `test_training_data.py` covers raster alignment, class mapping, nodata, checksums, and region-level split behavior.

Executed verification:

| Check | Result |
|---|---|
| Focused acquisition and training-data tests | **18 passed** |
| Full backend `python -m pytest -q` | **230 passed**, 4 warnings, 22.17 s |
| Backend `python -m compileall -q .` | Passed |
| Backend and AI module imports | Passed (`api.routes`, `api.schemas`, `config`, `processing.imagery`, satellite catalog/acquisition, and AI inference/training/acquisition modules) |
| Acquisition CLI `--help` | Passed; no network access or acquisition on help |
| `python -m pip check` | Passed: no broken requirements |
| Frontend tests | **72 passed** across 12 files |
| Frontend production build | Passed; Vite transformed 814 modules. Output was written to a unique directory outside the repository and then removed; removal was confirmed. |
| `git diff --check` | Passed from the repository root. |

The first frontend-test invocation included `--minWorkers=1`, which this installed Vitest version does not recognize. It failed at CLI argument parsing before tests ran. The supported stable-worker command was rerun as `npm test -- --pool=threads --maxWorkers=1` and all 72 tests passed. The four backend warnings were: one Starlette/httpx test-client deprecation, two Alembic `path_separator` deprecations, and one Rasterio `NotGeoreferencedWarning` in an existing imagery test. No dependency was installed or changed.

Commands run from the indicated project directories:

```text
backend:  python -m pytest tests/test_acquire_pilot.py tests/test_training_data.py -q
backend:  python -m pytest -q
backend:  python -m compileall -q .
backend:  python -c "import api.routes, api.schemas, config, processing.imagery, satellite.catalog, satellite.acquisition, geoai.inference, geoai.training_data, geoai.acquire_pilot, geoai.train_landcover"
backend:  python -m geoai.acquire_pilot --help
backend:  python -m pip check
frontend: npm test -- --pool=threads --maxWorkers=1
frontend: npm run build -- --outDir E:\satellite-vision-data\phase21-frontend-build-20261011 --emptyOutDir
repo:     git diff --check
```

The production-build directory was verified to contain `index.html` and then removed. Git remained on `main` with no staged changes; no changes were committed, pushed, deployed, migrated, or applied to production.

Focused Pylance diagnostics are **not clean**: no diagnostics were reported for the new test file; the new acquisition module retains three `reportCallIssue` notices (two for Rasterio `Window` construction and one for the `geographic_independence_evidence` keyword). The actual acquisition successfully created all six windows and invoked the current `prepare_training_dataset` implementation with that keyword. The filesystem definition in `training_data.py` includes the keyword, but Pylance hover returned a stale signature omitting it; this matches the contradictory signature diagnostic recorded in Phase 20. These diagnostics are reported as unresolved/stale tooling output, not as proof that the files are clean. A workspace-wide Pylance diagnostic count was not rerun.

## Scope and remaining limitations

- This is a small, three-region acquisition pilot, not a production dataset. All regions are in one part of Tamil Nadu and the class distribution is imbalanced.
- Scene cloud metadata is not a substitute for pixel-level cloud/shadow screening; this pipeline did not apply a cloud mask.
- WorldCover is weak supervision; no independent reference labels, model metrics, scientific validation, or accuracy claims were produced.
- The spatial hold-out is based on measured, separated image-window bounds, but two regions share one Sentinel STAC item. It is not a scene-independent or temporally independent evaluation.
- Source-object ETags are not cryptographic hashes; only the persisted crops and prepared samples have SHA-256 checksums.
- The model was **not** trained, no checkpoint was created, and no real satellite-image inference was run.
