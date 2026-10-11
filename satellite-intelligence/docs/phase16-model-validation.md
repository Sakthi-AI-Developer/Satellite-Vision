# Phase 16 — Local Pixel-Level Satellite Segmentation Model Review

## Decision

**No model is selected for the existing general land-cover map feature.** The strongest verified local Sentinel-2 segmentation candidate found is NASA/IBM's Prithvi-EO-2.0-300M-TL fine-tuned on Sen1Floods11. It is a real single-date georeferenced semantic segmenter, but it predicts only flood/water versus non-water (with an ignore/no-data label). It is not a general land-cover model. Its required six-band Sentinel-2 L1C input and large runtime/checkpoint also differ from the current application input.

The Phase 13 scikit-learn adapter, API, and readiness behavior are unchanged. It remains `model_not_ready` without an operator model manifest and does not claim to be a pretrained satellite model. No model, dataset, or weights were downloaded; no real-image inference or scientific evaluation was run.

## Current application fit

The acquisition pathway currently produces only B02, B03, B04, and B08, from Copernicus Sentinel-2 L2A. The existing AI request accepts one saved scene, and the classifier emits a class-index raster and confidence raster. The selected product goal is a general, pixel-level, georeferenced land-cover map.

## Candidate findings

| Candidate | Task, labels, and input | Preprocessing and resolution | Licenses and provenance | Size, runtime, and fit |
|---|---|---|---|---|
| [Prithvi-EO-2.0-300M-TL-Sen1Floods11](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) | UperNet semantic segmentation fine-tuned from a 300M-parameter Prithvi temporal ViT. The model card specifies single-timestamp segmentation for this fine-tune; classes are no-water (0), water/flood (1), and no-data/clouds (-1). Six Sentinel-2 bands: BLUE/B02, GREEN/B03, RED/B04, NIR_NARROW/B8A, SWIR_1/B11, SWIR_2/B12. It is flood mapping, not general land cover. | Fine-tuning config uses `constant_scale: 0.0001` on Sentinel-2 L1C reflectance stored scaled by 10000. The model card inference script uses 512 × 512 sliding windows and preserves the source GeoTIFF metadata for the output raster; training uses 224 × 224 random crops. Sen1Floods11 sample rasters are WGS84 at 10 m and six 10 m bands are present in the finetune config; the dataset repository describes its source S2 L1C chips as 10 m, UInt16, TOA reflectance scaled by 10000. Current app acquisitions are L2A and lack B8A/B11/B12. | Hugging Face model metadata declares **Apache-2.0** for the model repository; no separate weight-specific grant was verified. Official [Prithvi source code](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) is MIT-licensed; that code license is not treated as a substitute for model-weight terms. The official [Sen1Floods11 repository](https://github.com/cloudtostreet/Sen1Floods11) describes the dataset and label source but does not state a dataset license in the inspected README; dataset rights are **unverified**. | Fine-tuned checkpoint listed as **1,276,843,350 bytes** with Hugging Face LFS SHA-256 **`76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270`**. Hash and size are repository metadata only, not locally verified. Requires PyTorch, TerraTorch, Lightning, and associated model stack. Selected environment has no PyTorch. Weights alone are about 1.28 GB; no authoritative deployment RAM or CPU latency figure was found, and actual memory is unmeasured. No local deployment recommendation without resource testing. **Rejected for general land cover** because of task, band, and L1C/L2A mismatch. |
| [Prithvi-EO-2.0-300M foundation checkpoint](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) | 3D ViT masked-autoencoder reconstruction foundation model, not a land-cover segmentation head. Its official card describes six HLS bands: Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2, with temporal input support. | HLS V2, 30 m product, reflectance units, input files are time-ordered. It is not a ready segmentation model and not the app's four-band Sentinel-2 L2A GeoTIFF. | HF model card declares Apache-2.0; [project code](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) is MIT. These facts do not establish all upstream training data terms. | Checkpoint listed as **1,326,658,396 bytes**, LFS SHA-256 **`faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7`**. PyTorch/TerraTorch ecosystem; no CPU RAM/latency measurement. **Rejected**: output task is reconstruction; using it as segmentation requires a separate, compatible trained head and labels. |
| [Sentinel-2 U-Net ONNX lead](https://huggingface.co/ivanalkhayat/sentinel2-unet-segmentation-onnx) | Repository title suggests single-scene segmentation, but the README contains only license metadata. It does not document architecture beyond the name, input bands/order, output labels/classes, channel shape, or training dataset. | Spatial resolution, normalization, and preprocessing are not documented in its model card/repository README. Compatibility with Sentinel-2, georeferenced data, or this project's four bands cannot be established. | HF metadata declares MIT at repository level. No distinct weight license, training-data license, dataset identity, or provenance is documented. Repository includes an ONNX file and a TorchScript file; do not load the TorchScript artifact because its executable artifact provenance is not established. | ONNX file listed as **19,621,360 bytes**, LFS SHA-256 `c3ab14ed456d3cf2b3e61a5d0309ed3d70ea481177c735b0b83372726db24008`. TorchScript file: **69,255,244 bytes**, LFS SHA-256 `44023ac57cf75a2c429b2f642bed342d6bf2f6a2cdc988e7d97138ac2c31e5b1`. These are listing metadata only. Runtime is not specified; ONNX Runtime is not installed. **Rejected as unverifiable**, notwithstanding its name and small ONNX size. |
| [BIFOLD BigEarthNet v2.0 ResNet-50 S2 v0.2.0](https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0) | 19-class patch-level multi-label classification, not pixel segmentation. Ten-channel input. | 120 × 120 patch per published config. Current model card does not document v0.2.0 band ordering and scaling. Existing four-band input cannot be substituted. | Model card declares MIT for repository; separate checkpoint license and training data usage terms were not verified. Dataset labels derive from CORINE 2018; see Phase 15 report for details. | **94,523,860 bytes**, listed LFS SHA-256 `f55f8a5416d326c7e3139a97c6866e606dfc2e5cde30d52e2ccc47ce5f8b9be5`; PyTorch/configilm. **Rejected**: wrong output semantics and unverified input/licensing requirements. |
| [FTW three-class Sentinel-2 model](https://huggingface.co/torchgeo/ftw) | Two-date field/background/boundary segmentation, not general single-date land cover. | Eight channels, B04/B03/B02/B08 per date; inputs divided by 3000. | Commercial checkpoint declared CC-BY-4.0 by upstream registry; FTW code MIT; FTW dataset terms remain unverified. | **53,233,430 bytes**, metadata SHA-256 `74f7eb810e871bb4a299bf4dc7e2bb59636333d73c7eba76321f1d13d8b3000b`; PyTorch/Lightning stack. **Rejected** for needing two dates and a different task. See [Phase 14 report](./phase14-model-selection.md). |

### Prithvi-Sen1Floods11 sources

- [Official model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) supplies the task, label meanings, six bands, single-timestamp fine-tuning statement, and inference entry point.
- Its [fine-tuning configuration](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/blob/main/config.yaml) specifies `SemanticSegmentationTask`, UperNet, six bands, two output classes, and scaling `0.0001`.
- Its [inference implementation](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/blob/main/inference.py) uses 512-pixel sliding windows and writes predictions using source geospatial metadata.
- The [Sen1Floods11 dataset repository](https://github.com/cloudtostreet/Sen1Floods11) documents 446 labelled 512 × 512 chips, 11 events, 14 biomes, 357 ecoregions, the water/no-water/no-data labels, and imagery characteristics. No dataset license was found in the inspected README; do not treat data as cleared for evaluation or redistribution.
- Weight size and expected SHA-256 above come from the [Hugging Face model repository API](https://huggingface.co/api/models/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/tree/main?recursive=true), not from a downloaded file.

## Why no integration is prepared

Prithvi-Sen1Floods11 demonstrates that compatible single-date Sentinel-2 pixel segmentation is technically possible, but it is a flood detector rather than a general land-cover predictor. Extending its classes or relabeling water/non-water predictions as land-cover classes would be false. It also expects six L1C bands and the current acquisition flow supplies four L2A bands. Even for a separate flood feature, dataset licensing, operator approval for the 1.28 GB checkpoint, runtime dependencies, CPU/RAM bounds, and local checksum verification must be addressed before integration.

The generic Sentinel-2 U-Net lead is not usable evidence: a repository name and a top-level MIT tag do not prove band compatibility, model behavior, checkpoint license, training-data rights, or a trustworthy inference pathway.

## Smallest feasible next step

Make the product's target explicit before any adapter work:

1. If general land cover remains required, locate a released pixel-level land-cover checkpoint (not a foundation backbone alone) with a documented Sentinel-2 input contract, known class map, separate code/weight/data rights, and verifiable artifact provenance. No such candidate was verified in this review.
2. If flood extent is acceptable as a distinct feature, obtain operator approval for a separate Prithvi-Sen1Floods11 adapter and its optional ~1.28 GB artifact; first resolve Sen1Floods11 data rights and deployment RAM/CPU limits, then acquire six matching L1C bands or establish scientifically valid L2A compatibility. Keep generic land-cover readiness and outputs separate.
3. In either case, establish a license-cleared held-out reference set and split by event/geographic area before reporting metrics.

## Evaluation protocol after a model is selected

- Confirm written data-use rights, label semantics, spatial reference, and temporal alignment for the ground truth before loading it.
- Split by non-overlapping geographic tiles/events, keeping every chip and date from a site in one partition. Freeze the test set before threshold or preprocessing selection.
- Record product level, source scene IDs/dates, cloud/no-data masks, native band names/order, resampling, reflectance scale/offset, chip/windowing, model/config/weight digest, and dependency versions.
- For pixel segmentation, report per-class precision, recall, and F1; confusion matrix; overall accuracy; class IoU and mean IoU; valid-pixel coverage; and boundary F1 where boundaries are a target. Report sample counts and class imbalance; choose thresholds only on validation.
- Report upstream published metrics separately from locally reproduced metrics. Do not present synthetic tests or the upstream Sen1Floods metrics as validation of this app or a general land-cover model.

## Completion status

| Item | Status |
|---|---|
| Candidates and authoritative sources researched | Yes; task-specific and unsuitable leads documented |
| General land-cover segmentation model selected | No; selection remains blocked |
| Model adapter or dependency changes implemented | No |
| Checkpoint downloaded and integrity verified locally | No |
| Real Sentinel-2 imagery processed by a pretrained model | No |
| Scientific evaluation completed | No |
