# Phase 15 — Sentinel-2 AI Model Selection and Integration Readiness

## Decision

**No pretrained model is selected for integration in this phase.** The most relevant verifiable local model candidate is the BIFOLD BigEarthNet v2.0 ResNet-50 Sentinel-2 patch classifier. It is a single-scene multispectral classifier, but it does not match the current product contract:

- The app currently acquires B02, B03, B04, and B08 only; the candidate checkpoint has 10 input channels.
- The candidate emits 19 multi-label predictions for a 120 × 120 patch, not a per-pixel class raster. Repeating patch-level outputs across pixels would fabricate spatial precision.
- The model card does not establish the precise v0.2.0 per-band preprocessing or a separate license grant for the weight artifact. BigEarthNet/CORINE data terms also need verification for the intended use.
- The model requires an additional PyTorch/configilm runtime that is not installed in the selected environment. CPU peak memory and latency have not been measured for this service.

The existing scikit-learn interface is retained and unchanged. Its test fixture is synthetic and is not evidence of pretrained-model performance.

## Current application contract

The Copernicus acquisition module requests B02/B03/B04/B08. `GET /api/imagery/ai-models` and `POST /api/imagery/scenes/{scene_id}/ai-inference` expose a single saved scene. The current inference adapter returns a class-index GeoTIFF and a per-pixel maximum-probability GeoTIFF, preserving the source grid. No paired-date field-boundary workflow is present.

## Candidate comparison

| Candidate | Task/output | Sentinel-2 input and preprocessing | Data and domain | Licenses and artifact | Runtime/deployment fit | Decision |
|---|---|---|---|---|---|---|
| [BIFOLD BigEarthNet v2.0 ResNet-50 S2 v0.2.0](https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0) | Single-image, 19-class **multi-label patch classification**, not per-pixel segmentation. Model card reports a 120 × 120 input and 19 outputs in its config. | Config reports 10 channels and 120 × 120. The model card links to the reBEN training repository but does not specify the v0.2.0 band sequence, band scaling, or cross-resolution resampling in the card. The older v0.1.1 card lists B02/B03/B04/B08/B05/B06/B07/B11/B12/B8A but warns that its order is incompatible with later versions; do not copy that order to v0.2.0 without upstream confirmation. The current app only supplies four 10 m bands. | BigEarthNet v2.0 uses Sentinel-2 L2A imagery from June 2017–May 2018 over 10 European countries and derives labels from CORINE Land Cover 2018. Geographic and temporal domain is therefore European and dated. | Hugging Face card metadata declares **MIT** for the model repository. The card does not identify a distinct license for the checkpoint separate from that metadata. Training data use/redistribution terms for the relevant imagery and CORINE labels have not been established for this deployment. The 94,523,860-byte `model.safetensors` file is listed with LFS SHA-256 `f55f8a5416d326c7e3139a97c6866e606dfc2e5cde30d52e2ccc47ce5f8b9be5`; repository metadata only, not locally checked. | ResNet-50, 23,622,035 F32 parameters per model metadata. Requires PyTorch and `configilm` per its model card. PyTorch is not installed in the selected Pylance environment; peak CPU memory/latency and deployment RAM budget are unknown. | **Best candidate for a separate patch-label feature only**, after resolving band/preprocessing and license questions. It is not compatible with the current four-band pixel-map interface. |
| [FTW three-class Sentinel-2 model](https://huggingface.co/torchgeo/ftw) | Two-date field-boundary semantic segmentation: background, field, boundary. It is not general land-cover classification. | Two aligned dates, eight channels: B04/B03/B02/B08 per date; upstream preprocessing divides by 3000. | FTW dataset description reports 24 countries/four continents and spatially blocked splits, but local compatible ground truth and applicable dataset terms are not established. | Upstream model registry identifies a commercial CC-BY-4.0 checkpoint. Three-class file metadata lists 53,233,430 bytes and SHA-256 `74f7eb810e871bb4a299bf4dc7e2bb59636333d73c7eba76321f1d13d8b3000b`; not locally checked. FTW code repository is MIT. | PyTorch/Lightning/segmentation-models-pytorch; no runtime or checkpoint is installed. CPU needs are unmeasured. | Rejected for current single-scene task; requires an explicit paired-date field-boundary feature. See [Phase 14 research](./phase14-model-selection.md). |
| [Google Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) | Single-Sentinel-2-image, pixel-level 9-class land-cover probabilities and label product. Closest task shape to the current map. | Official dataset page says predictions derive from Sentinel-2 L1C and describes cloud/shadow filtering and probabilities. It does not publish the complete model band order, normalization, or standalone weights for local reproduction. | Near-real-time global coverage; class probabilities may be weak for classes whose interpretation depends on temporal context. | Hosted prediction dataset is published through Google Earth Engine; independently usable model code/weights and their license/checksum are not provided by this product page. This is not a verified local open-weight candidate and would add a hosted inference/data-service dependency. | Not a local CPU model; needs Earth Engine access and its terms/configuration. This project has no such integration. | Rejected for the requested local, offline-capable open-source model integration. |

### BigEarthNet checkpoint evidence

The [v0.2.0 model card](https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0) reports a held-out BigEarthNet test split with macro/micro average precision 0.713706/0.858878 and macro/micro F1 0.650478/0.764773. These are upstream-reported metrics, not independently reproduced here and not evidence of performance on this project’s scenes. Model metadata and file sizes were read from the [Hugging Face model API](https://huggingface.co/api/models/BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0) and [file listing](https://huggingface.co/api/models/BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0/tree/main?recursive=true). The artifact was not downloaded.

The BigEarthNet v2.0 project page describes the dataset, acquisition period, countries, and CORINE-derived labels: [BigEarthNet](https://bigearth.net/). The [reBEN training repository](https://git.tu-berlin.de/rsim/reben-training-scripts) was not accessible in this research session (HTTP 403), so its full preprocessing and license details could not be independently checked. No weight or dataset license beyond the model card's repository-level MIT declaration is assumed.

## Integration readiness

No adapter, dependency, model configuration, or weight was added. Readiness remains disabled because no candidate meets the current task, input, label-licensing, and inference-runtime requirements together. No automatic weight download or model loading was attempted.

If the product owner chooses a patch-level multi-label output, integration still requires:

1. Written upstream confirmation of v0.2.0 band order, reflectance scaling, and treatment/resampling of 10 m and 20 m input bands.
2. Verification of checkpoint and training-data licenses for the intended deployment/use.
3. An operator-approved private artifact path, maximum artifact size, and local digest verification against an authoritative checksum.
4. Selection of a trustworthy safetensors-only loading path and dependency versions; do not load pickle/joblib as pretrained weights.
5. Measured CPU peak memory and latency under the application's pixel/request bounds before enabling readiness.
6. An explicit separate API/UI contract that reports patch-level multi-label scores and does not represent them as per-pixel masks.

If the desired result remains a georeferenced per-pixel land-cover map, continue searching for a reproducible local segmentation checkpoint whose task, band requirements, preprocessing, code/weight/data licenses, and artifact provenance can all be verified. Do not substitute Dynamic World hosted outputs or FTW paired-date field boundaries for that requirement.

## Scientific evaluation protocol

Do not report model accuracy until this evaluation is run on a license-cleared, independent ground-truth sample.

1. **Ground truth and label mapping:** Confirm the ground-truth provider, license, label definitions, acquisition dates, spatial resolution, and compatibility with the chosen output task. For BigEarthNet, use patch-level CLC labels only if their data-use terms are confirmed; do not treat them as pixel masks.
2. **Spatially independent split:** Assign entire non-overlapping Sentinel-2 MGRS tiles or geographic blocks to train/validation/test before creating patches. Keep neighboring or overlapping patches and all dates from the same AOI in one split. Freeze the test set before threshold/model selection to limit spatial and temporal leakage.
3. **Preprocessing reproducibility:** Record scene/product identifiers, L1C/L2A processing level, dates, cloud masking, band sequence, native resolution, resampling method, reflectance scaling, nodata policy, chip size, and normalization. Reject mismatched or incomplete inputs; do not synthesize missing bands.
4. **Patch-level multi-label metrics, if that task is approved:** Report per-class precision, recall, and F1 at a threshold selected only on validation; micro/macro averages, overall average precision, confusion/co-occurrence summary, sample counts, class prevalence, and uncertainty intervals. Do not convert patch scores into pixel maps.
5. **Pixel-level segmentation metrics, if that task is selected:** Report per-class precision/recall/F1, confusion matrix, overall accuracy, mean IoU, class IoU, boundary F1 (if field boundaries are the target), and valid-pixel coverage. Mask clouds/nodata and document spatial resolution and ground-truth alignment.
6. **Reproducibility:** Save the dataset version/split manifest, code and dependency versions, model/checkpoint SHA-256, configuration, inference environment, and metric script. Report upstream published metrics separately from locally reproduced results. Synthetic rasters are test fixtures only.

## Status

| Completion item | Status |
|---|---|
| Model researched | Yes; candidates and task/input mismatches documented |
| Model selected for current application contract | No; none verified as compatible end-to-end |
| Model dependencies configured | No |
| Checkpoint provisioned and verified | No; no download attempted |
| Inference on real Sentinel-2 imagery | No |
| Scientific evaluation completed | No |

## Product decision recorded

The product owner selected **pixel-level georeferenced land-cover classes**. Continue searching for a locally runnable, appropriately licensed segmentation checkpoint that matches the acquired bands and has verifiable preprocessing and provenance. BigEarthNet's patch-level classifier is not selected for this output; do not spatially upsample its patch scores into a pixel map. No model integration is justified until a compatible segmentation candidate and its licenses, weights, and resource requirements are verified.
