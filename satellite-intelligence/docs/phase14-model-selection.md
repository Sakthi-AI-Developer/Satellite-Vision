# Phase 14 Sentinel-2 Model Selection

## Decision

**No model is integrated or enabled by this assessment.** The best verified candidate is the three-class Fields of The World (FTW) Sentinel-2 temporal field-boundary segmentation model. Its published input and output are materially different from the current single-scene, four-band scikit-learn land-cover inference contract. Integrating it safely requires an explicit paired-scene workflow, an operator-provisioned checkpoint, and compatible licensed evaluation data. No checkpoint or FTW evaluation dataset is present in the backend's expected `data/models` and `data/ftw` locations, and the paired-scene API does not exist.

The existing scikit-learn adapter remains unchanged. Its synthetic test fixture is not a pretrained satellite model and is not evidence of real-world performance.

## Candidate review

### Fields of The World (FTW), three-class Sentinel-2 model

- **Official project and implementation:** [FTW baselines](https://github.com/fieldsoftheworld/ftw-baselines); the project code is MIT-licensed according to its [LICENSE](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/LICENSE).
- **Model card and artifact repository:** [TorchGeo FTW model card](https://huggingface.co/torchgeo/ftw) and its [repository file listing](https://huggingface.co/api/models/torchgeo/ftw/tree/main?recursive=true).
- **Architecture and task:** U-Net with an EfficientNet-B3 encoder, trained for semantic segmentation of agricultural field boundaries. The released three-class mapping is background/neither, field, and boundary; it is not a general land-cover taxonomy.
- **Input:** Two temporal Sentinel-2 observations stacked into eight channels. The upstream model configuration and band mappings specify B04, B03, B02, B08 for each observation. The documented preprocessing divides input values by 3000. The observations must be spatially compatible; duplicating one date to fill the second-date channels would not be valid.
- **Dataset and labels:** The [FTW dataset description](https://source.coop/kerner-lab/fields-of-the-world) describes more than 70,000 samples across 24 countries, multi-date multispectral Sentinel-2 imagery, field-boundary labels, and 3×3 blocked tile groups split 80/10/10 into train/validation/test. This published split description is not an independent leakage audit. The specific data license and redistribution terms for any evaluation subset have not been verified for this project.
- **Weights and provenance:** The [FTW release configuration](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/configs/release/3_class/cc-by-ftw.yaml) is the CC-BY three-class checkpoint configuration. The [upstream model registry](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py) identifies the commercial-use variant as CC-BY-4.0; the model card distinguishes the `commercial` and `noncommercial` checkpoint folders. The commercial three-class TorchGeo checkpoint is listed as **53,233,430 bytes** with LFS SHA-256 **`74f7eb810e871bb4a299bf4dc7e2bb59636333d73c7eba76321f1d13d8b3000b`**. These are repository metadata values only: the checkpoint was not downloaded, and its digest was not verified against a local file.
- **Runtime and resource fit:** The upstream release configuration uses Lightning and segmentation-models-pytorch with a GPU training setup. The backend currently has no PyTorch/TorchGeo runtime dependency or model artifact. CPU inference time and peak memory for this app’s raster sizes have not been measured.

Other FTW release variants are not selected: the two-class version omits the boundary class, while the `noncommercial` checkpoint is not appropriate for an application that may be used commercially.

## Compatibility and verification blockers

1. The existing AI API and UI accept one saved scene at a time. FTW needs two dates and eight correctly ordered channels. The application does not currently collect and validate an aligned, same-area scene pair for this model.
2. The existing Phase 13 inference workflow is a single-scene scikit-learn classifier. Reusing that contract or relabeling its output would misrepresent the model task.
3. No checkpoint is installed. No weights were fetched; operator approval and a documented artifact configuration/size limit are required before provisioning any model file. Readiness and inference therefore remain unverified.
4. No FTW ground-truth evaluation subset is present in the backend's expected `data/ftw` location. Its applicable data license has not been established here, so no dataset was downloaded or used. This does not rule out data stored outside that location.
5. No real-scene inference, geospatial output comparison, or scientific evaluation has run. Precision, recall, F1, confusion matrix, and accuracy are **not available**. Synthetic test fixtures cannot fill this evidence gap.

## Evidence status

| Requirement | Status |
|---|---|
| Candidate identity and model task | Verified from upstream project configuration and documentation |
| Sentinel-2 input compatibility | Verified for paired B04/B03/B02/B08 imagery; incompatible with the current single-scene request |
| Code license | MIT, verified from the FTW repository |
| Commercial checkpoint variant and declared license | Upstream registry identifies the commercial three-class checkpoint as CC-BY-4.0; local artifact not inspected |
| Checkpoint size and expected SHA-256 | Verified from Hugging Face repository metadata; not verified against downloaded bytes |
| Model artifact installed | No |
| Inference on real satellite imagery | No |
| Evaluation against compatible ground truth | No |
| Scientific metrics | Not measured |

## Requirements before implementation can be validated

An operator must provide or authorize:

1. A private, local location for the commercial three-class checkpoint, with its expected SHA-256 and the existing model-size bound checked against the published 53,233,430-byte artifact.
2. Two owner-scoped saved scenes covering the same AOI, with B04/B03/B02/B08 bands, compatible grids, acquisition metadata, and documented scaling/nodata handling.
3. A ground-truth evaluation subset whose license and label mapping are confirmed, plus a held-out spatial split suitable for this evaluation.
4. Approval to add and maintain an optional PyTorch/TorchGeo-compatible inference runtime, then to measure CPU memory and latency limits before exposing the model as ready.

Until those conditions are met, the safe status is **candidate documented; integration, artifact installation, real-image inference, and scientific validation incomplete**.
