# Label Platform

This repository is a local-first model-in-the-loop AI annotation platform integrating **Label Studio OSS** with Dockerized ML backends for model-assisted labeling, segmentation quality instrumentation, prompt-stability uncertainty, human correction delta tracking, and lightweight correction-risk learning.

The project includes separate paths for text classification, image classification, and image segmentation HITL. MobileSAM is the validated segmentation runtime for SAM-compatible mask pre-labeling; SAM2 is only future-ready wording in this repository and is not implemented.

## Project Highlights

- Local-first Label Studio OSS annotation platform with Dockerized ML services.
- Model-assisted text classification, image classification, and image segmentation workflows.
- MobileSAM-compatible segmentation backend with Label Studio BrushLabels/RLE output.
- Segmentation quality metadata for mask area, bbox alignment, RLE validity, and review flags.
- Prompt-stability uncertainty estimation by perturbing bbox prompts and measuring mask agreement.
- Human correction delta dataset comparing model-generated masks with human-edited masks.
- Lightweight scikit-learn correction-risk predictor trained from human correction deltas when enough data exists.
- Active-review queue generation that ranks segmentation samples with correction risk, uncertainty, quality flags, geometry heuristics, and lightweight diversity signals.
- Reproducible smoke tests and validation gates for backend health, metadata, artifacts, and prediction formats.

## Implemented vs Future Work

Implemented:
- Local Label Studio OSS stack
- Dockerized ML backend and trainer service
- Text classification training/evaluation path
- Image classification training/evaluation path
- Image segmentation HITL workflow
- MobileSAM-compatible segmentation backend
- BrushLabels/RLE mask output
- Docker MobileSAM smoke validation
- Segmentation quality metadata
- Prompt-stability uncertainty instrumentation when enabled
- Human correction delta dataset
- scikit-learn correction-risk predictor when enough correction data exists
- Active-review queue generation for segmentation review prioritization

Future work:
- True active learning acquisition loop
- Automatic Label Studio task import for selected review items
- Diversity-aware sampling with learned embeddings
- Label-quality auditing / second-review recommendation
- SAM2 runtime backend
- Model comparison dashboard
- Multi-class segmentation
- Larger-scale annotation operations

## Image Segmentation Benchmark v0.1

Benchmark v0.1 evaluates whether the Layer 1-5 segmentation feedback loop finds likely major corrections better than random review. The MVP supports COCO val2017 only. It downloads only COCO `val2017` images plus `instances_val2017.json`, not the full COCO 2017 train/test corpus.

Prepare local data explicitly:

```bash
python -m image_segmentation.benchmark.download_data \
  --dataset coco_val2017 \
  --output-dir data/external
```

Data is stored under:

```text
data/external/coco/
  val2017/
  annotations/instances_val2017.json
```

Run preflight without downloading anything implicitly:

```bash
python -m image_segmentation.benchmark.preflight \
  --config configs/benchmark_v0_1.coco.yaml
```

Build a manifest:

```bash
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.coco.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_manifest.jsonl
```

Run the benchmark:

```bash
python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1 \
  --backend mobile_sam_or_existing_backend \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk
```

Outputs:
- `predictions.jsonl`: model pre-label masks with Layer 1 quality/review metadata and optional Layer 2 uncertainty.
- `correction_delta_dataset.jsonl`: model-vs-COCO-GT correction deltas, marked as `public_gt_simulated_human_annotation`.
- `review_queue.jsonl`: Layer 5 priority queue with delta only in `evaluation_only`.
- `evaluation_report.json` and `evaluation_report.md`: machine and human-readable metrics.

Report metrics:
- model/GT IoU, Dice, precision, and recall measure pre-label mask quality against public ground truth.
- major correction rate and severity distribution summarize how often simulated human correction is substantial.
- grouped dataset/tag metrics show behavior on small, border-touching, elongated, or crowded cases.
- precision@10/20% and recall@10/20% measure how many major corrections are found near the top of the review queue.
- lift@10/20% compares full Layer 5 priority against the random baseline major-correction rate.
- average precision summarizes ranking quality over all major-correction samples.

Current limitations and TODO:
- COCO val2017 is the only fully implemented dataset loader.
- LVIS, DIS5K, COD10K, CAMO, and Open Images are scaffolded for later loaders.
- COCO has no explicit low-contrast, occlusion, or truncation metadata in this MVP.
- Placeholder backend remains a deterministic fallback; MobileSAM requires the existing backend dependencies/checkpoints.

Boundaries:
- This is not a production-grade active learning platform yet.
- The review queue does not automatically modify Label Studio tasks or retrain models.
- It does not fine-tune MobileSAM.
- It does not implement SAM2.
- It does not automatically determine label correctness.
- Image detection and text NER remain deterministic rule-based demo paths.

## Repository Structure

```text
.
├── demo_data/
│   ├── local-files/images/
│   ├── model_state/
│   └── tasks/
├── infra/
│   └── docker-compose.yml
├── label_configs/
├── scripts/
├── services/
│   ├── ml-backend/
│   └── trainer/
└── tests/
```

## Prerequisites (Windows 10 + WSL2 + Docker Desktop)

1. Install WSL2 (Ubuntu recommended).
2. Install Docker Desktop and enable:
   - `Use the WSL 2 based engine`
   - WSL integration for your Ubuntu distro
3. Work from WSL filesystem paths (for example `/home/<you>/projects/...`).

## Start Stack

```bash
cp .env.example .env
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
docker compose --env-file .env -f infra/docker-compose.yml ps
```

Local URLs:
- Label Studio OSS: `http://localhost:18080`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`
- ML backend health: `http://localhost:9090/health`
- Trainer health: `http://localhost:9091/health`

## Label Studio Setup (Image + Text Demo)

1. Open `http://localhost:18080` and sign in.
2. Create project and paste one XML config from `label_configs/`.
3. Import task JSON (`demo_data/tasks/...`).
4. For image tasks using `/data/local-files/?d=...`, configure source storage:
   - `Project Settings` -> `Cloud Storage` -> `Add Source Storage` -> `Local Files`
   - Absolute local path: `/label-studio/files/images`
   - Enable `Treat every bucket object as a source file`
5. Connect ML backend:
   - `Project Settings` -> `Model` -> `Connect Model`
   - URL: `http://ml-backend:9090`

Why local image paths resolve:
- compose mount: `demo_data/local-files` -> `/label-studio/files`
- Label Studio document root: `/label-studio/files`
- task URL `d=images/demo_blue.png` resolves to `/label-studio/files/images/demo_blue.png`

## Phase 4 Text Classification (Real Model)

### Training data sources

Trainer can build text-classification datasets from:
- exported Label Studio JSON (`dataset_path`)
- explicit `samples` payload
- webhook-collected text+label events (`training_dataset_events.jsonl`)

Default demo workflow uses the hand-curated Label Studio export dataset only.

### Quality guards before training

Training now validates dataset quality before fitting:
- minimum total samples: `8`
- minimum per-class samples: `3`
- max label imbalance ratio: `3.0`

If validation fails, trainer returns a clear `train_error` message and keeps fallback behavior.

### Model settings (lightweight)

- `TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2), stop_words='english', sublinear_tf=True)`
- `LogisticRegression(class_weight='balanced', solver='liblinear', C=3.0, max_iter=1500)`

### Artifact location

- `demo_data/model_state/text_classifier/classifier.joblib`
- `demo_data/model_state/text_classifier/vectorizer.joblib`
- `demo_data/model_state/text_classifier/metadata.json`
- `demo_data/model_state/text_classifier/last_training_dataset.jsonl`

Included demo export dataset (balanced, readable):
- `demo_data/tasks/text_classification_labeled_export.json`
- Current size: `48` labeled examples (`24` Positive, `24` Negative)
- Includes a targeted hard-negative block (subtle dissatisfaction phrasing) for robustness checks
- Includes a nuanced-positive rebalance block (`8` calm positive examples) to reduce positive under-representation

## Retrain + Inspect Workflow

Trigger retrain:

```bash
scripts/trigger_text_classifier_retrain.sh
```

Inspect metadata summary:

```bash
scripts/inspect_text_classifier_metadata.sh
```

Run one deterministic local gate command (recommended final check):

```bash
scripts/run_text_classifier_validation_gate.sh
```

If localhost HTTP access is restricted in your environment, run:
`TEXT_GATE_MODE=container scripts/run_text_classifier_validation_gate.sh`

Run fixed regression probes (recommended):

```bash
scripts/run_text_classifier_regression_probes.sh
```

Compare predictions before/after retrain (optional):

```bash
scripts/compare_text_predictions_before_after.sh
```

Regression probes are a compact safety check over canonical examples. The script exits nonzero if required label expectations fail, so you can quickly catch demo regressions after retraining.
The gate command chains retrain + metadata fetch + probes + artifact checks and exits nonzero on failure.
Common failures:
- retrain API error: trainer could not build a valid dataset/model
- probe failure: one canonical text no longer matches expected label
- artifact check failure: expected model files were not persisted

## Phase 4 Image Classification (Real Model)

Image classification now has a real lightweight training path using demo local files and Label Studio-style export JSON.

Training dataset:
- `demo_data/tasks/image_classification_labeled_export.json`
- `demo_data/tasks/image_classification_eval_manifest.json` (fixed independent eval image list)
- `demo_data/tasks/image_classification_training_candidates.jsonl` (webhook/API captured training candidates)
- `demo_data/tasks/image_classification_non_eval_review_tasks.json` (small non-eval human-review pool)
- resolves image references like `/data/local-files/?d=images/demo_blue.png`
- reads actual image pixels from mounted local root
- uses a fixed split marker per task in `meta.dataset_split` (`train` / `eval`)
- fixed eval set id: `image-cls-eval-v1` (kept out of training for stable cross-version metrics)
- deterministic synthetic data generator: `scripts/generate_image_classification_demo_dataset.py`
- default generated size: `90 Product train`, `90 Other train`, `15 Product eval`, `15 Other eval`

Model approach:
- handcrafted color/statistical features from each image
- `KNeighborsClassifier` classifier (scikit-learn)
- deterministic and fast for local demos
- eval metrics are computed on the fixed eval subset; serving model fits train subset only
- metadata includes per-eval prediction rows and explicit `Product/Other` 2x2 confusion summary
- retrain merges base export + training candidates, and filters out fixed-eval images from train candidates

Image artifact location:
- `demo_data/model_state/image_classifier/classifier.joblib`
- `demo_data/model_state/image_classifier/metadata.json`
- `demo_data/model_state/image_classifier/last_training_dataset.jsonl`

Image retrain + inspect:

```bash
scripts/trigger_image_classifier_retrain.sh
scripts/inspect_image_classifier_metadata.sh
scripts/compare_image_predictions_before_after.sh
```

Export review tasks (misclassified or low-confidence eval cases) for Label Studio manual review:

```bash
scripts/export_image_review_tasks_from_metadata.sh
```

Create a small non-eval review pool to prove human corrections can enter retrain:

```bash
scripts/create_non_eval_image_review_tasks.py
```

Bootstrap/import a reproducible Label Studio human-review project:

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'
scripts/bootstrap_label_studio_image_review.py
scripts/import_image_review_tasks_to_label_studio.py
scripts/import_image_review_tasks_to_label_studio.py --tasks-path demo_data/tasks/image_classification_non_eval_review_tasks.json
```

Or run the automated portion of the flow check:

```bash
scripts/run_label_studio_human_review_flow_check.sh
```

Detailed guide: `docs/label_studio_human_review_flow.md`.

Webhook candidate ingest uses Label Studio API task fetch in trainer:
- `LABEL_STUDIO_URL` (default `http://label-studio:8080`)
- `LABEL_STUDIO_API_TOKEN` (required for API fetch on webhook flow)
- `IMAGE_TRAINING_CANDIDATES_PATH` (default `/demo-tasks/image_classification_training_candidates.jsonl`)

Run fixed image regression probes:

```bash
scripts/run_image_classifier_regression_probes.sh
```

Run one deterministic image validation gate (recommended final check):

```bash
scripts/run_image_classifier_validation_gate.sh
```

If localhost HTTP access is restricted in your environment, run:
`IMAGE_GATE_MODE=container scripts/run_image_classifier_validation_gate.sh`

Image probes are a compact safety check over canonical demo images. The probe script exits nonzero if required label expectations fail.
The image gate chains retrain + metadata fetch + probes + artifact checks and exits nonzero on failure.
Common failures:
- retrain API error or wrong mode: image training did not run as `real-image-classifier`
- probe failure: one canonical image no longer matches expected label
- artifact check failure: expected image model files were not persisted

Trainer/ML model metadata endpoints:
- `http://localhost:9091/models/image-classification/current`
- `http://localhost:9090/models/image-classification/current`

## Model-in-the-loop Segmentation Feedback Pipeline

The image segmentation path is structured as an inspectable feedback pipeline around Label Studio BrushLabels/RLE masks and MobileSAM-compatible pre-labels.

```text
Image task + bbox prompt
        ↓
Dockerized MobileSAM backend
        ↓
BrushLabels/RLE pre-label
        ↓
Layer 1: mask_quality + review metadata
        ↓
Layer 2: prompt-stability uncertainty, optional
        ↓
Human review/correction in Label Studio
        ↓
Layer 3: model-vs-human correction delta dataset
        ↓
Layer 4: correction-risk predictor
        ↓
Future: active review queue / active learning loop
```

Layer 1: Segmentation quality metadata
- Adds `mask_quality` and `review` metadata to BrushLabels/RLE predictions.
- Captures mask area, prompt bbox, mask bbox, bbox IoU, RLE length, border touching, and review flags.

Layer 2: Prompt-stability uncertainty
- Optional and disabled by default.
- Perturbs bbox prompts, runs the SAM-compatible backend over variants, computes pairwise mask IoU and disagreement area, and stores `uncertainty` metadata.

Layer 3: Human correction delta dataset
- Compares model-generated masks against human-corrected Label Studio annotations.
- Computes IoU, Dice, added/removed area, correction area ratio, bbox alignment, centroid shift, and correction severity.
- Persists `correction_delta_dataset.jsonl` and summarizes correction deltas in segmentation metadata.

Layer 4: Learned correction-risk predictor
- Uses scikit-learn `LogisticRegression`.
- Trains only when enough correction delta records exist.
- Uses pre-correction features from `mask_quality`, `review`, `uncertainty`, and geometry metadata.
- Predicts whether a pre-label is likely to require major human correction.
- Avoids target leakage by not using `delta` metrics as input features.

Layer 5: Active-review queue
- Combines correction-risk predictions, prompt-stability uncertainty, mask quality review flags, geometry heuristics, and a lightweight diversity signal.
- Produces ranked JSONL records with score components, review reasons, source metadata, and evaluation-only deltas.
- Does not modify Label Studio tasks, sample from an unlabeled pool, or retrain models from the queue.

This is active-learning-style review prioritization, not a full active learning acquisition loop.

## Image Segmentation HITL (SAM/MobileSAM Optional)

Image segmentation has a first-version Brush/mask human-review loop. It uses a single foreground label, `Object`, and keeps the Label Studio -> prediction -> human correction -> webhook -> training data -> retrain -> new pre-label contract testable. By default it uses the deterministic placeholder mask; set `IMAGE_SEG_BACKEND` to use a real SAM-compatible image backend for pre-labels.

Label config:
- `label_configs/image_segmentation.xml`

Artifact location:
- `demo_data/model_state/image_segmentation/metadata.json`
- `demo_data/model_state/image_segmentation/placeholder_model.json`
- `demo_data/model_state/image_segmentation/last_training_dataset.jsonl`
- `demo_data/model_state/image_segmentation/correction_delta_dataset.jsonl`
- `demo_data/model_state/image_segmentation/correction_risk/classifier.joblib`
- `demo_data/model_state/image_segmentation/correction_risk/metadata.json`
- `demo_data/model_state/image_segmentation/correction_risk/feature_names.json`
- `demo_data/model_state/image_segmentation/correction_risk/training_dataset.jsonl`
- `demo_data/model_state/image_segmentation/review_queue.jsonl`

Runtime pre-label backend:

```bash
# default, no heavyweight model dependency
export IMAGE_SEG_BACKEND=placeholder

# lightweight SAM-compatible path
export IMAGE_SEG_BACKEND=mobilesam
export IMAGE_SEG_MODEL_TYPE=vit_t
export IMAGE_SEG_CHECKPOINT=/model-state/image_segmentation/mobile_sam.pt
export IMAGE_SEG_DEVICE=cpu

# optional prompt-stability uncertainty instrumentation
export IMAGE_SEG_PROMPT_STABILITY_ENABLED=true
export IMAGE_SEG_PROMPT_STABILITY_VARIANTS=7
export IMAGE_SEG_PROMPT_STABILITY_JITTER=0.05

# SAM2 path
export IMAGE_SEG_BACKEND=sam2
export IMAGE_SEG_MODEL_ID=facebook/sam2-hiera-large
export IMAGE_SEG_DEVICE=cuda
```

The ML backend expects the optional model libraries to be installed in the runtime image when `IMAGE_SEG_BACKEND` is `mobilesam`, `sam`, or `sam2`. If a real backend is requested but unavailable, prediction falls back to the placeholder mask and includes `requested_backend`, `backend_error`, and `fallback` in prediction confidence metadata. SAM-compatible backends choose one box prompt in this order: `task.data.bbox`/`task.data.box`, `task.meta.bbox`/`task.meta.box`, the best valid bbox from `data.candidates[]` or `meta.candidates[]`, the first valid Label Studio `rectanglelabels` result in predictions or annotations, then a center-box fallback. Candidate boxes with `score` or `confidence` use the highest-scored valid box; unscored candidates keep input order. List values are pixel `x_min, y_min, x_max, y_max`, and mapping values can use `x/y/width/height` or `x_min/y_min/x_max/y_max`. Percent boxes must set `unit`, `units`, or `coordinate_system` to `percent`, `percentage`, `pct`, or `%`; boxes with `normalized=true` use 0-1 coordinates; Label Studio `rectanglelabels` use percent `x/y/width/height`. Result `meta` records `prompt`, pixel `prompt_box`, and `mask_bbox` for traceability.

The segmentation backend now attaches lightweight segmentation metadata instrumentation to each BrushLabels/RLE prediction. The `mask_quality` fields record mask area, prompt/mask bbox alignment, RLE length, border-touching behavior, and image geometry; the quality-aware `review` metadata records simple flags such as `needs_review`, `review_priority`, and `review_reason`. This is a foundation for future uncertainty estimation, prompt-stability uncertainty, and human-correction learning; it is not a full active learning loop or autonomous label-quality assessment.

When enabled, the MobileSAM-compatible backend can perform prompt-stability uncertainty estimation by perturbing the selected prompt bbox, generating multiple candidate masks, and measuring mask stability through pairwise IoU and disagreement area. The resulting `uncertainty` metadata records fields such as `mean_pairwise_iou`, `min_pairwise_iou`, `disagreement_area_ratio`, `stable`, and `stability_bucket`. This remains lightweight uncertainty instrumentation, not a full active learning loop. Prompt-stability is disabled by default, so normal MobileSAM prediction remains a single-pass inference path unless `IMAGE_SEG_PROMPT_STABILITY_ENABLED=true` is set. SAM2 remains optional/future-ready in this repository; this change does not implement a SAM2 runtime. The metadata is a foundation for future correction-risk learning and active review queues.

The segmentation HITL flow records model-vs-human correction deltas when human-corrected BrushLabels masks are available. For each paired model prediction and human annotation, the trainer computes IoU, Dice, added/removed area, correction area ratio, bbox alignment, centroid shift, and correction severity. These records are persisted as a JSONL artifact and summarized in segmentation metadata. When enough labeled correction delta records exist, they also provide the supervised data for correction-risk training.

The segmentation trainer can train a lightweight correction-risk predictor from the human correction delta dataset. The predictor uses scikit-learn `LogisticRegression` with pre-correction metadata such as mask quality, review flags, prompt/mask geometry, and prompt-stability uncertainty to estimate whether a model-generated mask is likely to require major human correction. It trains only when enough correction delta records and both target classes are present. `scikit-learn` is a required dependency for this correction-risk training path. This does not fine-tune MobileSAM and is not a full active learning loop; it is a lightweight feedback model that can later support active review prioritization.

The segmentation trainer can now generate an active-review queue from existing prediction and feedback metadata. The queue combines correction-risk predictions, prompt-stability uncertainty, mask quality review flags, geometry heuristics, and a lightweight diversity signal to rank segmentation samples for human review. This is active-learning-style review prioritization, not a full active learning acquisition loop: it does not automatically sample from an unlabeled pool, modify Label Studio tasks, or retrain models from the queue. The queue is persisted to `demo_data/model_state/image_segmentation/review_queue.jsonl` and summarized in `metadata.json` under `review_queue`.

Optional Docker GPU MobileSAM runtime:

```bash
# prerequisite: Docker can access the GPU
docker run --rm --gpus all pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime nvidia-smi

# prerequisite: this checkpoint exists locally
# models/mobilesam/mobile_sam.pt
cd infra
docker compose build ml-backend-gpu
docker compose up ml-backend-gpu
```

The GPU service maps host `9092` to container `9090` by default; set `ML_BACKEND_GPU_PORT` to override the host port. It runs with `MODEL_STATE_PATH=/app/demo_data/model_state/current_image_segmentation_model.json`, `IMAGE_SEG_BACKEND=mobilesam`, `IMAGE_SEG_MODEL_TYPE=vit_t`, `IMAGE_SEG_CHECKPOINT=/app/models/mobilesam/mobile_sam.pt`, and `IMAGE_SEG_DEVICE=cuda`. The checkpoint is mounted from `../models:/app/models`; do not commit model weights or checkpoints to git. To connect Label Studio to the GPU backend, use `http://ml-backend-gpu:9090` as the ML backend URL. Real MobileSAM pre-labels include `model_version=mobilesam-seg-v0001`, `prediction_source=mobilesam-image-segmentation`, `backend=mobilesam`, prompt metadata, and a Brush RLE with length greater than zero; they should not include placeholder fallback metadata. Label Studio may not preserve top-level prediction confidence in every database view, so use `model_version`, result `meta`, and backend response logs for traceability.

If Docker Desktop on WSL reports a port-forwarding error while publishing host ports `9091` or `9092`, leave the container URLs unchanged and override only the host ports for local checks, for example `TRAINER_PORT=19091 ML_BACKEND_GPU_PORT=19092 docker compose up -d trainer ml-backend-gpu`. Label Studio and the services still communicate over the Docker network with `trainer:9091` and `ml-backend-gpu:9090`.

Run the Docker MobileSAM smoke test after the segmentation review project has imported tasks and has a MobileSAM prediction. Fresh Label Studio databases may assign a different project ID, so use the project ID printed by the bootstrap/import scripts instead of hard-coding project `3`. The host health URL is usually `http://127.0.0.1:9092`, while Label Studio must use the Docker-network backend URL `http://ml-backend-gpu:9090`.

```bash
python scripts/smoke_mobilesam_segmentation_docker.py \
  --compose-dir infra \
  --project-id <segmentation-project-id> \
  --expected-backend-url http://ml-backend-gpu:9090 \
  --expected-model-version mobilesam-seg-v0001 \
  --host-ml-backend-url http://127.0.0.1:9092 \
  --out-dir /tmp
```

If host `9092` is not forwarded in WSL, use `--host-ml-backend-url http://127.0.0.1:19092`. The smoke test checks Docker/compose, container health, checkpoint/model state, Label Studio DB state, latest prediction metadata including `mask_quality`, `review`, `needs_review`, `review_priority`, and `review_reason`, and writes the decoded mask plus overlay PNGs to `/tmp`; failures are reported per check. Before running it, verify the bind mounts expose `/app/models/mobilesam/mobile_sam.pt` and `/app/demo_data/model_state/current_image_segmentation_model.json` inside `ml-backend-gpu`.

Prompt-stability uncertainty is optional in the smoke script. Start the backend with `IMAGE_SEG_PROMPT_STABILITY_ENABLED=true`, generate a fresh prediction, then add `--enable-prompt-stability` to require `uncertainty` metadata in the latest persisted prediction. The default smoke command does not require this because prompt-stability performs multiple MobileSAM inference calls.

Recent local smoke example with a fresh Label Studio database created project `1` and passed with `host_backend_url=http://127.0.0.1:9092`, `project_backend_url=http://ml-backend-gpu:9090`, `model_version=mobilesam-seg-v0001`, `prediction_id=1`, `task_id=1`, `brushlabels=True`, `rle=True`, `choices=False`, and MobileSAM/backend/`prompt_box`/`mask_quality`/`review` metadata present. The decoded overlay sanity check reported `prompt_bbox=[179.0, 47.0, 284.0, 118.0]`, `mask_bbox=[176, 44, 286, 121]`, and wrote `/tmp/mobilesam_prediction_1_mask.png` plus `/tmp/mobilesam_prediction_1_overlay.png`.

Bootstrap/import a segmentation review project:

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'
scripts/bootstrap_label_studio_image_segmentation_review.py
scripts/import_image_segmentation_review_tasks_to_label_studio.py
```

Trigger and inspect segmentation retrain:

```bash
scripts/trigger_image_segmentation_retrain.sh
scripts/inspect_image_segmentation_metadata.sh
```

Run the deterministic segmentation gate:

```bash
scripts/run_image_segmentation_validation_gate.sh
```

Segmentation metadata endpoints:
- `http://localhost:9091/models/image-segmentation/current`
- `http://localhost:9090/models/image-segmentation/current`

## How To Read Metadata

Inspect full metadata:

```bash
curl -sS http://localhost:9091/models/text-classification/current
```

Key sections:
- `dataset.quality`
  - `label_distribution`, `imbalance_ratio`, validation pass/errors
- `dataset.split`
  - split strategy and train/eval class distributions (`fixed_eval_set` when markers are present)
- `dataset.fixed_eval`
  - whether fixed eval is enabled, manifest check status, and fixed eval filenames
- `dataset.candidates`
  - training candidate file path, parse stats, and eval-leakage skipped count
- `metrics.eval_predictions`
  - one row per eval image: image path, true/predicted label, confidence, correct/incorrect
- `metrics.eval_misclassified`
  - misclassified eval sample list for quick review
- `metrics.confusion_matrix_product_other`
  - explicit 2x2 counts: `Product->Product`, `Product->Other`, `Other->Product`, `Other->Other`
- `metrics`
  - `accuracy`, `macro_f1`, `per_label`, `confusion_matrix`, confidence summary
- `model_details.top_terms_by_class`
  - strongest weighted features per class for explainability

## Prediction Confidence / Uncertainty

Text-classification predictions now include confidence details:
- `prediction_source`
- `confidence.confidence`
- `confidence.confidence_bucket`
- `confidence.uncertain`
- `confidence.top_probabilities`

Uncertainty rules are simple and inspectable:
- low top probability (`< 0.60`) OR
- small margin between top-2 probabilities (`< 0.10`)

## Validation Commands

```bash
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
docker compose --env-file .env -f infra/docker-compose.yml ps
curl -sS http://localhost:9090/health
curl -sS http://localhost:9091/health
scripts/trigger_text_classifier_retrain.sh
scripts/inspect_text_classifier_metadata.sh
scripts/run_text_classifier_validation_gate.sh
scripts/run_text_classifier_regression_probes.sh
scripts/compare_text_predictions_before_after.sh
scripts/trigger_image_classifier_retrain.sh
scripts/inspect_image_classifier_metadata.sh
scripts/compare_image_predictions_before_after.sh
scripts/run_image_classifier_regression_probes.sh
scripts/run_image_classifier_validation_gate.sh
```

## COCO1000 Natural Validation

COCO150 was useful for plumbing, but it had only about 8 `major_correction` positives and the OOF metrics had very high variance. COCO1000 expands validation to 1000 COCO instance annotation samples so the correction-risk model and Layer 5 review queue can be checked with more positive examples before changing weights.

COCO1000 means 1000 instance annotations, not necessarily 1000 unique images. It is a natural benchmark: samples are selected from COCO metadata for category balance and image diversity, not from model failure, delta, GT quality, `model_human_iou`, `major_correction`, or `correction_area_ratio`.

Build the manifest:

```bash
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.coco1000.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_manifest.jsonl \
  --summary-output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_manifest_summary.json
```

Run MobileSAM. CPU can take a while, so use resume:

```bash
export IMAGE_SEG_BACKEND=mobilesam
export IMAGE_SEG_CHECKPOINT=/home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt
export IMAGE_SEG_DEVICE=cpu

python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam \
  --backend mobile_sam \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true
```

Run OOF risk evaluation:

```bash
python -m image_segmentation.benchmark.crossfit_risk_evaluation \
  --delta-dataset demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam/correction_delta_dataset.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_oof \
  --n-splits 5 \
  --random-seed 42 \
  --bootstrap-iters 1000
```

Optional held-out split:

```bash
python -m image_segmentation.benchmark.make_eval_splits \
  --delta-dataset demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam/correction_delta_dataset.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_splits \
  --strategy stratified \
  --train-ratio 0.7 \
  --random-seed 42
```

Read the result conservatively. Prefer at least 30 positives, and 50 is better. Initial evidence requires OOF lift@20 > 1.5, OOF AP above the base rate, precision@20 above random expected precision, and bootstrap intervals that are not too wide. If the lift CI covers 1.0, the result is still unstable.

Do not tune Layer 5 weights before COCO1000 OOF evidence. Do not use evaluation diagnostics for ranking, risk-model features, or manifest sampling. Do not auto-download data for this validation path.

## Tests

```bash
python -m unittest tests/test_phase4_text_training.py -v
python -m unittest tests/test_phase4_image_training.py -v
python -m unittest tests/test_text_classifier_regression_probes.py -v
python -m unittest tests/test_image_classifier_regression_probes.py -v
```

## Stop

```bash
docker compose --env-file .env -f infra/docker-compose.yml down
```
