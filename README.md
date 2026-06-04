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
- `review_queue.jsonl`: Layer 5 priority queue with delta only in `evaluation_only`. Experimental `shadow_scores` may include boundary/shape, learned-fusion, and gated-fusion scores; these are shadow-only fields and do not affect default priority score, rank, bucket, or review queue sorting.
- `evaluation_report.json` and `evaluation_report.md`: machine and human-readable metrics.

Shadow-only learned boundary/shape observation is documented in `docs/segmentation_shadow_scoring_rollout.md`. The live runner can process one queue or multiple windows, writes production-mode monitoring with `--no-labels`, verifies default field and ordering equality, aggregates drift/alert status, and exports safe disagreement packets for human review:

```bash
.venv/bin/python -m image_segmentation.benchmark.live_shadow_rollout \
  --input-dir <production_queue_dir> \
  --window-glob "review_queue_*.jsonl" \
  --output-root demo_data/model_state/image_segmentation/benchmark/production_shadow_expanded_observation \
  --artifact-dir <artifact_dir> \
  --enable-learned-boundary-shape-shadow \
  --production-mode \
  --no-labels \
  --max-windows 5
```

This path is shadow-only: it must not modify default Layer 5 weights, default review queue sorting, or prediction-time feature matrices. Expanded observation writes per-window reports plus `expanded_shadow_multi_window_summary.json/.md` and `production_shadow_expanded_rollout_decision_report.md`. Optional human feedback import is offline analysis only via `image_segmentation.benchmark.summarize_shadow_human_feedback`.

Broader shadow observation uses the same runner with at least 10 windows and writes `production_shadow_broader_observation`, `broader_shadow_multi_window_summary.json/.md`, `human_review_task_packet/`, and `production_shadow_broader_rollout_decision_report.md`. Broader summaries flag p95 drift outliers above `1.0` baseline standard deviations so dataset or queue-mix shifts can be inspected before any promotion discussion.

## Segmentation Balanced Human Review Pilot

The first real-human validation pilot uses four balanced groups with 15 samples each, for 60 Label Studio segmentation tasks:

- `high_learned_low_current`
- `high_current_low_learned`
- `top_learned`
- `control_current_top`

This pilot validates whether learned boundary/shape shadow scoring adds useful review signal. It does not change default Layer 5 weights, default `review_queue.jsonl` sorting, or production scoring.

Prepare the pilot from a shadow-scored queue:

```bash
.venv/bin/python -m image_segmentation.benchmark.prepare_balanced_pilot_review \
  --review-queue <shadow_scored_review_queue.jsonl> \
  --output-dir demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04 \
  --per-group 15 \
  --pilot-id segmentation_balanced_pilot_2026_06_04
```

Import into Label Studio:

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'

scripts/bootstrap_label_studio_image_segmentation_review.py
scripts/import_image_segmentation_review_tasks_to_label_studio.py \
  --tasks-path demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/label_studio_tasks.json \
  --dry-run
scripts/import_image_segmentation_review_tasks_to_label_studio.py \
  --tasks-path demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/label_studio_tasks.json
```

After review, fill `human_review_outcome` in `review_assignment.jsonl` or the assignment CSV, then summarize offline feedback:

```bash
.venv/bin/python -m image_segmentation.benchmark.summarize_shadow_human_feedback \
  --review-packet demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/review_assignment.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/human_feedback_summary
```

Human feedback remains offline-only and must not feed production scoring or default sorting.

Report metrics:
- model/GT IoU, Dice, precision, and recall measure pre-label mask quality against public ground truth.
- major correction rate and severity distribution summarize how often simulated human correction is substantial.
- grouped dataset/tag metrics show behavior on small, border-touching, elongated, or crowded cases.
- precision@10/20% and recall@10/20% measure how many major corrections are found near the top of the review queue.
- lift@10/20% compares full Layer 5 priority against the random baseline major-correction rate.
- average precision summarizes ranking quality over all major-correction samples.

Current limitations and TODO:
- COCO val2017 and LVIS val use COCO-style instance annotations; DIS5K, COD10K, CAMO, and Open Images subset support use local manifest/preflight loaders.
- COCO has no explicit low-contrast, occlusion, or truncation metadata in this MVP.
- Placeholder backend remains a deterministic fallback; MobileSAM requires the existing backend dependencies/checkpoints.

### Additional Dataset Setup

List supported datasets:

```bash
python -m image_segmentation.benchmark.list_datasets
```

General principles: `download_data` is always an explicit command. `build_manifest` and `run_benchmark` never download data implicitly. Manual datasets require you to confirm license terms and download sources yourself. `data/external/` is ignored by git and must not be committed. Ground truth and correction deltas are only for benchmark/evaluation; they must not enter priority scoring. Natural validation datasets should not be selected based on model failures.

LVIS val is used for long-tail categories, small objects, and crowded multi-instance scenes. It downloads annotations to `data/external/lvis/annotations/lvis_v1_val.json` and reuses COCO images from `data/external/coco/val2017`.

```bash
python -m image_segmentation.benchmark.download_data --dataset lvis_val --output-dir data/external --dry-run
python -m image_segmentation.benchmark.download_data --dataset lvis_val --output-dir data/external
python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.lvis500.yaml
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.lvis500.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_manifest.jsonl
```

DIS5K is for high-fidelity boundaries and complex foreground object contours. The first implementation is manual placement:

```text
data/external/dis5k/
  images/
  masks/
```

```bash
python -m image_segmentation.benchmark.download_data --dataset dis5k --output-dir data/external
python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.dis5k300.yaml
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.dis5k300.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_manifest.jsonl
```

COD10K and CAMO are for camouflaged, low-contrast objects and prompt-stability uncertainty stress tests. They reuse the mask-folder loader and add `low_contrast` plus `camouflaged_object` tags by default. The recommended configs are `configs/benchmark_v0_1.cod10k300.yaml` and `configs/benchmark_v0_1.camo250.yaml`. The local CAMO split is 250 image/mask pairs. COD10K directories may include `COD10K-NonCAM-*` files with empty masks; the benchmark config filters those out and only samples `COD10K-CAM-*` without moving or deleting data. Manual placement is expected:

```text
data/external/cod10k/images/
data/external/cod10k/masks/
data/external/camo/images/
data/external/camo/masks/
```

```bash
python -m image_segmentation.benchmark.download_data --dataset cod10k --output-dir data/external
python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.cod10k300.yaml
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.cod10k300.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_cod10k300_manifest.jsonl

python -m image_segmentation.benchmark.download_data --dataset camo --output-dir data/external
python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.camo250.yaml
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.camo250.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_camo250_manifest.jsonl
```

Open Images V7 segmentations are for occlusion, truncation, group-object, and instance-segmentation variety. Use only a validation subset at first; do not download the full dataset. The optional FiftyOne path requires `pip install fiftyone` and exports to:

```text
data/external/open_images_v7/validation/
  images/
  segmentations/
  metadata.json
  annotations.jsonl
```

```bash
python -m image_segmentation.benchmark.download_data \
  --dataset open_images_v7_segmentations \
  --output-dir data/external \
  --split validation \
  --max-samples 1000 \
  --use-fiftyone true

python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.open_images500.yaml
```

License and source notes: COCO follows COCO image/annotation terms. LVIS annotations are CC BY 4.0 while images follow their source terms. DIS5K, COD10K, CAMO, and Open Images require checking their current dataset licenses and image-level terms before use.

### LVIS500 Validation

LVIS500 validates whether the COCO1000 finding transfers to a longer-tail, more crowded, smaller-object instance segmentation dataset. COCO1000 showed the MobileSAM benchmark, correction deltas, OOF correction-risk signal, and Layer 5 prioritization are viable; LVIS is the next stress test before any default weight change. Do not run Layer 6 and do not change default Layer 5 weights before LVIS results are reviewed.

Prepare LVIS val annotations. LVIS uses the existing COCO val2017 image directory at `data/external/coco/val2017`; this command downloads annotations only.

```bash
python -m image_segmentation.benchmark.download_data \
  --dataset lvis_val \
  --output-dir data/external
```

Preflight and build the LVIS500 manifest:

```bash
python -m image_segmentation.benchmark.preflight \
  --config configs/benchmark_v0_1.lvis500.yaml \
  --json-output demo_data/model_state/image_segmentation/benchmark/lvis500_preflight.json

python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.lvis500.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_manifest.jsonl \
  --summary-output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_manifest_summary.json
```

Run the MobileSAM GPU benchmark without prompt-stability uncertainty:

```bash
export IMAGE_SEG_BACKEND=mobilesam
export IMAGE_SEG_CHECKPOINT=/home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt
export IMAGE_SEG_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

python -m image_segmentation.benchmark.preflight_backend \
  --backend mobile_sam \
  --checkpoint /home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt \
  --device cuda

python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu \
  --backend mobile_sam \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true
```

Run OOF risk evaluation, weight ablation, and COCO-vs-LVIS comparison:

```bash
python -m image_segmentation.benchmark.crossfit_risk_evaluation \
  --delta-dataset demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu/correction_delta_dataset.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof \
  --n-splits 5 \
  --random-seed 42 \
  --bootstrap-iters 1000

python -m image_segmentation.benchmark.ablate_review_weights \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof/oof_review_queue.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof_weight_ablation \
  --bootstrap-iters 1000 \
  --random-seed 42

python -m image_segmentation.benchmark.compare_benchmark_runs \
  --left-name COCO1000 \
  --left-report demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu/evaluation_report.json \
  --left-oof demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof/oof_summary.json \
  --left-ablation demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --right-name LVIS500 \
  --right-report demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu/evaluation_report.json \
  --right-oof demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof/oof_summary.json \
  --right-ablation demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --output demo_data/model_state/image_segmentation/benchmark/coco1000_vs_lvis500_comparison.md
```

Interpretation: rare-category correction rate tells you whether long-tail LVIS categories are harder, but do not use GT-only frequency metadata in production scoring unless comparable category metadata is available at prediction time. If `risk_heavy` beats current on both COCO1000 and LVIS500, keep the default unchanged and mark `risk_heavy` as a strong experimental candidate. If LVIS disagrees with COCO, validate DIS5K or COD10K next before tuning defaults.

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

## Running COCO1000 MobileSAM On GPU

Use GPU for the MobileSAM COCO1000 benchmark. CPU execution can be impractically slow.

```bash
export IMAGE_SEG_BACKEND=mobilesam
export IMAGE_SEG_CHECKPOINT=/home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt
export IMAGE_SEG_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0
```

Preflight the backend before running the benchmark:

```bash
python -m image_segmentation.benchmark.preflight_backend \
  --backend mobile_sam \
  --checkpoint /home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt \
  --device cuda
```

If CUDA is unavailable, `resolved_device` is not `cuda`, or the model falls back to CPU, stop and fix the environment before continuing. Do not accept placeholder, bbox-rect, or silent CPU fallback for this validation.

Run the COCO100 sanity benchmark first:

```bash
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.coco100.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco100_manifest.jsonl \
  --summary-output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco100_manifest_summary.json

python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco100_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco100_mobile_sam_gpu \
  --backend mobile_sam \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true
```

Then run COCO300:

```bash
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.coco300.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco300_manifest.jsonl \
  --summary-output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco300_manifest_summary.json

python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco300_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco300_mobile_sam_gpu \
  --backend mobile_sam \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true
```

Finally run COCO1000 with resume:

```bash
python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu \
  --backend mobile_sam \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true
```

Runtime details are written to `runtime_metadata.json` in each benchmark output directory. It records requested/resolved device, CUDA availability, GPU name, elapsed time, seconds/sample, resume count, and peak CUDA memory.

Run OOF evaluation after COCO1000 finishes:

```bash
python -m image_segmentation.benchmark.crossfit_risk_evaluation \
  --delta-dataset demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu/correction_delta_dataset.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof \
  --n-splits 5 \
  --random-seed 42 \
  --bootstrap-iters 1000
```

Keep `--enable-prompt-stability false` for COCO1000. Do not tune Layer 5 weights before reviewing the COCO1000 OOF evidence.

## Layer 5 Weight Ablation

COCO1000 OOF showed that `full_priority` is effective, but `correction_risk_only` was stronger. Weight ablation checks whether the current Layer 5 fusion is diluting the risk signal without rerunning MobileSAM or retraining the risk model.

Run COCO1000 OOF weight ablation:

```bash
python -m image_segmentation.benchmark.ablate_review_weights \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof/oof_review_queue.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof_weight_ablation \
  --bootstrap-iters 1000 \
  --random-seed 42
```

The command writes:
- `weight_ablation.json`
- `weight_ablation.md`
- `preset_rankings/*.review_queue.jsonl`

Interpretation:
- `current_full_priority` is the current behavior and remains the default.
- `risk_only` being best means the learned correction-risk signal dominates this split.
- `risk_heavy` close to `risk_only` and above current may be a better engineering compromise because it keeps quality, uncertainty, geometry, and diversity in the score.
- `uncertainty_heavy` should only be interpreted on runs that include uncertainty metadata.

Run COCO300 uncertainty sanity:

```bash
python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco300_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco300_mobile_sam_uncertainty_gpu \
  --backend mobile_sam \
  --enable-prompt-stability true \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true

python -m image_segmentation.benchmark.ablate_review_weights \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco300_mobile_sam_uncertainty_gpu/review_queue.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco300_mobile_sam_uncertainty_gpu_weight_ablation \
  --bootstrap-iters 1000 \
  --random-seed 42
```

Experimental presets live in `configs/review_weight_presets.yaml`. The review queue can load a preset with `IMAGE_SEG_REVIEW_WEIGHT_PRESET` and `IMAGE_SEG_REVIEW_WEIGHT_PRESETS_FILE`, but do not replace the default before validating on a second dataset or another run. COCO is broad but may not reflect harder segmentation datasets such as LVIS, DIS5K, or COD10K.

## DIS5K300 Boundary-Stress Validation

### Purpose

DIS5K300 is a foreground segmentation and high-detail boundary stress test. It is not long-tail category validation. It tests fine-grained boundary quality, `correction_area_ratio`, thin structures, high boundary complexity, and cases where the bbox prompt is correct but the mask boundary still needs correction.

DIS5K is a manual dataset. The CLI prints placement instructions but does not download large files or hard-code unstable Google Drive/Baidu links:

```bash
python -m image_segmentation.benchmark.download_data \
  --dataset dis5k \
  --output-dir data/external \
  --dry-run
```

### Dataset Layout

Expected local layout:

```text
data/external/dis5k/
  images/
  masks/
```

Images and masks are paired by filename stem. Masks should be binary masks, or convertible to binary with `foreground > 0`. Image and mask sizes must match by default. The benchmark does not automatically download DIS5K and does not resize GT masks unless a config explicitly allows that behavior.

### Preflight

```bash
python -m image_segmentation.benchmark.preflight \
  --config configs/benchmark_v0_1.dis5k300.yaml \
  --json-output demo_data/model_state/image_segmentation/benchmark/dis5k300_preflight.json
```

Preflight checks `images_dir`, `masks_dir`, pairing count, missing image/mask counts, empty masks, image/mask size mismatch, valid bbox/area, and `sample_check`.

### Build Manifest

```bash
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.dis5k300.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_manifest.jsonl \
  --summary-output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_manifest_summary.json
```

The manifest includes `mask_path`, `category_name=foreground_object`, `gt_bbox_xyxy`, `difficulty_tags`, and `boundary_metadata`. `boundary_metadata` is GT-derived; it is for diagnostics and evaluation breakdown only. It must not be used for production priority scoring or risk features unless equivalent prediction-time features are implemented.

### Run MobileSAM GPU Benchmark

```bash
export IMAGE_SEG_BACKEND=mobilesam
export IMAGE_SEG_CHECKPOINT=/home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt
export IMAGE_SEG_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu \
  --backend mobile_sam \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true
```

Keep prompt-stability disabled for the full DIS5K300 run. Use GPU, and do not allow fallback. Outputs include `predictions.jsonl`, `correction_delta_dataset.jsonl`, `review_queue.jsonl`, `evaluation_report.json`, `evaluation_report.md`, and `runtime_metadata.json`.

### Boundary Metrics

Boundary metrics are written as evaluation deltas:

- boundary IoU: overlap between tolerated model and human boundary bands.
- boundary F1: harmonic mean of boundary precision and recall.
- boundary precision: fraction of model boundary matched by the GT boundary band.
- boundary recall: fraction of GT boundary matched by the model boundary band.
- boundary error area ratio: symmetric mask difference area divided by image area.

High IoU but low boundary F1 means the region is roughly correct but the boundary is poor. High `high_boundary_complexity` or `thin_structure` failure rates suggest boundary-aware prediction-time features may be needed.

### Prediction-Time Boundary/Shape Features

The benchmark also records prediction-time boundary/shape features derived only from the predicted mask and image size: `pred_area_ratio`, `pred_bbox_area_ratio`, `pred_extent`, `pred_aspect_ratio`, `pred_touches_border`, `pred_boundary_complexity`, `pred_boundary_density`, `pred_component_count`, `pred_largest_component_ratio`, `pred_hole_count`, and `pred_thinness_proxy`.

These features are stored in `prediction_features` and `mask_quality.prediction_time_boundary_shape`. They are safe candidates for Layer 5 priority experiments because they do not use GT masks, IoU/Dice, boundary deltas, correction labels, or human masks.

Evaluation-only fields include `boundary_metadata`, `evaluation_only.delta`, IoU/Dice, boundary IoU/F1/precision/recall, correction deltas, correction severity, and the `major_correction` label. They may be used for offline diagnostics and evaluation labels, but must not become production scoring or learned-fusion input features.

Run prediction-time feature diagnostics:

```bash
python -m image_segmentation.benchmark.diagnose_prediction_features \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu/review_queue.jsonl \
  --output-dir /tmp/dis5k300_boundary_shape_diagnostics
```

Outputs are `prediction_feature_diagnostics.json`, `prediction_feature_diagnostics.md`, and `prediction_feature_bins.jsonl`. The report includes missingness, quantiles, univariate AP/ROC-AUC/PR-AUC, Spearman rank correlation, binned major-correction rates, and a direction suggestion for each feature.

### OOF Risk Evaluation

```bash
python -m image_segmentation.benchmark.crossfit_risk_evaluation \
  --delta-dataset demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu/correction_delta_dataset.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof \
  --n-splits 5 \
  --random-seed 42 \
  --bootstrap-iters 1000
```

Initial effectiveness standard:

```text
OOF lift@20 > 1.5
OOF AP > base rate
OOF precision@20 > random expected precision
```

### Weight Ablation

```bash
python -m image_segmentation.benchmark.ablate_review_weights \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof/oof_review_queue.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof_weight_ablation \
  --bootstrap-iters 1000 \
  --random-seed 42
```

Focus on `current_full_priority`, `risk_heavy`, `risk_only`, and `no_diversity`.

The `boundary_shape_experimental` preset adds a nonzero `boundary_shape_score` component while leaving the default Layer 5 weights unchanged. `boundary_shape_rank_score` and `boundary_shape_calibrated_score` compare rank-normalized and robust log/clipped prediction-time boundary scores. These are experimental comparisons only; do not treat them as production defaults until they are stable across datasets.

Run leakage-safe OOF learned fusion:

```bash
python -m image_segmentation.benchmark.learn_boundary_shape_fusion \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu/review_queue.jsonl \
  --output-dir /tmp/dis5k300_boundary_shape_learned_fusion \
  --n-splits 5 \
  --random-seed 42
```

The learned fusion compares `current_full_priority`, `risk_only`, `risk_heavy`, `boundary_shape_experimental`, `boundary_shape_rank_score`, `boundary_shape_calibrated_score`, `learned_boundary_shape_only`, and `learned_current_plus_boundary_shape`. Each fold fits scaler/model only on the train fold and predicts the held-out fold. If scikit-learn is unavailable, a numpy logistic fallback is used.

### Boundary/Shape Shadow Scoring

Boundary/shape shadow scoring is a production-adjacent, shadow-only rollout path for prediction-time boundary features and learned boundary/shape artifacts. It is disabled by default in review queue generation. Enabling it adds `shadow_scores` and `shadow_score_metadata` to queue items after default Layer 5 scoring and ordering are complete.

Feature flags:

```bash
SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW=false
SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW=false
SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_DIR=
SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_REGISTRY=
SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_VERSION=
SEGMENTATION_BOUNDARY_SHAPE_SHADOW_VERSION=boundary_shape_shadow_v1
SEGMENTATION_BOUNDARY_SHAPE_FAIL_OPEN=true
```

Shadow output keeps a stable contract: `shadow_only: true`, `affects_default_ranking: false`, `score_version`, calibrated/rank boundary scores, learned scores when explicitly configured, and artifact metadata including validation status, missing safe feature counts, and inference errors. Shadow scores never update `priority_score`, `rank`, `priority_bucket`, default Layer 5 weights, or the default sorting key.

Learned artifacts can be loaded from an explicit artifact dir or from a registry:

```text
artifacts/segmentation/boundary_shape/
  learned_boundary_shape_only_v1/
    learned_boundary_shape_only_model.pkl
    learned_current_plus_boundary_shape_model.pkl
    feature_schema.json
    model_metadata.json
    validation_report.json
  CURRENT
```

`CURRENT` contains the active version name. Rollback is a config or `CURRENT` switch. Artifact metadata must stay shadow-only and must declare `affects_default_ranking: false`. Feature schemas reject GT-derived or evaluation-only fields such as GT masks/paths, IoU/Dice, boundary metrics, correction deltas, severity/major-correction labels, dataset GT-only metadata, and `boundary_metadata`.

Fail-safe behavior: missing artifact config produces null learned scores with `artifact_validation_status: not_configured`; missing paths, load failures, schema mismatch, or inference errors produce null learned scores and error metadata while the main queue generation continues when `SEGMENTATION_BOUNDARY_SHAPE_FAIL_OPEN=true`. Explicit validation uses `python -m image_segmentation.benchmark.learn_boundary_shape_fusion --validate-artifact ...` or `validate_artifact_for_shadow(...)` and raises on schema/load failures.

Run monitoring:

```bash
python -m image_segmentation.benchmark.compare_shadow_scores \
  --review-queue <review_queue.shadow_scored.jsonl> \
  --output-dir <output_dir>/shadow_monitoring \
  --window-id <production-window-id> \
  --baseline <previous_shadow_monitoring.json> \
  --production-mode \
  --no-labels
```

Monitoring writes `shadow_monitoring.json`, `shadow_monitoring.md`, and `shadow_disagreement_examples.jsonl` with coverage, score distributions and drift, current-vs-shadow rank agreement, top-k overlap/Jaccard, disagreement volume/examples, latency fields, alert checks, and safety metadata. Latency includes queue generation latency when supplied, shadow scoring total time, monitoring time, review packet export time, artifact load latency when separable, and per-item p50/p95/p99/mean/max timing; unavailable values remain `null`. Production mode does not read labels or `evaluation_only` fields. Offline label metrics are allowed only outside production mode and are marked `evaluation_only`. Promotion requires real shadow feedback, clean artifact validation, stable cross-dataset agreement, no leakage findings, and an explicit review; this rollout is not a default sorting or weight promotion.

The broader human review task packet exports safe-only JSONL groups: `high_learned_low_current`, `high_current_low_learned`, `top_learned`, and `control_current_top`. Human feedback rows may include `group`, `human_review_outcome`, `reviewer`, `reviewed_at`, and `notes`; the offline summary reports group correction rates, hit rates, learned lift versus control, unclear rate, examples, and inter-reviewer agreement. Human outcomes never enter production scoring or default sorting.

When broader rollout triggers non-hard drift, missing-feature, or top-k alerts, use the offline root-cause tools before expanding further:

```bash
python -m image_segmentation.benchmark.analyze_shadow_drift --multi-window-root <broader_root> --baseline-window window_001 --windows window_004 window_007 --output-dir <broader_root>/drift_root_cause
python -m image_segmentation.benchmark.analyze_shadow_missing_features --multi-window-root <broader_root> --windows window_001 window_004 --output-dir <broader_root>/missing_feature_analysis
python -m image_segmentation.benchmark.analyze_shadow_topk_jaccard --multi-window-root <broader_root> --baseline-window window_001 --windows window_002 window_004 window_005 window_007 --output-dir <broader_root>/topk_jaccard_analysis
python -m image_segmentation.benchmark.build_shadow_human_review_assignment --task-packet-dir <broader_root>/human_review_task_packet --output-dir <broader_root>/human_review_assignment
```

These reports use safe fields only. Non-hard alerts block expansion until explained, but they do not require rollback when default fields/order are unchanged, labels are not read, artifact validation is clean, and shadow scores remain isolated.

Replay and review packet:

```bash
python -m image_segmentation.benchmark.apply_shadow_scores \
  --review-queue <input_review_queue.jsonl> \
  --output <output_review_queue.shadow_scored.jsonl> \
  --artifact-dir <artifact_dir> \
  --enable-learned-boundary-shape-shadow

python -m image_segmentation.benchmark.export_shadow_review_packet \
  --review-queue <output_review_queue.shadow_scored.jsonl> \
  --output-dir <output_dir>/shadow_review_packet \
  --top-k 50 \
  --production-mode
```

Initial alert thresholds: learned null rate above `0.05`, any active artifact status not `valid`, inference error rate above `0.01`, missing safe feature p95 above `0`, queue generation latency regression above `10%`, configured shadow scoring p95 or artifact load latency thresholds, p95 distribution shift above `3` baseline std, top100 Jaccard relative change above `50%`, any `shadow_only != true`, or any `affects_default_ranking != false`.

Before enabling shadow rollout, confirm artifact validation passes, feature flags default off, fail-open behavior is tested, default sorting equality is tested, monitoring works, rollback is tested, and alert thresholds are defined. During rollout, track learned coverage, null rate, load failures, inference errors, distribution drift/outliers, top-k overlap, disagreement volume, human feedback group hit rates, learned lift versus control, and latency overhead. Roll back by disabling learned shadow, disabling boundary/shape shadow, clearing artifact env vars, switching artifact version, changing registry `CURRENT`, or redeploying the previous config. Promotion requires stable live coverage, low null/error rates, no latency regression, no leakage findings, stable distributions, useful human-reviewed disagreement outcomes, and label-backed lift on representative production traffic.

Full runbook: `docs/segmentation_shadow_scoring_rollout.md`.

Schema-aware shadow analysis is available for broader rollout windows. Run `classify_shadow_windows` first, then use `--schema-aware --classification <shadow_window_classification.json>` with `analyze_shadow_drift`, `analyze_shadow_topk_jaccard`, and `analyze_shadow_missing_features`. The schema-aware reports distinguish `stability_alert` from `compatibility_warning`: old schema fallback, mixed schema, and zero same-sample overlap are not treated as default-promotion evidence. Same shadow traffic may resume cautiously only when hard safety alerts are absent; small expansion, broader expansion, default sorting, and Layer 5 weight promotion remain held until comparable-window stability and human feedback support them.

Controlled comparable-window validation starts with `select_comparable_shadow_windows`, then runs the same schema-aware drift/top-k/missing-feature tools against `selected_shadow_window_classification.json`. Use `prepare_controlled_human_review_assignment` to create an offline safe-field-only review package. If controlled full-feature windows still show stability alerts, hold small expansion; if they are stable but feedback is missing, continue only same-scope shadow observation.

### Three-Way Comparison

```bash
python -m image_segmentation.benchmark.compare_benchmark_runs \
  --run COCO1000:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu/evaluation_report.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof/oof_summary.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --run LVIS500:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu/evaluation_report.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof/oof_summary.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --run DIS5K300:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu/evaluation_report.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof/oof_summary.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --output demo_data/model_state/image_segmentation/benchmark/coco_lvis_dis5k_comparison.md
```

The three-way comparison answers which dataset is hardest, whether `risk_heavy` beats current across all runs, whether DIS5K exposes boundary failures, and whether boundary prediction-time features are needed.

### Optional DIS5K500

Run DIS5K500 only when DIS5K300 evidence is insufficient: `positive_count < 30`, bootstrap CI is too wide, ablation conclusions are unstable, or boundary diagnostics have too few samples.

Template config:

```yaml
benchmark_id: benchmark_v0_1_dis5k500
random_seed: 42
max_samples_total: 500

datasets:
  dis5k:
    enabled: true
    images_dir: data/external/dis5k/images
    masks_dir: data/external/dis5k/masks
    max_samples: 500

sampling:
  strategy: stratified
  image_diversity: true
  include_tags:
    - small_object
    - touches_border
    - elongated_object
    - thin_structure
    - high_boundary_complexity
```

Then run the same `build_manifest`, `run_benchmark`, `crossfit_risk_evaluation`, and `ablate_review_weights` commands with `dis5k500` output paths.

### Do Not

- Do not use GT-derived `boundary_metadata` for production priority scoring.
- Do not use boundary delta metrics as risk features.
- Do not treat `boundary_shape_experimental` as a default-weight change.
- Do not change default Layer 5 weights based on DIS5K alone.
- Do not run Layer 6.
- Do not enable prompt-stability for full DIS5K300 unless explicitly doing an uncertainty sanity run.

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
