# Label Platform (Phase 4 Text + Image Classifiers)

This repository is a local-first model-assisted annotation demo built on **Label Studio OSS**.

Implemented:
- Phase 1: local runtime stack
- Phase 2: image/text pre-annotation demo
- Phase 3: minimal HITL retrain loop + persistent state
- Phase 4: real trainable text classification (`scikit-learn` TF-IDF + LogisticRegression)
- Phase 4 extension: real trainable image classification (`scikit-learn` + lightweight image features)

Current tuning pass focus:
- improved text-classification dataset quality checks
- improved evaluation and inspectability metadata
- clearer prediction confidence/uncertainty output

Still intentionally demo-level:
- image detection and text NER remain deterministic rule-based
- no heavy training infrastructure
- no active learning automation

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
- resolves image references like `/data/local-files/?d=images/demo_blue.png`
- reads actual image pixels from mounted local root
- uses a fixed split marker per task in `meta.dataset_split` (`train` / `eval`)
- fixed eval set id: `image-cls-eval-v1` (kept out of training for stable cross-version metrics)

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
