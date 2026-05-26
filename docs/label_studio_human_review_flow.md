# Label Studio Human Review Flow

This flow turns image-classifier mistakes or low-confidence eval cases into a repeatable human-in-the-loop correction loop:

1. Export review tasks from current image-classifier metadata.
2. Bootstrap a Label Studio image classification review project.
3. Import review tasks with dedupe.
4. Correct or confirm labels in Label Studio.
5. Let the webhook call trainer.
6. Trainer fetches the full Label Studio task through the API and appends candidates.
7. Retrain merges base export plus candidates while filtering fixed eval images.
8. Run the image validation gate to verify quality and eval-leakage safety.

This is intentionally scoped to image classification review. Image detection and text NER remain demo fallback paths.

## Prerequisites

- Windows 10 + WSL2 + Docker Desktop.
- Docker Desktop WSL integration enabled.
- Stack started from this repository.
- Label Studio account created at `http://localhost:18080`.
- Label Studio API token generated from the UI.
- Label Studio 1.23+ may issue JWT-style API tokens. The helper scripts and trainer support these by refreshing the JWT token automatically before API calls.

## Environment Variables

Host-side helper scripts default to local Label Studio at `http://localhost:18080`. Docker-compose service-to-service defaults are different and use container service names.

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'
export LABEL_STUDIO_TIMEOUT_SECONDS=10
export LABEL_STUDIO_IMAGE_REVIEW_PROJECT_TITLE='Image Classification Human Review'
export LABEL_STUDIO_IMAGE_REVIEW_PROJECT_DESCRIPTION='Human review project for image classification model mistakes and low-confidence cases.'
export LABEL_STUDIO_ML_BACKEND_URL=http://ml-backend:9090
export LABEL_STUDIO_TRAINER_WEBHOOK_URL=http://host.docker.internal:9091/webhook/label-studio
export IMAGE_REVIEW_TASKS_PATH=demo_data/tasks/image_classification_review_tasks.json
export IMAGE_TRAINING_CANDIDATES_PATH=demo_data/tasks/image_classification_training_candidates.jsonl
export IMAGE_EVAL_MANIFEST_PATH=demo_data/tasks/image_classification_eval_manifest.json
export IMAGE_LABEL_CONFIG_PATH=label_configs/image_classification.xml
```

Optional skip flags:

```bash
export LABEL_STUDIO_SKIP_ML_BACKEND_SETUP=1
export LABEL_STUDIO_SKIP_WEBHOOK_SETUP=1
```

Do not put a real token in committed files. For Label Studio 1.23+, use the UI-provided token as-is; if it is a JWT refresh token, the scripts will exchange it for a short-lived access token internally.

## Start Services

```bash
cp .env.example .env
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
docker compose --env-file .env -f infra/docker-compose.yml ps
```

Health checks:

```bash
curl -sS http://localhost:9090/health
curl -sS http://localhost:9091/health
```

## Bootstrap Project

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'
scripts/bootstrap_label_studio_image_review.py
```

The script will:

- Create or reuse `Image Classification Human Review`.
- Apply `label_configs/image_classification.xml`.
- Try to connect ML backend `http://ml-backend:9090`.
- Try to configure webhook `http://host.docker.internal:9091/webhook/label-studio`.
- Print a project URL like `http://localhost:18080/projects/<project_id>/data`.

If ML backend or webhook API endpoints differ in your Label Studio version, the project still remains usable and the script prints manual setup instructions.

## Export Review Tasks

```bash
scripts/export_image_review_tasks_from_metadata.sh
```

Default output:

```text
demo_data/tasks/image_classification_review_tasks.json
```

## Generate Expanded Demo Data

The image demo dataset can be regenerated deterministically without external downloads:

```bash
scripts/generate_image_classification_demo_dataset.py
```

Default output:

- `90` Product train images.
- `90` Other train images.
- `15` Product fixed-eval images.
- `15` Other fixed-eval images.
- `demo_data/tasks/image_classification_labeled_export.json`.
- `demo_data/tasks/image_classification_eval_manifest.json`.
- `demo_data/tasks/image_classification_regression_probes.json`.

The generator uses a fixed seed and lightweight synthetic patterns. Product samples contain a crisp foreground object, while Other samples use textured backgrounds without a product-like center object. This keeps the demo small, reproducible, and easy to inspect.

## Generate Non-Eval Review Tasks

Eval-derived review tasks prove leakage protection. Non-eval review tasks prove that human corrections can enter the next retrain input.

```bash
scripts/create_non_eval_image_review_tasks.py
```

Default output:

```text
demo_data/tasks/image_classification_non_eval_review_tasks.json
```

The script reads the labeled export, filters out every filename in `image_classification_eval_manifest.json`, and writes a concise 3-5 task review pool. It prints:

- total non-eval candidates.
- eval-filtered count.
- invalid skipped count.
- final written task count.
- output path.

## Import Review Tasks

```bash
scripts/import_image_review_tasks_to_label_studio.py
```

Useful variants:

```bash
scripts/import_image_review_tasks_to_label_studio.py --dry-run
scripts/import_image_review_tasks_to_label_studio.py --project-id <project_id>
scripts/import_image_review_tasks_to_label_studio.py --tasks-path demo_data/tasks/image_classification_review_tasks.json
scripts/import_image_review_tasks_to_label_studio.py --tasks-path demo_data/tasks/image_classification_non_eval_review_tasks.json
```

The import script validates `data.image`, skips invalid rows, lists existing project tasks, and dedupes by stable image key.

## One-Command Flow Check

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'
scripts/run_label_studio_human_review_flow_check.sh
```

Options:

```bash
scripts/run_label_studio_human_review_flow_check.sh --dry-run-import
scripts/run_label_studio_human_review_flow_check.sh --skip-export
scripts/run_label_studio_human_review_flow_check.sh --no-ml-backend --no-webhook
```

The script runs the automated part and then stops at the manual review checkpoint.

## Human Correction In Label Studio

1. Open the project URL printed by bootstrap/import, usually `http://localhost:18080/projects/<project_id>/data`.
2. Select an imported image review task.
3. Choose `Product` or `Other`.
4. Click `Submit` for a new annotation or `Update` after editing an annotation.
5. Repeat for a few examples.

Expected behavior after saving:

- Label Studio sends annotation created/updated webhook to trainer.
- Trainer does not trust the webhook payload alone.
- Trainer calls Label Studio task API with `LABEL_STUDIO_API_TOKEN`; both legacy tokens and Label Studio 1.23 JWT tokens are supported.
- Trainer parses the full task annotation result.
- Trainer appends a JSONL candidate row.

## Confirm Candidates

Default candidates file:

```text
demo_data/tasks/image_classification_training_candidates.jsonl
```

Check it:

```bash
tail -n 20 demo_data/tasks/image_classification_training_candidates.jsonl
```

Expected JSONL fields:

```json
{
  "task_id": 901,
  "project_id": 77,
  "image": "/data/local-files/?d=images/demo_product_red.png",
  "label": "Product",
  "source": "label_studio_webhook_task_fetch",
  "annotation_id": 9801,
  "updated_at": "2026-05-12T09:00:00.000000Z"
}
```

Duplicate annotation saves may append multiple rows. Retrain dedupes later by resolved image path and train/eval split, keeping the latest seen sample.

## Retrain

Use the existing image retrain script:

```bash
scripts/trigger_image_classifier_retrain.sh
scripts/inspect_image_classifier_metadata.sh
```

The trainer reads:

- Base export: `/demo-tasks/image_classification_labeled_export.json` inside the trainer container.
- Candidates: `IMAGE_TRAINING_CANDIDATES_PATH`.
- Eval manifest: `IMAGE_CLS_EVAL_MANIFEST_PATH` or `IMAGE_EVAL_MANIFEST_PATH` for helper scripts.

Fixed eval images are filtered out from candidate training samples during dataset build.

## Validation Gate

```bash
scripts/run_image_classifier_validation_gate.sh
```

If host localhost access is blocked, use container mode:

```bash
IMAGE_GATE_MODE=container scripts/run_image_classifier_validation_gate.sh
```

The gate retrains, inspects metadata, runs regression probes, checks persisted artifacts, and verifies fixed eval filenames do not leak into train files.

## Verified Local Smoke Test

Last verified on 2026-05-16 with Label Studio `1.23.0` on Docker Desktop + WSL2:

- Project `Image Classification Human Review` was created/reused as `project_id=1`.
- ML backend `http://ml-backend:9090` connected successfully.
- Webhook `http://host.docker.internal:9091/webhook/label-studio` was created/updated successfully.
- Three image review tasks were imported, and a second import dry run skipped all three as duplicates.
- The expanded deterministic image dataset contains `180` train images and `30` fixed-eval images.
- A Label Studio UI annotation update produced one eval JSONL training candidate:

```json
{
  "task_id": 1,
  "project_id": 1,
  "image": "/data/local-files/?d=images/demo_other_lowlight.png",
  "label": "Product",
  "source": "label_studio_webhook_task_fetch"
}
```

- A non-eval review candidate was also retained as a fixture:

```json
{
  "task_id": "image-non-eval-review-002",
  "project_id": 1,
  "image": "/data/local-files/?d=images/demo_green.png",
  "label": "Other",
  "source": "label_studio_ui_non_eval_smoke"
}
```

The eval candidate belongs to the fixed eval manifest and must be skipped. The non-eval candidate is allowed through the filter. The image gate should report:

```text
candidates used after eval filter: >= 1
candidate eval leakage skipped: >= 1
```

This is the key portfolio story: first the Label Studio correction loop was built, then eval leakage was proven safe, then a non-eval review task proved that human correction candidates can enter retrain, and finally the gate locked the quality boundary.

Validation commands that passed in this state:

```bash
python3 -m unittest discover -s tests -v
scripts/run_image_classifier_validation_gate.sh
scripts/run_text_classifier_validation_gate.sh
```

To reproduce the non-eval path manually:

```bash
scripts/generate_image_classification_demo_dataset.py
scripts/create_non_eval_image_review_tasks.py
scripts/import_image_review_tasks_to_label_studio.py --tasks-path demo_data/tasks/image_classification_non_eval_review_tasks.json
```

Then open Label Studio, annotate or update one non-eval review task, confirm `image_classification_training_candidates.jsonl` gains a non-eval row, and run:

```bash
scripts/trigger_image_classifier_retrain.sh
scripts/inspect_image_classifier_metadata.sh
scripts/run_image_classifier_validation_gate.sh
```

## Troubleshooting

API token missing:

```text
LABEL_STUDIO_API_TOKEN is required
```

Create/copy a token from Label Studio UI, then export it in the current shell.

Label Studio connection failed:

```bash
docker compose --env-file .env -f infra/docker-compose.yml ps
curl -sS http://localhost:18080/health
```

Project duplicated:

- The bootstrap script reuses exact title matches.
- Check `LABEL_STUDIO_IMAGE_REVIEW_PROJECT_TITLE` if you intentionally changed the title.

Webhook not triggered:

- Confirm webhook URL is reachable from the Label Studio container, not from the host.
- In this Docker Desktop + WSL2 setup, prefer `http://host.docker.internal:9091/webhook/label-studio`; Label Studio validates this URL and the container can reach the host-published trainer port.
- The Docker service URL `http://trainer:9091/webhook/label-studio` is reachable inside the compose network, but Label Studio 1.23 rejects the single-label hostname in webhook URL validation.
- If Label Studio runs directly on the host instead of in compose, `http://localhost:9091/webhook/label-studio` can also work.

ML backend connection failed:

- In compose, Label Studio should use `http://ml-backend:9090`.
- From host tools, health is `http://localhost:9090/health`.
- API endpoint differences across Label Studio versions may require manual connection in Project Settings -> Model.

Candidates remain empty:

- Confirm webhook exists and is active.
- Confirm annotations were submitted or updated.
- Confirm trainer has `LABEL_STUDIO_API_TOKEN` in compose environment if the running container was started before token configuration; restart trainer if needed.
- Check trainer response warnings from webhook calls.

Annotation payload parsing failed:

- The image label config must use `Choices name="image_label" toName="image"` with values `Product` and `Other`.
- Empty labels, cancelled annotations, unsupported labels, or non-image tasks are skipped with warnings.

Eval leakage blocked:

- If `candidate_eval_leakage_skipped` is greater than zero, a candidate image matched the fixed eval manifest and was not used for training.
- This is expected safety behavior.

Docker networking gotcha:

- Host scripts talk to Label Studio via `http://localhost:18080`.
- Label Studio container talks to ML backend and trainer via service names: `http://ml-backend:9090` and `http://trainer:9091`.
- `localhost` inside a container is that container itself, not your WSL host.
- For webhook configuration, Label Studio 1.23 validates the URL and rejects single-label hostnames such as `trainer`; use `http://host.docker.internal:9091/webhook/label-studio` on Docker Desktop + WSL2.

## Demo Talk Track

One-minute version:

> This demo closes the image-classification HITL loop. The model exports eval mistakes or low-confidence cases into Label Studio, a reviewer corrects Product/Other labels, and a webhook sends the event back to trainer. Trainer fetches the full Label Studio task through the API, writes a candidate JSONL row, retrains with base data plus candidates, and the validation gate confirms fixed eval samples did not leak into training.

Three-minute technical version:

> The platform runs Label Studio, ML backend, trainer, and storage in Docker Compose. For image classification, training uses a lightweight scikit-learn KNN over handcrafted image features, with a fixed independent eval manifest. The new bootstrap script creates or reuses a Label Studio review project, applies the existing XML label config, and idempotently configures the ML backend and webhook. The import script reads generated review tasks, validates `data.image`, checks existing project tasks, and imports only new images. During human review, Label Studio sends annotation events to trainer. Trainer treats the webhook as a trigger only: it fetches the complete task over the Label Studio API, parses the current Product/Other annotation, appends a candidate row, and later retrain merges candidates after filtering any eval-manifest images. The gate then retrains, probes predictions, checks artifacts, and verifies fixed eval filenames stay out of train files.
