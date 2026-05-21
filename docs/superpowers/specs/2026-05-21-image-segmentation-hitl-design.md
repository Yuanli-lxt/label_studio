# Image Segmentation HITL Loop Design

Date: 2026-05-21

## Goal

Add a first-version image segmentation human-in-the-loop flow that mirrors the existing image classification loop:

1. Import images into a Label Studio segmentation review project.
2. Produce model predictions as Brush/mask pre-labels.
3. Let a human correct the mask with Label Studio Brush tooling.
4. Receive Label Studio webhooks.
5. Fetch the full task from Label Studio and append training candidates.
6. Retrain a segmentation model artifact from export data plus candidates.
7. Use the new model version for the next pre-label cycle.

This first version intentionally prioritizes data contracts, project automation, and repeatable verification over a production-grade segmentation algorithm. The model is a deterministic placeholder that can later be replaced by SAM, YOLO-seg, Detectron2, or a custom model.

## Scope

In scope:

- New Label Studio config for image segmentation using `BrushLabels`.
- Single foreground class: `Object`.
- New `image_segmentation` task type in trainer and ML backend routing.
- Placeholder segmentation prediction that emits a stable `brushlabels` mask result.
- Webhook candidate ingest for Brush/mask annotations.
- Segmentation training candidate JSONL file.
- Segmentation retrain endpoint path and metadata artifact.
- Minimal persisted model artifact or metadata sufficient for the ML backend to identify the active segmentation model version.
- Bootstrap/import helper scripts for segmentation review tasks.
- Unit tests for parsing, candidate append/read, training metadata, and ML backend predictions.
- A validation script that exercises the deterministic local flow without requiring manual Label Studio UI actions.

Out of scope for this first version:

- Production-grade segmentation model quality.
- GPU support.
- External model weight downloads.
- Multi-class segmentation.
- Polygon support.
- Active learning ranking beyond simple deterministic review fixtures.

## Existing Context

The repository already has:

- `services/ml-backend/app.py` with Label Studio `/predict` support for text, image classification, image detection, and text NER.
- `services/trainer/app.py` with real text and image classification training paths.
- Image classification HITL assets:
  - Label config.
  - Bootstrap/import scripts.
  - Label Studio webhook task fetch.
  - Candidate JSONL append/read.
  - Retrain metadata.
  - Validation gate.

The segmentation implementation should follow those patterns rather than introducing a new framework.

## Proposed Architecture

### Label Studio Config

Add `label_configs/image_segmentation.xml`:

```xml
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="mask_label" toName="image">
    <Label value="Object" background="#ff6b6b"/>
  </BrushLabels>
</View>
```

The control names are stable API names:

- image object: `image`
- segmentation control: `mask_label`
- label: `Object`

### Task Type

Add task type normalization for `image_segmentation`:

- Manual endpoints:
  - `/train/image-segmentation`
  - `/retrain/image-segmentation`
- Payload values:
  - `task_type=image_segmentation`
  - `task=image_segmentation`
- ML backend mode inference:
  - If Label Studio config contains `BrushLabels` attached to `Image`, mode is `image_segmentation`.

Existing image classification, text classification, detection, and NER behavior must remain unchanged.

### Prediction Contract

The ML backend returns one prediction per task:

```json
{
  "model_version": "image-seg-v0001",
  "score": 0.65,
  "result": [
    {
      "id": "task_seg",
      "from_name": "mask_label",
      "to_name": "image",
      "type": "brushlabels",
      "original_width": 320,
      "original_height": 240,
      "image_rotation": 0,
      "value": {
        "format": "rle",
        "rle": [...],
        "brushlabels": ["Object"]
      }
    }
  ],
  "prediction_source": "placeholder-image-segmentation",
  "confidence": {
    "prediction_source": "placeholder-image-segmentation",
    "confidence": 0.65,
    "confidence_bucket": "medium",
    "uncertain": false
  }
}
```

The placeholder mask should be deterministic and image-size-aware when the local image file resolves. If the image cannot be resolved, it should fall back to the existing default image size constants. The first placeholder can be a centered foreground region encoded as Label Studio Brush RLE.

### Brush RLE Handling

Add focused helper functions for RLE handling:

- Encode a binary mask to Label Studio Brush RLE.
- Decode Label Studio Brush RLE to a compact internal mask summary if needed by tests.
- Validate annotation results for:
  - `type == "brushlabels"`
  - `value.brushlabels` includes `Object`
  - `value.rle` is present and non-empty
  - `original_width` and `original_height` are positive integers

The first training path does not need to reconstruct full mask tensors for model fitting. It should preserve the raw RLE and dimensions so a future real trainer can consume the candidates without changing the webhook contract.

### Candidate Data Contract

Add default candidate path:

```text
demo_data/tasks/image_segmentation_training_candidates.jsonl
```

Each JSONL row:

```json
{
  "task_id": 901,
  "project_id": 77,
  "image": "/data/local-files/?d=images/demo_product_red.png",
  "label": "Object",
  "rle": [1, 2, 3],
  "original_width": 320,
  "original_height": 240,
  "source": "label_studio_webhook_task_fetch",
  "annotation_id": 9801,
  "updated_at": "2026-05-21T09:00:00Z"
}
```

Candidate ingest should fetch the full Label Studio task, just like image classification. The webhook payload alone is not trusted as the source of training data.

Duplicate saves may append duplicate rows. Retrain should dedupe by resolved `image_path`, label, and split, keeping the latest candidate seen.

### Training Behavior

Add a placeholder segmentation training path:

- Build samples from:
  - Manual `samples` payload.
  - Label Studio export JSON if `dataset_path` is provided.
  - Candidate JSONL.
- Validate minimum viable data:
  - at least one valid image mask sample by default.
  - positive dimensions.
  - non-empty RLE.
  - label must normalize to `Object`.
- Persist:
  - metadata JSON.
  - last training dataset JSONL.
  - optional placeholder artifact JSON that stores model version and simple mask generation settings.

Default artifact directory:

```text
demo_data/model_state/image_segmentation/
```

Container path defaults should mirror the classification naming style:

- `IMAGE_SEG_MODEL_ARTIFACTS_DIR`
- `IMAGE_SEG_TRAINING_CANDIDATES_PATH`

Metadata should include:

- `task_type: image_segmentation`
- `model_version: image-seg-vNNNN`
- dataset totals and candidate stats
- label distribution
- mask dimension summary
- artifact paths
- placeholder model details

The current model state should gain:

```json
"image_segmentation": {
  "active": true,
  "model_version": "image-seg-v0001",
  "trained_at": "...",
  "metadata_path": "..."
}
```

### Retrain Routing

`services/trainer/app.py` should route `image_segmentation` to `_run_image_segmentation_training`.

`services/ml-backend/app.py` should expose segmentation metadata at:

- `/models/image-segmentation`
- `/models/image-segmentation/current`

Trainer should expose the same routes.

### Bootstrap And Import Scripts

Add segmentation-specific scripts rather than overloading image classification script names:

- `scripts/bootstrap_label_studio_image_segmentation_review.py`
- `scripts/import_image_segmentation_review_tasks_to_label_studio.py`
- `scripts/trigger_image_segmentation_retrain.sh`
- `scripts/inspect_image_segmentation_metadata.sh`
- `scripts/run_image_segmentation_validation_gate.sh`

The scripts should reuse `scripts/lib/label_studio_client.py` where practical. The shared client may need generalized settings for segmentation project title, config path, and task path.

Default project title:

```text
Image Segmentation Human Review
```

Default task path:

```text
demo_data/tasks/image_segmentation_review_tasks.json
```

### Demo Data

Use existing local image files for the first validation gate. Add small segmentation review fixtures with `data.image` and optional `meta` only. Do not commit large binary mask files.

If base export fixtures are needed for tests, create compact JSON fixtures that store Brush RLE directly in Label Studio export shape.

## Error Handling

- If a segmentation webhook lacks a task id, return a warning and do not append a candidate.
- If Label Studio task fetch fails, record a warning in the training response.
- If no valid segmentation samples exist, return HTTP 422 for segmentation retrain instead of silently falling back to deterministic classification behavior.
- If a task contains non-Brush annotations, skip those annotations with parse stats.
- If image path resolution fails, skip the sample with a clear error in candidate or dataset stats.

## Testing Strategy

Follow TDD during implementation:

1. Add failing tests for `BrushLabels` config parsing and mode inference.
2. Add failing tests for placeholder mask prediction shape.
3. Add failing tests for parsing a full Label Studio segmentation task into a candidate row.
4. Add failing tests for candidate append/read and invalid row stats.
5. Add failing tests for training metadata and last dataset persistence.
6. Add failing tests for trainer and ML backend segmentation metadata endpoints.
7. Add failing tests for bootstrap/import helper behavior.
8. Add a validation gate script test or shell smoke check.

Targeted test files:

- `tests/test_image_segmentation_prediction.py`
- `tests/test_image_segmentation_training.py`
- `tests/test_image_segmentation_webhook_candidates.py`
- `tests/test_bootstrap_label_studio_image_segmentation_review.py`
- `tests/test_import_image_segmentation_review_tasks_to_label_studio.py`

Existing image classification tests must still pass.

## Review And Verification

Before claiming completion:

- Run focused segmentation tests.
- Run existing image classification tests touched by shared routing/client code.
- Run the segmentation validation gate.
- Run at least one existing image classification validation or unit test to prove no regression in the existing loop.
- Perform a code-review pass focused on:
  - task type routing conflicts
  - Label Studio result shape compatibility
  - candidate contract stability
  - accidental eval/candidate path cross-talk with image classification
  - dirty worktree isolation

## Implementation Order

1. Add tests for parsing and prediction.
2. Implement label config parsing and placeholder prediction.
3. Add tests for webhook candidate extraction.
4. Implement segmentation candidate append/read.
5. Add tests for training metadata.
6. Implement placeholder retrain and metadata persistence.
7. Add tests/scripts for bootstrap/import.
8. Add docs and validation gate.
9. Review and run verification.

## Open Decisions Locked For Version One

- Annotation shape: Brush/mask only.
- Label set: single foreground class `Object`.
- Model quality: placeholder only, replaceable later.
- Project structure: add segmentation path in parallel to image classification.
