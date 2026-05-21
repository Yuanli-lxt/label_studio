# Image Segmentation HITL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-version Label Studio image segmentation HITL loop using Brush/mask annotations, single `Object` label, placeholder retraining, and new-model pre-labeling.

**Architecture:** Add `image_segmentation` as a parallel task type beside existing image classification. The ML backend emits Label Studio-compatible `brushlabels` RLE predictions, while the trainer stores corrected Brush masks as candidate JSONL rows and persists placeholder segmentation metadata/artifacts for the next pre-label cycle.

**Tech Stack:** Python stdlib HTTP services, `unittest`, Pillow/numpy when available, existing Label Studio helper scripts, and `label-studio-converter` brush utilities for RLE compatibility.

---

## References

- Label Studio BrushLabels tag: https://labelstud.io/tags/brushlabels
- Label Studio pre-annotations guide, Brush RLE section: https://labelstud.io/guide/predictions.html

## File Structure

- Create `label_configs/image_segmentation.xml`: Label Studio BrushLabels config using `Object`.
- Modify `services/ml-backend/app.py`: parse `BrushLabels`, infer `image_segmentation`, load segmentation metadata/artifact, generate placeholder RLE predictions, expose segmentation model metadata routes.
- Modify `services/trainer/app.py`: normalize and route `image_segmentation`, parse full Label Studio Brush tasks, append/read segmentation candidates, train placeholder metadata/artifact, expose metadata routes.
- Modify `services/ml-backend/Dockerfile` and `services/trainer/Dockerfile`: install `label-studio-converter` if not already available.
- Modify `infra/docker-compose.yml`: add segmentation artifact/candidate environment variables.
- Modify `scripts/lib/label_studio_client.py`: generalize settings enough for segmentation project/task defaults without breaking classification helpers.
- Create `scripts/bootstrap_label_studio_image_segmentation_review.py`: bootstrap segmentation review project.
- Create `scripts/import_image_segmentation_review_tasks_to_label_studio.py`: import segmentation review tasks with dedupe.
- Create `scripts/trigger_image_segmentation_retrain.sh`: call trainer retrain endpoint.
- Create `scripts/inspect_image_segmentation_metadata.sh`: print segmentation metadata summary.
- Create `scripts/run_image_segmentation_validation_gate.sh`: deterministic gate for retrain, metadata, prediction, and artifacts.
- Create `demo_data/tasks/image_segmentation_review_tasks.json`: small importable image task fixture.
- Modify `README.md`: add segmentation workflow docs.
- Create tests:
  - `tests/test_image_segmentation_prediction.py`
  - `tests/test_image_segmentation_training.py`
  - `tests/test_image_segmentation_webhook_candidates.py`
  - `tests/test_bootstrap_label_studio_image_segmentation_review.py`
  - `tests/test_import_image_segmentation_review_tasks_to_label_studio.py`

## Task 1: Label Config Parsing And Prediction Mode

**Files:**
- Create: `label_configs/image_segmentation.xml`
- Modify: `services/ml-backend/app.py`
- Test: `tests/test_image_segmentation_prediction.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_segmentation_prediction.py` with:

```python
import importlib.util
import os
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_BACKEND_APP = ROOT / "services" / "ml-backend" / "app.py"


def load_backend(env_overrides):
    backup = {}
    for key, value in env_overrides.items():
        backup[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        module_name = f"test_image_seg_prediction_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, str(ML_BACKEND_APP))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        for key, previous in backup.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


class ImageSegmentationPredictionTests(unittest.TestCase):
    def _backend(self, tmp_dir):
        return load_backend(
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                "IMAGE_SEG_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_segmentation"),
                "IMAGE_LOCAL_FILES_ROOT": str(ROOT / "demo_data" / "local-files"),
            }
        )

    def test_parse_brushlabels_config_infers_image_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
            parsed = backend._parse_label_config(config)
            self.assertEqual("mask_label", parsed["brushlabels"]["name"])
            self.assertEqual("image", parsed["brushlabels"]["to_name"])
            self.assertEqual(["Object"], parsed["brushlabels"]["labels"])
            self.assertEqual(
                "image_segmentation",
                backend._infer_mode(parsed, [{"data": {"image": "/data/local-files/?d=images/demo_blue.png"}}]),
            )

    def test_predict_returns_brushlabels_rle_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
            response = backend._predict(
                {
                    "label_config": config,
                    "tasks": [
                        {
                            "id": "seg-1",
                            "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                        }
                    ],
                }
            )
            self.assertEqual("image_segmentation", response["mode"])
            prediction = response["results"][0]
            self.assertEqual("placeholder-image-segmentation", prediction["prediction_source"])
            result = prediction["result"][0]
            self.assertEqual("brushlabels", result["type"])
            self.assertEqual("mask_label", result["from_name"])
            self.assertEqual("image", result["to_name"])
            self.assertGreater(result["original_width"], 0)
            self.assertGreater(result["original_height"], 0)
            self.assertEqual("rle", result["value"]["format"])
            self.assertEqual(["Object"], result["value"]["brushlabels"])
            self.assertIsInstance(result["value"]["rle"], list)
            self.assertGreater(len(result["value"]["rle"]), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests/test_image_segmentation_prediction.py -v
```

Expected: FAIL because `label_configs/image_segmentation.xml` is missing and `parsed["brushlabels"]` is not defined.

- [ ] **Step 3: Add Label Studio config**

Create `label_configs/image_segmentation.xml`:

```xml
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="mask_label" toName="image">
    <Label value="Object" background="#ff6b6b"/>
  </BrushLabels>
</View>
```

- [ ] **Step 4: Implement parsing and prediction**

In `services/ml-backend/app.py`, add `IMAGE_SEG_MODEL_ARTIFACTS_DIR`, `_IMAGE_SEG_METADATA_PATH`, and `_IMAGE_SEG_ARTIFACT_PATH` constants near the image classifier constants. Extend `_default_state()` with `image_segmentation`.

Update `_parse_label_config()` to include:

```python
parsed = {
    "image_name": "image",
    "text_name": "text",
    "choices": None,
    "rectanglelabels": None,
    "brushlabels": None,
    "labels": None,
}
```

Add:

```python
parsed["brushlabels"] = _extract_control(
    root,
    "BrushLabels",
    default_name="mask_label",
    default_to_name=parsed["image_name"],
)
```

Update `_extract_control()` so `BrushLabels` uses child tag `Label`:

```python
if control_tag == "Choices":
    child_tag = "Choice"
else:
    child_tag = "Label"
```

Update `_infer_mode()` before rectangle detection:

```python
brush = parsed.get("brushlabels")
if brush and brush.get("to_name") == image_name:
    return "image_segmentation"
```

Add `_load_image_segmentation_metadata()`, `_image_dimensions_for_task()`, `_make_placeholder_segmentation_mask()`, `_mask_to_rle()`, and `_image_segmentation()` using `label_studio_converter.brush.mask2rle` when available. If import fails, use a tiny fallback RLE list `[0, 1]` only for tests and mark metadata `rle_encoder: fallback-minimal`.

Add routing in `_predict()`:

```python
elif mode == "image_segmentation":
    predictions.append(_image_segmentation(task, parsed, state))
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
python -m unittest tests/test_image_segmentation_prediction.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add label_configs/image_segmentation.xml services/ml-backend/app.py tests/test_image_segmentation_prediction.py
git commit -m "Add image segmentation prelabel prediction"
```

## Task 2: Trainer Segmentation Webhook Candidates

**Files:**
- Modify: `services/trainer/app.py`
- Test: `tests/test_image_segmentation_webhook_candidates.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_segmentation_webhook_candidates.py` with tests mirroring `tests/test_image_webhook_candidates.py`, but using BrushLabels:

```python
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER_APP = ROOT / "services" / "trainer" / "app.py"


def install_stubs():
    backups = {}
    def put(name, module):
        backups[name] = sys.modules.get(name)
        sys.modules[name] = module
    np = types.ModuleType("numpy")
    np.float32 = float
    np.asarray = lambda x, dtype=None: x
    np.vstack = lambda rows: rows
    np.histogram = lambda data, bins=8, range=None, density=True: ([0.0] * bins, None)
    put("numpy", np)
    joblib = types.ModuleType("joblib")
    joblib.dump = lambda *args, **kwargs: None
    joblib.load = lambda *args, **kwargs: None
    put("joblib", joblib)
    for name in [
        "sklearn",
        "sklearn.feature_extraction",
        "sklearn.feature_extraction.text",
        "sklearn.linear_model",
        "sklearn.metrics",
        "sklearn.model_selection",
        "sklearn.neighbors",
    ]:
        module = types.ModuleType(name)
        put(name, module)
    sys.modules["sklearn.feature_extraction.text"].TfidfVectorizer = object
    sys.modules["sklearn.linear_model"].LogisticRegression = object
    sys.modules["sklearn.metrics"].accuracy_score = lambda *args, **kwargs: 0.0
    sys.modules["sklearn.metrics"].classification_report = lambda *args, **kwargs: {}
    sys.modules["sklearn.metrics"].confusion_matrix = lambda *args, **kwargs: []
    sys.modules["sklearn.metrics"].f1_score = lambda *args, **kwargs: 0.0
    sys.modules["sklearn.model_selection"].train_test_split = lambda data, **kwargs: (data, [])
    sys.modules["sklearn.neighbors"].KNeighborsClassifier = object
    return backups


def restore_stubs(backups):
    for name, previous in backups.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load_trainer(env_overrides):
    stubs = install_stubs()
    env_backup = {}
    for key, value in env_overrides.items():
        env_backup[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        module_name = f"test_image_seg_webhook_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, str(TRAINER_APP))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        restore_stubs(stubs)
        for key, previous in env_backup.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


class ImageSegmentationWebhookCandidateTests(unittest.TestCase):
    def _module(self, tmp_dir):
        return load_trainer(
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                "IMAGE_SEG_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_segmentation"),
                "IMAGE_LOCAL_FILES_ROOT": str(ROOT / "demo_data" / "local-files"),
                "IMAGE_SEG_TRAINING_CANDIDATES_PATH": str(tmp_dir / "image_seg_candidates.jsonl"),
                "LABEL_STUDIO_URL": "http://label-studio:8080",
                "LABEL_STUDIO_API_TOKEN": "token-123",
                "LABEL_STUDIO_TIMEOUT_SECONDS": "3.0",
            }
        )

    def test_parse_full_task_brush_annotation_to_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._module(Path(tmp))
            task = {
                "id": 901,
                "project": 77,
                "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                "annotations": [
                    {
                        "id": 9801,
                        "was_cancelled": False,
                        "updated_at": "2026-05-21T09:00:00Z",
                        "result": [
                            {
                                "from_name": "mask_label",
                                "to_name": "image",
                                "type": "brushlabels",
                                "original_width": 320,
                                "original_height": 240,
                                "image_rotation": 0,
                                "value": {
                                    "format": "rle",
                                    "rle": [0, 1, 2, 3],
                                    "brushlabels": ["Object"],
                                },
                            }
                        ],
                    }
                ],
            }
            candidate = trainer._parse_image_segmentation_sample_from_full_task(task)
            self.assertEqual(901, candidate["task_id"])
            self.assertEqual(77, candidate["project_id"])
            self.assertEqual("Object", candidate["label"])
            self.assertEqual([0, 1, 2, 3], candidate["rle"])
            self.assertEqual(320, candidate["original_width"])
            self.assertEqual(240, candidate["original_height"])

    def test_append_and_read_segmentation_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._module(Path(tmp))
            trainer._append_image_segmentation_training_candidate(
                {
                    "task_id": 901,
                    "project_id": 77,
                    "image": "/data/local-files/?d=images/demo_blue.png",
                    "label": "Object",
                    "rle": [0, 1, 2, 3],
                    "original_width": 320,
                    "original_height": 240,
                    "source": "label_studio_webhook_task_fetch",
                    "annotation_id": 9801,
                    "updated_at": "2026-05-21T09:00:00Z",
                }
            )
            samples, stats = trainer._read_image_segmentation_training_candidate_samples()
            self.assertEqual(1, stats["used"])
            self.assertEqual(1, len(samples))
            self.assertEqual("Object", samples[0]["label"])
            self.assertTrue(samples[0]["image_path"].endswith("demo_blue.png"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m unittest tests/test_image_segmentation_webhook_candidates.py -v
```

Expected: FAIL because segmentation parsing and candidate functions do not exist.

- [ ] **Step 3: Implement candidate parsing and storage**

In `services/trainer/app.py`, add constants:

```python
IMAGE_SEG_MODEL_ARTIFACTS_DIR = os.getenv("IMAGE_SEG_MODEL_ARTIFACTS_DIR", "/model-state/image_segmentation")
IMAGE_SEG_TRAINING_CANDIDATES_PATH = os.getenv(
    "IMAGE_SEG_TRAINING_CANDIDATES_PATH",
    "/demo-tasks/image_segmentation_training_candidates.jsonl",
)
_IMAGE_SEG_METADATA_PATH = os.path.join(IMAGE_SEG_MODEL_ARTIFACTS_DIR, "metadata.json")
_IMAGE_SEG_ARTIFACT_PATH = os.path.join(IMAGE_SEG_MODEL_ARTIFACTS_DIR, "placeholder_model.json")
_IMAGE_SEG_LAST_DATASET_PATH = os.path.join(IMAGE_SEG_MODEL_ARTIFACTS_DIR, "last_training_dataset.jsonl")
```

Add:

```python
def _normalize_image_segmentation_label(value):
    label = _normalize_label(value)
    if not label:
        return None
    return "Object" if label.lower() == "object" else None
```

Add `_extract_brush_mask_result(result_items)` that returns the first non-empty `Object` Brush result with `rle`, `original_width`, and `original_height`.

Add `_parse_image_segmentation_sample_from_full_task(task, fallback_project_id=None)`, `_append_image_segmentation_training_candidate(candidate)`, and `_read_image_segmentation_training_candidate_samples()` following the classification candidate functions but preserving `rle` and dimensions.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m unittest tests/test_image_segmentation_webhook_candidates.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/trainer/app.py tests/test_image_segmentation_webhook_candidates.py
git commit -m "Capture image segmentation webhook candidates"
```

## Task 3: Placeholder Segmentation Retrain And Metadata

**Files:**
- Modify: `services/trainer/app.py`
- Test: `tests/test_image_segmentation_training.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_segmentation_training.py` with dynamic module loading like Task 2 and these core assertions:

```python
class ImageSegmentationTrainingTests(unittest.TestCase):
    def test_train_placeholder_segmentation_writes_metadata_and_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._module(Path(tmp))
            samples = [
                {
                    "image": "/data/local-files/?d=images/demo_blue.png",
                    "image_path": str(ROOT / "demo_data" / "local-files" / "images" / "demo_blue.png"),
                    "label": "Object",
                    "rle": [0, 1, 2, 3],
                    "original_width": 320,
                    "original_height": 240,
                    "task_id": 1,
                    "annotation_id": 10,
                    "source": "test",
                }
            ]
            metadata = trainer._train_placeholder_image_segmentation(samples, training_run=3, dataset_context={"candidate_stats": {"total": 1, "used": 1, "skipped": 0, "errors": []}})
            self.assertEqual("image_segmentation", metadata["task_type"])
            self.assertEqual("image-seg-v0003", metadata["model_version"])
            self.assertEqual(1, metadata["dataset"]["total_size"])
            self.assertEqual({"Object": 1}, metadata["dataset"]["quality"]["label_distribution"])
            self.assertTrue(Path(trainer._IMAGE_SEG_METADATA_PATH).exists())
            self.assertTrue(Path(trainer._IMAGE_SEG_ARTIFACT_PATH).exists())
            self.assertTrue(Path(trainer._IMAGE_SEG_LAST_DATASET_PATH).exists())

    def test_run_training_routes_image_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._module(Path(tmp))
            outcome = trainer._run_training(
                {
                    "task_type": "image_segmentation",
                    "samples": [
                        {
                            "image": "/data/local-files/?d=images/demo_blue.png",
                            "label": "Object",
                            "rle": [0, 1, 2, 3],
                            "original_width": 320,
                            "original_height": 240,
                        }
                    ],
                },
                trigger="manual-train",
            )
            self.assertTrue(outcome["ok"])
            self.assertEqual("image_segmentation", outcome["task_type"])
            self.assertEqual("placeholder-image-segmentation", outcome["mode"])
            self.assertEqual("image-seg-v0001", outcome["metadata"]["model_version"])
            self.assertTrue(outcome["next_state"]["image_segmentation"]["active"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m unittest tests/test_image_segmentation_training.py -v
```

Expected: FAIL because training functions and task routing do not exist.

- [ ] **Step 3: Implement dataset build and placeholder training**

In `services/trainer/app.py`, add:

- `_normalize_manual_image_segmentation_samples(samples)`
- `_parse_image_segmentation_export_task(task)`
- `_read_image_segmentation_samples_from_export(path)`
- `_dedupe_image_segmentation_samples(samples)`
- `_build_image_segmentation_dataset(payload)`
- `_image_segmentation_dataset_quality_report(samples)`
- `_train_placeholder_image_segmentation(samples, training_run, dataset_context=None)`
- `_run_image_segmentation_training(payload, trigger)`

Training metadata must use:

```python
model_version = f"image-seg-v{training_run:04d}"
metadata = {
    "model_version": model_version,
    "training_run": training_run,
    "training_run_id": f"train-{training_run:04d}",
    "trained_at": trained_at,
    "task_type": "image_segmentation",
    "labels": ["Object"],
    "dataset": {
        "total_size": len(samples),
        "source": "label_studio_brush_masks",
        "snapshot_path": _IMAGE_SEG_LAST_DATASET_PATH,
        "quality": quality,
        "candidates": {
            "path": context.get("candidates_path"),
            "stats": context.get("candidate_stats"),
        },
    },
    "model_details": {
        "name": "placeholder_center_mask",
        "rle_format": "label_studio_brush",
        "replaceable_with": ["SAM", "YOLO-seg", "Detectron2", "custom"],
    },
    "artifacts": {
        "metadata_path": _IMAGE_SEG_METADATA_PATH,
        "placeholder_model_path": _IMAGE_SEG_ARTIFACT_PATH,
    },
}
```

Update `_normalize_task_type()` to include `image_segmentation`, and route it in `_run_training()`.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m unittest tests/test_image_segmentation_training.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/trainer/app.py tests/test_image_segmentation_training.py
git commit -m "Add placeholder image segmentation retraining"
```

## Task 4: Metadata Endpoints And Backend Model Version

**Files:**
- Modify: `services/ml-backend/app.py`
- Modify: `services/trainer/app.py`
- Test: extend `tests/test_image_segmentation_prediction.py` and `tests/test_image_segmentation_training.py`

- [ ] **Step 1: Write failing endpoint/version tests**

Add to prediction tests:

```python
def test_prediction_uses_trained_segmentation_metadata_version(self):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        artifacts = tmp_dir / "image_segmentation"
        artifacts.mkdir(parents=True)
        (artifacts / "metadata.json").write_text(
            json.dumps({"model_version": "image-seg-v0042", "task_type": "image_segmentation"}),
            encoding="utf-8",
        )
        backend = self._backend(tmp_dir)
        config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
        response = backend._predict({"label_config": config, "tasks": [{"id": 1, "data": {"image": "/data/local-files/?d=images/demo_blue.png"}}]})
        self.assertEqual("image-seg-v0042", response["model_version"])
        self.assertEqual("image-seg-v0042", response["results"][0]["model_version"])
```

Add to training tests:

```python
def test_read_image_segmentation_metadata_returns_dict(self):
    with tempfile.TemporaryDirectory() as tmp:
        trainer = self._module(Path(tmp))
        Path(trainer._IMAGE_SEG_METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(trainer._IMAGE_SEG_METADATA_PATH).write_text('{"task_type":"image_segmentation"}', encoding="utf-8")
        self.assertEqual("image_segmentation", trainer._read_image_segmentation_metadata()["task_type"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m unittest tests/test_image_segmentation_prediction.py tests/test_image_segmentation_training.py -v
```

Expected: FAIL because metadata loading/routes are incomplete.

- [ ] **Step 3: Implement metadata reads and GET routes**

In both services, add segmentation metadata to `/health`, `/models/image-segmentation`, and `/models/image-segmentation/current`.

In ML backend `_image_segmentation()`, prefer metadata model version if present:

```python
metadata = _load_image_segmentation_metadata() or {}
model_version = metadata.get("model_version") or state.get("image_segmentation", {}).get("model_version") or state["model_version"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests/test_image_segmentation_prediction.py tests/test_image_segmentation_training.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/ml-backend/app.py services/trainer/app.py tests/test_image_segmentation_prediction.py tests/test_image_segmentation_training.py
git commit -m "Expose image segmentation model metadata"
```

## Task 5: Label Studio Bootstrap And Import Scripts

**Files:**
- Modify: `scripts/lib/label_studio_client.py`
- Create: `scripts/bootstrap_label_studio_image_segmentation_review.py`
- Create: `scripts/import_image_segmentation_review_tasks_to_label_studio.py`
- Create: `demo_data/tasks/image_segmentation_review_tasks.json`
- Test: `tests/test_bootstrap_label_studio_image_segmentation_review.py`
- Test: `tests/test_import_image_segmentation_review_tasks_to_label_studio.py`

- [ ] **Step 1: Write failing script tests**

Create bootstrap test similar to `tests/test_bootstrap_label_studio_image_review.py`, asserting the segmentation config path and title are used:

```python
from scripts.bootstrap_label_studio_image_segmentation_review import bootstrap_image_segmentation_review

class BootstrapImageSegmentationReviewTests(unittest.TestCase):
    def test_bootstrap_uses_segmentation_label_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "seg.xml"
            cfg.write_text("<View><BrushLabels name='mask_label' toName='image'/><Image name='image' value='$image'/></View>", encoding="utf-8")
            base = LabelStudioSettings.from_env(require_token=False)
            settings = base.__class__(**{**base.__dict__, "image_label_config_path": cfg, "image_review_project_title": "Image Segmentation Human Review"})
            client = FakeBootstrapClient()
            result = bootstrap_image_segmentation_review(client, settings)
        self.assertEqual(42, result["project_id"])
        self.assertIn("<BrushLabels", client.project_args["label_config"])
```

Create import test similar to classification import, loading `demo_data/tasks/image_segmentation_review_tasks.json` and deduping `data.image`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m unittest tests/test_bootstrap_label_studio_image_segmentation_review.py tests/test_import_image_segmentation_review_tasks_to_label_studio.py -v
```

Expected: FAIL because scripts and fixture do not exist.

- [ ] **Step 3: Add scripts and fixture**

Create `demo_data/tasks/image_segmentation_review_tasks.json`:

```json
[
  {
    "id": "seg-review-001",
    "data": {
      "image": "/data/local-files/?d=images/demo_blue.png"
    },
    "meta": {
      "review_type": "image_segmentation_smoke"
    }
  },
  {
    "id": "seg-review-002",
    "data": {
      "image": "/data/local-files/?d=images/demo_product_red.png"
    },
    "meta": {
      "review_type": "image_segmentation_smoke"
    }
  }
]
```

Create bootstrap/import scripts by copying the classification script structure and changing names, defaults, and messages to segmentation. Use `label_configs/image_segmentation.xml` as the default config path and `Image Segmentation Human Review` as the title.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests/test_bootstrap_label_studio_image_segmentation_review.py tests/test_import_image_segmentation_review_tasks_to_label_studio.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/label_studio_client.py scripts/bootstrap_label_studio_image_segmentation_review.py scripts/import_image_segmentation_review_tasks_to_label_studio.py demo_data/tasks/image_segmentation_review_tasks.json tests/test_bootstrap_label_studio_image_segmentation_review.py tests/test_import_image_segmentation_review_tasks_to_label_studio.py
git commit -m "Add Label Studio image segmentation review scripts"
```

## Task 6: Validation Scripts, Docker Env, And Docs

**Files:**
- Modify: `infra/docker-compose.yml`
- Modify: `services/ml-backend/Dockerfile`
- Modify: `services/trainer/Dockerfile`
- Create: `scripts/trigger_image_segmentation_retrain.sh`
- Create: `scripts/inspect_image_segmentation_metadata.sh`
- Create: `scripts/run_image_segmentation_validation_gate.sh`
- Modify: `README.md`

- [ ] **Step 1: Write the validation scripts**

Create `scripts/trigger_image_segmentation_retrain.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
DATASET_PATH="${IMAGE_SEG_DATASET_PATH:-}"

payload='{"task_type":"image_segmentation"}'
if [[ -n "$DATASET_PATH" ]]; then
  payload="{\"task_type\":\"image_segmentation\",\"dataset_path\":\"$DATASET_PATH\"}"
fi

curl -sS -X POST "$TRAINER_URL/retrain/image-segmentation" \
  -H 'Content-Type: application/json' \
  -d "$payload"
echo
```

Create `scripts/inspect_image_segmentation_metadata.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
curl -sS "$TRAINER_URL/models/image-segmentation/current"
echo
```

Create `scripts/run_image_segmentation_validation_gate.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
ML_BACKEND_URL="${ML_BACKEND_URL:-http://localhost:9090}"

echo "[INFO] trigger image segmentation retrain"
"$ROOT/scripts/trigger_image_segmentation_retrain.sh" >/tmp/image_seg_retrain.json
python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("/tmp/image_seg_retrain.json").read_text())
assert data.get("accepted") is True, data
assert data.get("task_type") == "image_segmentation", data
assert data.get("mode") == "placeholder-image-segmentation", data
PY

echo "[INFO] inspect trainer metadata"
curl -sS "$TRAINER_URL/models/image-segmentation/current" >/tmp/image_seg_metadata.json
python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("/tmp/image_seg_metadata.json").read_text())
assert data.get("task_type") == "image_segmentation", data
assert data.get("model_version", "").startswith("image-seg-v"), data
assert data.get("dataset", {}).get("total_size", 0) >= 1, data
PY

echo "[INFO] check ML backend prediction"
python - <<'PY'
import json
import urllib.request
config = open("label_configs/image_segmentation.xml", encoding="utf-8").read()
payload = {
    "label_config": config,
    "tasks": [{"id": "gate-seg", "data": {"image": "/data/local-files/?d=images/demo_blue.png"}}],
}
req = urllib.request.Request(
    "http://localhost:9090/predict",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=5) as resp:
    data = json.loads(resp.read().decode())
assert data["mode"] == "image_segmentation", data
result = data["results"][0]["result"][0]
assert result["type"] == "brushlabels", result
assert result["value"]["format"] == "rle", result
assert result["value"]["brushlabels"] == ["Object"], result
PY

echo "[OK] image segmentation validation gate passed"
```

- [ ] **Step 2: Add Docker/env configuration**

Add env vars in `infra/docker-compose.yml`:

```yaml
IMAGE_SEG_MODEL_ARTIFACTS_DIR: /model-state/image_segmentation
IMAGE_SEG_TRAINING_CANDIDATES_PATH: /demo-tasks/image_segmentation_training_candidates.jsonl
```

Add `label-studio-converter` to both Dockerfiles if it is not already installed.

- [ ] **Step 3: Update README**

Add an `Image Segmentation HITL` section with commands:

```bash
scripts/bootstrap_label_studio_image_segmentation_review.py
scripts/import_image_segmentation_review_tasks_to_label_studio.py
scripts/trigger_image_segmentation_retrain.sh
scripts/inspect_image_segmentation_metadata.sh
scripts/run_image_segmentation_validation_gate.sh
```

Document that first version uses Brush/mask, single `Object`, and placeholder retraining.

- [ ] **Step 4: Run script syntax checks**

```bash
bash -n scripts/trigger_image_segmentation_retrain.sh
bash -n scripts/inspect_image_segmentation_metadata.sh
bash -n scripts/run_image_segmentation_validation_gate.sh
```

Expected: no output and exit code 0.

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/trigger_image_segmentation_retrain.sh scripts/inspect_image_segmentation_metadata.sh scripts/run_image_segmentation_validation_gate.sh
git add infra/docker-compose.yml services/ml-backend/Dockerfile services/trainer/Dockerfile scripts/trigger_image_segmentation_retrain.sh scripts/inspect_image_segmentation_metadata.sh scripts/run_image_segmentation_validation_gate.sh README.md
git commit -m "Add image segmentation validation workflow"
```

## Task 7: Regression Review And Final Verification

**Files:**
- Review all files changed in previous tasks.

- [ ] **Step 1: Run focused tests**

```bash
python -m unittest tests/test_image_segmentation_prediction.py -v
python -m unittest tests/test_image_segmentation_webhook_candidates.py -v
python -m unittest tests/test_image_segmentation_training.py -v
python -m unittest tests/test_bootstrap_label_studio_image_segmentation_review.py -v
python -m unittest tests/test_import_image_segmentation_review_tasks_to_label_studio.py -v
```

Expected: all PASS.

- [ ] **Step 2: Run related existing tests**

```bash
python -m unittest tests/test_phase4_image_training.py -v
python -m unittest tests/test_image_webhook_candidates.py -v
python -m unittest tests/test_bootstrap_label_studio_image_review.py -v
python -m unittest tests/test_import_image_review_tasks_to_label_studio.py -v
```

Expected: all PASS.

- [ ] **Step 3: Run static syntax checks**

```bash
python -m py_compile services/ml-backend/app.py services/trainer/app.py scripts/bootstrap_label_studio_image_segmentation_review.py scripts/import_image_segmentation_review_tasks_to_label_studio.py
bash -n scripts/trigger_image_segmentation_retrain.sh
bash -n scripts/inspect_image_segmentation_metadata.sh
bash -n scripts/run_image_segmentation_validation_gate.sh
```

Expected: no output and exit code 0.

- [ ] **Step 4: Run Docker/local validation gate if services are available**

```bash
scripts/run_image_segmentation_validation_gate.sh
```

Expected: `[OK] image segmentation validation gate passed`.

If services are not running, start them first:

```bash
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
```

- [ ] **Step 5: Review diff for routing regressions**

Run:

```bash
git diff --stat HEAD~6..HEAD
git diff HEAD~6..HEAD -- services/ml-backend/app.py services/trainer/app.py scripts/lib/label_studio_client.py
```

Check:

- `BrushLabels` detection runs before `RectangleLabels` and `Choices`.
- Existing `image_classification` routes still normalize exactly.
- Segmentation candidates use `IMAGE_SEG_TRAINING_CANDIDATES_PATH`, not image classification candidate path.
- Metadata routes for image classification remain unchanged.
- No real API tokens or generated model binaries are committed.

- [ ] **Step 6: Commit any review fixes**

If review found fixes:

```bash
git add <fixed-files>
git commit -m "Review image segmentation HITL integration"
```

If no fixes are needed, do not create an empty commit.

## Self-Review Notes

- Spec coverage: tasks cover config, prediction, webhook fetch, candidates, retrain, metadata routes, bootstrap/import scripts, docs, and validation.
- TDD: every behavior task starts with a failing `unittest` before implementation.
- Risk controls: final verification reruns existing image classification tests to catch cross-talk.
