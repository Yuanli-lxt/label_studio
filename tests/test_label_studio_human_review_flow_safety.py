import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

from scripts.import_image_review_tasks_to_label_studio import import_review_tasks
from scripts.lib.label_studio_client import LabelStudioClient, LabelStudioSettings

ROOT = Path(__file__).resolve().parents[1]
TRAINER_APP = ROOT / "services" / "trainer" / "app.py"


class FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
        self.text = json.dumps(payload if payload is not None else {})

    def json(self):
        return self.payload


class SafetyImportSession:
    def __init__(self):
        self.headers = {}
        self.import_called = False

    def request(self, method, url, timeout=None, headers=None, **kwargs):
        if method == "GET" and url.endswith("/api/projects/"):
            return FakeResponse({"results": [{"id": 55, "title": "Image Classification Human Review"}]})
        if method == "GET" and url.endswith("/api/tasks/"):
            return FakeResponse({"results": [], "next": None})
        if method == "POST" and url.endswith("/api/projects/55/import"):
            self.import_called = True
            return FakeResponse({"task_count": len(kwargs["json"])})
        raise AssertionError(f"unexpected request: {method} {url}")


def install_lightweight_stubs():
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

    sklearn = types.ModuleType("sklearn")
    put("sklearn", sklearn)
    feature_extraction = types.ModuleType("sklearn.feature_extraction")
    feature_text = types.ModuleType("sklearn.feature_extraction.text")
    feature_text.TfidfVectorizer = object
    put("sklearn.feature_extraction", feature_extraction)
    put("sklearn.feature_extraction.text", feature_text)
    linear_model = types.ModuleType("sklearn.linear_model")
    linear_model.LogisticRegression = object
    put("sklearn.linear_model", linear_model)
    metrics = types.ModuleType("sklearn.metrics")
    metrics.accuracy_score = lambda *args, **kwargs: 0.0
    metrics.classification_report = lambda *args, **kwargs: {}
    metrics.confusion_matrix = lambda *args, **kwargs: []
    metrics.f1_score = lambda *args, **kwargs: 0.0
    put("sklearn.metrics", metrics)
    model_selection = types.ModuleType("sklearn.model_selection")
    model_selection.train_test_split = lambda data, test_size=1, random_state=42, stratify=None: (data, [])
    put("sklearn.model_selection", model_selection)
    neighbors = types.ModuleType("sklearn.neighbors")
    neighbors.KNeighborsClassifier = object
    put("sklearn.neighbors", neighbors)
    return backups


def restore_stubs(backups):
    for name, previous in backups.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load_trainer(env):
    backups = install_lightweight_stubs()
    old_env = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        spec = importlib.util.spec_from_file_location("safety_trainer", str(TRAINER_APP))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        restore_stubs(backups)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class HumanReviewFlowSafetyTests(unittest.TestCase):
    def test_review_import_dry_run_does_not_modify_eval_manifest_or_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            manifest = tmp_dir / "image_eval_manifest.json"
            original = {"eval_set_id": "image-cls-eval-v1", "images": [{"filename": "eval.png", "label": "Product"}]}
            manifest.write_text(json.dumps(original), encoding="utf-8")
            tasks = tmp_dir / "review.json"
            tasks.write_text(
                json.dumps([{"id": "eval-review", "data": {"image": "/data/local-files/?d=images/eval.png"}}]),
                encoding="utf-8",
            )
            session = SafetyImportSession()
            client = LabelStudioClient(LabelStudioSettings.from_env(require_token=False), session=session)
            result = import_review_tasks(client, tasks, project_id=55, dry_run=True)

            self.assertEqual(original, json.loads(manifest.read_text(encoding="utf-8")))
            self.assertFalse(session.import_called)
            self.assertEqual(1, result["planned_import"])
            self.assertEqual(0, result["imported"])

    def test_eval_manifest_candidate_is_filtered_before_training_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = tmp_dir / "local-files"
            (image_root / "images").mkdir(parents=True)
            source = ROOT / "demo_data" / "local-files" / "images"
            for name in ["demo_blue.png", "demo_product_red.png"]:
                shutil.copy2(source / name, image_root / "images" / name)

            manifest = tmp_dir / "image_eval_manifest.json"
            manifest.write_text(
                json.dumps({"eval_set_id": "image-cls-eval-v1", "images": [{"filename": "demo_blue.png", "label": "Product"}]}),
                encoding="utf-8",
            )
            candidates = tmp_dir / "candidates.jsonl"
            rows = [
                {
                    "task_id": 1,
                    "project_id": 9,
                    "image": "/data/local-files/?d=images/demo_blue.png",
                    "label": "Product",
                    "source": "label_studio_webhook_task_fetch",
                    "annotation_id": 10,
                    "updated_at": "2026-05-12T00:00:00Z",
                },
                {
                    "task_id": 2,
                    "project_id": 9,
                    "image": "/data/local-files/?d=images/demo_product_red.png",
                    "label": "Product",
                    "source": "label_studio_webhook_task_fetch",
                    "annotation_id": 11,
                    "updated_at": "2026-05-12T00:00:00Z",
                },
            ]
            candidates.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            trainer = load_trainer(
                {
                    "MODEL_STATE_PATH": str(tmp_dir / "state.json"),
                    "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text"),
                    "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image"),
                    "TRAINING_EVENT_LOG_PATH": str(tmp_dir / "events.jsonl"),
                    "TRAINING_DATASET_EVENTS_PATH": str(tmp_dir / "text_events.jsonl"),
                    "IMAGE_LOCAL_FILES_ROOT": str(image_root),
                    "IMAGE_CLS_EVAL_MANIFEST_PATH": str(manifest),
                    "IMAGE_TRAINING_CANDIDATES_PATH": str(candidates),
                }
            )
            samples, context = trainer._build_image_classification_dataset({"samples": []})

        filenames = {Path(sample["image_path"]).name for sample in samples}
        self.assertNotIn("demo_blue.png", filenames)
        self.assertIn("demo_product_red.png", filenames)
        self.assertEqual(1, context["candidate_eval_leakage_skipped"])


if __name__ == "__main__":
    unittest.main()
