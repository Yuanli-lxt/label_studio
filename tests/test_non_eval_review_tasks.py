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
NON_EVAL_TASKS = ROOT / "demo_data" / "tasks" / "image_classification_non_eval_review_tasks.json"
EVAL_MANIFEST = ROOT / "demo_data" / "tasks" / "image_classification_eval_manifest.json"
LABELED_EXPORT = ROOT / "demo_data" / "tasks" / "image_classification_labeled_export.json"
IMAGE_ROOT = ROOT / "demo_data" / "local-files"


def _install_lightweight_stubs():
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


def _restore_stubs(backups):
    for name, previous in backups.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load_trainer_module(env_overrides):
    backups = _install_lightweight_stubs()
    env_backup = {}
    for key, value in env_overrides.items():
        env_backup[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        module_name = f"test_non_eval_review_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, str(TRAINER_APP))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_stubs(backups)
        for key, previous in env_backup.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


class NonEvalReviewTasksTests(unittest.TestCase):
    def test_non_eval_review_tasks_do_not_overlap_eval_manifest(self):
        tasks = json.loads(NON_EVAL_TASKS.read_text(encoding="utf-8"))
        manifest = json.loads(EVAL_MANIFEST.read_text(encoding="utf-8"))
        eval_filenames = {row["filename"] for row in manifest["images"]}

        self.assertGreaterEqual(len(tasks), 3)
        self.assertLessEqual(len(tasks), 5)
        for task in tasks:
            data = task.get("data", {})
            meta = task.get("meta", {})
            image = data.get("image", "")
            filename = meta.get("filename")
            self.assertTrue(image.startswith("/data/local-files/?d=images/"))
            self.assertIsInstance(filename, str)
            self.assertNotIn(filename, eval_filenames)
            self.assertEqual("non_eval_training_candidate_review", meta.get("review_reason"))

    def test_non_eval_candidate_survives_eval_filter_and_eval_candidate_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            candidates = tmp_dir / "candidates.jsonl"
            candidates.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in [
                        {
                            "task_id": 1,
                            "project_id": 1,
                            "image": "/data/local-files/?d=images/demo_other_lowlight.png",
                            "label": "Product",
                            "source": "label_studio_webhook_task_fetch",
                            "annotation_id": 1,
                            "updated_at": "2026-05-16T06:09:55.048892Z",
                        },
                        {
                            "task_id": "image-non-eval-review-002",
                            "project_id": 1,
                            "image": "/data/local-files/?d=images/demo_green.png",
                            "label": "Other",
                            "source": "label_studio_ui_non_eval_smoke",
                            "annotation_id": "non-eval-smoke-001",
                            "updated_at": "2026-05-16T06:30:00Z",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            trainer = load_trainer_module(
                {
                    "MODEL_STATE_PATH": str(tmp_dir / "state.json"),
                    "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text"),
                    "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image"),
                    "TRAINING_EVENT_LOG_PATH": str(tmp_dir / "events.jsonl"),
                    "TRAINING_DATASET_EVENTS_PATH": str(tmp_dir / "text_events.jsonl"),
                    "IMAGE_LOCAL_FILES_ROOT": str(IMAGE_ROOT),
                    "IMAGE_CLS_EVAL_MANIFEST_PATH": str(EVAL_MANIFEST),
                    "IMAGE_TRAINING_CANDIDATES_PATH": str(candidates),
                }
            )
            samples, context = trainer._build_image_classification_dataset(
                {"dataset_path": str(LABELED_EXPORT)}
            )

        self.assertEqual(2, context["candidate_stats"]["total"])
        self.assertEqual(2, context["candidate_stats"]["used"])
        self.assertGreaterEqual(context["candidate_used_after_eval_filter"], 1)
        self.assertGreaterEqual(context["candidate_eval_leakage_skipped"], 1)

        by_filename = {Path(sample["image_path"]).name: sample for sample in samples}
        self.assertIn("demo_green.png", by_filename)
        self.assertEqual("label_studio_ui_non_eval_smoke", by_filename["demo_green.png"].get("source"))
        self.assertNotEqual(
            "label_studio_webhook_task_fetch",
            by_filename.get("demo_other_lowlight.png", {}).get("source"),
        )


if __name__ == "__main__":
    unittest.main()
