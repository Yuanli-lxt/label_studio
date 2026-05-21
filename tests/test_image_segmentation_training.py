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
        module_name = f"test_image_seg_training_{uuid.uuid4().hex}"
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


class ImageSegmentationTrainingTests(unittest.TestCase):
    def _module(self, tmp_dir: Path):
        image_root = ROOT / "demo_data" / "local-files"
        return load_trainer_module(
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                "IMAGE_SEG_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_seg_artifacts"),
                "TRAINING_EVENT_LOG_PATH": str(tmp_dir / "training_events.jsonl"),
                "TRAINING_DATASET_EVENTS_PATH": str(tmp_dir / "dataset_events.jsonl"),
                "IMAGE_LOCAL_FILES_ROOT": str(image_root),
                "IMAGE_CLS_EVAL_MANIFEST_PATH": str(tmp_dir / "image_eval_manifest.json"),
                "IMAGE_TRAINING_CANDIDATES_PATH": str(tmp_dir / "image_candidates.jsonl"),
                "IMAGE_SEG_TRAINING_CANDIDATES_PATH": str(tmp_dir / "image_seg_candidates.jsonl"),
            }
        )

    def test_train_placeholder_segmentation_writes_metadata_and_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            module = self._module(tmp_dir)
            sample = {
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

            metadata = module._train_placeholder_image_segmentation(
                [sample],
                training_run=3,
                dataset_context={
                    "candidate_stats": {"total": 1, "used": 1, "skipped": 0, "errors": []},
                    "candidates_path": str(tmp_dir / "candidates.jsonl"),
                },
            )

            self.assertEqual("image_segmentation", metadata["task_type"])
            self.assertEqual("image-seg-v0003", metadata["model_version"])
            self.assertEqual(1, metadata["dataset"]["total_size"])
            self.assertEqual({"Object": 1}, metadata["dataset"]["quality"]["label_distribution"])
            self.assertTrue(Path(module._IMAGE_SEG_METADATA_PATH).exists())
            self.assertTrue(Path(module._IMAGE_SEG_ARTIFACT_PATH).exists())
            self.assertTrue(Path(module._IMAGE_SEG_LAST_DATASET_PATH).exists())

    def test_manual_segmentation_sample_falls_back_to_image_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            module = self._module(tmp_dir)
            valid_path = str(ROOT / "demo_data" / "local-files" / "images" / "demo_blue.png")
            payload = {
                "task_type": "image_segmentation",
                "samples": [
                    {
                        "image": "/missing.png",
                        "image_path": valid_path,
                        "label": "Object",
                        "rle": [0, 1, 2, 3],
                        "original_width": 320,
                        "original_height": 240,
                    }
                ],
            }

            normalized = module._normalize_manual_image_segmentation_samples(payload["samples"])
            dataset_samples, _ = module._build_image_segmentation_dataset(payload)

            self.assertEqual(1, len(normalized))
            self.assertEqual(valid_path, normalized[0]["image_path"])
            self.assertEqual(valid_path, normalized[0]["image"])
            self.assertEqual(1, len(dataset_samples))
            self.assertEqual(valid_path, dataset_samples[0]["image_path"])

    def test_train_placeholder_segmentation_rejects_invalid_rle_and_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            module = self._module(tmp_dir)
            valid_path = str(ROOT / "demo_data" / "local-files" / "images" / "demo_blue.png")
            base_sample = {
                "image": "/data/local-files/?d=images/demo_blue.png",
                "image_path": valid_path,
                "label": "Object",
                "rle": [0, 1, 2, 3],
                "original_width": 320,
                "original_height": 240,
            }

            empty_rle = dict(base_sample, rle=[])
            bad_dimensions = dict(base_sample, original_width=0, original_height=0)

            with self.assertRaisesRegex(ValueError, "rle"):
                module._train_placeholder_image_segmentation([empty_rle], training_run=1)

            with self.assertRaisesRegex(ValueError, "dimensions"):
                module._train_placeholder_image_segmentation([bad_dimensions], training_run=1)

    def test_run_training_routes_image_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            module = self._module(tmp_dir)

            outcome = module._run_training(
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

    def test_normalize_and_resolve_task_type_accepts_image_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = self._module(Path(tmp))

            self.assertEqual("image_segmentation", module._normalize_task_type("image_segmentation"))
            self.assertEqual("image_segmentation", module._normalize_task_type("image-segmentation"))
            self.assertEqual("image_segmentation", module._resolve_task_type({"task_type": "image_segmentation"}))


if __name__ == "__main__":
    unittest.main()
