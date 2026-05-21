import importlib.util
import json
import os
import shutil
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
        module_name = f"test_image_seg_webhook_{uuid.uuid4().hex}"
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


class ImageSegmentationWebhookCandidatesTests(unittest.TestCase):
    def _prepare_image_root(self, tmp_dir: Path) -> Path:
        image_root = tmp_dir / "local-files"
        images_dir = image_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        source = ROOT / "demo_data" / "local-files" / "images"
        shutil.copy2(source / "demo_blue.png", images_dir / "demo_blue.png")
        return image_root

    def _module(self, tmp_dir: Path, image_root: Path):
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
                "LABEL_STUDIO_URL": "http://label-studio:8080",
                "LABEL_STUDIO_API_TOKEN": "test-token",
                "LABEL_STUDIO_TIMEOUT_SECONDS": "4.5",
            }
        )

    def _full_segmentation_task(self, task_id=901, project_id=77, annotation_id=9801):
        return {
            "id": task_id,
            "project": project_id,
            "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
            "annotations": [
                {
                    "id": annotation_id,
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

    def test_parse_full_task_brush_annotation_to_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            module = self._module(tmp_dir, image_root)

            parsed = module._parse_image_segmentation_sample_from_full_task(
                self._full_segmentation_task()
            )

            self.assertEqual(901, parsed["task_id"])
            self.assertEqual(77, parsed["project_id"])
            self.assertEqual("/data/local-files/?d=images/demo_blue.png", parsed["image"])
            self.assertEqual("Object", parsed["label"])
            self.assertEqual([0, 1, 2, 3], parsed["rle"])
            self.assertEqual(320, parsed["original_width"])
            self.assertEqual(240, parsed["original_height"])
            self.assertEqual(9801, parsed["annotation_id"])

    def test_append_and_read_segmentation_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            module = self._module(tmp_dir, image_root)
            candidate = {
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

            module._append_image_segmentation_training_candidate(candidate)
            samples, stats = module._read_image_segmentation_training_candidate_samples()

            self.assertEqual(1, stats["used"])
            self.assertEqual(1, len(samples))
            self.assertEqual("Object", samples[0]["label"])
            self.assertEqual([0, 1, 2, 3], samples[0]["rle"])
            self.assertEqual(320, samples[0]["original_width"])
            self.assertEqual(240, samples[0]["original_height"])
            self.assertTrue(samples[0]["image_path"].endswith("demo_blue.png"))

    def test_webhook_task_fetch_and_segmentation_candidate_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            module = self._module(tmp_dir, image_root)
            full_task = self._full_segmentation_task(task_id=900, project_id=30)

            class Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps(full_task).encode("utf-8")

            captured = {}

            def fake_urlopen(req, timeout=None):
                captured["url"] = req.full_url
                captured["auth"] = req.get_header("Authorization")
                captured["timeout"] = timeout
                return Resp()

            original = module.urllib.request.urlopen
            module.urllib.request.urlopen = fake_urlopen
            try:
                sample, warning = module._maybe_record_image_segmentation_candidate_from_webhook(
                    {"task": {"id": 900}, "project": {"id": 30}}
                )
            finally:
                module.urllib.request.urlopen = original

            self.assertIsNone(warning)
            self.assertEqual("Object", sample["label"])
            self.assertIn("/api/tasks/900/", captured["url"])
            self.assertIn("project=30", captured["url"])
            self.assertEqual("Token test-token", captured["auth"])
            self.assertEqual(4.5, captured["timeout"])

            rows = [
                json.loads(line)
                for line in Path(module.IMAGE_SEG_TRAINING_CANDIDATES_PATH).read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(1, len(rows))
            self.assertEqual(900, rows[0]["task_id"])
            self.assertEqual(30, rows[0]["project_id"])
            self.assertEqual("Object", rows[0]["label"])

    def test_deterministic_fallback_preserves_active_segmentation_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            module = self._module(tmp_dir, image_root)
            segmentation_state = {
                "active": True,
                "model_version": "image-seg-v7",
                "trained_at": "2026-05-21T08:00:00Z",
                "artifact_path": str(tmp_dir / "image_seg_artifacts" / "model.joblib"),
                "metadata_path": str(tmp_dir / "image_seg_artifacts" / "metadata.json"),
            }
            current = module._default_state()
            current["training_run"] = 3
            current["image_segmentation"] = segmentation_state
            module._save_state(current)

            outcome = module._run_deterministic_fallback({}, "unit-test")

            self.assertEqual(segmentation_state, outcome["next_state"]["image_segmentation"])
            persisted = json.loads(Path(module.MODEL_STATE_PATH).read_text(encoding="utf-8"))
            self.assertEqual(segmentation_state, persisted["image_segmentation"])


if __name__ == "__main__":
    unittest.main()
