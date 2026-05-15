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

    # Minimal numpy stub for import-time references.
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
        module_name = f"test_image_webhook_{uuid.uuid4().hex}"
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


class ImageWebhookCandidatesTests(unittest.TestCase):
    def _prepare_image_root(self, tmp_dir: Path) -> Path:
        image_root = tmp_dir / "local-files"
        images_dir = image_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        source = ROOT / "demo_data" / "local-files" / "images"
        for name in ["demo_blue.png", "demo_gray.png", "demo_product_red.png"]:
            shutil.copy2(source / name, images_dir / name)
        return image_root

    def _module(self, tmp_dir: Path, image_root: Path):
        return load_trainer_module(
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                "TRAINING_EVENT_LOG_PATH": str(tmp_dir / "training_events.jsonl"),
                "TRAINING_DATASET_EVENTS_PATH": str(tmp_dir / "dataset_events.jsonl"),
                "IMAGE_LOCAL_FILES_ROOT": str(image_root),
                "IMAGE_CLS_EVAL_MANIFEST_PATH": str(tmp_dir / "image_eval_manifest.json"),
                "IMAGE_TRAINING_CANDIDATES_PATH": str(tmp_dir / "image_candidates.jsonl"),
                "LABEL_STUDIO_URL": "http://label-studio:8080",
                "LABEL_STUDIO_API_TOKEN": "token-123",
                "LABEL_STUDIO_TIMEOUT_SECONDS": "3.0",
            }
        )

    def test_webhook_task_id_fetch_full_task_and_append_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            module = self._module(tmp_dir, image_root)

            full_task = {
                "id": 901,
                "project": 77,
                "data": {"image": "/data/local-files/?d=images/demo_product_red.png"},
                "annotations": [
                    {
                        "id": 9801,
                        "was_cancelled": False,
                        "updated_at": "2026-05-12T09:00:00.000000Z",
                        "result": [
                            {
                                "from_name": "image_label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["Product"]},
                            }
                        ],
                    }
                ],
            }

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
                sample, warning = module._maybe_record_image_candidate_from_webhook(
                    {"task": {"id": 901}, "project": {"id": 77}}
                )
            finally:
                module.urllib.request.urlopen = original

            self.assertIsNone(warning)
            self.assertEqual("Product", sample["label"])
            self.assertIn("/api/tasks/901/", captured["url"])
            self.assertIn("project=77", captured["url"])
            self.assertEqual("Token token-123", captured["auth"])
            self.assertEqual(3.0, captured["timeout"])

            rows = [
                json.loads(line)
                for line in Path(module.IMAGE_TRAINING_CANDIDATES_PATH).read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(1, len(rows))
            self.assertEqual(901, rows[0]["task_id"])
            self.assertEqual(77, rows[0]["project_id"])

    def test_parse_image_classification_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            module = self._module(tmp_dir, image_root)

            task = {
                "id": 3001,
                "project": 66,
                "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                "annotations": [
                    {
                        "id": 4001,
                        "was_cancelled": True,
                        "updated_at": "2026-05-12T09:00:00.000000Z",
                        "result": [],
                    },
                    {
                        "id": 4002,
                        "was_cancelled": False,
                        "updated_at": "2026-05-12T09:10:00.000000Z",
                        "result": [
                            {
                                "from_name": "image_label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["Other"]},
                            }
                        ],
                    },
                ],
            }

            parsed = module._parse_image_classification_sample_from_full_task(task)
            self.assertEqual(3001, parsed["task_id"])
            self.assertEqual(66, parsed["project_id"])
            self.assertEqual("Other", parsed["label"])
            self.assertEqual(4002, parsed["annotation_id"])

    def test_eval_manifest_samples_not_in_train_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            module = self._module(tmp_dir, image_root)

            manifest = {
                "eval_set_id": "image-cls-eval-v1",
                "images": [{"filename": "demo_blue.png", "label": "Product"}],
            }
            Path(module.IMAGE_CLS_EVAL_MANIFEST_PATH).write_text(json.dumps(manifest), encoding="utf-8")

            candidates = [
                {
                    "task_id": 1,
                    "project_id": 1,
                    "image": "/data/local-files/?d=images/demo_blue.png",
                    "label": "Product",
                    "source": "label_studio_webhook_task_fetch",
                    "annotation_id": 11,
                    "updated_at": "2026-05-12T09:20:00.000000Z",
                },
                {
                    "task_id": 2,
                    "project_id": 1,
                    "image": "/data/local-files/?d=images/demo_product_red.png",
                    "label": "Product",
                    "source": "label_studio_webhook_task_fetch",
                    "annotation_id": 12,
                    "updated_at": "2026-05-12T09:21:00.000000Z",
                },
            ]
            Path(module.IMAGE_TRAINING_CANDIDATES_PATH).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates),
                encoding="utf-8",
            )

            samples, ctx = module._build_image_classification_dataset({"samples": []})
            filenames = {Path(sample["image_path"]).name for sample in samples}
            self.assertIn("demo_product_red.png", filenames)
            self.assertNotIn("demo_blue.png", filenames)
            self.assertEqual(1, ctx["candidate_eval_leakage_skipped"])


if __name__ == "__main__":
    unittest.main()
