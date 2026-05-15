import importlib.util
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER_APP = ROOT / "services" / "trainer" / "app.py"
ML_BACKEND_APP = ROOT / "services" / "ml-backend" / "app.py"


def load_module(module_path: Path, env_overrides: dict):
    backup = {}
    for key, value in env_overrides.items():
        backup[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        module_name = f"test_dynamic_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
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


class ImageClassifierTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("joblib") is None or importlib.util.find_spec("sklearn") is None:
            raise unittest.SkipTest("joblib/scikit-learn not installed in host Python")
        if importlib.util.find_spec("PIL") is None:
            raise unittest.SkipTest("Pillow not installed in host Python")

    def _prepare_image_root(self, tmp_dir: Path) -> Path:
        image_root = tmp_dir / "local-files"
        images_dir = image_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        source_dir = ROOT / "demo_data" / "local-files" / "images"
        file_names = [
            "demo_blue.png",
            "demo_blue_dark.png",
            "demo_blue_light.png",
            "demo_green.png",
            "demo_green_dark.png",
            "demo_gray.png",
            "demo_product_red.png",
        ]
        for name in file_names:
            shutil.copy2(source_dir / name, images_dir / name)
        return image_root

    def _trainer_module(self, tmp_dir: Path, image_root: Path):
        return load_module(
            TRAINER_APP,
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                "TRAINING_EVENT_LOG_PATH": str(tmp_dir / "training_events.jsonl"),
                "TRAINING_DATASET_EVENTS_PATH": str(tmp_dir / "dataset_events.jsonl"),
                "IMAGE_LOCAL_FILES_ROOT": str(image_root),
                "IMAGE_CLS_MIN_TOTAL_SAMPLES": "6",
                "IMAGE_CLS_MIN_PER_CLASS_SAMPLES": "2",
                "IMAGE_CLS_MAX_IMBALANCE_RATIO": "3.0",
                "IMAGE_CLS_EVAL_SPLIT_RATIO": "0.25",
                "IMAGE_CLS_MIN_EVAL_SAMPLES": "2",
                "IMAGE_CLS_EVAL_MANIFEST_PATH": str(tmp_dir / "image_eval_manifest.json"),
                "IMAGE_TRAINING_CANDIDATES_PATH": str(tmp_dir / "image_training_candidates.jsonl"),
                "LABEL_STUDIO_URL": "http://label-studio:8080",
                "LABEL_STUDIO_API_TOKEN": "test-token",
                "LABEL_STUDIO_TIMEOUT_SECONDS": "2.0",
            },
        )

    def _ml_backend_module(self, tmp_dir: Path, image_root: Path):
        return load_module(
            ML_BACKEND_APP,
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                "IMAGE_LOCAL_FILES_ROOT": str(image_root),
            },
        )

    def _write_image_export(self, path: Path):
        payload = [
            {
                "id": 1,
                "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                "annotations": [
                    {
                        "id": 11,
                        "was_cancelled": False,
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
            },
            {
                "id": 2,
                "data": {"image": "/data/local-files/?d=images/demo_blue_dark.png"},
                "annotations": [
                    {
                        "id": 12,
                        "was_cancelled": False,
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
            },
            {
                "id": 3,
                "data": {"image": "/data/local-files/?d=images/demo_blue_light.png"},
                "annotations": [
                    {
                        "id": 13,
                        "was_cancelled": False,
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
            },
            {
                "id": 4,
                "data": {"image": "/data/local-files/?d=images/demo_green.png"},
                "annotations": [
                    {
                        "id": 14,
                        "was_cancelled": False,
                        "result": [
                            {
                                "from_name": "image_label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["Other"]},
                            }
                        ],
                    }
                ],
            },
            {
                "id": 5,
                "data": {"image": "/data/local-files/?d=images/demo_green_dark.png"},
                "annotations": [
                    {
                        "id": 15,
                        "was_cancelled": False,
                        "result": [
                            {
                                "from_name": "image_label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["Other"]},
                            }
                        ],
                    }
                ],
            },
            {
                "id": 6,
                "data": {"image": "/data/local-files/?d=images/demo_gray.png"},
                "annotations": [
                    {
                        "id": 16,
                        "was_cancelled": False,
                        "result": [
                            {
                                "from_name": "image_label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["Other"]},
                            }
                        ],
                    }
                ],
            },
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_image_export_with_fixed_split(self, path: Path):
        payload = [
            {
                "id": 101,
                "meta": {"dataset_split": "train", "eval_set_id": "test-eval-v1"},
                "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                "annotations": [
                    {
                        "id": 201,
                        "was_cancelled": False,
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
            },
            {
                "id": 102,
                "meta": {"dataset_split": "train", "eval_set_id": "test-eval-v1"},
                "data": {"image": "/data/local-files/?d=images/demo_blue_dark.png"},
                "annotations": [
                    {
                        "id": 202,
                        "was_cancelled": False,
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
            },
            {
                "id": 103,
                "meta": {"dataset_split": "eval", "eval_set_id": "test-eval-v1"},
                "data": {"image": "/data/local-files/?d=images/demo_blue_light.png"},
                "annotations": [
                    {
                        "id": 203,
                        "was_cancelled": False,
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
            },
            {
                "id": 104,
                "meta": {"dataset_split": "train", "eval_set_id": "test-eval-v1"},
                "data": {"image": "/data/local-files/?d=images/demo_green.png"},
                "annotations": [
                    {
                        "id": 204,
                        "was_cancelled": False,
                        "result": [
                            {
                                "from_name": "image_label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["Other"]},
                            }
                        ],
                    }
                ],
            },
            {
                "id": 105,
                "meta": {"dataset_split": "train", "eval_set_id": "test-eval-v1"},
                "data": {"image": "/data/local-files/?d=images/demo_green_dark.png"},
                "annotations": [
                    {
                        "id": 205,
                        "was_cancelled": False,
                        "result": [
                            {
                                "from_name": "image_label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["Other"]},
                            }
                        ],
                    }
                ],
            },
            {
                "id": 106,
                "meta": {"dataset_split": "eval", "eval_set_id": "test-eval-v1"},
                "data": {"image": "/data/local-files/?d=images/demo_gray.png"},
                "annotations": [
                    {
                        "id": 206,
                        "was_cancelled": False,
                        "result": [
                            {
                                "from_name": "image_label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": ["Other"]},
                            }
                        ],
                    }
                ],
            },
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_build_image_dataset_from_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            module = self._trainer_module(tmp_dir, image_root)

            export_path = tmp_dir / "image_export.json"
            self._write_image_export(export_path)

            samples, ctx = module._build_image_classification_dataset({"dataset_path": str(export_path)})
            self.assertEqual(6, len(samples))
            self.assertEqual(str(export_path), ctx["dataset_path"])
            self.assertEqual({"Product", "Other"}, {sample["label"] for sample in samples})
            self.assertTrue(all(Path(sample["image_path"]).exists() for sample in samples))

    def test_train_image_classifier_writes_artifacts_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            trainer = self._trainer_module(tmp_dir, image_root)

            export_path = tmp_dir / "image_export.json"
            self._write_image_export(export_path)
            samples, ctx = trainer._build_image_classification_dataset({"dataset_path": str(export_path)})

            metadata = trainer._train_real_image_classifier(samples, training_run=1, dataset_context=ctx)
            self.assertEqual("image_classification", metadata["task_type"])
            self.assertEqual(6, metadata["dataset"]["total_size"])
            self.assertIn("accuracy", metadata["metrics"])
            self.assertIn("macro_f1", metadata["metrics"])
            self.assertTrue(Path(trainer._IMAGE_CLASSIFIER_PATH).exists())
            self.assertTrue(Path(trainer._IMAGE_METADATA_PATH).exists())

    def test_train_image_classifier_uses_fixed_eval_split_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            trainer = self._trainer_module(tmp_dir, image_root)

            manifest = {
                "eval_set_id": "test-eval-v1",
                "images": [
                    {"filename": "demo_blue_light.png", "label": "Product"},
                    {"filename": "demo_gray.png", "label": "Other"},
                ],
            }
            (tmp_dir / "image_eval_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            export_path = tmp_dir / "image_export_fixed_split.json"
            self._write_image_export_with_fixed_split(export_path)
            samples, ctx = trainer._build_image_classification_dataset({"dataset_path": str(export_path)})

            metadata = trainer._train_real_image_classifier(samples, training_run=7, dataset_context=ctx)
            split = metadata["dataset"]["split"]
            fixed_eval = metadata["dataset"]["fixed_eval"]

            self.assertEqual("fixed_eval_set", split["strategy"])
            self.assertEqual(4, split["train_size"])
            self.assertEqual(2, split["eval_size"])
            self.assertEqual({"Other": 2, "Product": 2}, split["train_label_distribution"])
            self.assertEqual({"Other": 1, "Product": 1}, split["eval_label_distribution"])
            self.assertTrue(fixed_eval["enabled"])
            self.assertEqual("test-eval-v1", fixed_eval["eval_set_id"])
            self.assertTrue(fixed_eval["manifest_present"])
            self.assertTrue(fixed_eval["manifest_checked"])
            self.assertEqual(
                ["demo_blue_light.png", "demo_gray.png"],
                fixed_eval["eval_filenames"],
            )
            self.assertEqual("train_split_only", metadata["model_details"]["classifier"]["serving_fit"])
            self.assertIn("confusion_matrix_product_other", metadata["metrics"])
            self.assertIn("eval_predictions", metadata["metrics"])
            self.assertIn("eval_misclassified", metadata["metrics"])

    def test_train_image_classifier_fails_on_eval_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            trainer = self._trainer_module(tmp_dir, image_root)

            manifest = {
                "eval_set_id": "test-eval-v1",
                "images": [
                    {"filename": "demo_blue_light.png", "label": "Product"},
                    {"filename": "demo_green.png", "label": "Other"},
                ],
            }
            (tmp_dir / "image_eval_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            export_path = tmp_dir / "image_export_fixed_split.json"
            self._write_image_export_with_fixed_split(export_path)
            samples, ctx = trainer._build_image_classification_dataset({"dataset_path": str(export_path)})

            with self.assertRaisesRegex(ValueError, "mismatch with manifest"):
                trainer._train_real_image_classifier(samples, training_run=8, dataset_context=ctx)

    def test_parse_image_classification_sample_from_full_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            trainer = self._trainer_module(tmp_dir, image_root)

            task = {
                "id": 777,
                "project": 12,
                "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                "annotations": [
                    {
                        "id": 9001,
                        "was_cancelled": False,
                        "updated_at": "2026-05-12T08:00:00.000000Z",
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

            sample = trainer._parse_image_classification_sample_from_full_task(task)
            self.assertEqual(777, sample["task_id"])
            self.assertEqual(12, sample["project_id"])
            self.assertEqual("/data/local-files/?d=images/demo_blue.png", sample["image"])
            self.assertEqual("Product", sample["label"])
            self.assertEqual(9001, sample["annotation_id"])

    def test_append_and_read_image_training_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            trainer = self._trainer_module(tmp_dir, image_root)

            candidate = {
                "task_id": 7001,
                "project_id": 11,
                "image": "/data/local-files/?d=images/demo_product_red.png",
                "label": "Product",
                "source": "label_studio_webhook_task_fetch",
                "annotation_id": 9101,
                "updated_at": "2026-05-12T08:10:00.000000Z",
            }
            trainer._append_image_training_candidate(candidate)
            samples, stats = trainer._read_image_training_candidate_samples()
            self.assertEqual(1, len(samples))
            self.assertEqual(1, stats["used"])
            self.assertEqual("Product", samples[0]["label"])
            self.assertTrue(str(samples[0]["image_path"]).endswith("demo_product_red.png"))

    def test_build_image_dataset_excludes_eval_manifest_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            trainer = self._trainer_module(tmp_dir, image_root)

            manifest = {
                "eval_set_id": "image-cls-eval-v1",
                "images": [
                    {"filename": "demo_blue.png", "label": "Product"},
                    {"filename": "demo_gray.png", "label": "Other"},
                ],
            }
            (tmp_dir / "image_eval_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            candidates_path = Path(trainer.IMAGE_TRAINING_CANDIDATES_PATH)
            rows = [
                {
                    "task_id": 8001,
                    "project_id": 20,
                    "image": "/data/local-files/?d=images/demo_blue.png",
                    "label": "Product",
                    "source": "label_studio_webhook_task_fetch",
                    "annotation_id": 9201,
                    "updated_at": "2026-05-12T08:30:00.000000Z",
                },
                {
                    "task_id": 8002,
                    "project_id": 20,
                    "image": "/data/local-files/?d=images/demo_product_red.png",
                    "label": "Product",
                    "source": "label_studio_webhook_task_fetch",
                    "annotation_id": 9202,
                    "updated_at": "2026-05-12T08:31:00.000000Z",
                },
            ]
            candidates_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            samples, ctx = trainer._build_image_classification_dataset({"samples": []})
            file_names = {Path(sample["image_path"]).name for sample in samples}
            self.assertIn("demo_product_red.png", file_names)
            self.assertNotIn("demo_blue.png", file_names)
            self.assertEqual(1, ctx["candidate_eval_leakage_skipped"])

    def test_webhook_task_fetch_and_candidate_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)
            trainer = self._trainer_module(tmp_dir, image_root)

            task_payload = {
                "id": 900,
                "project": 30,
                "data": {"image": "/data/local-files/?d=images/demo_product_red.png"},
                "annotations": [
                    {
                        "id": 9500,
                        "was_cancelled": False,
                        "updated_at": "2026-05-12T08:40:00.000000Z",
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

            class _Resp:
                def __init__(self, payload):
                    self._payload = payload

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps(self._payload).encode("utf-8")

            captured = {}

            def fake_urlopen(req, timeout=None):
                captured["url"] = req.full_url
                captured["auth"] = req.get_header("Authorization")
                captured["timeout"] = timeout
                return _Resp(task_payload)

            original = trainer.urllib.request.urlopen
            trainer.urllib.request.urlopen = fake_urlopen
            try:
                sample, warning = trainer._maybe_record_image_candidate_from_webhook(
                    {"task": {"id": 900}, "project": {"id": 30}}
                )
            finally:
                trainer.urllib.request.urlopen = original

            self.assertIsNone(warning)
            self.assertIsNotNone(sample)
            self.assertIn("/api/tasks/900/", captured["url"])
            self.assertIn("project=30", captured["url"])
            self.assertEqual("Token test-token", captured["auth"])
            self.assertEqual(2.0, captured["timeout"])

            candidates_path = Path(trainer.IMAGE_TRAINING_CANDIDATES_PATH)
            lines = [line for line in candidates_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(1, len(lines))
            row = json.loads(lines[0])
            self.assertEqual(900, row["task_id"])
            self.assertEqual(30, row["project_id"])
            self.assertEqual("Product", row["label"])

    def test_ml_backend_image_fallback_and_trained_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_root = self._prepare_image_root(tmp_dir)

            backend_fallback = self._ml_backend_module(tmp_dir, image_root)
            parsed = {
                "image_name": "image",
                "choices": {
                    "name": "image_label",
                    "to_name": "image",
                    "labels": ["Product", "Other"],
                },
            }
            state = backend_fallback._default_state()

            fallback_pred = backend_fallback._image_classification(
                {"id": "img-fallback", "data": {"image": "/data/local-files/?d=images/demo_blue.png"}},
                parsed,
                state,
            )
            self.assertEqual("demo-rule-fallback", fallback_pred["prediction_source"])

            trainer = self._trainer_module(tmp_dir, image_root)
            export_path = tmp_dir / "image_export.json"
            self._write_image_export(export_path)
            samples, ctx = trainer._build_image_classification_dataset({"dataset_path": str(export_path)})
            trainer._train_real_image_classifier(samples, training_run=2, dataset_context=ctx)

            backend_trained = self._ml_backend_module(tmp_dir, image_root)
            trained_pred = backend_trained._image_classification(
                {"id": "img-trained", "data": {"image": "/data/local-files/?d=images/demo_blue.png"}},
                parsed,
                state,
            )
            self.assertEqual("trained-image-classifier", trained_pred["prediction_source"])
            self.assertTrue(trained_pred["model_version"].startswith("image-cls-v"))
            self.assertIn("confidence", trained_pred)


if __name__ == "__main__":
    unittest.main()
