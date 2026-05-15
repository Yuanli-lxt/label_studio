import importlib.util
import json
import os
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


class TextClassifierQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("joblib") is None or importlib.util.find_spec("sklearn") is None:
            raise unittest.SkipTest("joblib/scikit-learn not installed in host Python")

    def _trainer_module(self, tmp_dir: Path):
        return load_module(
            TRAINER_APP,
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "MODEL_ARTIFACTS_DIR": str(tmp_dir / "artifacts"),
                "TRAINING_EVENT_LOG_PATH": str(tmp_dir / "training_events.jsonl"),
                "TRAINING_DATASET_EVENTS_PATH": str(tmp_dir / "dataset_events.jsonl"),
                "TEXT_CLS_MIN_TOTAL_SAMPLES": "8",
                "TEXT_CLS_MIN_PER_CLASS_SAMPLES": "3",
                "TEXT_CLS_MAX_IMBALANCE_RATIO": "3.0",
            },
        )

    def _ml_backend_module(self, tmp_dir: Path):
        return load_module(
            ML_BACKEND_APP,
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "MODEL_ARTIFACTS_DIR": str(tmp_dir / "artifacts"),
            },
        )

    def _balanced_samples(self):
        positives = [
            "excellent support and helpful workflow",
            "great quality and useful automation",
            "good model output and clear interface",
            "reliable release and fast response",
            "happy team with stable performance",
        ]
        negatives = [
            "awful output and poor reliability",
            "bad workflow and slow interface",
            "disappointing quality and noisy predictions",
            "frustrating bugs and unstable release",
            "unhappy team due to broken behavior",
        ]

        samples = []
        for idx, text in enumerate(positives, start=1):
            samples.append({"text": text, "label": "Positive", "task_id": idx})
        for idx, text in enumerate(negatives, start=101):
            samples.append({"text": text, "label": "Negative", "task_id": idx})
        return samples

    def _write_demo_export(self, path: Path):
        payload = [
            {
                "id": 1,
                "data": {"text": "simple positive text"},
                "annotations": [
                    {
                        "id": 11,
                        "was_cancelled": False,
                        "result": [
                            {
                                "from_name": "text_label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": ["Positive"]},
                            }
                        ],
                    }
                ],
            },
            {
                "id": 2,
                "data": {"text": "simple negative text"},
                "annotations": [
                    {
                        "id": 12,
                        "was_cancelled": False,
                        "result": [
                            {
                                "from_name": "text_label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": ["Negative"]},
                            }
                        ],
                    }
                ],
            },
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_dataset_quality_validation_catches_small_or_unbalanced_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = self._trainer_module(Path(tmp))

            too_small = [
                {"text": "good example", "label": "Positive"},
                {"text": "bad example", "label": "Negative"},
            ]
            report_small = module._dataset_quality_report(too_small)
            self.assertFalse(report_small["validation"]["passed"])
            self.assertTrue(any("at least" in err for err in report_small["validation"]["errors"]))

            imbalanced = [
                {"text": f"positive sample {idx}", "label": "Positive"}
                for idx in range(1, 9)
            ] + [
                {"text": "negative sample 1", "label": "Negative"},
                {"text": "negative sample 2", "label": "Negative"},
            ]
            report_imbalanced = module._dataset_quality_report(imbalanced)
            self.assertFalse(report_imbalanced["validation"]["passed"])
            self.assertIn("Negative", report_imbalanced["label_distribution"])

    def test_training_metadata_contains_distribution_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = self._trainer_module(Path(tmp))
            metadata = module._train_real_text_classifier(self._balanced_samples(), training_run=1)

            self.assertEqual("text_classification", metadata["task_type"])
            self.assertIn("quality", metadata["dataset"])
            self.assertIn("label_distribution", metadata["dataset"]["quality"])
            self.assertTrue(metadata["dataset"]["quality"]["validation"]["passed"])

            metrics = metadata["metrics"]
            self.assertIn("accuracy", metrics)
            self.assertIn("macro_f1", metrics)
            self.assertIn("per_label", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("confidence", metrics)

            self.assertTrue(Path(module._CLASSIFIER_PATH).exists())
            self.assertTrue(Path(module._VECTORIZER_PATH).exists())
            self.assertTrue(Path(module._METADATA_PATH).exists())

    def test_build_dataset_uses_demo_export_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            module = self._trainer_module(tmp_dir)

            export_path = tmp_dir / "demo_export.json"
            self._write_demo_export(export_path)

            samples, ctx = module._build_text_classification_dataset(
                {
                    "dataset_path": str(export_path),
                },
                webhook_sample=None,
            )
            self.assertEqual(2, len(samples))
            self.assertEqual(str(export_path), ctx["dataset_path"])
            self.assertEqual(
                {"Positive", "Negative"},
                {sample["label"] for sample in samples},
            )

    def test_training_metadata_records_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = self._trainer_module(Path(tmp))
            context = {
                "dataset_path": "/demo-tasks/text_classification_labeled_export.json",
            }

            metadata = module._train_real_text_classifier(
                self._balanced_samples(),
                training_run=2,
                dataset_context=context,
            )
            dataset = metadata["dataset"]
            self.assertEqual(
                "/demo-tasks/text_classification_labeled_export.json",
                dataset["source_path"],
            )

    def test_ml_backend_fallback_when_artifacts_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = self._ml_backend_module(Path(tmp))

            state = module._default_state()
            parsed = {
                "text_name": "text",
                "choices": {
                    "name": "text_label",
                    "to_name": "text",
                    "labels": ["Positive", "Negative"],
                },
            }
            task = {"id": "txt-1", "data": {"text": "I love this helpful release"}}

            prediction = module._text_classification(task, parsed, state)
            self.assertEqual("demo-rule-fallback", prediction["prediction_source"])
            self.assertEqual(state["model_version"], prediction["model_version"])
            self.assertIn("confidence", prediction)
            label = prediction["result"][0]["value"]["choices"][0]
            self.assertEqual("Positive", label)

    def test_ml_backend_loads_trained_artifacts_and_returns_confidence_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            trainer = self._trainer_module(tmp_dir)
            trainer._train_real_text_classifier(self._balanced_samples(), training_run=3)

            backend = self._ml_backend_module(tmp_dir)
            state = backend._default_state()
            parsed = {
                "text_name": "text",
                "choices": {
                    "name": "text_label",
                    "to_name": "text",
                    "labels": ["Positive", "Negative"],
                },
            }

            prediction = backend._text_classification(
                {"id": "txt-2", "data": {"text": "excellent and useful quality"}},
                parsed,
                state,
            )

            self.assertEqual("trained-text-classifier", prediction["prediction_source"])
            self.assertTrue(prediction["model_version"].startswith("text-cls-v"))
            self.assertIn("confidence", prediction)
            self.assertIn("top_probabilities", prediction["confidence"])
            self.assertGreaterEqual(len(prediction["confidence"]["top_probabilities"]), 1)


if __name__ == "__main__":
    unittest.main()
