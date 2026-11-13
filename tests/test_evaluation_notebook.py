import contextlib
import csv
import io
import json
import shutil
import tempfile
import types
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils import load_generation_config


NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "vietnamese_summarization_evaluation.ipynb"


def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            codes.append("".join(cell["source"]))
    return codes


class EvaluationNotebookTests(unittest.TestCase):
    def execute(self, source, namespace):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    def fixture(self, root):
        output = root / "outputs" / "new_run"
        validation = output / "validation" / "decoding_comparison_fixture"
        validation.mkdir(parents=True)
        config = {"seed": 42, "paths": {"subset_manifest": "subset.json", "processed_manifest": "processed.json"}}
        generation = {"num_beams": 4, "max_new_tokens": 128, "do_sample": False}
        summary = {"best_epoch": 2, "best_global_step": 500, "best_validation_loss": 2.1}
        hashes = {}
        for name in ["src/textRank.py", "src/inference.py", "src/preprocessing.py", "src/metrics.py", "src/utils.py"]:
            hashes[name] = "source_hash"
        source = {"git_commit": "evaluation_commit", "runtime_source_sha256": hashes}
        comparison = {
            "checkpoint_run_id": "new_run", "comparison_id": validation.name,
            "train_config": config, "validation_sample_ids": ["v1", "v2"],
            "best_epoch": 2, "best_global_step": 500, "best_validation_loss": 2.1,
            "candidates": {"beam": {"generation_config": generation}}, "source": source,
        }
        textrank_report = {
            "checkpoint_run_id": "new_run", "comparison_id": validation.name,
            "train_config": config, "validation_sample_ids": ["v1", "v2"],
            "textrank_config": {"damping": 0.85}, "source": source,
        }
        for name, value in [("decoding_comparison.json", comparison), ("textrank_validation.json", textrank_report)]:
            (validation / name).write_text(json.dumps(value), encoding="utf-8")
        for name in ["config.json", "training_summary.json", "subset.json", "processed.json"]:
            (output / name).write_text("{}", encoding="utf-8")
        rows = [
            {"id": "t1", "document": "source one", "summary": "reference one"},
            {"id": "t2", "document": "source two", "summary": "reference two"},
        ]

        def load_test(drive_root, approved):
            decisions = list(output.glob("test_evaluation_*/decoding_selection.json"))
            self.assertEqual(len(decisions), 1)
            decision = json.loads(decisions[0].read_text())
            self.assertEqual(decision["selected_method"], "beam")
            self.assertEqual(decision["textrank_config"], {"damping": 0.85})
            return rows, {"sample_count": 2}

        evaluate = types.SimpleNamespace(
            record_evaluation_source=Mock(return_value=source),
            load_fixed_test_source=Mock(side_effect=load_test),
            source_ids_fingerprint=Mock(return_value="tokens_hash"),
        )
        metric = {"rouge1": 0.3, "rouge2": 0.2, "rougeL": 0.25}
        percent = {"rouge1": 30.0, "rouge2": 20.0, "rougeL": 25.0}
        scoring = Mock(return_value={
            "sample_count": 2, "mean_f1": metric, "mean_f1_percent": percent,
            "per_sample": [metric, metric], "settings": {"unicode": True},
        })
        namespace = {
            "Path": Path, "json": json, "csv": csv, "shutil": shutil,
            "datetime": datetime, "timezone": timezone,
            "evaluation_run_id": "new_run", "validation_comparison_id": "",
            "drive_root": root, "output_dir": output, "repo_dir": root,
            "train_config": config, "config_path": root / "runtime_train.json",
            "checkpoint_dir": root / "checkpoints" / "new_run", "run_summary": summary,
            "validation_rows": [{"id": "v1"}, {"id": "v2"}],
            "src": types.SimpleNamespace(evaluate=evaluate),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=Mock(return_value=True), empty_cache=Mock())),
            "gc": types.SimpleNamespace(collect=Mock()),
            "load_generation_config": load_generation_config,
            "load_train_config": Mock(return_value=config),
            "load_summarizer": Mock(return_value=(object(), object(), {"epoch": 2, "global_step": 500, "validation_loss": 2.1})),
            "set_seed": Mock(), "manifests": {"subset_manifest": {}, "processed_manifest": {}},
            "summarize_textrank": Mock(return_value={
                "summary": "extracted", "used_input_ids": [2, 1], "textrank_config": {"damping": 0.85},
                "summary_was_truncated": False, "converged": True, "elapsed_seconds": 0.01,
            }),
            "summarize_text": Mock(return_value={
                "summary": "generated", "used_input_ids": [2, 1], "generation_config": generation,
                "input_token_count": 2, "used_input_token_count": 2, "was_truncated": False, "elapsed_seconds": 0.1,
            }),
            "score_summaries": scoring,
        }
        return namespace

    def enable_test(self, namespace):
        source = notebook_code()[5]
        source = source.replace("evaluate_test = False", "evaluate_test = True", 1)
        source = source.replace('selected_method = ""', 'selected_method = "beam"', 1)
        self.execute(source, namespace)

    def test_test_disabled_by_default_without_reading_data(self):
        namespace = {}
        for source in notebook_code()[5:8]:
            self.execute(source, namespace)
        self.assertFalse(namespace["evaluate_test"])
        self.assertNotIn("test_dir", namespace)

    def test_validation_adds_textrank_with_matching_sources_and_saves_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = self.fixture(root)
            namespace["validation_rows"] = [
                {"id": "v1", "document": "first source", "summary": "first reference"},
                {"id": "v2", "document": "second source", "summary": "second reference"},
            ]
            result_dir = root / "outputs" / "decoding_comparison_new"
            result_dir.mkdir()
            comparison = {
                "comparison_id": result_dir.name, "output_dir": str(result_dir),
                "validation_sample_ids": ["v1", "v2"], "proposed_method": "beam",
                "candidates": {
                    "greedy": {"mean_f1_percent": {"rougeL": 20}},
                    "beam": {"mean_f1_percent": {"rougeL": 21}},
                },
            }
            greedy = []
            for row in namespace["validation_rows"]:
                greedy.append(json.dumps({"id": row["id"], "source_ids_sha256": "tokens_hash"}))
            (result_dir / "greedy_predictions.jsonl").write_text("\n".join(greedy), encoding="utf-8")
            (result_dir / "decoding_comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
            namespace["generation_path"] = root / "generation.json"
            namespace["src"].evaluate.run_decoding_comparison = Mock(return_value=comparison)
            namespace["summarize_textrank"].return_value["was_truncated"] = False
            transformers = types.ModuleType("transformers")
            transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=Mock(return_value=object()))
            colab = types.ModuleType("google.colab")
            colab.files = types.SimpleNamespace(download=Mock())
            google = types.ModuleType("google")
            google.colab = colab
            with patch.dict("sys.modules", {"transformers": transformers, "google": google, "google.colab": colab}):
                self.execute(notebook_code()[4], namespace)
            saved = namespace["output_dir"] / "validation" / result_dir.name
            report = json.loads((saved / "textrank_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(report["validation_sample_ids"], ["v1", "v2"])
            self.assertFalse(report["test_content_read"])
            lines = (saved / "textrank_predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["id"] for line in lines], ["v1", "v2"])
            colab.files.download.assert_called_once()
            namespace["src"].evaluate.load_fixed_test_source.assert_not_called()

    def test_reconnected_evaluation_freezes_decision_then_scores_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.fixture(Path(directory))
            self.enable_test(namespace)
            namespace["src"].evaluate.load_fixed_test_source.assert_not_called()
            self.execute(notebook_code()[6], namespace)
            namespace["src"].evaluate.load_fixed_test_source.assert_called_once()
            self.assertEqual(namespace["score_summaries"].call_count, 2)
            test_dir = namespace["test_dir"]
            records = []
            for line in (test_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
                records.append(json.loads(line))
            self.assertEqual([row["id"] for row in records], ["t1", "t2"])
            self.assertEqual(records[0]["textrank"], "extracted")
            self.assertEqual(records[0]["vit5"], "generated")
            report = json.loads((test_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["results"]), {"textrank", "vit5"})
            self.assertTrue(report["test_previously_inspected_in_project"])
            colab = types.ModuleType("google.colab")
            colab.files = types.SimpleNamespace(download=Mock())
            google = types.ModuleType("google")
            google.colab = colab
            with patch.dict("sys.modules", {"google": google, "google.colab": colab}):
                self.execute(notebook_code()[7], namespace)
            colab.files.download.assert_called_once()
            with zipfile.ZipFile(namespace["archive_path"]) as archive:
                self.assertIn("comparison.csv", archive.namelist())
                self.assertIn("textrank_validation.json", archive.namelist())

    def test_missing_textrank_validation_prevents_test_release(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.fixture(Path(directory))
            for path in namespace["output_dir"].glob("validation/*/textrank_validation.json"):
                path.unlink()
            with self.assertRaises(RuntimeError):
                self.enable_test(namespace)
            namespace["src"].evaluate.load_fixed_test_source.assert_not_called()

    def test_changed_textrank_source_requires_new_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.fixture(Path(directory))
            source = namespace["src"].evaluate.record_evaluation_source.return_value
            source["runtime_source_sha256"]["src/textRank.py"] = "changed_source"
            with self.assertRaises(RuntimeError):
                self.enable_test(namespace)
            namespace["src"].evaluate.load_fixed_test_source.assert_not_called()

    def test_changed_choice_after_decision_does_not_open_test(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.fixture(Path(directory))
            self.enable_test(namespace)
            namespace["selected_method"] = "greedy"
            with self.assertRaises(RuntimeError):
                self.execute(notebook_code()[6], namespace)
            namespace["src"].evaluate.load_fixed_test_source.assert_not_called()

    def test_different_source_tokens_are_rejected_without_publishing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = self.fixture(Path(directory))
            self.enable_test(namespace)
            namespace["summarize_text"].return_value["used_input_ids"] = [3, 1]
            with self.assertRaises(RuntimeError):
                self.execute(notebook_code()[6], namespace)
            self.assertFalse((namespace["test_dir"] / "metrics.json").exists())
