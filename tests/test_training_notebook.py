import contextlib
import copy
import csv
import io
import json
import math
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.train import write_training_history


NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "vietnamese_summarization_train.ipynb"


def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            codes.append("".join(cell["source"]))
    return codes


class TrainingNotebookTests(unittest.TestCase):
    def fixture(self, directory, resume):
        root = Path(directory)
        output = root / "outputs" / "new_run"
        output.mkdir(parents=True)
        config = {"seed": 42, "num_epochs": 3, "gradient_accumulation_steps": 1}
        profile = {
            "run_id": "new_run",
            "train_config": config,
            "source": {"git_commit": "fixture_commit"},
            "source_changes": {"patch_file": None, "untracked_snapshot_files": []},
            "data": {"train_sample_count": 2, "validation_sample_count": 1},
            "paths": {"output_dir": str(output), "checkpoint_dir": str(root / "checkpoints" / "new_run")},
        }
        previous_row = {
            "epoch": 1, "global_step": 1, "train_loss": 3.0, "validation_loss": 2.0,
            "train_sample_count": 2, "validation_sample_count": 1,
            "train_valid_target_tokens": 10, "validation_valid_target_tokens": 5,
            "learning_rate": 0.00005, "train_seconds": 1.0,
            "validation_seconds": 0.1, "epoch_compute_seconds": 1.1,
        }
        progress = {
            "completed_epochs": 1, "global_step": 1, "training_history": [previous_row],
            "best_epoch": 1, "best_global_step": 1, "best_validation_loss": 2.0,
        }
        (output / "config.json").write_text(json.dumps(profile), encoding="utf-8")
        optimizer = types.SimpleNamespace(state={}, param_groups=[{"lr": 0.00005}])
        losses = [2.0, 1.8, 2.1]
        if resume:
            losses = losses[1:]
        reports = []
        for loss in losses:
            reports.append({"validation_loss": loss, "sample_count": 1, "valid_target_token_count": 5})

        def train_epoch(model, loader, optimizer, global_step, config_path):
            return {
                "train_loss": 3.0 - global_step * 0.5, "global_step": global_step + 1,
                "train_sample_count": 2, "train_valid_target_tokens": 10, "train_seconds": 1.0,
            }

        def update_best(model, tokenizer, checkpoint, candidate, validation, config_path):
            updated = copy.deepcopy(candidate)
            loss = validation["validation_loss"]
            changed = "best_validation_loss" not in updated or loss < updated["best_validation_loss"]
            if changed:
                updated["best_validation_loss"] = loss
                updated["best_epoch"] = updated["completed_epochs"]
                updated["best_global_step"] = updated["global_step"]
            return updated, {"updated": changed}

        train = types.SimpleNamespace(
            create_training_run=Mock(return_value=profile),
            load_vit5_model=Mock(return_value=object()),
            create_optimizer=Mock(return_value=optimizer),
            load_last_checkpoint=Mock(return_value=(object(), object(), optimizer, progress, {})),
            collect_run_source=Mock(return_value=({"git_commit": "fixture_commit"}, b"", {})),
            train_epoch=Mock(side_effect=train_epoch),
            validate_model=Mock(side_effect=reports),
            update_best_checkpoint=Mock(side_effect=update_best),
            save_last_checkpoint=Mock(),
            write_training_history=write_training_history,
        )
        namespace = {
            "Path": Path, "json": json, "csv": csv, "math": math, "time": time,
            "src": types.SimpleNamespace(train=train),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=Mock(), empty_cache=Mock())),
            "gc": types.SimpleNamespace(collect=Mock()), "set_seed": Mock(),
            "resume_run_id": "new_run" if resume else "", "drive_root": root,
            "repo_dir": root, "target_commit": "fixture_commit", "train_config": config,
            "config_path": root / "configs/train.json", "tokenizer": object(),
            "data_loaders": {"train": [object()], "validation": [object()]},
            "tokenized_splits": {
                "train": [{"id": "train_1"}, {"id": "train_2"}],
                "validation": [{"id": "val_1"}],
            },
        }
        return namespace, train

    def execute(self, code, namespace):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, str(NOTEBOOK), "exec"), namespace)

    def test_fresh_run_trains_all_epochs_and_writes_its_own_summary(self):
        codes = notebook_code()
        with tempfile.TemporaryDirectory() as directory:
            namespace, train = self.fixture(directory, False)
            self.execute(codes[5], namespace)
            self.execute(codes[6], namespace)
            self.execute(codes[7], namespace)
            train.create_training_run.assert_called_once()
            train.load_vit5_model.assert_called_once()
            train.load_last_checkpoint.assert_not_called()
            self.assertEqual(train.train_epoch.call_count, 3)
            self.assertEqual(train.validate_model.call_count, 3)
            self.assertEqual(train.save_last_checkpoint.call_count, 3)
            summary = namespace["training_summary"]
            self.assertEqual(summary["run_id"], "new_run")
            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(summary["global_step"], 3)
            self.assertEqual(namespace["report_run_id"], "new_run")
            saved = json.loads((namespace["output_dir"] / "training_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, summary)

    def test_resume_runs_only_remaining_epochs_and_preserves_history(self):
        codes = notebook_code()
        with tempfile.TemporaryDirectory() as directory:
            namespace, train = self.fixture(directory, True)
            self.execute(codes[5], namespace)
            self.execute(codes[6], namespace)
            self.execute(codes[7], namespace)
            train.create_training_run.assert_not_called()
            train.load_vit5_model.assert_not_called()
            train.load_last_checkpoint.assert_called_once()
            self.assertEqual(train.train_epoch.call_count, 2)
            self.assertEqual(train.train_epoch.call_args_list[0].args[3], 1)
            history = namespace["training_summary"]["history"]
            self.assertEqual([r["epoch"] for r in history], [1, 2, 3])
            self.assertEqual(history[0]["train_loss"], 3.0)

    def test_mismatched_source_prevents_training(self):
        codes = notebook_code()
        with tempfile.TemporaryDirectory() as directory:
            namespace, train = self.fixture(directory, False)
            self.execute(codes[5], namespace)
            train.collect_run_source.return_value = ({"git_commit": "changed"}, b"", {})
            with self.assertRaises(RuntimeError):
                self.execute(codes[6], namespace)
            train.train_epoch.assert_not_called()

    def test_test_is_disabled_by_default_without_loading_test_data(self):
        source = notebook_code()[12]
        self.execute(source, {})

    def test_optional_test_freezes_decision_and_scores_the_new_run(self):
        source = notebook_code()[12]
        source = source.replace("evaluate_test = False", "evaluate_test = True", 1)
        source = source.replace('selected_method = ""', 'selected_method = "beam"', 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("subset.json").write_text("{}", encoding="utf-8")
            root.joinpath("processed.json").write_text("{}", encoding="utf-8")
            generation = {"num_beams": 4, "do_sample": False, "max_new_tokens": 128}
            rows = [
                {"id": "test_1", "document": "first", "summary": "ref one"},
                {"id": "test_2", "document": "second", "summary": "ref two"},
            ]

            def load_test(drive_root, approved):
                files = list(root.glob("test_evaluation_*/decoding_selection.json"))
                self.assertEqual(len(files), 1)
                decision = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertEqual(decision["run_id"], "new_run")
                return rows, {"sha256": "fixture"}

            score = {"sample_count": 2, "mean_f1": {"rougeL": 0.5}, "mean_f1_percent": {"rougeL": 50.0}, "settings": {}}
            namespace = {
                "Path": Path, "json": json, "output_dir": root, "drive_root": root,
                "run_profile": {"run_id": "new_run"},
                "train_config": {"seed": 42, "paths": {"subset_manifest": "subset.json", "processed_manifest": "processed.json"}},
                "comparison_report": {
                    "checkpoint_run_id": "new_run", "comparison_id": "new_validation",
                    "best_epoch": 2, "best_global_step": 2, "best_validation_loss": 1.8,
                    "candidates": {"beam": {"generation_config": generation}},
                },
                "src": types.SimpleNamespace(evaluate=types.SimpleNamespace(load_fixed_test_source=Mock(side_effect=load_test))),
                "checkpoint_dir": root / "checkpoints/new_run", "config_path": root / "train.json",
                "load_summarizer": Mock(return_value=(object(), object(), {"epoch": 2, "global_step": 2, "validation_loss": 1.8})),
                "summarize_text": Mock(return_value={"summary": "generated", "used_input_ids": [1]}),
                "set_seed": Mock(), "gc": types.SimpleNamespace(collect=Mock()),
                "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=Mock())),
            }
            with patch("src.baseline.summarize_lead_three", return_value={"summary": "baseline", "used_input_ids": [1]}):
                with patch("src.metrics.score_summaries", return_value=score) as scoring:
                    self.execute(source, namespace)
            self.assertEqual(scoring.call_count, 2)
            self.assertEqual(scoring.call_args.args[0], ["ref one", "ref two"])
            test_dir = namespace["test_dir"]
            records = [json.loads(line) for line in (test_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["id"] for r in records], ["test_1", "test_2"])
            saved = json.loads((test_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["run_id"], "new_run")
            self.assertEqual(saved["generation_config"], generation)
