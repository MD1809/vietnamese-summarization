import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from app.demo_data import load_demo_evaluation


class DemoEvaluationTests(unittest.TestCase):
    def fixture(self, root, evaluation_id="test_evaluation_20261113_092101_564105"):
        profile = {"run_id": "new_run", "train_config": {"max_target_length": 128}}
        summary = {"best_epoch": 2, "best_global_step": 500, "best_validation_loss": 2.1}
        directory = root / "outputs" / profile["run_id"] / evaluation_id
        directory.mkdir(parents=True, exist_ok=True)
        generation = {"num_beams": 1, "max_new_tokens": 128, "do_sample": False}
        textrank = {"damping": 0.85, "tolerance": 0.000001, "max_iterations": 100}
        decision = {
            "run_id": profile["run_id"], "comparison_id": "validation_fixture",
            "selected_method": "greedy", "generation_config": generation, "textrank_config": textrank,
        }
        hashes = {}
        project = Path(__file__).resolve().parent.parent
        for name in ["src/textRank.py", "src/inference.py", "src/preprocessing.py"]:
            content = (project / name).read_bytes().replace(b"\r\n", b"\n")
            hashes[name] = hashlib.sha256(content).hexdigest()
        metrics = {
            "run_id": profile["run_id"], "run_status": "completed",
            "train_config": profile["train_config"], "decision": decision,
            "generation_config": generation, "textrank_config": textrank,
            "source": {"runtime_source_sha256": hashes},
        }
        metrics.update(summary)
        comparison = {
            "checkpoint_run_id": profile["run_id"], "comparison_id": "validation_fixture",
            "train_config": profile["train_config"],
            "candidates": {"greedy": {"generation_config": generation}},
        }
        comparison.update(summary)
        tr_validation = {
            "checkpoint_run_id": profile["run_id"], "comparison_id": "validation_fixture",
            "train_config": profile["train_config"], "textrank_config": textrank,
        }
        for name, value in [
            ("metrics.json", metrics), ("decoding_selection.json", decision),
            ("generation.json", generation), ("decoding_comparison.json", comparison),
            ("textrank_validation.json", tr_validation),
        ]:
            (directory / name).write_text(json.dumps(value), encoding="utf-8")
        return profile, summary, directory

    def change(self, directory, name, key, value):
        path = directory / name
        content = json.loads(path.read_text(encoding="utf-8"))
        content[key] = value
        path.write_text(json.dumps(content), encoding="utf-8")

    def test_uses_evaluated_greedy_instead_of_original_beam_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, summary, expected = self.fixture(root)
            profile["generation_config"] = {"num_beams": 4, "max_new_tokens": 128, "do_sample": False}
            metrics, decision, generation, selected = load_demo_evaluation(root, profile, summary)
            self.assertEqual(selected, expected)
            self.assertEqual(generation["num_beams"], 1)
            self.assertEqual(decision["selected_method"], "greedy")
            self.assertEqual(profile["generation_config"]["num_beams"], 4)

    def test_latest_completed_evaluation_is_selected_and_incomplete_one_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, "test_evaluation_20261112_000000_000000")
            profile, summary, expected = self.fixture(root)
            incomplete = root / "outputs" / "new_run" / "test_evaluation_20261114_000000_000000"
            incomplete.mkdir()
            selected = load_demo_evaluation(root, profile, summary)[3]
            self.assertEqual(selected, expected)

    def test_missing_latest_generation_does_not_fall_back_to_older_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, "test_evaluation_20261112_000000_000000")
            profile, summary, newest = self.fixture(root)
            (newest / "generation.json").unlink()
            with self.assertRaises(FileNotFoundError):
                load_demo_evaluation(root, profile, summary)

    def test_wrong_run_or_best_is_rejected(self):
        for key, value in [("run_id", "other_run"), ("best_epoch", 3)]:
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    profile, summary, result = self.fixture(root)
                    self.change(result, "metrics.json", key, value)
                    with self.assertRaises(ValueError):
                        load_demo_evaluation(root, profile, summary)

    def test_generation_or_textrank_mismatch_is_rejected(self):
        for filename, key, value in [
            ("generation.json", "num_beams", 4),
            ("textrank_validation.json", "textrank_config", {"damping": 0.9}),
        ]:
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    profile, summary, result = self.fixture(root)
                    self.change(result, filename, key, value)
                    with self.assertRaises(ValueError):
                        load_demo_evaluation(root, profile, summary)

    def test_changed_algorithm_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, summary, result = self.fixture(root)
            self.change(result, "metrics.json", "source", {"runtime_source_sha256": {
                "src/textRank.py": "changed", "src/inference.py": "changed", "src/preprocessing.py": "changed",
            }})
            with self.assertRaises(ValueError):
                load_demo_evaluation(root, profile, summary)

    def test_explicit_evaluation_and_absent_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, summary, expected = self.fixture(root)
            self.fixture(root, "test_evaluation_20261114_000000_000000")
            self.assertEqual(load_demo_evaluation(root, profile, summary, expected.name)[3], expected)
            profile["run_id"] = "run_without_evaluation"
            with self.assertRaises(FileNotFoundError):
                load_demo_evaluation(root, profile, summary)
