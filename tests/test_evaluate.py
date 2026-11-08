import hashlib
import csv
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data import _split_manifest_fingerprint
from src.evaluate import (
    DECODING_RELEASE_COMMIT,
    DECODING_REPORT_DIR,
    choose_decoding,
    load_fixed_test_source,
    run_decoding_comparison,
    run_test_evaluation,
    verify_test_release,
)
from src.utils import load_train_config


class DecodingChoiceTests(unittest.TestCase):
    def candidates(self, greedy, beam):
        return {
            "greedy": {"sample_count": 2, "mean_f1": {"rougeL": greedy}},
            "beam": {"sample_count": 2, "mean_f1": {"rougeL": beam}},
        }

    def test_beam_is_selected_only_for_a_higher_rouge_l(self):
        self.assertEqual(choose_decoding(self.candidates(0.3, 0.4)), "beam")

    def test_exact_tie_selects_greedy(self):
        self.assertEqual(choose_decoding(self.candidates(0.4, 0.4)), "greedy")

    def test_higher_greedy_score_selects_greedy(self):
        self.assertEqual(choose_decoding(self.candidates(0.5, 0.4)), "greedy")

    def test_invalid_scores_or_different_counts_are_rejected(self):
        for value in [float("nan"), float("inf"), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                choose_decoding(self.candidates(0.4, value))
        candidates = self.candidates(0.4, 0.5)
        candidates["beam"]["sample_count"] = 1
        with self.assertRaises(ValueError):
            choose_decoding(candidates)


class DecodingComparisonTests(unittest.TestCase):
    def prepare(self, root):
        config = load_train_config()
        self.config = config
        self.rows = [
            {"id": "v1", "group_id": "g1", "document": "document one", "summary": "reference one"},
            {"id": "v2", "group_id": "g2", "document": "document two", "summary": "reference two"},
        ]
        train = [{"id": "t1", "group_id": "gt", "document": "train document", "summary": "train reference"}]
        subset = {"dataset": config["dataset"], "seed": config["seed"], "splits": {}}
        files = {}
        for name, rows in [("train", train), ("validation", self.rows)]:
            ids = []
            lines = []
            for row in rows:
                ids.append(row["id"])
                lines.append(json.dumps(row))
            content = ("\n".join(lines) + "\n").encode("utf-8")
            relative_path = config["paths"][name + "_file"]
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            subset["splits"][name] = {"sample_ids": ids, "pair_count": len(ids)}
            files[name] = {
                "path": relative_path,
                "pair_count": len(rows),
                "byte_count": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        processed = {
            "dataset": config["dataset"],
            "seed": config["seed"],
            "subset_manifest_sha256": _split_manifest_fingerprint(subset),
            "test_lock": {"locked": True},
            "files": files,
        }
        for key, content in [("subset_manifest", subset), ("processed_manifest", processed)]:
            path = root / config["paths"][key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content), encoding="utf-8")
        self.test_path = root / config["paths"]["test_file"]
        self.test_path.write_bytes(b"LOCKED TEST MUST NOT BE READ")
        self.selection = {
            "validation_sample_ids": ["v1", "v2"],
            "epoch": 2,
            "global_step": 500,
            "validation_loss": 2.1,
        }
        self.change_source = False
        self.wrong_score_count = False

    def inference(self, text, model, tokenizer, config_path, generation_config_path):
        generation = json.loads(Path(generation_config_path).read_text())
        source_ids = [7, 1]
        if text == "document two":
            source_ids = [8, 1]
        if self.change_source and generation["num_beams"] == 4:
            source_ids = [9, 1]
        return {
            "summary": "prediction " + str(generation["num_beams"]),
            "used_input_ids": source_ids,
            "input_token_count": 2,
            "used_input_token_count": 2,
            "was_truncated": False,
            "elapsed_seconds": float(generation["num_beams"]),
            "generation_config": generation,
            "generated_ids": [0, 5, 1],
        }

    def scores(self, references, predictions):
        self.assertEqual(references, ["reference one", "reference two"])
        value = 0.4
        if predictions[0] == "prediction 4":
            value = 0.5
        mean = {"rouge1": value, "rouge2": value, "rougeL": value}
        percent = {}
        sample_scores = {}
        for name in mean:
            percent[name] = mean[name] * 100
            sample_scores[name] = {"precision": value, "recall": value, "f1": value}
        count = len(predictions)
        if self.wrong_score_count:
            count = count - 1
        return {
            "sample_count": count,
            "per_sample": [sample_scores, sample_scores],
            "mean_f1": mean,
            "mean_f1_percent": percent,
        }

    def run_comparison(self, root):
        with patch("src.evaluate.load_summarizer", return_value=(object(), object(), self.selection)):
            with patch("src.evaluate.set_seed"):
                with patch("src.evaluate.version", return_value="test-version"):
                    with patch("src.evaluate.summarize_text", side_effect=self.inference) as generate:
                        with patch("src.evaluate.score_summaries", side_effect=self.scores):
                            with patch("src.evaluate.print"):
                                report = run_decoding_comparison(Path.cwd(), root, "saved_run")
        return report, generate

    def test_same_validation_source_warmup_artifacts_and_no_test_read_or_config_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            generation_path = Path("configs/generation.json")
            original = generation_path.read_bytes()
            read_bytes = Path.read_bytes
            read_text = Path.read_text
            reads = []

            def checked_bytes(path, *args, **kwargs):
                reads.append(path)
                self.assertNotEqual(path, self.test_path)
                return read_bytes(path, *args, **kwargs)

            def checked_text(path, *args, **kwargs):
                reads.append(path)
                self.assertNotEqual(path, self.test_path)
                return read_text(path, *args, **kwargs)

            with patch.object(Path, "read_bytes", checked_bytes):
                with patch.object(Path, "read_text", checked_text):
                    report, generate = self.run_comparison(root)
            self.assertNotIn(self.test_path, reads)
            self.assertEqual(generate.call_count, 6)
            for call in generate.call_args_list:
                self.assertIn(call.args[0], ["document one", "document two"])
                self.assertNotIn("labels", call.kwargs)
                self.assertNotIn("reference", call.kwargs)
            self.assertEqual(report["proposed_method"], "beam")
            self.assertFalse(report["generation_config_updated"])
            self.assertEqual(generation_path.read_bytes(), original)
            self.assertEqual(report["candidates"]["greedy"]["total_inference_seconds"], 2.0)
            self.assertEqual(report["candidates"]["beam"]["total_inference_seconds"], 8.0)
            output = Path(report["output_dir"])
            for method, beams in [("greedy", 1), ("beam", 4)]:
                candidate = report["candidates"][method]
                self.assertEqual(candidate["generation_config"], {"num_beams": beams, "max_new_tokens": 128, "do_sample": False})
                rows = []
                for line in (output / candidate["predictions_file"]).read_text(encoding="utf-8").splitlines():
                    rows.append(json.loads(line))
                self.assertEqual([row["id"] for row in rows], ["v1", "v2"])
            self.assertEqual(json.loads((output / "decoding_comparison.json").read_text()), report)
            self.assertEqual(len(json.loads((output / "decoding_examples.json").read_text())), 2)

    def test_changed_source_ids_between_methods_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            self.change_source = True
            with self.assertRaisesRegex(ValueError, "ID nguồn"):
                self.run_comparison(root)

    def test_validation_ids_must_match_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            self.selection["validation_sample_ids"] = ["different"]
            with self.assertRaisesRegex(ValueError, "chọn best"):
                self.run_comparison(root)

    def test_scoring_must_cover_every_validation_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            self.wrong_score_count = True
            with self.assertRaisesRegex(ValueError, "đủ validation"):
                self.run_comparison(root)


class FixedTestEvaluationTests(unittest.TestCase):
    def prepare(self, directory):
        self.repo = Path(directory) / "repo"
        self.drive = Path(directory) / "drive"
        config = load_train_config()
        generation = {"num_beams": 4, "max_new_tokens": 128, "do_sample": False}
        self.rows = [
            {"id": "x1", "group_id": "gx1", "document": "test document one", "summary": "reference one"},
            {"id": "x2", "group_id": "gx2", "document": "test document two", "summary": "reference two"},
        ]
        self.test_path = self.drive / config["paths"]["test_file"]
        self.test_path.parent.mkdir(parents=True)
        content = self.encode_rows(self.rows)
        self.test_path.write_bytes(content)
        subset = {
            "dataset": config["dataset"],
            "seed": config["seed"],
            "splits": {
                "train": {"sample_ids": ["t1"], "group_ids": ["gt"]},
                "validation": {"sample_ids": ["v1"], "group_ids": ["gv"]},
                "test": {"sample_ids": ["x1", "x2"], "group_ids": ["gx1", "gx2"], "pair_count": 2},
            },
        }
        metadata = {
            "path": config["paths"]["test_file"],
            "pair_count": 2,
            "byte_count": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        processed = {
            "dataset": config["dataset"],
            "seed": config["seed"],
            "subset_manifest_sha256": _split_manifest_fingerprint(subset),
            "files": {"test": metadata},
            "test_lock": {"locked": True, "sha256": metadata["sha256"]},
        }
        decision = {
            "decision": "approved_by_user",
            "generation_config_updated": True,
            "selected_generation_config": generation,
            "checkpoint_run_id": "saved_run",
            "comparison_id": "approved_comparison",
        }
        comparison = {
            "comparison_id": "approved_comparison",
            "best_epoch": 2,
            "best_global_step": 500,
            "best_validation_loss": 2.1,
        }
        self.approved = {
            "train_config": config,
            "generation_config": generation,
            "subset": subset,
            "processed": processed,
            "decision": decision,
            "comparison": comparison,
        }
        paths = {
            "train_config": "configs/train.json",
            "generation_config": "configs/generation.json",
            "subset": config["paths"]["subset_manifest"],
            "processed": config["paths"]["processed_manifest"],
            "decision": DECODING_REPORT_DIR + "/decoding_selection.json",
            "comparison": DECODING_REPORT_DIR + "/decoding_comparison.json",
        }
        self.release_files = {}
        for key, relative_path in paths.items():
            text = json.dumps(self.approved[key])
            path = self.repo / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            self.release_files[relative_path] = text
        self.blocked_target = ""
        self.changed_source = False
        self.changed_generation = False
        self.wrong_count = False

    def encode_rows(self, rows):
        lines = []
        for row in rows:
            lines.append(json.dumps(row))
        return ("\n".join(lines) + "\n").encode("utf-8")

    def git(self, args, **options):
        stdout = ""
        code = 0
        if args[1] == "remote":
            stdout = "https://github.com/MD1809/vietnamese-summarization.git\n"
        elif args[1] == "fetch":
            pass
        elif args[1] == "merge-base":
            if args[-1] == self.blocked_target:
                code = 1
        elif args[1] == "show":
            self.assertTrue(args[2].startswith(DECODING_RELEASE_COMMIT + ":"))
            relative_path = args[2].split(":", 1)[1]
            stdout = self.release_files[relative_path]
        else:
            self.fail("Unexpected git command: " + str(args))
        return types.SimpleNamespace(stdout=stdout, returncode=code)

    def source_info(self, text):
        if text == "test document two":
            return {"used_input_ids": [8] * 1023 + [1], "used_input_token_count": 1024, "input_token_count": 1500, "was_truncated": True}
        return {"used_input_ids": [7, 1], "used_input_token_count": 2, "input_token_count": 2, "was_truncated": False}

    def baseline(self, text, tokenizer, config_path):
        result = self.source_info(text)
        result.update({
            "summary": "baseline " + text,
            "summary_token_count": 5,
            "summary_was_truncated": False,
            "selected_sentence_count": 1,
            "elapsed_seconds": 0.01,
        })
        return result

    def inference(self, text, model, tokenizer, config_path, generation_config_path):
        result = self.source_info(text)
        generation = json.loads(Path(generation_config_path).read_text())
        if self.changed_generation:
            generation["num_beams"] = 1
        if self.changed_source:
            result["used_input_ids"][0] = 99
        result.update({
            "summary": "vit5 " + text,
            "generated_ids": [0, 5, 1],
            "generation_config": generation,
            "elapsed_seconds": 1.0,
        })
        return result

    def scores(self, references, predictions):
        self.assertEqual(references, ["reference one", "reference two"])
        self.assertEqual(len(predictions), 2)
        f1 = 0.3
        if predictions[0].startswith("vit5"):
            f1 = 0.4
        mean = {"rouge1": f1, "rouge2": f1, "rougeL": f1}
        scores = {}
        percent = {}
        for name in mean:
            scores[name] = {"precision": f1, "recall": f1, "f1": f1}
            percent[name] = f1 * 100
        return {"sample_count": 1 if self.wrong_count else 2, "per_sample": [scores, scores], "mean_f1": mean, "mean_f1_percent": percent}

    def run_evaluation(self):
        selection = {"epoch": 2, "global_step": 500, "validation_loss": 2.1}
        with patch("src.evaluate.subprocess.run", side_effect=self.git):
            with patch("src.evaluate.load_summarizer", return_value=(object(), object(), selection)):
                with patch("src.evaluate.set_seed"), patch("src.evaluate.version", return_value="test-version"):
                    with patch("src.evaluate.record_evaluation_source", return_value={"git_commit": DECODING_RELEASE_COMMIT}):
                        with patch("src.evaluate.summarize_lead_three", side_effect=self.baseline) as baseline:
                            with patch("src.evaluate.summarize_text", side_effect=self.inference) as generate:
                                with patch("src.evaluate.score_summaries", side_effect=self.scores):
                                    with patch("src.evaluate.choose_decoding", side_effect=AssertionError("Do not tune on test")):
                                        with patch("src.evaluate.print"):
                                            report = run_test_evaluation(self.repo, self.drive, "saved_run")
        return report, baseline, generate

    def test_unpublished_or_old_checkout_stops_before_test_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            for target in ["HEAD", "origin/main"]:
                self.blocked_target = target
                with patch("src.evaluate.subprocess.run", side_effect=self.git):
                    with patch("src.evaluate.load_fixed_test_source") as loader:
                        with self.assertRaisesRegex(RuntimeError, "push G6.4"):
                            run_test_evaluation(self.repo, self.drive, "saved_run")
                loader.assert_not_called()

    def test_changed_release_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            path = self.repo / "configs/generation.json"
            path.write_text(json.dumps({"num_beams": 1, "max_new_tokens": 128, "do_sample": False}))
            with patch("src.evaluate.subprocess.run", side_effect=self.git):
                with self.assertRaisesRegex(ValueError, "thay đổi sau khi chốt"):
                    verify_test_release(self.repo, "saved_run")

    def test_approved_release_and_fixed_test_are_loaded_without_manifest_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            before = copy.deepcopy(self.approved)
            with patch("src.evaluate.subprocess.run", side_effect=self.git):
                approved = verify_test_release(self.repo, "saved_run")
            rows, metadata = load_fixed_test_source(self.drive, approved)
            self.assertEqual(rows, self.rows)
            self.assertEqual(metadata["pair_count"], 2)
            self.assertEqual(approved, before)

    def test_changed_test_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            self.test_path.write_bytes(self.test_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Nội dung hoặc kích thước"):
                load_fixed_test_source(self.drive, self.approved)

    def test_wrong_test_ids_order_groups_count_and_empty_text_are_rejected(self):
        for case in ["id", "duplicate", "order", "group", "count", "empty"]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                self.prepare(directory)
                rows = copy.deepcopy(self.rows)
                if case == "id":
                    rows[0]["id"] = "different"
                elif case == "duplicate":
                    rows[1]["id"] = rows[0]["id"]
                elif case == "order":
                    rows.reverse()
                elif case == "group":
                    rows[0]["group_id"] = "different"
                elif case == "count":
                    rows.pop()
                else:
                    rows[0]["document"] = " "
                content = self.encode_rows(rows)
                self.test_path.write_bytes(content)
                metadata = self.approved["processed"]["files"]["test"]
                metadata["byte_count"] = len(content)
                metadata["sha256"] = hashlib.sha256(content).hexdigest()
                self.approved["processed"]["test_lock"]["sha256"] = metadata["sha256"]
                with self.assertRaises(ValueError):
                    load_fixed_test_source(self.drive, self.approved)

    def test_test_outputs_preserve_configs_and_do_not_select_or_train(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            config_bytes = (self.repo / "configs/generation.json").read_bytes()
            test_bytes = self.test_path.read_bytes()
            report, baseline, generate = self.run_evaluation()
            self.assertEqual(baseline.call_count, 3)
            self.assertEqual(generate.call_count, 3)
            for call in generate.call_args_list:
                self.assertIn(call.args[0], ["test document one", "test document two"])
                self.assertNotIn("labels", call.kwargs)
                self.assertNotIn("reference", call.kwargs)
            self.assertEqual(report["truncated_source_count"], 1)
            self.assertEqual(report["truncated_source_percent"], 50.0)
            self.assertEqual(report["results"]["vit5"]["total_inference_seconds"], 2.0)
            self.assertEqual(report["results"]["baseline"]["total_inference_seconds"], 0.02)
            self.assertFalse(report["test_used_for_tuning"])
            self.assertFalse(report["training_performed"])
            self.assertTrue(report["test_content_read"])
            self.assertEqual((self.repo / "configs/generation.json").read_bytes(), config_bytes)
            self.assertEqual(self.test_path.read_bytes(), test_bytes)
            output = Path(report["output_dir"])
            rows = []
            for line in (output / "predictions.jsonl").read_text().splitlines():
                rows.append(json.loads(line))
            self.assertEqual([row["id"] for row in rows], ["x1", "x2"])
            self.assertEqual(json.loads((output / "metrics.json").read_text()), report)
            with (output / "comparison.csv").open(newline="") as file:
                csv_rows = list(csv.DictReader(file))
            self.assertEqual([row["method"] for row in csv_rows], ["baseline", "vit5"])
            for row in csv_rows:
                self.assertEqual(int(row["sample_count"]), 2)
                self.assertEqual(float(row["rougeL_f1_percent"]), report["results"][row["method"]]["mean_f1_percent"]["rougeL"])

    def test_test_source_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            self.changed_source = True
            with self.assertRaisesRegex(ValueError, "cùng ID nguồn test"):
                self.run_evaluation()

    def test_generation_change_during_test_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            self.changed_generation = True
            with self.assertRaisesRegex(ValueError, "Tham số sinh đã đổi"):
                self.run_evaluation()

    def test_every_test_prediction_must_be_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            self.wrong_count = True
            with self.assertRaisesRegex(ValueError, "Chưa chấm đủ"):
                self.run_evaluation()


if __name__ == "__main__":
    unittest.main()
