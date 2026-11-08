import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data import _split_manifest_fingerprint
from src.evaluate import choose_decoding, run_decoding_comparison
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


if __name__ == "__main__":
    unittest.main()
