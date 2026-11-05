import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.data import _split_manifest_fingerprint
from src.tokenization import count_truncated_samples, measure_truncation, save_truncation_report


class TruncationTests(unittest.TestCase):
    def prepare_files(self, directory):
        root = Path(directory)
        config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
        config["max_source_length"] = 3
        config["max_target_length"] = 2
        config_path = root / "train.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        rows = {
            "train": [
                {"id": "a", "document": "a b", "summary": "x"},
                {"id": "b", "document": "a b c", "summary": "x y"},
                {"id": "c", "document": "a b c d", "summary": "x"},
            ],
            "validation": [{"id": "v", "document": "a", "summary": "x y z"}],
        }
        subset = {"dataset": config["dataset"], "seed": config["seed"], "splits": {}}
        files = {}
        for split_name in rows:
            ids = []
            lines = []
            for pair in rows[split_name]:
                ids.append(pair["id"])
                lines.append(json.dumps(pair))
            relative_path = config["paths"][split_name + "_file"]
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            content = ("\n".join(lines) + "\n").encode("utf-8")
            path.write_bytes(content)
            subset["splits"][split_name] = {"sample_ids": ids, "pair_count": len(ids)}
            files[split_name] = {
                "path": relative_path,
                "pair_count": len(ids),
                "byte_count": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        manifest_dir = root / "data/manifests"
        manifest_dir.mkdir(parents=True)
        processed = {
            "dataset": config["dataset"],
            "seed": config["seed"],
            "subset_manifest_sha256": _split_manifest_fingerprint(subset),
            "files": files,
            "test_lock": {"locked": True},
        }
        (manifest_dir / "wikilingua_subsets.json").write_text(json.dumps(subset), encoding="utf-8")
        (manifest_dir / "wikilingua_processed.json").write_text(json.dumps(processed), encoding="utf-8")
        return root, config_path, rows

    def tokenizer_module(self):
        def encode(**kwargs):
            if "text" in kwargs:
                text = kwargs["text"]
            else:
                text = kwargs["text_target"]
            return {"input_ids": [1] * (len(text.split()) + 1)}

        tokenizer = Mock(side_effect=encode)
        auto = Mock()
        auto.from_pretrained.return_value = tokenizer
        return types.SimpleNamespace(AutoTokenizer=auto), tokenizer, auto

    def test_strict_boundary_and_percentage(self):
        result = count_truncated_samples([511, 512, 513, 1024], 512)
        self.assertEqual(result["truncated_count"], 2)
        self.assertEqual(result["truncated_percentage"], 50.0)
        self.assertEqual(count_truncated_samples([127, 128, 129], 128)["truncated_count"], 1)

    def test_invalid_limit_and_empty_data(self):
        for limit in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                count_truncated_samples([5], limit)
        with self.assertRaises(ValueError):
            count_truncated_samples([], 512)

    def test_train_validation_only_and_original_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = self.prepare_files(directory)
            paths = list((root / "data/processed").glob("*.jsonl"))
            original_bytes = {}
            for path in paths:
                original_bytes[path] = path.read_bytes()
            module, tokenizer, auto = self.tokenizer_module()
            with patch.dict(sys.modules, {"transformers": module}):
                report = measure_truncation(root, config_path)
            self.assertEqual(report["splits"]["train"]["document"]["truncated_count"], 2)
            self.assertEqual(report["splits"]["train"]["summary"]["truncated_count"], 1)
            self.assertEqual(report["splits"]["validation"]["document"]["truncated_count"], 0)
            self.assertEqual(report["splits"]["validation"]["summary"]["truncated_count"], 1)
            self.assertEqual(tokenizer.call_count, 8)
            self.assertFalse((root / "data/processed/test.jsonl").exists())
            self.assertIs(report["checks"]["test_content_read"], False)
            self.assertEqual(report["decision_status"], "pending_review")
            for path in paths:
                self.assertEqual(path.read_bytes(), original_bytes[path])
            for call in tokenizer.call_args_list:
                self.assertIs(call.kwargs["add_special_tokens"], True)
                self.assertIs(call.kwargs["truncation"], False)
                self.assertIs(call.kwargs["padding"], False)
                self.assertNotIn("max_length", call.kwargs)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            auto.from_pretrained.assert_called_once_with(
                config["model"]["id"], revision=config["model"]["revision"], use_fast=False,
            )

    def test_task_prefix_affects_only_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = self.prepare_files(directory)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["task_prefix"] = "prefix "
            config_path.write_text(json.dumps(config), encoding="utf-8")
            module, tokenizer, auto = self.tokenizer_module()
            with patch.dict(sys.modules, {"transformers": module}):
                report = measure_truncation(root, config_path)
            self.assertEqual(tokenizer.call_args_list[0].kwargs["text"], "prefix a b")
            self.assertEqual(tokenizer.call_args_list[1].kwargs["text_target"], "x")
            self.assertEqual(report["splits"]["train"]["document"]["truncated_count"], 3)

    def test_changed_file_hash_is_rejected_before_tokenizer_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = self.prepare_files(directory)
            path = root / "data/processed/train.jsonl"
            path.write_bytes(path.read_bytes() + b" ")
            module, tokenizer, auto = self.tokenizer_module()
            with patch.dict(sys.modules, {"transformers": module}):
                with self.assertRaisesRegex(ValueError, "Prepared file changed"):
                    measure_truncation(root, config_path)
            auto.from_pretrained.assert_not_called()

    def test_wrong_ids_are_rejected_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = self.prepare_files(directory)
            path = root / "data/processed/train.jsonl"
            path.write_bytes(path.read_bytes().replace(b'"id": "a"', b'"id": "z"'))
            manifest_path = root / "data/manifests/wikilingua_processed.json"
            processed = json.loads(manifest_path.read_text(encoding="utf-8"))
            processed["files"]["train"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(processed), encoding="utf-8")
            module, tokenizer, auto = self.tokenizer_module()
            with patch.dict(sys.modules, {"transformers": module}):
                with self.assertRaisesRegex(ValueError, "sample IDs"):
                    measure_truncation(root, config_path)

    def test_prefix_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = self.prepare_files(directory)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            del config["task_prefix"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            module, tokenizer, auto = self.tokenizer_module()
            with patch.dict(sys.modules, {"transformers": module}):
                with self.assertRaisesRegex(ValueError, "task_prefix"):
                    measure_truncation(root, config_path)

    def test_report_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = self.prepare_files(directory)
            module, tokenizer, auto = self.tokenizer_module()
            with patch.dict(sys.modules, {"transformers": module}):
                report = measure_truncation(root, config_path)
            path = root / "data/manifests/truncation_test.json"
            saved = save_truncation_report(report, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), saved)
            self.assertNotIn("created_at_utc", report)


if __name__ == "__main__":
    unittest.main()
