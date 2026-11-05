import hashlib
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.data import _split_manifest_fingerprint
from src.tokenization import (
    count_truncated_samples,
    measure_truncation,
    save_truncation_report,
    tokenize_batch,
    tokenize_train_validation,
    validate_tokenized_split,
    create_seq2seq_collator,
    collate_tokenized_samples,
    validate_padded_batch,
    decode_padded_labels,
)


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
                pair["group_id"] = "url-" + pair["id"]
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


class MappingDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])
        self.map_options = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def map(self, function, **options):
        self.map_options = options
        batch = {}
        for column in self.column_names:
            batch[column] = []
            for row in self.rows:
                batch[column].append(row[column])
        encoded = function(batch, **options["fn_kwargs"])
        rows = []
        for index in range(len(self.rows)):
            row = {}
            for column in self.column_names:
                if column not in options["remove_columns"]:
                    row[column] = self.rows[index][column]
            for column in encoded:
                row[column] = encoded[column][index]
            rows.append(row)
        return MappingDataset(rows)


class TokenizePairsTests(unittest.TestCase):
    def test_labels_come_from_targets_and_options_are_explicit(self):
        sources = {"input_ids": [[20, 21, 1], [30, 1]], "attention_mask": [[1, 1, 1], [1, 1]]}
        targets = {"input_ids": [[80, 1], [90, 91, 1]]}
        tokenizer = Mock(side_effect=[sources, targets])
        batch = {"document": ["source one", "source two"], "summary": ["summary one", "summary two"]}
        output = tokenize_batch(batch, tokenizer, 1024, 128, "")
        self.assertEqual(output["input_ids"], sources["input_ids"])
        self.assertEqual(output["labels"], targets["input_ids"])
        source_options = tokenizer.call_args_list[0].kwargs
        target_options = tokenizer.call_args_list[1].kwargs
        self.assertEqual(source_options["text"], batch["document"])
        self.assertEqual(target_options["text_target"], batch["summary"])
        self.assertEqual(source_options["max_length"], 1024)
        self.assertEqual(target_options["max_length"], 128)
        for options in [source_options, target_options]:
            self.assertIs(options["truncation"], True)
            self.assertIs(options["padding"], False)
            self.assertIs(options["add_special_tokens"], True)
            self.assertNotIn("return_tensors", options)

    def test_prefix_is_applied_only_to_documents(self):
        tokenizer = Mock(side_effect=[
            {"input_ids": [[10, 1]], "attention_mask": [[1, 1]]},
            {"input_ids": [[80, 1]]},
        ])
        tokenize_batch({"document": ["source"], "summary": ["target"]}, tokenizer, 7, 4, "prefix ")
        self.assertEqual(tokenizer.call_args_list[0].kwargs["text"], ["prefix source"])
        self.assertEqual(tokenizer.call_args_list[1].kwargs["text_target"], ["target"])
        self.assertEqual(tokenizer.call_args_list[0].kwargs["max_length"], 7)

    def test_rejects_misaligned_batch_and_invalid_limits(self):
        tokenizer = Mock()
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            tokenize_batch({"document": ["a"], "summary": []}, tokenizer, 1024, 128, "")
        for limit in [0, True, -1]:
            with self.assertRaises(ValueError):
                tokenize_batch({"document": ["a"], "summary": ["b"]}, tokenizer, limit, 128, "")
        tokenizer.assert_not_called()

    def test_unpadded_lengths_and_exact_limit_are_valid(self):
        original = [{"id": "a", "group_id": "A"}, {"id": "b", "group_id": "B"}]
        tokenized = [
            {"id": "a", "group_id": "A", "input_ids": [2] * 1024, "attention_mask": [1] * 1024, "labels": [3] * 128},
            {"id": "b", "group_id": "B", "input_ids": [4, 1], "attention_mask": [1, 1], "labels": [5, 1]},
        ]
        validate_tokenized_split(tokenized, original, 1024, 128)

    def test_validation_detects_wrong_identity_lengths_masks_and_labels(self):
        original = [{"id": "a", "group_id": "A"}]
        valid = {"id": "a", "group_id": "A", "input_ids": [2, 1], "attention_mask": [1, 1], "labels": [3, 1]}
        changes = [
            ("id", "wrong"),
            ("input_ids", [2] * 1025),
            ("labels", [3] * 129),
            ("labels", [-100, 1]),
            ("attention_mask", [1]),
            ("attention_mask", [1, 0]),
            ("labels", []),
        ]
        for key, value in changes:
            with self.subTest(field=key, value=value):
                bad = valid.copy()
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_tokenized_split([bad], original, 1024, 128)
        with self.assertRaisesRegex(ValueError, "number of samples"):
            validate_tokenized_split([], original, 1024, 128)

    def tokenizer_module(self):
        tokenizer = Mock(side_effect=[
            {"input_ids": [[10, 11, 1], [10, 1], [20, 1]], "attention_mask": [[1, 1, 1], [1, 1], [1, 1]]},
            {"input_ids": [[70, 1], [80, 1], [90, 1]]},
            {"input_ids": [[40, 1]], "attention_mask": [[1, 1]]},
            {"input_ids": [[50, 1]]},
        ])
        auto = Mock()
        auto.from_pretrained.return_value = tokenizer
        return types.SimpleNamespace(AutoTokenizer=auto), tokenizer, auto

    def test_map_pipeline_preserves_ids_and_uses_only_train_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = TruncationTests().prepare_files(directory)
            original = copy.deepcopy(rows)
            raw = {}
            for split_name in rows:
                raw[split_name] = MappingDataset(rows[split_name])
            load = Mock(return_value=raw)
            module, tokenizer, auto = self.tokenizer_module()
            modules = {"datasets": types.SimpleNamespace(load_dataset=load), "transformers": module}
            with patch.dict(sys.modules, modules):
                tokenized, returned_tokenizer = tokenize_train_validation(
                    root, config_path, load_from_cache_file=False,
                )
            self.assertIs(returned_tokenizer, tokenizer)
            self.assertEqual(set(tokenized), {"train", "validation"})
            self.assertEqual(rows, original)
            self.assertEqual(set(load.call_args.kwargs["data_files"]), {"train", "validation"})
            for split_name in rows:
                self.assertTrue(raw[split_name].map_options["batched"])
                self.assertIs(raw[split_name].map_options["load_from_cache_file"], False)
                self.assertEqual(set(tokenized[split_name].column_names), {"id", "group_id", "input_ids", "attention_mask", "labels"})
                for index in range(len(rows[split_name])):
                    self.assertEqual(tokenized[split_name][index]["id"], rows[split_name][index]["id"])
            self.assertEqual(tokenized["train"][0]["labels"], [70, 1])
            self.assertEqual(tokenized["train"][0]["input_ids"], [10, 11, 1])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            auto.from_pretrained.assert_called_once_with(
                config["model"]["id"], revision=config["model"]["revision"], use_fast=False,
            )

    def test_cached_raw_data_must_match_verified_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = TruncationTests().prepare_files(directory)
            stale = copy.deepcopy(rows)
            stale["train"][0]["document"] = "old cached source"
            raw = {}
            for split_name in stale:
                raw[split_name] = MappingDataset(stale[split_name])
            module, tokenizer, auto = self.tokenizer_module()
            modules = {"datasets": types.SimpleNamespace(load_dataset=Mock(return_value=raw)), "transformers": module}
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, "verified JSONL"):
                    tokenize_train_validation(root, config_path)
            tokenizer.assert_not_called()


class ListTensor:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.shape = (len(rows), len(rows[0]))

    def tolist(self):
        return copy.deepcopy(self.rows)


class PaddingTests(unittest.TestCase):
    def examples(self):
        samples = [
            {"id": "a", "group_id": "A", "input_ids": [10, 11, 1], "attention_mask": [1, 1, 1], "labels": [80, 1]},
            {"id": "b", "group_id": "B", "input_ids": [20, 1], "attention_mask": [1, 1], "labels": [90, 91, 1]},
        ]
        batch = {
            "input_ids": ListTensor([[10, 11, 1], [20, 1, 0]]),
            "attention_mask": ListTensor([[1, 1, 1], [1, 1, 0]]),
            "labels": ListTensor([[80, 1, -100], [90, 91, 1]]),
        }
        tokenizer = types.SimpleNamespace(pad_token_id=0, padding_side="right")
        return samples, batch, tokenizer

    def test_collator_uses_dynamic_padding_and_no_model(self):
        samples, batch, tokenizer = self.examples()
        constructor = Mock(return_value="collator")
        module = types.SimpleNamespace(DataCollatorForSeq2Seq=constructor)
        with patch.dict(sys.modules, {"transformers": module}):
            self.assertEqual(create_seq2seq_collator(tokenizer), "collator")
        constructor.assert_called_once_with(
            tokenizer=tokenizer, model=None, padding=True,
            label_pad_token_id=-100, return_tensors="pt",
        )

    def test_collator_strips_metadata_and_copies_lists(self):
        samples, batch, tokenizer = self.examples()
        original = copy.deepcopy(samples)

        def collator(features):
            self.assertEqual(set(features[0]), {"input_ids", "attention_mask", "labels"})
            features[0]["labels"].append(-100)
            features[0]["input_ids"].append(0)
            return batch

        self.assertIs(collate_tokenized_samples(samples, collator), batch)
        self.assertEqual(samples, original)

    def test_dynamic_batch_shapes_and_padding_are_valid(self):
        samples, batch, tokenizer = self.examples()
        checks = validate_padded_batch(batch, samples, tokenizer)
        self.assertTrue(all(checks.values()))

    def test_rejects_bad_padding_masks_and_changed_tokens(self):
        changes = [
            ("input_ids", 1, 2, 9),
            ("input_ids", 0, 0, 9),
            ("attention_mask", 1, 2, 1),
            ("attention_mask", 0, 0, 0),
            ("labels", 0, 2, 0),
            ("labels", 1, 0, -100),
        ]
        for field, row, column, value in changes:
            with self.subTest(field=field, row=row, column=column):
                samples, batch, tokenizer = self.examples()
                batch[field].rows[row][column] = value
                with self.assertRaises(ValueError):
                    validate_padded_batch(batch, samples, tokenizer)

    def test_rejects_static_max_padding_shape(self):
        samples, batch, tokenizer = self.examples()
        batch["input_ids"].shape = (2, 1024)
        with self.assertRaisesRegex(ValueError, "dynamic batch length"):
            validate_padded_batch(batch, samples, tokenizer)

    def test_single_sample_does_not_need_padding(self):
        samples, batch, tokenizer = self.examples()
        single_batch = {}
        for field in ["input_ids", "attention_mask", "labels"]:
            single_batch[field] = ListTensor([samples[0][field]])
        self.assertTrue(all(validate_padded_batch(single_batch, samples[:1], tokenizer).values()))

    def test_decode_excludes_minus_100_without_changing_labels(self):
        samples, batch, tokenizer = self.examples()
        original = batch["labels"].tolist()
        decoder = Mock()
        decoder.decode.side_effect = ["first summary", "second summary"]
        decoded = decode_padded_labels(batch["labels"], decoder)
        self.assertEqual(decoded, ["first summary", "second summary"])
        self.assertEqual(decoder.decode.call_args_list[0].args[0], [80, 1])
        self.assertEqual(decoder.decode.call_args_list[1].args[0], [90, 91, 1])
        self.assertIs(decoder.decode.call_args_list[0].kwargs["skip_special_tokens"], True)
        self.assertEqual(batch["labels"].tolist(), original)

    def test_empty_batch_and_unsupported_tokenizer_are_rejected(self):
        samples, batch, tokenizer = self.examples()
        with self.assertRaises(ValueError):
            collate_tokenized_samples([], Mock())
        with self.assertRaises(ValueError):
            validate_padded_batch(batch, [], tokenizer)
        constructor = Mock()
        module = types.SimpleNamespace(DataCollatorForSeq2Seq=constructor)
        with patch.dict(sys.modules, {"transformers": module}):
            with self.assertRaises(ValueError):
                create_seq2seq_collator(types.SimpleNamespace(pad_token_id=None, padding_side="right"))
            with self.assertRaises(ValueError):
                create_seq2seq_collator(types.SimpleNamespace(pad_token_id=0, padding_side="left"))
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
