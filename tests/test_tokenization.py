import hashlib
import copy
import json
import random
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
    create_data_loader,
    create_train_validation_loaders,
    validate_loader_epoch,
    load_tokenization_source,
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
        manifest_dir = root / "data/manifests/full"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        processed = {
            "dataset": config["dataset"],
            "seed": config["seed"],
            "subset_manifest_sha256": _split_manifest_fingerprint(subset),
            "files": files,
            "test_lock": {"locked": True},
        }
        (manifest_dir / "wikilingua_full.json").write_text(json.dumps(subset), encoding="utf-8")
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
            paths = list((root / "data/processed/full").glob("*.jsonl"))
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
            self.assertFalse((root / "data/processed/full/test.jsonl").exists())
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
            path = root / "data/processed/full/train.jsonl"
            path.write_bytes(path.read_bytes() + b" ")
            module, tokenizer, auto = self.tokenizer_module()
            with patch.dict(sys.modules, {"transformers": module}):
                with self.assertRaisesRegex(ValueError, "Prepared file changed"):
                    measure_truncation(root, config_path)
            auto.from_pretrained.assert_not_called()

    def test_wrong_ids_are_rejected_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = self.prepare_files(directory)
            path = root / "data/processed/full/train.jsonl"
            path.write_bytes(path.read_bytes().replace(b'"id": "a"', b'"id": "z"'))
            manifest_path = root / "data/manifests/full/wikilingua_processed.json"
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


class TestGenerator:
    def __init__(self):
        self.random = random.Random()
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        self.random.seed(seed)


class TestDataLoader:
    def __init__(self, dataset, **options):
        self.dataset = dataset
        self.options = options
        self.generator = options["generator"]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.options["shuffle"]:
            self.generator.random.shuffle(indices)
        size = self.options["batch_size"]
        for start in range(0, len(indices), size):
            samples = []
            for index in indices[start:start + size]:
                samples.append(self.dataset[index])
            if len(samples) < size and self.options["drop_last"]:
                continue
            yield self.options["collate_fn"](samples)


class DataLoaderTests(unittest.TestCase):
    def examples(self):
        samples = []
        for index in range(11):
            samples.append({
                "id": f"sample-{index}",
                "group_id": f"group-{index}",
                "input_ids": [index + 10, 1],
                "attention_mask": [1, 1],
                "labels": [index + 80, 1],
            })
        tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=1, padding_side="right")
        return samples, tokenizer

    def modules(self):
        def collate(features):
            output = {}
            for field in ["input_ids", "attention_mask", "labels"]:
                rows = []
                for feature in features:
                    rows.append(list(feature[field]))
                output[field] = ListTensor(rows)
            return output

        data_module = types.SimpleNamespace(DataLoader=TestDataLoader)
        utils_module = types.SimpleNamespace(data=data_module)
        torch_module = types.SimpleNamespace(Generator=TestGenerator, utils=utils_module)
        transformers_module = types.SimpleNamespace(DataCollatorForSeq2Seq=Mock(return_value=collate))
        return {
            "torch": torch_module,
            "torch.utils": utils_module,
            "torch.utils.data": data_module,
            "transformers": transformers_module,
        }

    def test_settings_use_config_and_independent_generators(self):
        samples, tokenizer = self.examples()
        with patch.dict(sys.modules, self.modules()):
            loaders = create_train_validation_loaders({"train": samples, "validation": samples[:4]}, tokenizer)
        self.assertIs(loaders["train"].options["shuffle"], True)
        self.assertIs(loaders["validation"].options["shuffle"], False)
        self.assertIsNot(loaders["train"].generator, loaders["validation"].generator)
        for loader in loaders.values():
            self.assertEqual(loader.options["batch_size"], 1)
            self.assertEqual(loader.generator.seed, 42)
            self.assertEqual(loader.options["num_workers"], 0)
            self.assertIs(loader.options["drop_last"], False)

    def test_two_fresh_train_loaders_reproduce_first_epoch(self):
        samples, tokenizer = self.examples()
        with patch.dict(sys.modules, self.modules()):
            first = create_data_loader(samples, tokenizer, 3, True, 42)
            second = create_data_loader(samples, tokenizer, 3, True, 42)
            first_report = validate_loader_epoch(first, samples, tokenizer)
            second_report = validate_loader_epoch(second, samples, tokenizer)
        self.assertEqual(first_report["sample_ids"], second_report["sample_ids"])
        self.assertEqual(first_report["sample_count"], 11)
        self.assertEqual(first_report["batch_count"], 4)

    def test_next_epoch_advances_shuffle_and_keeps_all_samples(self):
        samples, tokenizer = self.examples()
        with patch.dict(sys.modules, self.modules()):
            loader = create_data_loader(samples, tokenizer, 3, True, 42)
            first = validate_loader_epoch(loader, samples, tokenizer)
            second = validate_loader_epoch(loader, samples, tokenizer)
        self.assertNotEqual(first["sample_ids"], second["sample_ids"])
        self.assertEqual(set(first["sample_ids"]), set(second["sample_ids"]))

    def test_validation_order_and_partial_final_batch(self):
        samples, tokenizer = self.examples()
        with patch.dict(sys.modules, self.modules()):
            loader = create_data_loader(samples, tokenizer, 3, False, 42)
            report = validate_loader_epoch(loader, samples, tokenizer, keep_order=True)
            batches = list(loader)
        expected_ids = []
        for sample in samples:
            expected_ids.append(sample["id"])
        self.assertEqual(report["sample_ids"], expected_ids)
        self.assertEqual(len(batches[-1]["sample_ids"]), 2)
        self.assertEqual(set(batches[0]), {"sample_ids", "group_ids", "model_inputs"})
        self.assertEqual(set(batches[0]["model_inputs"]), {"input_ids", "attention_mask", "labels"})

    def test_validation_does_not_change_train_generator(self):
        samples, tokenizer = self.examples()
        splits = {"train": samples, "validation": samples[:4]}
        with patch.dict(sys.modules, self.modules()):
            loaders = create_train_validation_loaders(splits, tokenizer)
            repeated = create_train_validation_loaders(splits, tokenizer)
            validate_loader_epoch(loaders["train"], samples, tokenizer)
            validate_loader_epoch(repeated["train"], samples, tokenizer)
            validate_loader_epoch(loaders["validation"], samples[:4], tokenizer, keep_order=True)
            actual = validate_loader_epoch(loaders["train"], samples, tokenizer)
            expected = validate_loader_epoch(repeated["train"], samples, tokenizer)
        self.assertEqual(actual["sample_ids"], expected["sample_ids"])

    def test_detects_missing_repeated_samples_and_wrong_order(self):
        samples, tokenizer = self.examples()
        with patch.dict(sys.modules, self.modules()):
            loader = create_data_loader(samples, tokenizer, 3, False, 42)
            batches = list(loader)
        with self.assertRaisesRegex(ValueError, "omitted"):
            validate_loader_epoch(batches[:-1], samples, tokenizer)
        with self.assertRaisesRegex(ValueError, "repeated"):
            validate_loader_epoch(batches + [batches[0]], samples, tokenizer)
        with self.assertRaisesRegex(ValueError, "sample order"):
            validate_loader_epoch(list(reversed(batches)), samples, tokenizer, keep_order=True)

    def test_detects_missing_eos_and_wrong_group_identity(self):
        samples, tokenizer = self.examples()
        with patch.dict(sys.modules, self.modules()):
            loader = create_data_loader(samples, tokenizer, 3, False, 42)
            batches = list(loader)
        for field in ["input_ids", "labels"]:
            broken = copy.deepcopy(samples)
            broken[0][field][-1] = 999
            with self.assertRaisesRegex(ValueError, "EOS"):
                validate_loader_epoch(batches, broken, tokenizer)
        batches[0]["group_ids"][0] = "wrong-group"
        with self.assertRaisesRegex(ValueError, "group identity"):
            validate_loader_epoch(batches, samples, tokenizer)

    def test_rejects_invalid_parameters_and_test_split(self):
        samples, tokenizer = self.examples()
        with patch.dict(sys.modules, self.modules()):
            for batch_size in [0, -1, True]:
                with self.assertRaises(ValueError):
                    create_data_loader(samples, tokenizer, batch_size, True, 42)
            with self.assertRaises(ValueError):
                create_data_loader([], tokenizer, 1, False, 42)
            with self.assertRaises(ValueError):
                create_data_loader(samples, tokenizer, 1, "yes", 42)
            with self.assertRaises(ValueError):
                create_data_loader(samples, tokenizer, 1, False, -1)
        with self.assertRaisesRegex(ValueError, "test remains locked"):
            create_train_validation_loaders({"train": samples, "validation": samples, "test": samples}, tokenizer)


class CacheSettingsTests(unittest.TestCase):
    def test_manifest_paths_are_read_from_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = TruncationTests().prepare_files(directory)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ["subset_manifest", "processed_manifest"]:
                original_path = root / config["paths"][key]
                relative_path = "custom-manifests/" + key + ".json"
                changed_path = root / relative_path
                changed_path.parent.mkdir(parents=True, exist_ok=True)
                changed_path.write_bytes(original_path.read_bytes())
                original_path.write_text("{}", encoding="utf-8")
                config["paths"][key] = relative_path
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded_config, loaded_rows, source_files, fingerprint = load_tokenization_source(root, config_path)
            self.assertEqual(loaded_rows, rows)
            self.assertEqual(loaded_config["paths"], config["paths"])

    def test_missing_manifest_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = TruncationTests().prepare_files(directory)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            del config["paths"]["subset_manifest"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "paths.subset_manifest"):
                load_tokenization_source(root, config_path)

    def test_cache_bypass_passes_updated_settings_to_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, rows = TruncationTests().prepare_files(directory)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["max_source_length"] = 5
            config["max_target_length"] = 4
            config["task_prefix"] = "prefix "
            config_path.write_text(json.dumps(config), encoding="utf-8")
            raw = {}
            for split_name in rows:
                raw[split_name] = MappingDataset(rows[split_name])
            module, tokenizer, auto = TokenizePairsTests().tokenizer_module()
            modules = {"datasets": types.SimpleNamespace(load_dataset=Mock(return_value=raw)), "transformers": module}
            with patch.dict(sys.modules, modules):
                tokenize_train_validation(root, config_path, load_from_cache_file=False)
            for split_name in raw:
                options = raw[split_name].map_options
                self.assertIs(options["load_from_cache_file"], False)
                self.assertEqual(options["fn_kwargs"]["max_source_length"], 5)
                self.assertEqual(options["fn_kwargs"]["max_target_length"], 4)
                self.assertEqual(options["fn_kwargs"]["task_prefix"], "prefix ")
                self.assertNotIn("new_fingerprint", options)
                self.assertNotIn("cache_file_name", options)
            self.assertTrue(tokenizer.call_args_list[0].kwargs["text"][0].startswith("prefix "))


if __name__ == "__main__":
    unittest.main()
