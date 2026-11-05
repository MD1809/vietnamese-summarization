import json
import copy
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.data import (
    clean_wikilingua_pairs,
    convert_wikilingua_sections,
    make_sample_id,
    save_cleaning_manifest,
    save_split_manifest,
    save_subset_manifest,
    select_fixed_subsets,
    restore_fixed_subsets,
    build_dataset_statistics,
    export_prepared_datasets,
    summarize_token_lengths,
    verify_prepared_datasets,
    split_wikilingua_pairs,
    validate_dataset_splits,
)
from src.preprocessing import normalize_pairs, normalize_text


class ConvertSectionsTests(unittest.TestCase):
    def test_alignment_and_source_positions(self):
        rows = [
            {
                "url": "article-a",
                "article": {
                    "section_name": ["first", "second"],
                    "document": ["source-a0", "source-a1"],
                    "summary": ["summary-a0", "summary-a1"],
                },
            },
            {
                "url": "article-b",
                "article": {
                    "section_name": ["first"],
                    "document": ["source-b0"],
                    "summary": ["summary-b0"],
                },
            },
        ]
        pairs = convert_wikilingua_sections(rows)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[1], {
            "source_row_index": 0,
            "source_section_index": 1,
            "url": "article-a",
            "section_name": "second",
            "document": "source-a1",
            "summary": "summary-a1",
        })
        self.assertEqual(pairs[2]["source_row_index"], 1)
        self.assertEqual(pairs[2]["source_section_index"], 0)
        self.assertEqual(pairs[2]["summary"], "summary-b0")

    def test_mismatched_lengths_report_source(self):
        rows = [{
            "url": "article-a",
            "article": {
                "section_name": ["first", "second"],
                "document": ["source-a0", "source-a1"],
                "summary": ["summary-a0"],
            },
        }]
        with self.assertRaisesRegex(ValueError, "row=0, url=article-a"):
            convert_wikilingua_sections(rows)

    def test_missing_list_is_not_silently_skipped(self):
        rows = [{
            "url": "article-a",
            "article": {"section_name": [], "document": []},
        }]
        with self.assertRaisesRegex(ValueError, "article.summary"):
            convert_wikilingua_sections(rows)

    def test_preserves_content_and_empty_values(self):
        rows = [{
            "url": "article-a",
            "article": {
                "section_name": ["first", "second"],
                "document": ["  Keep  spacing.\n", ""],
                "summary": ["Keep CASE!", None],
            },
        }]
        pairs = convert_wikilingua_sections(rows)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["document"], "  Keep  spacing.\n")
        self.assertEqual(pairs[0]["summary"], "Keep CASE!")
        self.assertEqual(pairs[1]["document"], "")
        self.assertIsNone(pairs[1]["summary"])

    def test_empty_section_lists(self):
        rows = [{
            "url": "article-a",
            "article": {"section_name": [], "document": [], "summary": []},
        }]
        self.assertEqual(convert_wikilingua_sections(rows), [])


class NormalizeTextTests(unittest.TestCase):
    def test_combining_marks_become_nfc(self):
        self.assertEqual(normalize_text("Tie\u0302\u0301ng Vie\u0323\u0302t"), "Tiếng Việt")

    def test_whitespace_preserves_information(self):
        text = " \tKHÔNG  bỏ 2 bước!\r\nGiữ dấu: 3,5.\u00a0 "
        self.assertEqual(normalize_text(text), "KHÔNG bỏ 2 bước! Giữ dấu: 3,5.")

    def test_normalization_is_idempotent(self):
        text = "  Tie\u0302\u0301ng\nViệt!  "
        normalized = normalize_text(text)
        self.assertEqual(normalize_text(normalized), normalized)

    def test_empty_and_whitespace_only_strings(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(" \t\n"), "")

    def test_direct_text_function_rejects_non_string(self):
        with self.assertRaises(TypeError):
            normalize_text(None)

    def test_pairs_preserve_metadata_and_original(self):
        pair = {
            "url": "original-url",
            "source_row_index": 4,
            "source_section_index": 2,
            "section_name": "  Original section  ",
            "document": "  Giữ  dấu. ",
            "summary": " KHÔNG\nbỏ 2 bước! ",
        }
        result = normalize_pairs([pair])
        self.assertEqual(len(result), 1)
        self.assertIsNot(result[0], pair)
        self.assertEqual(pair["document"], "  Giữ  dấu. ")
        self.assertEqual(pair["summary"], " KHÔNG\nbỏ 2 bước! ")
        self.assertEqual(result[0]["document"], "Giữ dấu.")
        self.assertEqual(result[0]["summary"], "KHÔNG bỏ 2 bước!")
        for field in ["url", "source_row_index", "source_section_index", "section_name"]:
            self.assertEqual(result[0][field], pair[field])

    def test_pairs_keep_invalid_missing_and_empty_values(self):
        pairs = [
            {"document": None, "summary": 12},
            {"document": "", "summary": " \t"},
            {"url": "missing-fields"},
        ]
        result = normalize_pairs(pairs)
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0]["document"])
        self.assertEqual(result[0]["summary"], 12)
        self.assertEqual(result[1], {"document": "", "summary": ""})
        self.assertEqual(result[2], {"url": "missing-fields"})
        self.assertEqual(normalize_pairs(result), result)


class CleanPairsTests(unittest.TestCase):
    def make_pair(self, url="A", document="source", summary="summary", section=0):
        return {
            "url": url,
            "document": document,
            "summary": summary,
            "source_row_index": 0,
            "source_section_index": section,
            "section_name": "original section",
        }

    def test_id_does_not_depend_on_row_position(self):
        pair = self.make_pair()
        moved = pair.copy()
        moved["source_row_index"] = 99
        self.assertEqual(make_sample_id(pair), make_sample_id(moved))
        self.assertNotEqual(make_sample_id(pair), make_sample_id(self.make_pair(section=1)))
        self.assertEqual(len(make_sample_id(pair)), 67)

    def test_filters_first_reason_only_and_preserves_original(self):
        invalid = self.make_pair()
        invalid["document"] = None
        invalid["summary"] = None
        wrong_type = self.make_pair()
        wrong_type["summary"] = 12
        empty_url = self.make_pair(url=" ")
        missing = self.make_pair()
        del missing["summary"]
        pairs = [invalid, wrong_type, empty_url, missing, self.make_pair()]
        cleaned, report = clean_wikilingua_pairs(pairs)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report["removed_counts_by_reason"], {
            "missing_document": 1,
            "invalid_type_summary": 1,
            "empty_url": 1,
            "missing_summary": 1,
        })
        self.assertEqual(report["input_pair_count"], report["retained_pair_count"] + report["removed_pair_count"])
        self.assertNotIn("id", pairs[-1])

    def test_links_survive_duplicate_removal_and_overlap(self):
        pairs = [
            self.make_pair("A", "source-x", "same"),
            self.make_pair("B", "source-x", "same"),
            self.make_pair("B", "source-y", "other", 1),
            self.make_pair("C", "source-y", "different"),
        ]
        cleaned, report = clean_wikilingua_pairs(pairs)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned[0]["group_id"], "A")
        self.assertEqual(report["removed_counts_by_reason"], {"exact_duplicate_pair": 1})
        removed = report["removed_samples"][0]
        self.assertEqual(removed["representative_id"], cleaned[0]["id"])
        self.assertEqual(removed["id"], make_sample_id(pairs[1]))
        groups = []
        for group in report["shared_source_groups"]:
            groups.append(group["group_ids"])
        self.assertIn(["A", "B"], groups)
        self.assertIn(["B", "C"], groups)

    def test_same_source_different_summary_is_retained(self):
        pairs = [self.make_pair(summary="one"), self.make_pair(summary="two", section=1)]
        cleaned, report = clean_wikilingua_pairs(pairs)
        self.assertEqual(len(cleaned), 2)
        self.assertNotEqual(cleaned[0]["id"], cleaned[1]["id"])
        self.assertEqual(report["removed_pair_count"], 0)

    def test_requires_normalized_input(self):
        with self.assertRaisesRegex(ValueError, "Normalize document"):
            clean_wikilingua_pairs([self.make_pair(document="  source ")])

    def test_empty_content_is_removed_after_normalization(self):
        pairs = normalize_pairs([self.make_pair(document=" \t\n")])
        cleaned, report = clean_wikilingua_pairs(pairs)
        self.assertEqual(cleaned, [])
        self.assertEqual(report["removed_counts_by_reason"], {"empty_document": 1})

    def test_repeated_occurrence_has_one_retained_id(self):
        pair = self.make_pair()
        cleaned, report = clean_wikilingua_pairs([pair, pair.copy()])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report["removed_pair_count"], 1)

    def test_report_json_round_trip(self):
        cleaned, report = clean_wikilingua_pairs([self.make_pair()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifests" / "cleaning.json"
            manifest = save_cleaning_manifest(report, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest)
            self.assertEqual(manifest["dataset"]["id"], "esdurmus/wiki_lingua")
        self.assertEqual(clean_wikilingua_pairs([])[0], [])


class SplitPairsTests(unittest.TestCase):
    def make_pairs(self):
        pairs = []
        for index in range(20):
            pairs.append({
                "id": f"sample-{index:02d}",
                "url": f"url-{index:02d}",
                "group_id": f"url-{index:02d}",
                "document": f"document-{index}",
                "summary": f"summary-{index}",
            })
        return pairs

    def make_report(self, pairs):
        return {"retained_pair_count": len(pairs), "shared_source_groups": []}

    def manual_splits(self, pairs):
        return {"train": pairs[:18], "validation": pairs[18:19], "test": pairs[19:]}

    def test_repeatability_input_order_and_coverage(self):
        pairs = self.make_pairs()
        report = self.make_report(pairs)
        splits, manifest = split_wikilingua_pairs(pairs, report)
        reversed_splits, reversed_manifest = split_wikilingua_pairs(list(reversed(pairs)), report)
        self.assertEqual(splits, reversed_splits)
        self.assertEqual(manifest, reversed_manifest)
        self.assertEqual(len(splits["train"]), 16)
        self.assertEqual(len(splits["validation"]), 2)
        self.assertEqual(len(splits["test"]), 2)
        self.assertEqual(manifest["seed"], 42)
        for passed in manifest["checks"].values():
            self.assertTrue(passed)

    def test_sections_of_one_url_stay_together(self):
        pairs = self.make_pairs()
        extra = pairs[0].copy()
        extra["id"] = "extra-section"
        extra["document"] = "another section"
        pairs.append(extra)
        splits, manifest = split_wikilingua_pairs(pairs, self.make_report(pairs))
        owner = None
        for split_name in splits:
            for pair in splits[split_name]:
                if pair["group_id"] == "url-00":
                    if owner is not None:
                        self.assertEqual(split_name, owner)
                    owner = split_name
        self.assertEqual(manifest["input_pair_count"], 21)

    def test_transitive_links_through_removed_url(self):
        pairs = self.make_pairs()
        report = self.make_report(pairs)
        report["shared_source_groups"] = [
            {"group_ids": ["url-00", "removed-bridge"]},
            {"group_ids": ["removed-bridge", "url-01"]},
        ]
        splits, manifest = split_wikilingua_pairs(pairs, report)
        owners = {}
        for split_name in splits:
            for pair in splits[split_name]:
                owners[pair["group_id"]] = split_name
        self.assertEqual(owners["url-00"], owners["url-01"])
        self.assertEqual(manifest["split_groups"]["removed-bridge"], ["removed-bridge", "url-00", "url-01"])

    def test_validator_rejects_document_leak(self):
        pairs = self.make_pairs()
        pairs[18]["document"] = pairs[0]["document"]
        with self.assertRaisesRegex(ValueError, "document overlaps"):
            validate_dataset_splits(self.manual_splits(pairs), pairs, [])

    def test_validator_rejects_group_leak(self):
        pairs = self.make_pairs()
        pairs[18]["url"] = pairs[0]["url"]
        pairs[18]["group_id"] = pairs[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "group overlaps"):
            validate_dataset_splits(self.manual_splits(pairs), pairs, [])

    def test_validator_rejects_linked_groups_across_splits(self):
        pairs = self.make_pairs()
        links = [{"group_ids": ["url-00", "removed"]}, {"group_ids": ["removed", "url-18"]}]
        with self.assertRaisesRegex(ValueError, "Linked article groups"):
            validate_dataset_splits(self.manual_splits(pairs), pairs, links)

    def test_validator_rejects_missing_and_repeated_ids(self):
        pairs = self.make_pairs()
        splits = self.manual_splits(pairs)
        splits["train"] = pairs[1:18]
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_dataset_splits(splits, pairs, [])
        splits = self.manual_splits(pairs)
        splits["test"] = [pairs[0]]
        with self.assertRaisesRegex(ValueError, "more than once"):
            validate_dataset_splits(splits, pairs, [])

    def test_too_few_groups_and_report_mismatch(self):
        pairs = self.make_pairs()[:3]
        with self.assertRaisesRegex(ValueError, "Not enough independent groups"):
            split_wikilingua_pairs(pairs, self.make_report(pairs))
        with self.assertRaisesRegex(ValueError, "report count"):
            split_wikilingua_pairs(pairs, self.make_report([]))

    def test_split_manifest_round_trip(self):
        pairs = self.make_pairs()
        splits, manifest = split_wikilingua_pairs(pairs, self.make_report(pairs))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifests" / "splits.json"
            saved = save_split_manifest(manifest, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), saved)
            self.assertIn("created_at_utc", saved)
            self.assertNotIn("created_at_utc", manifest)
        total = 0
        for split_name in splits:
            total = total + manifest["splits"][split_name]["pair_count"]
        self.assertEqual(total, 20)


class FixedSubsetTests(unittest.TestCase):
    def make_parent(self, linked=False):
        pairs = []
        for group in range(30):
            for section in range(3):
                url = f"url-{group:02d}"
                sample_id = f"sample-{group:02d}-{section}"
                pairs.append({
                    "id": sample_id,
                    "url": url,
                    "group_id": url,
                    "document": "document-" + sample_id,
                    "summary": "summary-" + sample_id,
                })
        report = {"retained_pair_count": len(pairs), "shared_source_groups": []}
        if linked:
            report["shared_source_groups"] = [
                {"group_ids": ["url-00", "removed-bridge"]},
                {"group_ids": ["removed-bridge", "url-01"]},
            ]
        return split_wikilingua_pairs(pairs, report)

    def targets(self):
        return {"train": 7, "validation": 4, "test": 4}

    def test_whole_groups_overshoot_and_parent_membership(self):
        splits, parent = self.make_parent()
        subsets, manifest = select_fixed_subsets(splits, parent, self.targets())
        for name in subsets:
            expected = 6
            if name == "train":
                expected = 9
            self.assertEqual(len(subsets[name]), expected)
            parent_ids = set(parent["splits"][name]["sample_ids"])
            for pair in subsets[name]:
                self.assertIn(pair["id"], parent_ids)
        self.assertTrue(all(manifest["checks"].values()))

    def test_repeatable_when_parent_input_order_changes(self):
        splits, parent = self.make_parent()
        subsets, manifest = select_fixed_subsets(splits, parent, self.targets())
        reversed_splits = {}
        for name in splits:
            reversed_splits[name] = list(reversed(splits[name]))
        repeated, repeated_manifest = select_fixed_subsets(reversed_splits, parent, self.targets())
        self.assertEqual(repeated, subsets)
        self.assertEqual(repeated_manifest, manifest)

    def test_restore_uses_saved_ids_without_random_selection(self):
        splits, parent = self.make_parent()
        subsets, manifest = select_fixed_subsets(splits, parent, self.targets())
        with patch("src.data.random.Random", side_effect=RuntimeError("Unexpected resampling")):
            restored = restore_fixed_subsets(splits, parent, manifest)
        self.assertEqual(restored, subsets)

    def test_linked_groups_remain_whole_including_removed_bridge(self):
        splits, parent = self.make_parent(linked=True)
        subsets, manifest = select_fixed_subsets(splits, parent, self.targets())
        for name in manifest["splits"]:
            urls = manifest["splits"][name]["group_ids"]
            self.assertEqual("url-00" in urls, "url-01" in urls)

    def test_rejects_partial_group_and_foreign_sample(self):
        splits, parent = self.make_parent()
        subsets, manifest = select_fixed_subsets(splits, parent, self.targets())
        partial = copy.deepcopy(manifest)
        partial["splits"]["train"]["sample_ids"].pop()
        with self.assertRaisesRegex(ValueError, "whole parent groups"):
            restore_fixed_subsets(splits, parent, partial)
        foreign = copy.deepcopy(manifest)
        foreign["splits"]["train"]["sample_ids"][0] = parent["splits"]["test"]["sample_ids"][0]
        with self.assertRaisesRegex(ValueError, "whole parent groups"):
            restore_fixed_subsets(splits, parent, foreign)

    def test_rejects_wrong_parent_and_ignores_save_timestamp(self):
        splits, parent = self.make_parent()
        subsets, manifest = select_fixed_subsets(splits, parent, self.targets())
        saved_parent = parent.copy()
        saved_parent["created_at_utc"] = "another save time"
        self.assertEqual(restore_fixed_subsets(splits, saved_parent, manifest), subsets)
        wrong_parent = copy.deepcopy(parent)
        wrong_parent["seed"] = 99
        with self.assertRaisesRegex(ValueError, "different parent"):
            restore_fixed_subsets(splits, wrong_parent, manifest)

    def test_invalid_targets_and_parent_input_are_rejected(self):
        splits, parent = self.make_parent()
        for value in [0, -1, True, 1000]:
            targets = self.targets()
            targets["train"] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    select_fixed_subsets(splits, parent, targets)
        broken = copy.deepcopy(splits)
        broken["train"][0]["id"] = "unknown-id"
        with self.assertRaisesRegex(ValueError, "parent split manifest"):
            select_fixed_subsets(broken, parent, self.targets())

    def test_does_not_modify_parent_or_share_pair_dictionaries(self):
        splits, parent = self.make_parent()
        original = copy.deepcopy(splits)
        subsets, manifest = select_fixed_subsets(splits, parent, self.targets())
        subsets["train"][0]["summary"] = "changed output"
        self.assertEqual(splits, original)

    def test_manifest_round_trip_restores_identical_subset(self):
        splits, parent = self.make_parent()
        subsets, manifest = select_fixed_subsets(splits, parent, self.targets())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifests" / "subsets.json"
            saved = save_subset_manifest(manifest, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved, loaded)
            self.assertEqual(restore_fixed_subsets(splits, parent, loaded), subsets)
            self.assertNotIn("created_at_utc", manifest)


class PreparedDatasetTests(unittest.TestCase):
    def make_data(self):
        splits, parent = FixedSubsetTests().make_parent()
        return select_fixed_subsets(
            splits,
            parent,
            {"train": 7, "validation": 4, "test": 4},
        )

    def make_statistics(self, subsets, manifest):
        def tokenize(**kwargs):
            if "text" in kwargs:
                text = kwargs["text"]
            else:
                text = kwargs["text_target"]
            return {"input_ids": [1] * (len(text.split()) + 1)}

        tokenizer = Mock(side_effect=tokenize)
        auto = Mock()
        auto.from_pretrained.return_value = tokenizer
        module = types.SimpleNamespace(AutoTokenizer=auto)
        cleaning = {"removed_counts_by_reason": {"empty_document": 1}}
        with patch.dict(sys.modules, {"transformers": module}):
            statistics = build_dataset_statistics(subsets, manifest, cleaning)
        return statistics, tokenizer, auto

    def test_nearest_rank_p95_and_mean(self):
        self.assertEqual(summarize_token_lengths(list(range(1, 21))), {
            "mean": 10.5, "p95": 19, "max": 20,
        })
        self.assertEqual(summarize_token_lengths(list(range(1, 2001)))["p95"], 1900)
        self.assertEqual(summarize_token_lengths([5]), {"mean": 5.0, "p95": 5, "max": 5})
        with self.assertRaises(ValueError):
            summarize_token_lengths([])

    def test_statistics_tokenizes_only_train_without_truncation(self):
        subsets, manifest = self.make_data()
        stats, tokenizer, auto = self.make_statistics(subsets, manifest)
        config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
        auto.from_pretrained.assert_called_once_with(
            config["model"]["id"], revision=config["model"]["revision"], use_fast=False,
        )
        self.assertEqual(tokenizer.call_count, len(subsets["train"]) * 2)
        call_index = 0
        for pair in subsets["train"]:
            source = tokenizer.call_args_list[call_index].kwargs
            target = tokenizer.call_args_list[call_index + 1].kwargs
            self.assertEqual(source["text"], pair["document"])
            self.assertEqual(target["text_target"], pair["summary"])
            for kwargs in [source, target]:
                self.assertIs(kwargs["padding"], False)
                self.assertIs(kwargs["truncation"], False)
                self.assertIs(kwargs["add_special_tokens"], True)
                self.assertNotIn("max_length", kwargs)
                self.assertNotIn("return_tensors", kwargs)
            call_index = call_index + 2
        self.assertEqual(stats["training_token_lengths"]["sample_count"], len(subsets["train"]))
        self.assertEqual(stats["removed_counts_by_reason"], {"empty_document": 1})

    def test_unicode_jsonl_roundtrip_and_idempotent_export(self):
        subsets, manifest = self.make_data()
        subsets["train"][0]["section_name"] = "Tiếng Việt\u2028giữ metadata"
        stats, tokenizer, auto = self.make_statistics(subsets, manifest)
        with tempfile.TemporaryDirectory() as directory:
            processed = export_prepared_datasets(subsets, manifest, stats, directory)
            repeated = export_prepared_datasets(subsets, manifest, stats, directory)
            self.assertEqual(processed, repeated)
            self.assertTrue(all(verify_prepared_datasets(directory, processed, manifest).values()))
            path = Path(directory) / "data/processed/train.jsonl"
            with path.open(encoding="utf-8") as stream:
                rows = []
                for line in stream:
                    rows.append(json.loads(line))
            self.assertEqual(rows, subsets["train"])
            self.assertTrue(processed["test_lock"]["locked"])

    def test_refuses_changed_test_and_preserves_other_files(self):
        subsets, manifest = self.make_data()
        stats, tokenizer, auto = self.make_statistics(subsets, manifest)
        with tempfile.TemporaryDirectory() as directory:
            processed = export_prepared_datasets(subsets, manifest, stats, directory)
            root = Path(directory)
            test_path = root / processed["files"]["test"]["path"]
            train_path = root / processed["files"]["train"]["path"]
            train_before = train_path.read_bytes()
            test_path.write_bytes(test_path.read_bytes() + b" ")
            damaged = test_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Prepared file changed: test"):
                export_prepared_datasets(subsets, manifest, stats, directory)
            self.assertEqual(test_path.read_bytes(), damaged)
            self.assertEqual(train_path.read_bytes(), train_before)

    def test_refuses_different_initial_file_before_any_write(self):
        subsets, manifest = self.make_data()
        stats, tokenizer, auto = self.make_statistics(subsets, manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data/processed/test.jsonl"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"existing content")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                export_prepared_datasets(subsets, manifest, stats, directory)
            self.assertEqual(path.read_bytes(), b"existing content")
            self.assertFalse((path.parent / "train.jsonl").exists())

    def test_locked_export_rejects_content_changes_with_same_ids(self):
        subsets, manifest = self.make_data()
        stats, tokenizer, auto = self.make_statistics(subsets, manifest)
        with tempfile.TemporaryDirectory() as directory:
            export_prepared_datasets(subsets, manifest, stats, directory)
            subsets["test"][0]["summary"] = "different content"
            with self.assertRaisesRegex(ValueError, "locked"):
                export_prepared_datasets(subsets, manifest, stats, directory)

    def test_rejects_wrong_ids_and_mismatched_statistics(self):
        subsets, manifest = self.make_data()
        stats, tokenizer, auto = self.make_statistics(subsets, manifest)
        with tempfile.TemporaryDirectory() as directory:
            wrong_stats = copy.deepcopy(stats)
            wrong_stats["splits"]["train"]["pair_count"] = 999
            with self.assertRaisesRegex(ValueError, "Statistics counts"):
                export_prepared_datasets(subsets, manifest, wrong_stats, directory)
            wrong_ids = copy.deepcopy(subsets)
            wrong_ids["train"][0]["id"] = "foreign-sample"
            with self.assertRaisesRegex(ValueError, "Export IDs"):
                export_prepared_datasets(wrong_ids, manifest, stats, directory)

    def test_verifier_detects_statistics_and_lock_changes(self):
        subsets, manifest = self.make_data()
        stats, tokenizer, auto = self.make_statistics(subsets, manifest)
        with tempfile.TemporaryDirectory() as directory:
            processed = export_prepared_datasets(subsets, manifest, stats, directory)
            wrong_lock = copy.deepcopy(processed)
            wrong_lock["test_lock"]["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "Test lock"):
                verify_prepared_datasets(directory, wrong_lock, manifest)
            stats_path = Path(directory) / processed["statistics"]["path"]
            stats_path.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "statistics file changed"):
                verify_prepared_datasets(directory, processed, manifest)


if __name__ == "__main__":
    unittest.main()
