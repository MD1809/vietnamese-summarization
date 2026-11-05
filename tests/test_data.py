import json
import tempfile
import unittest
from pathlib import Path

from src.data import (
    clean_wikilingua_pairs,
    convert_wikilingua_sections,
    make_sample_id,
    save_cleaning_manifest,
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


if __name__ == "__main__":
    unittest.main()
