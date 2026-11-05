import unittest

from src.data import convert_wikilingua_sections
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


if __name__ == "__main__":
    unittest.main()
