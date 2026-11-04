import unittest

from src.data import convert_wikilingua_sections


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


if __name__ == "__main__":
    unittest.main()
