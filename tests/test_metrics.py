import importlib.util
import unittest
from unittest.mock import patch

from src.metrics import (
    VietnameseRougeTokenizer,
    get_rouge_settings,
    score_summary,
    score_summaries,
)


class RougeTokenizerTests(unittest.TestCase):
    def test_vietnamese_accents_case_punctuation_and_numbers(self):
        tokens = VietnameseRougeTokenizer().tokenize("TIẾNG VIỆT: học_sâu, 2026! Đúng?")
        self.assertEqual(tokens, ["tiếng", "việt", "học", "sâu", "2026", "đúng"])

    def test_composed_and_decomposed_unicode_are_equivalent(self):
        tokenizer = VietnameseRougeTokenizer()
        self.assertEqual(tokenizer.tokenize("Tôi học"), ["tôi", "học"])
        self.assertEqual(tokenizer.tokenize("Tôi học"), tokenizer.tokenize("Tôi học"))

    def test_no_stemming_and_empty_tokens(self):
        tokenizer = VietnameseRougeTokenizer()
        self.assertEqual(tokenizer.tokenize("running runs"), ["running", "runs"])
        self.assertEqual(tokenizer.tokenize(" \n!? "), [])

    def test_non_string_input_is_rejected(self):
        with self.assertRaises(TypeError):
            VietnameseRougeTokenizer().tokenize(None)

    def test_invalid_corpus_is_rejected_before_scoring(self):
        with patch("src.metrics.create_rouge_scorer") as create:
            for references, predictions in [([], []), (["a"], [])]:
                with self.assertRaises(ValueError):
                    score_summaries(references, predictions)
            for references, predictions in [("a", ["a"]), (["a"], "a")]:
                with self.assertRaises(TypeError):
                    score_summaries(references, predictions)
        create.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("rouge_score"), "rouge-score chưa được cài trong môi trường kiểm tra")
class RougeScoreTests(unittest.TestCase):
    def test_identical_nonempty_text_scores_one(self):
        scores = score_summary("Tôi học tiếng Việt.", "TÔI HỌC TIẾNG VIỆT!")
        for name in ["rouge1", "rouge2", "rougeL"]:
            self.assertEqual(scores[name], {"precision": 1.0, "recall": 1.0, "f1": 1.0})

    def test_hand_calculated_example_and_argument_order(self):
        scores = score_summary("em thích học sâu", "em học sâu")
        for name in ["rouge1", "rougeL"]:
            self.assertEqual(scores[name]["precision"], 1.0)
            self.assertEqual(scores[name]["recall"], 3 / 4)
            self.assertAlmostEqual(scores[name]["f1"], 6 / 7)
        self.assertEqual(scores["rouge2"]["precision"], 1 / 2)
        self.assertEqual(scores["rouge2"]["recall"], 1 / 3)
        self.assertAlmostEqual(scores["rouge2"]["f1"], 0.4)

    def test_empty_sequences_and_no_overlap_score_zero(self):
        for reference, prediction in [("", ""), ("a b", ""), ("", "a b"), ("a b", "c d"), ("!?", "!?")]:
            with self.subTest(reference=reference, prediction=prediction):
                scores = score_summary(reference, prediction)
                for name in ["rouge1", "rouge2", "rougeL"]:
                    self.assertEqual(scores[name], {"precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_single_token_identical_text_has_no_bigrams(self):
        scores = score_summary("học", "học")
        self.assertEqual(scores["rouge1"]["f1"], 1.0)
        self.assertEqual(scores["rouge2"]["f1"], 0.0)
        self.assertEqual(scores["rougeL"]["f1"], 1.0)

    def test_repeated_ngrams_use_clipped_overlap_counts(self):
        scores = score_summary("a a b", "a a a")
        self.assertAlmostEqual(scores["rouge1"]["f1"], 2 / 3)
        self.assertEqual(scores["rouge2"]["f1"], 0.5)
        self.assertAlmostEqual(scores["rougeL"]["f1"], 2 / 3)

    def test_lcs_does_not_require_adjacent_tokens(self):
        scores = score_summary("a x b y c", "a b c")
        self.assertEqual(scores["rouge2"]["f1"], 0.0)
        self.assertAlmostEqual(scores["rougeL"]["f1"], 0.75)

    def test_mean_is_computed_from_each_f1_not_mean_precision_and_recall(self):
        result = score_summaries(["a b c d", "a"], ["a", "a b c d"])
        self.assertEqual(result["sample_count"], 2)
        self.assertAlmostEqual(result["mean_f1"]["rouge1"], 0.4)
        self.assertAlmostEqual(result["mean_f1_percent"]["rouge1"], 40.0)
        scores = result["per_sample"]
        mean_precision = (scores[0]["rouge1"]["precision"] + scores[1]["rouge1"]["precision"]) / 2
        mean_recall = (scores[0]["rouge1"]["recall"] + scores[1]["rouge1"]["recall"]) / 2
        wrong_f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
        self.assertNotAlmostEqual(result["mean_f1"]["rouge1"], wrong_f1)
        self.assertEqual(result["settings"], get_rouge_settings())


if __name__ == "__main__":
    unittest.main()
