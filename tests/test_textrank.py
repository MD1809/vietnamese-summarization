import json
import math
import tempfile
import unittest
from pathlib import Path

from src.inference import prepare_source
from src.textRank import rank_sentences, sentence_similarity, sentence_units, split_sentences, summarize_textrank
from src.utils import load_train_config


class SimpleTokenizer:
    def __init__(self):
        self.words = {}
        self.reverse = {}

    def __call__(self, text, **options):
        ids = []
        for word in text.split():
            if word not in self.words:
                index = len(self.words) + 2
                self.words[word] = index
                self.reverse[index] = word
            ids.append(self.words[word])
        if options.get("add_special_tokens", True):
            ids.append(1)
        if options.get("truncation", False) and len(ids) > options["max_length"]:
            ids = ids[:options["max_length"] - 1] + [1]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for index in ids:
            if index in self.reverse:
                words.append(self.reverse[index])
        return " ".join(words)


class TextRankTests(unittest.TestCase):
    def summarize_with_budget(self, text, budget, tokenizer=None):
        if tokenizer is None:
            tokenizer = SimpleTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            config = load_train_config()
            config["max_target_length"] = budget
            path = Path(directory) / "train.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            return summarize_textrank(text, tokenizer, path)

    def test_unicode_units_and_sentence_boundaries(self):
        self.assertEqual(sentence_units("Tôi TRỒNG đậu, 2 ngày!"), ["tôi", "trồng", "đậu", "2", "ngày"])
        self.assertEqual(split_sentences("Một. Hai! Ba? 3.14 là số."), ["Một.", "Hai!", "Ba?", "3.14 là số."])

    def test_similarity_matches_hand_calculation_and_counts_shared_units_once(self):
        expected = 2 / (math.log(3) + math.log(3))
        self.assertAlmostEqual(sentence_similarity("trồng cây xanh", "cây xanh đẹp"), expected)
        self.assertAlmostEqual(sentence_similarity("cây cây xanh", "cây khác"), 1 / (math.log(3) + math.log(2)))
        self.assertEqual(sentence_similarity("ĐẬU", "đậu"), 1.0)
        self.assertEqual(sentence_similarity("!!!", "đậu"), 0.0)

    def test_first_iteration_matches_weighted_pagerank_by_hand(self):
        ranking = rank_sentences(["đậu cây", "đậu xanh", "cây đẹp"], max_iterations=1)
        self.assertAlmostEqual(ranking["scores"][0], 0.05 + 0.85 * 2 / 3)
        self.assertAlmostEqual(ranking["scores"][1], 0.05 + 0.85 / 6)
        self.assertAlmostEqual(ranking["scores"][2], 0.05 + 0.85 / 6)
        self.assertFalse(ranking["converged"])

    def test_disconnected_nodes_keep_finite_normalized_scores(self):
        for sentences in [["đậu", "mèo", "sách"], ["đậu cây", "đậu xanh", "sách"]]:
            ranking = rank_sentences(sentences)
            self.assertTrue(ranking["converged"])
            self.assertAlmostEqual(sum(ranking["scores"]), 1.0)
            for score in ranking["scores"]:
                self.assertTrue(math.isfinite(score))
                self.assertGreater(score, 0)
        self.assertEqual(rank_sentences([])["scores"], [])

    def test_invalid_ranking_parameters_are_rejected(self):
        for options in [{"damping": 1}, {"damping": float("nan")}, {"tolerance": 0}, {"max_iterations": 0}]:
            with self.assertRaises(ValueError):
                rank_sentences(["câu"], **options)

    def test_empty_input_rejected_and_single_sentence_preserved(self):
        with self.assertRaises(ValueError):
            summarize_textrank(" \n\t", SimpleTokenizer())
        result = summarize_textrank("  Tôi trồng cây. ", SimpleTokenizer())
        self.assertEqual(result["summary"], "Tôi trồng cây.")
        self.assertEqual(result["sentence_scores"][0]["score"], 1.0)

    def test_equal_scores_choose_earlier_sentence_and_preserve_source_order(self):
        result = self.summarize_with_budget("Mèo ngủ. Sách đẹp. Trời mưa.", 4)
        self.assertEqual(result["summary"], "Mèo ngủ. Sách đẹp.")
        result = self.summarize_with_budget("Đẹp trời. Cây xanh. Cây đẹp.", 4)
        self.assertEqual(result["summary"], "Đẹp trời. Cây đẹp.")
        self.assertEqual(result["selected_sentence_indices"], [0, 2])

    def test_ranked_central_sentence_beats_first_sentence(self):
        result = self.summarize_with_budget("Sách mới. Đậu cây. Đậu xanh. Cây đẹp.", 2)
        self.assertEqual(result["summary"], "Đậu cây.")
        self.assertEqual(result["selected_sentence_indices"], [1])

    def test_exact_duplicate_sentences_are_only_kept_once(self):
        result = summarize_textrank("Trồng cây. TRỒNG CÂY. Tưới nước.", SimpleTokenizer())
        self.assertEqual(result["summary"], "Trồng cây. Tưới nước.")
        self.assertEqual(result["duplicate_sentence_count"], 1)
        self.assertEqual(result["selected_sentence_indices"], [0, 2])

    def test_source_matches_vit5_preparation_and_excludes_cut_off_text(self):
        tokenizer = SimpleTokenizer()
        text = " ".join(["đậu"] * 1100) + ". CÂU_NGOÀI_NGUỒN."
        result = summarize_textrank(text, tokenizer)
        encoded, source = prepare_source(text, tokenizer, load_train_config())
        self.assertEqual(result["used_input_ids"], source["used_input_ids"])
        self.assertTrue(result["was_truncated"])
        self.assertNotIn("CÂU_NGOÀI_NGUỒN", result["summary"])
        self.assertLessEqual(result["summary_token_count"], 128)

    def test_whole_sentence_selection_skips_oversized_sentence(self):
        result = self.summarize_with_budget("Đậu cây xanh đẹp tốt. Đậu xanh. Sách mới.", 2)
        self.assertEqual(result["summary"], "Đậu xanh.")
        self.assertFalse(result["summary_was_truncated"])

    def test_fallback_truncates_only_when_no_complete_sentence_fits(self):
        result = self.summarize_with_budget("Đậu xanh trồng trong chậu.", 3)
        self.assertEqual(result["summary"], "Đậu xanh trồng")
        self.assertTrue(result["summary_was_truncated"])
        self.assertEqual(result["summary_token_count"], 3)

    def test_fallback_recounts_after_decoding_changes_token_boundaries(self):
        class ExpandingTokenizer(SimpleTokenizer):
            def decode(self, ids, skip_special_tokens=True):
                text = super().decode(ids, skip_special_tokens)
                if len(ids) == 3:
                    text = text + " thêm"
                return text

        result = self.summarize_with_budget("Một hai ba bốn năm.", 3, ExpandingTokenizer())
        self.assertEqual(result["summary"], "Một hai")
        self.assertEqual(result["summary_token_count"], 2)
