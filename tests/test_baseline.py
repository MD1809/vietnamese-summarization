import unittest

from src.baseline import split_sentences, summarize_lead_three
from src.inference import prepare_source
from src.utils import load_train_config


class WordTokenizer:
    def __init__(self):
        self.word_ids = {}
        self.id_words = {}

    def __call__(self, text, **options):
        ids = []
        for word in text.split():
            if word not in self.word_ids:
                token_id = len(self.word_ids) + 2
                self.word_ids[word] = token_id
                self.id_words[token_id] = word
            ids.append(self.word_ids[word])
        if options.get("add_special_tokens", True):
            ids.append(1)
        if options.get("truncation", False):
            max_length = options["max_length"]
            if len(ids) > max_length:
                ids = ids[:max_length - 1] + [1]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for token_id in ids:
            if token_id in self.id_words:
                words.append(self.id_words[token_id])
        return " ".join(words)


class BaselineTests(unittest.TestCase):
    def test_sentence_boundaries_keep_punctuation_and_require_whitespace(self):
        actual = split_sentences("Một. Hai! Ba? Bốn... Năm.6 chưa tách")
        self.assertEqual(actual, ["Một.", "Hai!", "Ba?", "Bốn...", "Năm.6 chưa tách"])

    def test_first_three_sentences_and_normalization(self):
        result = summarize_lead_three("  Tôi học.\nBạn học! Ai học? Câu bốn. ", WordTokenizer())
        self.assertEqual(result["summary"], "Tôi học. Bạn học! Ai học?")
        self.assertEqual(result["selected_sentence_count"], 3)
        self.assertFalse(result["summary_was_truncated"])

    def test_fewer_sentences_and_no_punctuation(self):
        for text, count in [("Một. Hai.", 2), ("Một câu", 1), ("Một câu.", 1)]:
            with self.subTest(text=text):
                result = summarize_lead_three(text, WordTokenizer())
                self.assertEqual(result["summary"], text)
                self.assertEqual(result["selected_sentence_count"], count)

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rỗng"):
            summarize_lead_three(" \n\t ", WordTokenizer())

    def test_source_ids_match_shared_inference_preparation(self):
        tokenizer = WordTokenizer()
        text = " ".join(["nguồn"] * 1500) + ". CÂU_NGOÀI_GIỚI_HẠN."
        result = summarize_lead_three(text, tokenizer)
        encoded, expected = prepare_source(text, tokenizer, load_train_config())
        self.assertEqual(result["used_input_ids"], expected["used_input_ids"])
        self.assertTrue(result["was_truncated"])
        self.assertEqual(result["used_input_token_count"], 1024)
        self.assertNotIn("CÂU_NGOÀI_GIỚI_HẠN", result["available_source_text"])
        self.assertNotIn("CÂU_NGOÀI_GIỚI_HẠN", result["summary"])

    def test_output_is_limited_to_128_content_tokens(self):
        text = " ".join(["từ"] * 200) + ". Câu hai. Câu ba. Câu bốn."
        result = summarize_lead_three(text, WordTokenizer())
        self.assertEqual(result["selected_sentence_count"], 3)
        self.assertEqual(result["summary_token_count"], 128)
        self.assertEqual(len(result["summary_ids"]), 128)
        self.assertTrue(result["summary_was_truncated"])
        self.assertFalse(result["was_truncated"])
        self.assertEqual(result["summary"], " ".join(["từ"] * 128))

    def test_decoded_summary_is_recounted_after_token_boundary_changes(self):
        class ExpandingTokenizer(WordTokenizer):
            def decode(self, ids, skip_special_tokens=True):
                text = super().decode(ids, skip_special_tokens)
                if len(ids) == 128:
                    text = text + " thêm"
                return text

        tokenizer = ExpandingTokenizer()
        text = " ".join(["từ"] * 200)
        result = summarize_lead_three(text, tokenizer)
        actual_ids = tokenizer(result["summary"], add_special_tokens=False)["input_ids"]
        self.assertEqual(result["summary_token_count"], len(actual_ids))
        self.assertLessEqual(len(actual_ids), 128)
        self.assertTrue(result["summary_was_truncated"])


if __name__ == "__main__":
    unittest.main()
