import re

from src.preprocessing import normalize_text


ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def measure_sentence_repetition(predictions):
    if not isinstance(predictions, (list, tuple)):
        raise TypeError("Lỗi: Danh sách dự đoán phải là list hoặc tuple.")
    repeated_sample_count = 0
    repeated_sentence_count = 0
    for prediction in predictions:
        normalized = normalize_text(prediction).lower()
        sentences = []
        for part in re.split(r"(?<=[.!?])\s+", normalized):
            sentence = part.strip()
            if sentence != "":
                sentences.append(sentence)
        duplicate_count = len(sentences) - len(set(sentences))
        if duplicate_count > 0:
            repeated_sample_count = repeated_sample_count + 1
            repeated_sentence_count = repeated_sentence_count + duplicate_count
    percentage = 0.0
    if len(predictions) > 0:
        percentage = repeated_sample_count / len(predictions) * 100
    return {
        "sample_count": len(predictions),
        "repeated_sample_count": repeated_sample_count,
        "repeated_sample_percent": percentage,
        "repeated_sentence_count": repeated_sentence_count,
        "rule": "NFC, lowercase and whitespace normalization; exact sentences split after . ! ? followed by whitespace",
    }


class VietnameseRougeTokenizer:
    def tokenize(self, text):
        normalized = normalize_text(text).lower()
        characters = []
        for character in normalized:
            if character.isalnum():
                characters.append(character)
            else:
                characters.append(" ")
        return "".join(characters).split()


def get_rouge_settings():
    return {
        "rouge_types": list(ROUGE_TYPES),
        "unicode_normalization": "NFC",
        "lowercase": True,
        "non_letter_number_handling": "replace with spaces",
        "token_unit": "whitespace-separated Unicode letters and numbers",
        "use_stemmer": False,
        "rouge_l_variant": "rougeL",
        "aggregation": "mean of per-sample F1; multiply by 100 for percentage scores",
        "empty_token_sequence_score": 0.0,
    }


def create_rouge_scorer():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(
        ROUGE_TYPES,
        use_stemmer=False,
        tokenizer=VietnameseRougeTokenizer(),
    )


def score_summary(reference, prediction, scorer=None):
    if scorer is None:
        scorer = create_rouge_scorer()
    raw_scores = scorer.score(reference, prediction)
    scores = {}
    for name in ROUGE_TYPES:
        score = raw_scores[name]
        scores[name] = {
            "precision": float(score.precision),
            "recall": float(score.recall),
            "f1": float(score.fmeasure),
        }
    return scores


def score_summaries(references, predictions):
    if not isinstance(references, (list, tuple)):
        raise TypeError("Lỗi: Danh sách tham chiếu phải là list hoặc tuple.")
    if not isinstance(predictions, (list, tuple)):
        raise TypeError("Lỗi: Danh sách dự đoán phải là list hoặc tuple.")
    if len(references) != len(predictions):
        raise ValueError("Lỗi: Số tham chiếu và dự đoán không khớp.")
    if len(references) == 0:
        raise ValueError("Lỗi: Không có mẫu để tính ROUGE trung bình.")
    scorer = create_rouge_scorer()
    per_sample = []
    totals = {}
    for name in ROUGE_TYPES:
        totals[name] = 0.0
    for reference, prediction in zip(references, predictions):
        scores = score_summary(reference, prediction, scorer)
        per_sample.append(scores)
        for name in ROUGE_TYPES:
            totals[name] = totals[name] + scores[name]["f1"]
    mean_f1 = {}
    mean_f1_percent = {}
    for name in ROUGE_TYPES:
        mean_f1[name] = totals[name] / len(per_sample)
        mean_f1_percent[name] = mean_f1[name] * 100
    return {
        "sample_count": len(per_sample),
        "per_sample": per_sample,
        "mean_f1": mean_f1,
        "mean_f1_percent": mean_f1_percent,
        "settings": get_rouge_settings(),
    }
