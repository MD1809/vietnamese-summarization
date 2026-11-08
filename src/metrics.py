from src.preprocessing import normalize_text


ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


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
