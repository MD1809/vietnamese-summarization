import math
import re
import time

from src.inference import prepare_source
from src.preprocessing import normalize_text
from src.utils import load_train_config


def split_sentences(text):
    sentences = []
    parts = re.split(r"(?<=[.!?])\s+", text)
    for part in parts:
        sentence = part.strip()
        if sentence != "":
            sentences.append(sentence)
    return sentences


def sentence_units(sentence):
    normalized = normalize_text(sentence)
    normalized = normalized.lower()
    units = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return units


def sentence_similarity(first, second):
    first_units = sentence_units(first)
    second_units = sentence_units(second)
    if len(first_units) == 0 or len(second_units) == 0:
        return 0.0
    first_unique_units = set(first_units)
    second_unique_units = set(second_units)
    shared_units = first_unique_units.intersection(second_unique_units)
    shared_count = len(shared_units)
    first_log_length = math.log(len(first_units))
    second_log_length = math.log(len(second_units))
    denominator = first_log_length + second_log_length
    if denominator == 0:
        return float(shared_count)
    similarity = shared_count / denominator
    return similarity


def rank_sentences(sentences, damping=0.85, tolerance=0.000001, max_iterations=100):
    if not math.isfinite(damping) or damping <= 0 or damping >= 1:
        raise ValueError("Lỗi: damping phải nằm giữa 0 và 1.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Lỗi: tolerance phải là số dương hữu hạn.")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Lỗi: max_iterations phải là số nguyên dương.")
    count = len(sentences)
    if count == 0:
        return {
            "scores": [],
            "iterations": 0,
            "converged": True,
            "edge_count": 0,
        }
    weights = []
    for index in range(count):
        row = []
        for column in range(count):
            row.append(0.0)
        weights.append(row)
    edge_count = 0
    for first in range(count):
        for second in range(first + 1, count):
            weight = sentence_similarity(sentences[first], sentences[second])
            weights[first][second] = weight
            weights[second][first] = weight
            if weight > 0:
                edge_count = edge_count + 1
    degrees = []
    scores = []
    for index in range(count):
        degree = sum(weights[index])
        degrees.append(degree)
        initial_score = 1.0 / count
        scores.append(initial_score)
    converged = False
    for iteration in range(1, max_iterations + 1):
        dangling_score = 0.0
        for index in range(count):
            if degrees[index] == 0:
                dangling_score = dangling_score + scores[index]
        updated = []
        for target in range(count):
            incoming = 0.0
            for source in range(count):
                if degrees[source] > 0:
                    weighted_score = weights[source][target] * scores[source]
                    contribution = weighted_score / degrees[source]
                    incoming = incoming + contribution
            base_score = (1.0 - damping) / count
            redistributed_score = dangling_score / count
            incoming_score = damping * (incoming + redistributed_score)
            score = base_score + incoming_score
            updated.append(score)
        difference = 0.0
        for index in range(count):
            sentence_difference = abs(updated[index] - scores[index])
            difference = difference + sentence_difference
        scores = updated
        if difference <= tolerance:
            converged = True
            break
    return {
        "scores": scores,
        "iterations": iteration,
        "converged": converged,
        "edge_count": edge_count,
    }


def content_token_ids(text, tokenizer):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )
    return encoded["input_ids"]


def summarize_textrank(
    text,
    tokenizer,
    config_path="configs/train.json",
    damping=0.85,
    tolerance=0.000001,
    max_iterations=100,
):
    started_at = time.perf_counter()
    config = load_train_config(config_path)
    encoded, source_info = prepare_source(text, tokenizer, config)
    available_source = tokenizer.decode(
        source_info["used_input_ids"],
        skip_special_tokens=True,
    )
    available_source = available_source.strip()
    original_sentences = split_sentences(available_source)
    sentences = []
    original_indices = []
    seen = set()
    for index in range(len(original_sentences)):
        sentence = original_sentences[index]
        identity = normalize_text(sentence)
        identity = identity.lower()
        if identity not in seen:
            seen.add(identity)
            sentences.append(sentence)
            original_indices.append(index)
    if len(sentences) == 0:
        raise ValueError("Lỗi: Không còn câu nào sau khi chuẩn bị nguồn.")
    ranking = rank_sentences(sentences, damping, tolerance, max_iterations)
    remaining_indices = []
    for index in range(len(sentences)):
        remaining_indices.append(index)
    ranked_indices = []
    while len(remaining_indices) > 0:
        best_index = remaining_indices[0]
        for index in remaining_indices:
            if ranking["scores"][index] > ranking["scores"][best_index]:
                best_index = index
        ranked_indices.append(best_index)
        remaining_indices.remove(best_index)
    budget = config["max_target_length"]
    selected = []
    for index in ranked_indices:
        candidate_indices = selected.copy()
        candidate_indices.append(index)
        candidate_indices.sort()
        candidate_sentences = []
        for candidate_index in candidate_indices:
            candidate_sentences.append(sentences[candidate_index])
        candidate = " ".join(candidate_sentences)
        candidate_ids = content_token_ids(candidate, tokenizer)
        candidate_token_count = len(candidate_ids)
        if candidate_token_count <= budget:
            selected.append(index)
    truncated = False
    if len(selected) == 0:
        best_index = ranked_indices[0]
        selected.append(best_index)
        best_sentence = sentences[best_index]
        best_sentence_ids = content_token_ids(best_sentence, tokenizer)
        kept_ids = []
        for index in range(min(budget, len(best_sentence_ids))):
            kept_ids.append(best_sentence_ids[index])
        summary = tokenizer.decode(kept_ids, skip_special_tokens=True)
        summary = summary.strip()
        summary_ids = content_token_ids(summary, tokenizer)
        while len(summary_ids) > budget:
            kept_ids.pop()
            summary = tokenizer.decode(kept_ids, skip_special_tokens=True)
            summary = summary.strip()
            summary_ids = content_token_ids(summary, tokenizer)
        truncated = True
    else:
        selected.sort()
        selected_sentences = []
        for index in selected:
            selected_sentences.append(sentences[index])
        summary = " ".join(selected_sentences)
    summary_ids = content_token_ids(summary, tokenizer)
    selected_original_indices = []
    for index in selected:
        selected_original_indices.append(original_indices[index])
    sentence_scores = []
    for index in range(len(sentences)):
        sentence = sentences[index]
        sentence_result = {
            "source_sentence_index": original_indices[index],
            "sentence": sentence,
            "score": ranking["scores"][index],
        }
        sentence_scores.append(sentence_result)
    result = {
        "baseline": "textrank",
        "summary": summary,
        "available_source_text": available_source,
        "selected_sentence_count": len(selected),
        "selected_sentence_indices": selected_original_indices,
        "sentence_scores": sentence_scores,
        "duplicate_sentence_count": len(original_sentences) - len(sentences),
        "summary_ids": summary_ids,
        "summary_token_count": len(summary_ids),
        "max_summary_tokens": budget,
        "summary_was_truncated": truncated,
        "textrank_config": {
            "similarity": "unique shared Unicode units divided by log sentence lengths",
            "token_unit": "Unicode letter/number units, usually Vietnamese syllables",
            "remove_stopwords": False,
            "damping": damping,
            "tolerance": tolerance,
            "max_iterations": max_iterations,
            "convergence_measure": "L1 difference between consecutive score vectors",
            "dangling_nodes": "redistribute score uniformly",
            "duplicate_handling": "keep first sentence after NFC, whitespace and lowercase",
            "tie_break": "earlier source sentence",
            "selection": "greedy whole sentences within content-token budget; restore source order",
            "overflow_fallback": "truncate highest-ranked sentence only if no whole sentence fits",
            "single_unit_similarity": "shared unit count when both log lengths are zero",
        },
        "iterations": ranking["iterations"],
        "converged": ranking["converged"],
        "edge_count": ranking["edge_count"],
        "elapsed_seconds": time.perf_counter() - started_at,
        "timing_scope": "source preparation, graph ranking and sentence selection; excludes tokenizer loading",
    }
    result.update(source_info)
    return result
