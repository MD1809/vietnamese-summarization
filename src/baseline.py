import re
import time

from src.inference import prepare_source
from src.utils import load_train_config


def split_sentences(text):
    sentences = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        sentence = part.strip()
        if sentence != "":
            sentences.append(sentence)
    return sentences


def summarize_lead_three(text, tokenizer, config_path="configs/train.json"):
    config = load_train_config(config_path)
    started_at = time.perf_counter()
    encoded, source_info = prepare_source(text, tokenizer, config)
    available_source = tokenizer.decode(
        source_info["used_input_ids"],
        skip_special_tokens=True,
    ).strip()
    sentences = split_sentences(available_source)
    selected_sentences = sentences[:3]
    selected_text = " ".join(selected_sentences)
    candidate_ids = tokenizer(
        selected_text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
    max_summary_tokens = config["max_target_length"]
    kept_ids = candidate_ids[:max_summary_tokens]
    summary = tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
    summary_ids = tokenizer(
        summary,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
    while len(summary_ids) > max_summary_tokens:
        kept_ids = kept_ids[:-1]
        summary = tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
        summary_ids = tokenizer(
            summary,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
    result = {
        "baseline": "lead_three",
        "summary": summary,
        "available_source_text": available_source,
        "selected_sentence_count": len(selected_sentences),
        "selected_text": selected_text,
        "candidate_token_count": len(candidate_ids),
        "summary_token_count": len(summary_ids),
        "max_summary_tokens": max_summary_tokens,
        "summary_was_truncated": len(kept_ids) < len(candidate_ids),
        "summary_ids": summary_ids,
        "elapsed_seconds": time.perf_counter() - started_at,
        "timing_scope": "normalization, tokenization, sentence selection and decode; excludes tokenizer loading",
    }
    result.update(source_info)
    return result
