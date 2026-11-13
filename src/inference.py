from pathlib import Path

from src.preprocessing import normalize_text
from src.utils import load_generation_config, load_train_config


def load_summarizer(checkpoint_run_dir, config_path="configs/train.json"):
    from src.train import load_best_checkpoint

    best_dir = Path(checkpoint_run_dir) / "best"
    required_files = [
        "config.json",
        "model.safetensors",
        "spiece.model",
        "tokenizer_config.json",
        "selection.json",
    ]
    for name in required_files:
        if not (best_dir / name).is_file():
            raise FileNotFoundError("Lỗi: Checkpoint best thiếu file: " + str(best_dir / name))
    return load_best_checkpoint(checkpoint_run_dir, config_path)


def prepare_source(text, tokenizer, config, return_tensors=None):
    normalized = normalize_text(text)
    if normalized == "":
        raise ValueError("Lỗi: Văn bản đầu vào rỗng hoặc chỉ có khoảng trắng.")
    source = config["task_prefix"] + normalized
    full_encoding = tokenizer(
        source,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )
    encoded = tokenizer(
        source,
        add_special_tokens=True,
        max_length=config["max_source_length"],
        truncation=True,
        padding=False,
        return_tensors=return_tensors,
    )
    if return_tensors == "pt":
        used_ids = encoded["input_ids"].tolist()[0]
    else:
        used_ids = encoded["input_ids"]
    source_info = {
        "normalized_text": normalized,
        "input_token_count": len(full_encoding["input_ids"]),
        "used_input_token_count": len(used_ids),
        "was_truncated": len(full_encoding["input_ids"]) > len(used_ids),
        "max_source_length": config["max_source_length"],
        "used_input_ids": used_ids,
    }
    return encoded, source_info


def summarize_text(
    text,
    model,
    tokenizer,
    config_path="configs/train.json",
    generation_config_path="configs/generation.json",
):
    import time
    import torch

    config = load_train_config(config_path)
    generation = load_generation_config(generation_config_path)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    encoded, source_info = prepare_source(text, tokenizer, config, "pt")
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            do_sample=generation["do_sample"],
            num_beams=generation["num_beams"],
            max_new_tokens=generation["max_new_tokens"],
            no_repeat_ngram_size=generation.get("no_repeat_ngram_size", 0),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generated_ids = generated.tolist()[0]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    elapsed = time.perf_counter() - started_at
    result = {
        "summary": summary,
        "generated_ids": generated_ids,
        "generation_config": generation,
        "model_device": str(device),
        "elapsed_seconds": elapsed,
        "timing_scope": "normalization, tokenization, generation and decode; excludes model loading",
    }
    result.update(source_info)
    return result
