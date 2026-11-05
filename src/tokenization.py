import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data import _prepared_file_path, _read_prepared_jsonl, _split_manifest_fingerprint
from src.utils import load_train_config


def count_truncated_samples(lengths, limit):
    if type(limit) is not int or limit <= 0:
        raise ValueError("Token limit must be a positive integer")
    if len(lengths) == 0:
        raise ValueError("Cannot measure truncation on empty data")
    truncated_count = 0
    for length in lengths:
        if length > limit:
            truncated_count = truncated_count + 1
    return {
        "limit": limit,
        "sample_count": len(lengths),
        "truncated_count": truncated_count,
        "truncated_percentage": truncated_count / len(lengths) * 100,
    }


def measure_truncation(project_root, config_path="configs/train.json"):
    from transformers import AutoTokenizer

    config = load_train_config(config_path)
    if not isinstance(config.get("task_prefix"), str):
        raise ValueError("Configuration must explicitly define task_prefix as a string")
    root = Path(project_root)
    manifest_dir = root / "data" / "manifests"
    processed = json.loads(
        (manifest_dir / "wikilingua_processed.json").read_text(encoding="utf-8")
    )
    subset = json.loads(
        (manifest_dir / "wikilingua_subsets.json").read_text(encoding="utf-8")
    )
    if processed["dataset"] != config["dataset"] or subset["dataset"] != config["dataset"]:
        raise ValueError("Prepared dataset does not match configuration")
    if processed["seed"] != config["seed"] or subset["seed"] != config["seed"]:
        raise ValueError("Prepared seed does not match configuration")
    if processed["subset_manifest_sha256"] != _split_manifest_fingerprint(subset):
        raise ValueError("Prepared data refers to a different subset")
    if processed["test_lock"]["locked"] is not True:
        raise ValueError("Test must remain locked during token limit inspection")

    splits = {}
    source_files = {}
    for split_name in ["train", "validation"]:
        metadata = processed["files"][split_name]
        if metadata["path"] != config["paths"][split_name + "_file"]:
            raise ValueError("Prepared path does not match configuration")
        path = _prepared_file_path(root, metadata["path"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Prepared file changed: {split_name}")
        if len(content) != metadata["byte_count"]:
            raise ValueError(f"Prepared file size changed: {split_name}")
        rows = _read_prepared_jsonl(path)
        ids = []
        for pair in rows:
            ids.append(pair["id"])
        expected = subset["splits"][split_name]
        if len(rows) != metadata["pair_count"] or len(rows) != expected["pair_count"]:
            raise ValueError("Prepared count does not match manifest")
        if ids != sorted(expected["sample_ids"]) or len(set(ids)) != len(ids):
            raise ValueError("Prepared sample IDs do not match fixed subset")
        splits[split_name] = rows
        source_files[split_name] = metadata.copy()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        use_fast=False,
    )
    results = {}
    for split_name in ["train", "validation"]:
        source_lengths = []
        target_lengths = []
        for pair in splits[split_name]:
            source = tokenizer(
                text=config["task_prefix"] + pair["document"],
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )
            target = tokenizer(
                text_target=pair["summary"],
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )
            source_lengths.append(len(source["input_ids"]))
            target_lengths.append(len(target["input_ids"]))
        results[split_name] = {
            "document": count_truncated_samples(source_lengths, config["max_source_length"]),
            "summary": count_truncated_samples(target_lengths, config["max_target_length"]),
        }

    return {
        "dataset": config["dataset"].copy(),
        "seed": config["seed"],
        "subset_manifest_sha256": processed["subset_manifest_sha256"],
        "source_files": source_files,
        "tokenization": {
            "model_id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "tokenizer_class": type(tokenizer).__name__,
            "task_prefix": config["task_prefix"],
            "add_special_tokens": True,
            "padding": False,
            "truncation_when_measuring": False,
            "use_fast": False,
            "summary_input": "text_target",
        },
        "splits": results,
        "decision_status": "pending_review",
        "checks": {
            "file_hash_check_passed": True,
            "fixed_sample_ids_check_passed": True,
            "test_remains_locked": True,
            "test_content_read": False,
        },
    }


def save_truncation_report(report, path):
    saved = report.copy()
    saved["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(saved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return saved
