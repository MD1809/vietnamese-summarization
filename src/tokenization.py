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


def load_tokenization_source(project_root, config_path="configs/train.json"):
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

    return config, splits, source_files, processed["subset_manifest_sha256"]


def measure_truncation(project_root, config_path="configs/train.json"):
    from transformers import AutoTokenizer

    config, splits, source_files, subset_fingerprint = load_tokenization_source(
        project_root, config_path
    )

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
        "subset_manifest_sha256": subset_fingerprint,
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


def tokenize_batch(batch, tokenizer, max_source_length, max_target_length, task_prefix):
    for limit in [max_source_length, max_target_length]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("Token limits must be positive integers")
    if not isinstance(task_prefix, str):
        raise ValueError("Task prefix must be a string")
    if len(batch["document"]) != len(batch["summary"]):
        raise ValueError("Document and summary batch sizes do not match")

    sources = []
    for document in batch["document"]:
        sources.append(task_prefix + document)

    encoded_sources = tokenizer(
        text=sources,
        max_length=max_source_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    encoded_targets = tokenizer(
        text_target=batch["summary"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    return {
        "input_ids": encoded_sources["input_ids"],
        "attention_mask": encoded_sources["attention_mask"],
        "labels": encoded_targets["input_ids"],
    }


def validate_tokenized_split(tokenized, original_rows, max_source_length, max_target_length):
    if len(tokenized) != len(original_rows):
        raise ValueError("Tokenization changed the number of samples")
    for index in range(len(original_rows)):
        row = tokenized[index]
        original = original_rows[index]
        if row["id"] != original["id"] or row["group_id"] != original["group_id"]:
            raise ValueError("Tokenization changed sample identity or order")
        if not 0 < len(row["input_ids"]) <= max_source_length:
            raise ValueError("Tokenized source length is invalid")
        if not 0 < len(row["labels"]) <= max_target_length:
            raise ValueError("Tokenized label length is invalid")
        if len(row["input_ids"]) != len(row["attention_mask"]):
            raise ValueError("Source IDs and attention mask lengths do not match")
        for value in row["attention_mask"]:
            if value != 1:
                raise ValueError("Unexpected padding before collation")
        for field in ["input_ids", "labels"]:
            for token_id in row[field]:
                if type(token_id) is not int or token_id < 0:
                    raise ValueError("Expected non-negative token IDs before collation")


def tokenize_train_validation(
    project_root,
    config_path="configs/train.json",
    load_from_cache_file=True,
):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    config, original_splits, source_files, subset_fingerprint = load_tokenization_source(
        project_root, config_path
    )
    data_files = {}
    for split_name in ["train", "validation"]:
        path = _prepared_file_path(project_root, source_files[split_name]["path"])
        data_files[split_name] = str(path)
    datasets = load_dataset("json", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        use_fast=False,
    )
    tokenized_splits = {}
    for split_name in ["train", "validation"]:
        dataset = datasets[split_name]
        original_rows = original_splits[split_name]
        if len(dataset) != len(original_rows):
            raise ValueError("Loaded dataset count does not match verified JSONL")
        for index in range(len(original_rows)):
            for field in ["id", "group_id", "document", "summary"]:
                if dataset[index][field] != original_rows[index][field]:
                    raise ValueError("Loaded dataset differs from verified JSONL")

        remove_columns = []
        for column in dataset.column_names:
            if column not in ["id", "group_id"]:
                remove_columns.append(column)
        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=remove_columns,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_source_length": config["max_source_length"],
                "max_target_length": config["max_target_length"],
                "task_prefix": config["task_prefix"],
            },
            load_from_cache_file=load_from_cache_file,
            desc="Tokenizing " + split_name,
        )
        validate_tokenized_split(
            tokenized,
            original_rows,
            config["max_source_length"],
            config["max_target_length"],
        )
        tokenized_splits[split_name] = tokenized
    return tokenized_splits, tokenizer
