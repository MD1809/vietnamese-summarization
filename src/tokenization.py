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


def create_seq2seq_collator(tokenizer):
    from transformers import DataCollatorForSeq2Seq

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    if tokenizer.padding_side != "right":
        raise ValueError("This ViT5 pipeline expects right padding")
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )


def collate_tokenized_samples(samples, collator):
    if len(samples) == 0:
        raise ValueError("Cannot collate an empty batch")
    features = []
    for sample in samples:
        feature = {}
        for field in ["input_ids", "attention_mask", "labels"]:
            feature[field] = list(sample[field])
        features.append(feature)
    return collator(features)


def validate_padded_batch(batch, samples, tokenizer):
    if len(samples) == 0:
        raise ValueError("Cannot validate an empty batch")
    if tokenizer.padding_side != "right":
        raise ValueError("This ViT5 pipeline expects right padding")
    longest_source = 0
    longest_target = 0
    for sample in samples:
        longest_source = max(longest_source, len(sample["input_ids"]))
        longest_target = max(longest_target, len(sample["labels"]))
    source_shape = (len(samples), longest_source)
    target_shape = (len(samples), longest_target)
    if tuple(batch["input_ids"].shape) != source_shape:
        raise ValueError("Source shape does not match dynamic batch length")
    if tuple(batch["attention_mask"].shape) != source_shape:
        raise ValueError("Attention mask shape does not match source shape")
    if tuple(batch["labels"].shape) != target_shape:
        raise ValueError("Label shape does not match dynamic target length")

    input_rows = batch["input_ids"].tolist()
    mask_rows = batch["attention_mask"].tolist()
    label_rows = batch["labels"].tolist()
    for index in range(len(samples)):
        sample = samples[index]
        source_length = len(sample["input_ids"])
        target_length = len(sample["labels"])
        if input_rows[index][:source_length] != sample["input_ids"]:
            raise ValueError("Source tokens changed during collation")
        if mask_rows[index][:source_length] != sample["attention_mask"]:
            raise ValueError("Original attention mask changed during collation")
        if label_rows[index][:target_length] != sample["labels"]:
            raise ValueError("Target tokens changed during collation")
        for position in range(source_length, longest_source):
            if input_rows[index][position] != tokenizer.pad_token_id:
                raise ValueError("Source padding must use pad_token_id")
            if mask_rows[index][position] != 0:
                raise ValueError("Attention mask must be zero at padded positions")
        for position in range(source_length):
            if mask_rows[index][position] != 1:
                raise ValueError("Attention mask must be one at original token positions")
        for position in range(target_length, longest_target):
            if label_rows[index][position] != -100:
                raise ValueError("Label padding must be -100")
    return {
        "dynamic_shape_check_passed": True,
        "original_tokens_check_passed": True,
        "source_padding_and_mask_check_passed": True,
        "label_padding_check_passed": True,
    }


def decode_padded_labels(labels, tokenizer):
    decoded = []
    for row in labels.tolist():
        token_ids = []
        for token_id in row:
            if token_id != -100:
                token_ids.append(token_id)
        decoded.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    return decoded


class Seq2SeqBatchCollator:
    def __init__(self, tokenizer):
        self.collator = create_seq2seq_collator(tokenizer)

    def __call__(self, samples):
        sample_ids = []
        group_ids = []
        for sample in samples:
            sample_ids.append(sample["id"])
            group_ids.append(sample["group_id"])
        return {
            "sample_ids": sample_ids,
            "group_ids": group_ids,
            "model_inputs": collate_tokenized_samples(samples, self.collator),
        }


def create_data_loader(dataset, tokenizer, batch_size, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("DataLoader batch_size must be a positive integer")
    if type(shuffle) is not bool:
        raise ValueError("DataLoader shuffle must be a boolean")
    if type(seed) is not int or seed < 0 or seed >= 2**32:
        raise ValueError("DataLoader seed must be an integer between 0 and 2**32 - 1")
    if len(dataset) == 0:
        raise ValueError("Cannot create a DataLoader from an empty dataset")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Seq2SeqBatchCollator(tokenizer),
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def create_train_validation_loaders(tokenized_splits, tokenizer, config_path="configs/train.json"):
    if set(tokenized_splits) != {"train", "validation"}:
        raise ValueError("Expected tokenized train and validation only; test remains locked")
    config = load_train_config(config_path)
    loaders = {}
    for split_name in ["train", "validation"]:
        shuffle = False
        if split_name == "train":
            shuffle = True
        loaders[split_name] = create_data_loader(
            tokenized_splits[split_name],
            tokenizer,
            config["batch_size"],
            shuffle,
            config["seed"],
        )
    return loaders


def validate_loader_epoch(loader, dataset, tokenizer, keep_order=False):
    samples_by_id = {}
    expected_ids = []
    for sample in dataset:
        sample_id = sample["id"]
        if sample_id in samples_by_id:
            raise ValueError("Duplicate sample ID in DataLoader input")
        samples_by_id[sample_id] = sample
        expected_ids.append(sample_id)

    seen_ids = []
    seen_set = set()
    batch_count = 0
    for batch in loader:
        batch_count = batch_count + 1
        if set(batch["model_inputs"]) != {"input_ids", "attention_mask", "labels"}:
            raise ValueError("Unexpected metadata or fields in model_inputs")
        if len(batch["sample_ids"]) != len(batch["group_ids"]):
            raise ValueError("Batch sample IDs and group IDs have different lengths")
        original_samples = []
        for index in range(len(batch["sample_ids"])):
            sample_id = batch["sample_ids"][index]
            if sample_id not in samples_by_id or sample_id in seen_set:
                raise ValueError("DataLoader returned an unknown or repeated sample ID")
            sample = samples_by_id[sample_id]
            if batch["group_ids"][index] != sample["group_id"]:
                raise ValueError("DataLoader changed sample group identity")
            if sample["input_ids"][-1] != tokenizer.eos_token_id:
                raise ValueError("Source EOS is missing before collation")
            if sample["labels"][-1] != tokenizer.eos_token_id:
                raise ValueError("Target EOS is missing before collation")
            original_samples.append(sample)
            seen_ids.append(sample_id)
            seen_set.add(sample_id)
        validate_padded_batch(batch["model_inputs"], original_samples, tokenizer)

    if seen_set != set(expected_ids):
        raise ValueError("DataLoader omitted input samples")
    if keep_order and seen_ids != expected_ids:
        raise ValueError("DataLoader changed validation sample order")
    return {
        "sample_count": len(seen_ids),
        "batch_count": batch_count,
        "sample_ids": seen_ids,
        "checks": {
            "complete_unique_samples_check_passed": True,
            "metadata_separation_check_passed": True,
            "padding_and_mask_check_passed": True,
            "eos_and_label_positions_check_passed": True,
            "requested_order_check_passed": True,
        },
    }
