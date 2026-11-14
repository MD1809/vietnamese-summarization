import json
import math
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def load_config(path):
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open(encoding="utf-8-sig") as config_file:
            config = json.load(config_file)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON in {config_path}: {error}")
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a JSON object: {config_path}")
    return config


def _positive_integer(config, key):
    value = config.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def load_train_config(path="configs/train.json"):
    config = load_config(path)
    expected_ids = {"model": "VietAI/vit5-base", "dataset": "esdurmus/wiki_lingua"}
    for section in expected_ids:
        expected_id = expected_ids[section]
        metadata = config.get(section)
        if not isinstance(metadata, dict):
            raise ValueError(f"{section} must be a JSON object")
        if metadata.get("id") != expected_id:
            raise ValueError(f"{section}.id must be {expected_id}")
        revision = metadata.get("revision")
        if not isinstance(revision, str):
            raise ValueError(f"{section}.revision must be a full 40-character commit SHA")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{section}.revision must be a full 40-character commit SHA")
    if config["dataset"].get("language") != "vietnamese":
        raise ValueError("dataset.language must be vietnamese")

    paths = config.get("paths")
    path_keys = ("train_file", "validation_file", "test_file", "output_dir", "checkpoint_dir")
    if not isinstance(paths, dict):
        raise ValueError("paths must be a JSON object")
    for key in path_keys:
        path_value = paths.get(key)
        if not isinstance(path_value, str):
            raise ValueError(f"paths.{key} must be a non-empty path string")
        if path_value.strip() == "":
            raise ValueError(f"paths.{key} must be a non-empty path string")

    seed = config.get("seed")
    if type(seed) is not int:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    integer_fields = [
        "max_source_length",
        "max_target_length",
        "batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
    ]
    for key in integer_fields:
        _positive_integer(config, key)
    for key in ("learning_rate", "weight_decay"):
        value = config.get(key)
        if type(value) not in (int, float):
            raise ValueError(f"{key} must be a finite number")
        if not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if value < 0:
            raise ValueError(f"{key} is outside the allowed range")
        if key == "learning_rate" and value == 0:
            raise ValueError(f"{key} is outside the allowed range")
    if config.get("optimizer") != "AdamW":
        raise ValueError("optimizer must be AdamW for the current project")
    if config.get("precision") not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if config.get("checkpoint_selection") != {"metric": "validation_loss", "mode": "min"}:
        raise ValueError("Select checkpoints by minimum validation_loss")
    if "early_stopping" in config:
        settings = config["early_stopping"]
        if not isinstance(settings, dict):
            raise ValueError("early_stopping must be a JSON object")
        _positive_integer(settings, "patience")
        min_delta = settings.get("min_delta")
        if type(min_delta) not in (int, float) or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("early_stopping.min_delta must be a finite nonnegative number")
    return config


def load_generation_config(path="configs/generation.json"):
    config = load_config(path)
    _positive_integer(config, "num_beams")
    _positive_integer(config, "max_new_tokens")
    if config.get("do_sample") is not False:
        raise ValueError("do_sample must be false for the planned greedy/beam comparison")
    no_repeat_ngram_size = config.get("no_repeat_ngram_size", 0)
    if type(no_repeat_ngram_size) is not int:
        raise ValueError("Lỗi: no_repeat_ngram_size phải là số nguyên.")
    if no_repeat_ngram_size < 0:
        raise ValueError("Lỗi: no_repeat_ngram_size không được âm.")
    return config


def set_seed(seed):
    if type(seed) is not int:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    try:
        import numpy as np
        import torch
    except ImportError:
        raise RuntimeError("set_seed requires NumPy and PyTorch in the active environment")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
