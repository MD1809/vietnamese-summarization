from src.utils import load_train_config


def load_vit5_model(config_path="configs/train.json"):
    import torch
    from transformers import AutoModelForSeq2SeqLM

    config = load_train_config(config_path)
    if config["precision"] != "fp32":
        raise ValueError("G4.1 expects the approved fp32 configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("Lỗi: GPU chưa khả dụng để tải ViT5.")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
    )
    model = model.to(device="cuda:0", dtype=torch.float32)
    model.eval()
    return model


def check_forward_batch(model, batch):
    import torch

    if set(batch["model_inputs"]) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Forward expects source IDs, mask and original labels only")
    inputs = batch["model_inputs"]
    for field in ["input_ids", "attention_mask", "labels"]:
        if len(inputs[field].shape) != 2:
            raise ValueError("Forward inputs must be two-dimensional tensors")
    batch_size = inputs["input_ids"].shape[0]
    target_length = inputs["labels"].shape[1]
    if batch_size == 0 or target_length == 0 or inputs["input_ids"].shape[1] == 0:
        raise ValueError("Cannot run forward on an empty batch")
    if inputs["attention_mask"].shape != inputs["input_ids"].shape:
        raise ValueError("Source IDs and mask shapes must match")
    if inputs["labels"].shape[0] != batch_size:
        raise ValueError("Source and label batch sizes must match")
    if len(batch["sample_ids"]) != batch_size or len(batch["group_ids"]) != batch_size:
        raise ValueError("Batch metadata count must match tensor batch size")

    label_rows = inputs["labels"].tolist()
    valid_tokens = 0
    masked_tokens = 0
    for row in label_rows:
        for token_id in row:
            if token_id == -100:
                masked_tokens = masked_tokens + 1
            else:
                valid_tokens = valid_tokens + 1
    if valid_tokens == 0:
        raise ValueError("Loss requires at least one non-masked target token")

    parameter = next(model.parameters())
    model_inputs = {}
    for field in ["input_ids", "attention_mask", "labels"]:
        model_inputs[field] = inputs[field].to(parameter.device)

    model.eval()
    with torch.no_grad():
        decoder_ids = model.prepare_decoder_input_ids_from_labels(
            labels=model_inputs["labels"].clone(),
        )
        decoder_rows = decoder_ids.tolist()
        for index in range(batch_size):
            expected = [model.config.decoder_start_token_id]
            for token_id in label_rows[index][:-1]:
                if token_id == -100:
                    expected.append(model.config.pad_token_id)
                else:
                    expected.append(token_id)
            if decoder_rows[index] != expected:
                raise ValueError("Decoder input does not match one right shift of labels")

        outputs = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
            return_dict=True,
            use_cache=False,
        )
        expected_shape = (batch_size, target_length, model.config.vocab_size)
        if tuple(outputs.logits.shape) != expected_shape:
            raise ValueError("Logits shape must be [batch, target length, vocabulary size]")
        if outputs.loss is None or len(outputs.loss.shape) != 0:
            raise ValueError("Forward must return a scalar loss")
        if not torch.isfinite(outputs.loss).all().item():
            raise ValueError("Forward loss contains NaN or Inf")
        if not torch.isfinite(outputs.logits).all().item():
            raise ValueError("Forward logits contain NaN or Inf")
        if outputs.loss.requires_grad or outputs.logits.requires_grad:
            raise ValueError("G4.1 forward inspection must not build a gradient graph")
        if model_inputs["labels"].tolist() != label_rows:
            raise ValueError("Original labels changed during forward inspection")
        loss_value = outputs.loss.item()

    return {
        "sample_ids": list(batch["sample_ids"]),
        "group_ids": list(batch["group_ids"]),
        "model_class": type(model).__name__,
        "model_device": str(parameter.device),
        "model_dtype": str(parameter.dtype),
        "mode": "eval with no_grad",
        "input_shape": list(model_inputs["input_ids"].shape),
        "attention_mask_shape": list(model_inputs["attention_mask"].shape),
        "labels_shape": list(model_inputs["labels"].shape),
        "logits_shape": list(outputs.logits.shape),
        "vocab_size": model.config.vocab_size,
        "loss": loss_value,
        "valid_target_token_count": valid_tokens,
        "masked_label_count": masked_tokens,
        "labels_preview": label_rows[0][:12],
        "decoder_input_ids_preview": decoder_rows[0][:12],
        "decoder_start_token_id": model.config.decoder_start_token_id,
        "pad_token_id": model.config.pad_token_id,
        "checks": {
            "logits_shape_check_passed": True,
            "finite_logits_and_loss_check_passed": True,
            "decoder_right_shift_check_passed": True,
            "original_labels_check_passed": True,
            "no_gradient_graph_check_passed": True,
        },
        "backward_run": False,
        "optimizer_step_run": False,
        "training_memory_verified": False,
    }


def create_optimizer(model, config_path="configs/train.json"):
    import torch

    config = load_train_config(config_path)
    if config["optimizer"] != "AdamW" or config["precision"] != "fp32":
        raise ValueError("G4.2 expects AdamW and fp32")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        foreach=False,
    )


def validate_model(model, validation_loader, expected_sample_ids):
    import torch

    expected_sample_ids = list(expected_sample_ids)
    if len(expected_sample_ids) == 0:
        raise ValueError("Validation requires a nonempty set of samples")
    if len(set(expected_sample_ids)) != len(expected_sample_ids):
        raise ValueError("Expected validation sample IDs must be unique")

    parameters_before = {}
    for name, parameter in model.named_parameters():
        parameters_before[name] = parameter.detach().cpu().clone()

    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    seen_ids = []
    model.eval()
    try:
        for batch in validation_loader:
            report = check_forward_batch(model, batch)
            token_count = report["valid_target_token_count"]
            total_loss = total_loss + report["loss"] * token_count
            total_tokens = total_tokens + token_count
            batch_count = batch_count + 1
            seen_ids.extend(report["sample_ids"])
            if batch_count % 50 == 0:
                print("Validation đã xử lý số batch:", batch_count, flush=True)

        if seen_ids != expected_sample_ids:
            raise ValueError("Validation sample IDs or order do not match the expected split")
        if total_tokens == 0:
            raise ValueError("Validation requires valid target tokens")
        for name, parameter in model.named_parameters():
            if not torch.equal(parameters_before[name], parameter.detach().cpu()):
                raise ValueError("Validation changed model weights: " + name)
    finally:
        model.train()

    return {
        "validation_loss": total_loss / total_tokens,
        "weighted_loss_sum": total_loss,
        "valid_target_token_count": total_tokens,
        "sample_count": len(seen_ids),
        "batch_count": batch_count,
        "sample_ids": seen_ids,
        "measurement_mode": "eval with no_grad",
        "complete_validation_in_order_check_passed": True,
        "finite_logits_and_loss_check_passed": True,
        "no_gradient_graph_check_passed": True,
        "weights_unchanged_check_passed": True,
        "parameter_tensor_count_checked": len(parameters_before),
        "model_training_after_validation": model.training,
        "backward_run": False,
        "optimizer_step_run": False,
        "test_content_read": False,
    }


def capture_random_state():
    import random
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_random_state(state):
    import random
    import numpy as np
    import torch

    if len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA device count does not match this runtime")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["name"],
        np.array(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    restored = capture_random_state()
    if restored["python"] != state["python"] or restored["numpy"] != state["numpy"]:
        raise ValueError("Python or NumPy random state was not restored")
    if not torch.equal(restored["torch_cpu"], state["torch_cpu"]):
        raise ValueError("PyTorch CPU random state was not restored")
    for index in range(len(state["torch_cuda"])):
        if not torch.equal(restored["torch_cuda"][index], state["torch_cuda"][index]):
            raise ValueError("PyTorch CUDA random state was not restored")


def save_training_checkpoint(
    model, tokenizer, optimizer, checkpoint_dir, progress, config_path="configs/train.json",
    train_loader=None,
):
    from pathlib import Path
    import torch

    config = load_train_config(config_path)
    for key in ["completed_epochs", "global_step"]:
        if type(progress.get(key)) is not int or progress[key] < 0:
            raise ValueError("Checkpoint progress must contain nonnegative integer counters")
    optimizer_state = optimizer.state_dict()
    if len(optimizer_state["state"]) == 0:
        raise ValueError("Checkpoint check requires an optimizer that has completed a step")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "train_config": config,
        "progress": dict(progress),
        "optimizer_state_dict": optimizer_state,
        "random_state": capture_random_state(),
    }
    if train_loader is not None:
        state["train_loader_state"] = capture_train_loader_state(train_loader)
    torch.save(state, checkpoint_dir / "training_state.pt")


def load_training_checkpoint(checkpoint_dir, config_path="configs/train.json", train_loader=None):
    from pathlib import Path
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = load_train_config(config_path)
    if config["precision"] != "fp32" or not torch.cuda.is_available():
        raise RuntimeError("Checkpoint check requires CUDA and the approved fp32 configuration")
    checkpoint_dir = Path(checkpoint_dir)
    state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    if state["train_config"] != config:
        raise ValueError("Checkpoint configuration does not match current train configuration")
    if state.get("train_loader_state") is not None and train_loader is None:
        raise ValueError("Checkpoint requires the training DataLoader for epoch-boundary resume")
    if train_loader is not None and state.get("train_loader_state") is None:
        raise ValueError("Checkpoint does not contain training DataLoader state")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, use_fast=False, local_files_only=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir, local_files_only=True)
    model = model.to(device="cuda:0", dtype=torch.float32)
    model.eval()
    optimizer = create_optimizer(model, config_path)
    saved_optimizer = state["optimizer_state_dict"]
    optimizer.load_state_dict(saved_optimizer)
    restored_optimizer = optimizer.state_dict()
    if restored_optimizer["param_groups"] != saved_optimizer["param_groups"]:
        raise ValueError("Optimizer parameter groups were not restored")
    if set(restored_optimizer["state"]) != set(saved_optimizer["state"]):
        raise ValueError("Optimizer parameter states were not restored")
    tensor_count = 0
    step_values = set()
    for parameter_id, saved_values in saved_optimizer["state"].items():
        restored_values = restored_optimizer["state"][parameter_id]
        if set(restored_values) != set(saved_values):
            raise ValueError("Optimizer state fields were not restored")
        for key, value in saved_values.items():
            if not torch.equal(value.cpu(), restored_values[key].detach().cpu()):
                raise ValueError("Optimizer state tensor was not restored: " + key)
            tensor_count = tensor_count + 1
        step_values.add(int(restored_values["step"].item()))
    restore_random_state(state["random_state"])
    if train_loader is not None:
        restore_train_loader_state(train_loader, state["train_loader_state"])
    report = {
        "optimizer_state_check_passed": True,
        "optimizer_state_tensor_count_checked": tensor_count,
        "optimizer_parameter_state_count": len(saved_optimizer["state"]),
        "optimizer_step_values": sorted(step_values),
        "random_state_check_passed": True,
        "restored_progress": dict(state["progress"]),
        "configuration_check_passed": True,
        "train_loader_state_restored": train_loader is not None,
        "model_device": str(next(model.parameters()).device),
        "model_dtype": str(next(model.parameters()).dtype),
    }
    return model, tokenizer, optimizer, dict(state["progress"]), report


def collect_run_source(project_root, allow_uncommitted=False):
    import subprocess
    from pathlib import Path

    project_root = Path(project_root).resolve()

    def git(arguments):
        return subprocess.run(
            ["git"] + arguments,
            cwd=project_root,
            capture_output=True,
            check=True,
        ).stdout

    actual_root = Path(git(["rev-parse", "--show-toplevel"]).decode("utf-8").strip()).resolve()
    if actual_root != project_root:
        raise ValueError("Run source must be the repository root")
    repo_url = git(["remote", "get-url", "origin"]).decode("utf-8").strip()
    if repo_url != "https://github.com/MD1809/vietnamese-summarization.git":
        raise ValueError("Run source has an unexpected origin URL")
    commit = git(["rev-parse", "HEAD"]).decode("utf-8").strip()
    remote_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
        cwd=project_root,
        capture_output=True,
    )
    if remote_check.returncode != 0:
        raise ValueError("Run source commit must be available on origin/main; push and fetch first")
    status = git(["status", "--porcelain=v1", "--untracked-files=all"]).decode("utf-8")
    dirty = status != ""
    if dirty and not allow_uncommitted:
        raise ValueError("Run source has uncommitted changes; commit/push before training")
    patch_content = git(["diff", "--binary", "HEAD"])
    untracked_files = {}
    paths = git(["ls-files", "--others", "--exclude-standard", "-z"]).decode("utf-8").split("\0")
    for relative_path in paths:
        if relative_path == "":
            continue
        allowed = relative_path in ["requirements.txt", "README.md", ".gitignore"]
        for prefix in ["src/", "configs/", "tests/", "notebooks/"]:
            if relative_path.startswith(prefix):
                allowed = True
        if not allowed:
            raise ValueError("Unexpected untracked file in runtime repository: " + relative_path)
        path = (project_root / relative_path).resolve()
        path.relative_to(project_root)
        untracked_files[relative_path] = path.read_bytes()
    return {
        "repo_url": repo_url,
        "git_commit": commit,
        "commit_on_fetched_origin_main": True,
        "has_uncommitted_changes": dirty,
        "status_porcelain": status,
        "allow_uncommitted": allow_uncommitted,
    }, patch_content, untracked_files


def create_training_run(project_root, drive_root, allow_uncommitted=False):
    import json
    import platform
    import shutil
    from datetime import datetime, timezone
    from importlib.metadata import version
    from pathlib import Path
    import torch
    from src.data import _prepared_file_path
    from src.tokenization import load_tokenization_source
    from src.utils import load_generation_config

    project_root = Path(project_root).resolve()
    drive_root = Path(drive_root).resolve()
    config_path = project_root / "configs" / "train.json"
    config, splits, source_files, subset_fingerprint = load_tokenization_source(
        drive_root, config_path,
    )
    source, patch_content, untracked_files = collect_run_source(project_root, allow_uncommitted)
    if not torch.cuda.is_available():
        raise RuntimeError("Run environment requires an available CUDA GPU")
    gpus = []
    for index in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(index)
        gpus.append({
            "index": index,
            "name": gpu.name,
            "total_vram_gib": round(gpu.total_memory / 1024**3, 2),
        })
    library_versions = {}
    for package in [
        "torch", "transformers", "datasets", "sentencepiece", "rouge-score",
        "pandas", "matplotlib", "gradio", "numpy",
    ]:
        library_versions[package] = version(package)
    now = datetime.now(timezone.utc)
    run_id = "vit5_finetuning_" + now.strftime("%Y%m%d_%H%M%S_%f")
    output_dir = _prepared_file_path(drive_root, config["paths"]["output_dir"]) / run_id
    checkpoint_dir = _prepared_file_path(drive_root, config["paths"]["checkpoint_dir"]) / run_id
    report = {
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "run_status": "setup_only",
        "source": source,
        "train_config": config,
        "generation_config": load_generation_config(project_root / "configs" / "generation.json"),
        "environment": {
            "python_version": platform.python_version(),
            "library_versions": library_versions,
            "pytorch_cuda_version": torch.version.cuda,
            "cuda_available": True,
            "gpus": gpus,
        },
        "data": {
            "subset_manifest_sha256": subset_fingerprint,
            "verified_source_files": source_files,
            "train_sample_count": len(splits["train"]),
            "validation_sample_count": len(splits["validation"]),
            "train_validation_readback_check_passed": True,
            "test_content_read": False,
        },
        "paths": {
            "drive_root": str(drive_root),
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "train_file": str(_prepared_file_path(drive_root, config["paths"]["train_file"])),
            "validation_file": str(_prepared_file_path(drive_root, config["paths"]["validation_file"])),
            "test_file": str(_prepared_file_path(drive_root, config["paths"]["test_file"])),
        },
        "source_changes": {
            "patch_file": None,
            "untracked_snapshot_files": [],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    if patch_content:
        (output_dir / "source_changes.patch").write_bytes(patch_content)
        report["source_changes"]["patch_file"] = "source_changes.patch"
    for relative_path, content in untracked_files.items():
        snapshot_path = output_dir / "source_snapshot" / relative_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(content)
        report["source_changes"]["untracked_snapshot_files"].append(relative_path)
    for key in ["subset_manifest", "processed_manifest"]:
        source_path = _prepared_file_path(drive_root, config["paths"][key])
        shutil.copyfile(source_path, output_dir / source_path.name)
        if (output_dir / source_path.name).read_bytes() != source_path.read_bytes():
            raise ValueError("Run manifest copy readback failed: " + key)
        report["paths"][key] = str(source_path)
    report_path = output_dir / "config.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    restored = json.loads(report_path.read_text(encoding="utf-8"))
    if restored != report:
        raise ValueError("Saved training run profile does not match the collected information")
    if patch_content and (output_dir / "source_changes.patch").read_bytes() != patch_content:
        raise ValueError("Source patch readback failed")
    for relative_path, content in untracked_files.items():
        if (output_dir / "source_snapshot" / relative_path).read_bytes() != content:
            raise ValueError("Untracked source snapshot readback failed")
    return report


def train_accumulated_group(model, batches, optimizer, config_path="configs/train.json"):
    import math
    import torch

    config = load_train_config(config_path)
    if len(batches) == 0 or len(batches) > config["gradient_accumulation_steps"]:
        raise ValueError("Accumulation group must contain 1 to the configured number of batches")
    max_grad_norm = config.get("max_grad_norm")
    if type(max_grad_norm) not in [int, float] or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("Accumulation requires a finite positive max_grad_norm")
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32 or config["precision"] != "fp32":
        raise ValueError("Accumulation expects the approved fp32 configuration")
    token_counts = []
    sample_ids = []
    for batch in batches:
        inputs = batch["model_inputs"]
        if set(inputs) != {"input_ids", "attention_mask", "labels"}:
            raise ValueError("Accumulation expects source IDs, mask and original labels only")
        if inputs["input_ids"].shape[0] != 1 or inputs["labels"].shape[0] != 1 or len(batch["sample_ids"]) != 1:
            raise ValueError("Accumulation expects physical batch size 1")
        count = 0
        for row in inputs["labels"].tolist():
            for token_id in row:
                if token_id != -100:
                    count = count + 1
        if count == 0:
            raise ValueError("Each accumulation batch requires valid target tokens")
        token_counts.append(count)
        sample_ids.extend(batch["sample_ids"])
    total_tokens = sum(token_counts)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    loss_weights = []
    for index, batch in enumerate(batches):
        model_inputs = {}
        for field in ["input_ids", "attention_mask", "labels"]:
            model_inputs[field] = batch["model_inputs"][field].to(parameter.device)
        outputs = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
            return_dict=True,
            use_cache=False,
        )
        loss = outputs.loss
        if loss is None or len(loss.shape) != 0 or not loss.requires_grad:
            raise ValueError("Accumulation requires a scalar loss with a gradient graph")
        if not torch.isfinite(loss).all().item():
            raise ValueError("Accumulation loss contains NaN or Inf")
        if model_inputs["labels"].tolist() != batch["model_inputs"]["labels"].tolist():
            raise ValueError("Original labels changed during accumulation")
        weight = token_counts[index] / total_tokens
        losses.append(loss.item())
        loss_weights.append(weight)
        scaled_loss = loss * weight
        scaled_loss.backward()
        del outputs, loss, scaled_loss, model_inputs

    selected_parameter = None
    selected_name = None
    for name, current_parameter in model.named_parameters():
        if current_parameter.grad is None:
            continue
        if not torch.isfinite(current_parameter.grad).all().item():
            raise ValueError("Accumulated gradient contains NaN or Inf: " + name)
        if current_parameter.grad.detach().abs().max().item() == 0:
            continue
        if selected_parameter is None or current_parameter.numel() < selected_parameter.numel():
            selected_parameter = current_parameter
            selected_name = name
    if selected_parameter is None:
        raise ValueError("No parameter has a nonzero accumulated gradient")
    norm_before = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=True, foreach=False,
    ).item()
    squared_norm_after = 0.0
    for current_parameter in model.parameters():
        if current_parameter.grad is not None:
            norm = current_parameter.grad.detach().norm(2).item()
            squared_norm_after = squared_norm_after + norm**2
    norm_after = math.sqrt(squared_norm_after)
    if not math.isfinite(norm_after) or norm_after > max_grad_norm + 0.00001:
        raise ValueError("Gradient norm after clipping exceeds the configured limit")
    parameter_before = selected_parameter.detach().clone()
    optimizer.step()
    parameter_after = selected_parameter.detach()
    if not torch.isfinite(parameter_after).all().item():
        raise ValueError("Updated parameter contains NaN or Inf")
    change = (parameter_after - parameter_before).abs().max().item()
    if change == 0:
        raise ValueError("Selected parameter did not change after accumulated update")
    group_loss = 0.0
    for index in range(len(losses)):
        group_loss = group_loss + losses[index] * loss_weights[index]
    return {
        "sample_ids": sample_ids,
        "microbatch_count": len(batches),
        "valid_target_token_counts": token_counts,
        "total_valid_target_tokens": total_tokens,
        "microbatch_losses": losses,
        "loss_weights": loss_weights,
        "token_weighted_group_loss": group_loss,
        "gradient_norm_before_clipping": norm_before,
        "gradient_norm_after_clipping": norm_after,
        "max_grad_norm": max_grad_norm,
        "clipping_reduced_norm": norm_before > max_grad_norm,
        "selected_parameter": selected_name,
        "parameter_change_max_abs": change,
        "finite_gradient_check_passed": True,
        "clipping_check_passed": True,
        "parameter_changed_check_passed": True,
        "optimizer_steps": 1,
        "original_labels_check_passed": True,
    }


def capture_train_loader_state(train_loader):
    if train_loader.generator is None or train_loader.num_workers != 0:
        raise ValueError("Epoch resume requires a DataLoader generator and num_workers=0")
    sample_ids = []
    for sample in train_loader.dataset:
        sample_ids.append(sample["id"])
    if len(sample_ids) == 0 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Epoch resume requires distinct training sample IDs")
    return {
        "sample_ids": sample_ids,
        "batch_size": train_loader.batch_size,
        "drop_last": train_loader.drop_last,
        "num_workers": train_loader.num_workers,
        "sampler_class": type(train_loader.sampler).__name__,
        "generator_state": train_loader.generator.get_state(),
    }


def restore_train_loader_state(train_loader, saved_state):
    import torch

    current = capture_train_loader_state(train_loader)
    for key in ["sample_ids", "batch_size", "drop_last", "num_workers", "sampler_class"]:
        if current[key] != saved_state[key]:
            raise ValueError("Training DataLoader does not match checkpoint: " + key)
    train_loader.generator.set_state(saved_state["generator_state"])
    if not torch.equal(train_loader.generator.get_state(), saved_state["generator_state"]):
        raise ValueError("Training DataLoader generator state was not restored")


def save_last_checkpoint(
    model, tokenizer, optimizer, checkpoint_run_dir, progress, train_loader,
    config_path="configs/train.json",
):
    import shutil
    from pathlib import Path
    import torch

    if progress.get("completed_epochs", 0) < 1:
        raise ValueError("Last checkpoint must be saved after completing an epoch")
    root = Path(checkpoint_run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    last = root / "last"
    pending = root / "last_pending"
    previous = root / "last_previous"
    for path in [last, pending, previous]:
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Checkpoint path must stay inside the selected run directory")
    if pending.exists() or previous.exists():
        raise RuntimeError("Checkpoint staging remains from an interrupted save; inspect before retrying")
    save_training_checkpoint(
        model, tokenizer, optimizer, pending, progress, config_path, train_loader=train_loader,
    )
    for file_name in ["config.json", "model.safetensors", "tokenizer_config.json", "spiece.model", "training_state.pt"]:
        path = pending / file_name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Incomplete pending checkpoint: " + file_name)
    state = torch.load(pending / "training_state.pt", map_location="cpu", weights_only=True)
    if state["progress"] != progress or state["train_config"] != load_train_config(config_path):
        raise ValueError("Pending checkpoint progress or configuration readback failed")
    restore_state = capture_train_loader_state(train_loader)
    if not torch.equal(state["train_loader_state"]["generator_state"], restore_state["generator_state"]):
        raise ValueError("Pending DataLoader generator state readback failed")
    del state
    if last.exists():
        last.rename(previous)
    try:
        pending.rename(last)
    except OSError:
        if previous.exists() and not last.exists():
            previous.rename(last)
        raise
    if previous.exists():
        if previous.is_symlink() or previous.resolve() != root / "last_previous":
            raise ValueError("Cannot remove a checkpoint backup outside the selected run directory")
        shutil.rmtree(previous)
    return {
        "checkpoint_dir": str(last),
        "saved_progress": dict(progress),
        "resume_boundary": "end_of_epoch",
        "pending_readback_check_passed": True,
        "train_loader_state_saved": True,
    }


def load_last_checkpoint(checkpoint_run_dir, train_loader, config_path="configs/train.json"):
    from pathlib import Path

    root = Path(checkpoint_run_dir).resolve()
    checkpoint_dir = root / "last"
    if not checkpoint_dir.is_dir() and (root / "last_previous").is_dir():
        checkpoint_dir = root / "last_previous"
    if checkpoint_dir.is_symlink() or checkpoint_dir.resolve().parent != root:
        raise ValueError("Resume checkpoint must stay inside the selected run directory")
    model, tokenizer, optimizer, progress, report = load_training_checkpoint(
        checkpoint_dir, config_path, train_loader=train_loader,
    )
    if progress["completed_epochs"] < 1:
        raise ValueError("Resume checkpoint must represent a completed epoch")
    report["loaded_checkpoint_dir"] = str(checkpoint_dir)
    report["resume_boundary"] = "end_of_epoch"
    report["next_epoch"] = progress["completed_epochs"] + 1
    model.train()
    return model, tokenizer, optimizer, progress, report


def update_best_checkpoint(
    model, tokenizer, checkpoint_run_dir, progress, validation_report,
    config_path="configs/train.json",
):
    import json
    import math
    import shutil
    from pathlib import Path

    config = load_train_config(config_path)
    if config["checkpoint_selection"] != {"metric": "validation_loss", "mode": "min"}:
        raise ValueError("Best checkpoint selection expects minimum validation_loss")
    loss = validation_report["validation_loss"]
    if type(loss) not in [int, float] or not math.isfinite(loss) or loss < 0:
        raise ValueError("Best checkpoint requires a finite nonnegative validation loss")
    for key in ["complete_validation_in_order_check_passed", "weights_unchanged_check_passed"]:
        if validation_report.get(key) is not True:
            raise ValueError("Best selection requires a checked validation report")
    validation_ids = list(validation_report["sample_ids"])
    if len(validation_ids) == 0 or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Best selection requires distinct validation sample IDs")
    if validation_report["sample_count"] != len(validation_ids) or validation_report["valid_target_token_count"] <= 0:
        raise ValueError("Validation sample or token counts are invalid")
    for key in ["completed_epochs", "global_step"]:
        if type(progress.get(key)) is not int or progress[key] < 1:
            raise ValueError("Best selection must follow a completed training epoch")
    root = Path(checkpoint_run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    best = root / "best"
    pending = root / "best_pending"
    previous = root / "best_previous"
    for path in [best, pending, previous]:
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Best checkpoint path must stay inside the run directory")
    selected = None
    if best.exists():
        selected = json.loads((best / "selection.json").read_text(encoding="utf-8"))
        if (
            selected["train_config"] != config
            or selected["validation_sample_ids"] != validation_ids
            or selected["valid_target_token_count"] != validation_report["valid_target_token_count"]
        ):
            raise ValueError("Best selection must use the same configuration and validation split")
        if not math.isfinite(selected["validation_loss"]):
            raise ValueError("Saved best validation loss is invalid")
    elif "best_validation_loss" in progress:
        raise ValueError("Progress refers to a best checkpoint that is missing")
    updated = selected is None or loss < selected["validation_loss"]
    if updated:
        if pending.exists() or previous.exists():
            raise RuntimeError("Best staging remains from an interrupted save; inspect before retrying")
        selected = {
            "metric": "validation_loss",
            "mode": "min",
            "validation_loss": loss,
            "epoch": progress["completed_epochs"],
            "global_step": progress["global_step"],
            "validation_sample_ids": validation_ids,
            "valid_target_token_count": validation_report["valid_target_token_count"],
            "train_config": config,
        }
        pending.mkdir()
        model.save_pretrained(pending, safe_serialization=True)
        tokenizer.save_pretrained(pending)
        metadata_path = pending / "selection.json"
        metadata_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        for name in ["config.json", "model.safetensors", "tokenizer_config.json", "spiece.model", "selection.json"]:
            path = pending / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Incomplete pending best checkpoint: " + name)
        if json.loads(metadata_path.read_text(encoding="utf-8")) != selected:
            raise ValueError("Best selection metadata readback failed")
        if best.exists():
            best.rename(previous)
        try:
            pending.rename(best)
        except OSError:
            if previous.exists() and not best.exists():
                previous.rename(best)
            raise
        if previous.exists():
            if previous.is_symlink() or previous.resolve() != root / "best_previous":
                raise ValueError("Cannot remove best backup outside the run directory")
            shutil.rmtree(previous)
    updated_progress = dict(progress)
    updated_progress["best_validation_loss"] = selected["validation_loss"]
    updated_progress["best_epoch"] = selected["epoch"]
    updated_progress["best_global_step"] = selected["global_step"]
    return updated_progress, {
        "updated": updated,
        "candidate_validation_loss": loss,
        "best_validation_loss": selected["validation_loss"],
        "best_epoch": selected["epoch"],
        "best_global_step": selected["global_step"],
        "validation_sample_count": len(validation_ids),
        "checkpoint_dir": str(best),
        "strict_lower_loss_rule_check_passed": True,
    }


def load_best_checkpoint(checkpoint_run_dir, config_path="configs/train.json"):
    import json
    import math
    from pathlib import Path
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    root = Path(checkpoint_run_dir).resolve()
    best = root / "best"
    if best.is_symlink() or best.resolve().parent != root:
        raise ValueError("Best checkpoint must stay inside the run directory")
    selection = json.loads((best / "selection.json").read_text(encoding="utf-8"))
    config = load_train_config(config_path)
    if selection["train_config"] != config or not math.isfinite(selection["validation_loss"]):
        raise ValueError("Best checkpoint metadata does not match current configuration")
    if config["precision"] != "fp32" or not torch.cuda.is_available():
        raise RuntimeError("Best checkpoint check requires CUDA and fp32")
    tokenizer = AutoTokenizer.from_pretrained(best, use_fast=False, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(best, local_files_only=True)
    model = model.to(device="cuda:0", dtype=torch.float32)
    model.eval()
    return model, tokenizer, selection


def train_epoch(model, train_loader, optimizer, global_step=0, config_path="configs/train.json"):
    import time
    import torch

    config = load_train_config(config_path)
    expected_ids = capture_train_loader_state(train_loader)["sample_ids"]
    seen_ids = []
    total_loss = 0.0
    total_tokens = 0
    update_count = 0
    device = next(model.parameters()).device
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    iterator = iter(train_loader)
    while True:
        group = []
        for index in range(config["gradient_accumulation_steps"]):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            group.append(batch)
        if len(group) == 0:
            break
        report = train_accumulated_group(model, group, optimizer, config_path)
        seen_ids.extend(report["sample_ids"])
        token_count = report["total_valid_target_tokens"]
        total_loss = total_loss + report["token_weighted_group_loss"] * token_count
        total_tokens = total_tokens + token_count
        global_step = global_step + 1
        update_count = update_count + 1
        if update_count % 25 == 0:
            print("Global step:", global_step, "Mẫu đã học:", len(seen_ids), "Train loss:", total_loss / total_tokens, flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started_at
    if len(seen_ids) != len(expected_ids) or set(seen_ids) != set(expected_ids):
        raise ValueError("Training epoch omitted or repeated samples")
    if total_tokens == 0:
        raise ValueError("Training epoch requires valid target tokens")
    expected_updates = (len(train_loader) + config["gradient_accumulation_steps"] - 1) // config["gradient_accumulation_steps"]
    if update_count != expected_updates:
        raise ValueError("Training epoch optimizer update count is incorrect")
    optimizer_steps = set()
    for state in optimizer.state.values():
        optimizer_steps.add(int(state["step"].item()))
    if optimizer_steps != {global_step}:
        raise ValueError("Optimizer state does not match global_step")
    return {
        "train_loss": total_loss / total_tokens,
        "train_sample_count": len(seen_ids),
        "train_valid_target_tokens": total_tokens,
        "optimizer_updates": update_count,
        "global_step": global_step,
        "train_seconds": elapsed,
        "complete_train_samples_check_passed": True,
    }


def write_training_history(output_dir, history):
    import csv
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "training_history.csv"
    pending = output_dir / "training_history.csv.tmp"
    fields = [
        "epoch", "global_step", "train_loss", "validation_loss",
        "train_sample_count", "validation_sample_count",
        "train_valid_target_tokens", "validation_valid_target_tokens",
        "learning_rate", "train_seconds", "validation_seconds", "epoch_compute_seconds",
    ]
    with pending.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    with pending.open(encoding="utf-8", newline="") as file:
        restored = list(csv.DictReader(file))
    expected = []
    for row in history:
        strings = {}
        for field in fields:
            strings[field] = str(row[field])
        expected.append(strings)
    if restored != expected:
        raise ValueError("Training CSV readback does not match checkpoint history")
    pending.replace(target)
    return target
