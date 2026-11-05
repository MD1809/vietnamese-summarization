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


def check_optimization_step(model, batch, optimizer):
    import torch

    inputs = batch["model_inputs"]
    if set(inputs) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Optimization expects source IDs, mask and original labels only")
    if inputs["input_ids"].shape[0] != 1 or inputs["labels"].shape[0] != 1:
        raise ValueError("G4.2 expects batch size 1")
    label_rows = inputs["labels"].tolist()
    valid_tokens = 0
    for row in label_rows:
        for token_id in row:
            if token_id != -100:
                valid_tokens = valid_tokens + 1
    if valid_tokens == 0:
        raise ValueError("Loss requires at least one non-masked target token")

    first_parameter = next(model.parameters())
    if first_parameter.dtype != torch.float32:
        raise ValueError("G4.2 expects fp32 model weights")
    model_inputs = {}
    for field in ["input_ids", "attention_mask", "labels"]:
        model_inputs[field] = inputs[field].to(first_parameter.device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        labels=model_inputs["labels"],
        return_dict=True,
        use_cache=False,
    )
    loss = outputs.loss
    if loss is None or len(loss.shape) != 0:
        raise ValueError("Optimization must return a scalar loss")
    if not torch.isfinite(loss).all().item():
        raise ValueError("Optimization loss contains NaN or Inf")
    if not loss.requires_grad:
        raise ValueError("Optimization loss must have a gradient graph")
    if model_inputs["labels"].tolist() != label_rows:
        raise ValueError("Original labels changed during optimization")
    loss.backward()

    selected_name = None
    selected_parameter = None
    gradient_tensor_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient_tensor_count = gradient_tensor_count + 1
        if not torch.isfinite(parameter.grad).all().item():
            raise ValueError("Gradient contains NaN or Inf: " + name)
        if parameter.grad.detach().abs().max().item() == 0:
            continue
        if selected_parameter is None or parameter.numel() < selected_parameter.numel():
            selected_name = name
            selected_parameter = parameter
    if selected_parameter is None:
        raise ValueError("No parameter has a nonzero gradient")

    gradient_max_abs = selected_parameter.grad.detach().abs().max().item()
    parameter_before = selected_parameter.detach().clone()
    optimizer.step()
    parameter_after = selected_parameter.detach()
    if not torch.isfinite(parameter_after).all().item():
        raise ValueError("Updated parameter contains NaN or Inf")
    parameter_change = (parameter_after - parameter_before).abs().max().item()
    if parameter_change == 0:
        raise ValueError("Selected parameter did not change after optimizer.step")

    return {
        "sample_ids": list(batch["sample_ids"]),
        "model_device": str(first_parameter.device),
        "model_dtype": str(first_parameter.dtype),
        "mode": "train",
        "input_shape": list(model_inputs["input_ids"].shape),
        "labels_shape": list(model_inputs["labels"].shape),
        "loss_before_update": loss.item(),
        "valid_target_token_count": valid_tokens,
        "gradient_tensor_count": gradient_tensor_count,
        "selected_parameter": selected_name,
        "gradient_max_abs": gradient_max_abs,
        "parameter_change_max_abs": parameter_change,
        "finite_gradients_check_passed": True,
        "nonzero_gradient_check_passed": True,
        "parameter_changed_check_passed": True,
        "original_labels_check_passed": True,
        "optimizer_steps": 1,
        "gradient_accumulation_applied": False,
        "test_content_read": False,
    }


def create_small_sample_batches(train_dataset, tokenizer, seed=42):
    import random
    from src.tokenization import create_data_loader

    if len(train_dataset) < 16:
        raise ValueError("G4.3 requires at least 16 training samples")
    random_generator = random.Random(seed)
    indices = random_generator.sample(range(len(train_dataset)), 16)
    selected_dataset = train_dataset.select(indices)
    loader = create_data_loader(
        selected_dataset,
        tokenizer,
        batch_size=1,
        shuffle=False,
        seed=seed,
    )
    return list(loader)


def measure_small_sample_loss(model, batches):
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        report = check_forward_batch(model, batch)
        token_count = report["valid_target_token_count"]
        total_loss = total_loss + report["loss"] * token_count
        total_tokens = total_tokens + token_count
    if total_tokens == 0:
        raise ValueError("Cannot measure loss without valid target tokens")
    return total_loss / total_tokens


def run_small_sample_check(model, batches, optimizer):
    if len(batches) != 16:
        raise ValueError("G4.3 expects exactly 16 fixed batches")
    sample_ids = []
    source_lengths = []
    target_lengths = []
    for batch in batches:
        if len(batch["sample_ids"]) != 1:
            raise ValueError("G4.3 expects one sample per batch")
        sample_ids.append(batch["sample_ids"][0])
        source_lengths.append(batch["model_inputs"]["input_ids"].shape[1])
        target_lengths.append(batch["model_inputs"]["labels"].shape[1])
    if len(set(sample_ids)) != 16:
        raise ValueError("G4.3 requires 16 distinct sample IDs")

    optimizer.zero_grad(set_to_none=True)
    initial_loss = measure_small_sample_loss(model, batches)
    if initial_loss <= 0:
        raise ValueError("Initial loss must be positive to compare relative decrease")
    print("Loss ban đầu:", initial_loss, flush=True)
    history = []
    optimization_steps = 0
    for pass_index in range(3):
        training_step_losses = []
        for batch in batches:
            step_report = check_optimization_step(model, batch, optimizer)
            training_step_losses.append(step_report["loss_before_update"])
            optimization_steps = optimization_steps + 1
        optimizer.zero_grad(set_to_none=True)
        measured_loss = measure_small_sample_loss(model, batches)
        history.append({
            "pass": pass_index + 1,
            "optimizer_steps_completed": optimization_steps,
            "training_step_losses": training_step_losses,
            "measured_train_loss": measured_loss,
        })
        print("Lượt:", pass_index + 1, "Loss đo lại:", measured_loss, flush=True)

    final_loss = history[-1]["measured_train_loss"]
    model.train()
    return {
        "sample_ids": sample_ids,
        "sample_count": len(sample_ids),
        "source_lengths": source_lengths,
        "target_lengths": target_lengths,
        "passes": 3,
        "optimizer_steps": optimization_steps,
        "initial_measured_train_loss": initial_loss,
        "history": history,
        "final_measured_train_loss": final_loss,
        "loss_decrease": initial_loss - final_loss,
        "loss_decrease_percent": (initial_loss - final_loss) / initial_loss * 100,
        "loss_decreased": final_loss < initial_loss,
        "measurement_mode": "eval with no_grad on the same 16 train samples",
        "finite_gradients_and_parameter_updates_check_passed": True,
        "gradient_accumulation_applied": False,
        "validation_used": False,
        "test_content_read": False,
    }


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
    torch.save({
        "train_config": config,
        "progress": dict(progress),
        "optimizer_state_dict": optimizer_state,
        "random_state": capture_random_state(),
    }, checkpoint_dir / "training_state.pt")


def load_training_checkpoint(checkpoint_dir, config_path="configs/train.json"):
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
    report = {
        "optimizer_state_check_passed": True,
        "optimizer_state_tensor_count_checked": tensor_count,
        "optimizer_parameter_state_count": len(saved_optimizer["state"]),
        "optimizer_step_values": sorted(step_values),
        "random_state_check_passed": True,
        "restored_progress": dict(state["progress"]),
        "configuration_check_passed": True,
        "model_device": str(next(model.parameters()).device),
        "model_dtype": str(next(model.parameters()).dtype),
    }
    return model, tokenizer, optimizer, dict(state["progress"]), report


def generate_checkpoint_probe(model, tokenizer, source_text, config_path="configs/train.json"):
    import torch
    from src.utils import load_generation_config

    config = load_train_config(config_path)
    generation_config = load_generation_config()
    if generation_config["do_sample"] or generation_config["num_beams"] != 1:
        raise ValueError("G4.5 expects greedy decoding without sampling")
    encoded = tokenizer(
        config["task_prefix"] + source_text,
        return_tensors="pt",
        max_length=config["max_source_length"],
        truncation=True,
        padding=False,
    )
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            do_sample=False,
            num_beams=1,
            max_new_tokens=generation_config["max_new_tokens"],
        )
    generated_ids = generated[0].tolist()
    return {
        "input_ids": encoded["input_ids"][0].tolist(),
        "attention_mask": encoded["attention_mask"][0].tolist(),
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generation_config": generation_config,
    }
