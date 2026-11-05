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
