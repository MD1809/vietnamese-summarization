import copy
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.train import (
    check_forward_batch,
    check_optimization_step,
    create_optimizer,
    create_small_sample_batches,
    load_vit5_model,
    measure_small_sample_loss,
    run_small_sample_check,
)


class TensorExample:
    def __init__(self, values, shape=None):
        self.values = copy.deepcopy(values)
        self.requires_grad = False
        self.device = "cpu"
        self.finite = True
        if shape is None:
            self.shape = (len(values), len(values[0]))
        else:
            self.shape = shape

    def to(self, device):
        moved = TensorExample(self.values, self.shape)
        moved.device = device
        return moved

    def clone(self):
        return TensorExample(self.values, self.shape)

    def tolist(self):
        return copy.deepcopy(self.values)

    def item(self):
        return self.values


class NoGradientExample:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        self.state["no_grad"] = True

    def __exit__(self, error_type, error, traceback):
        self.state["no_grad"] = False


class ModelExample:
    def __init__(self, state):
        self.state = state
        self.config = types.SimpleNamespace(vocab_size=16, decoder_start_token_id=0, pad_token_id=0)
        self.training = True
        self.call_kwargs = None
        self.decoder_override = None
        self.output = types.SimpleNamespace(
            loss=TensorExample(2.5, shape=()),
            logits=TensorExample(None, shape=(1, 5, 16)),
        )

    def parameters(self):
        return iter([types.SimpleNamespace(device="cuda:0", dtype="torch.float32")])

    def eval(self):
        self.training = False

    def prepare_decoder_input_ids_from_labels(self, labels):
        if self.decoder_override is not None:
            return self.decoder_override
        return TensorExample([[0, 8, 9, 1, 0]])

    def __call__(self, **kwargs):
        if self.training or not self.state["no_grad"]:
            raise AssertionError("Forward must run in eval/no_grad")
        self.call_kwargs = kwargs
        return self.output


class ForwardTests(unittest.TestCase):
    def examples(self):
        state = {"no_grad": False}

        def isfinite(tensor):
            finite = tensor.finite
            if tensor.shape == ():
                finite = finite and math.isfinite(tensor.item())
            result = Mock()
            result.all.return_value.item.return_value = finite
            return result

        def no_grad():
            return NoGradientExample(state)

        torch_module = types.SimpleNamespace(
            no_grad=Mock(side_effect=no_grad),
            isfinite=isfinite,
            cuda=types.SimpleNamespace(is_available=Mock(return_value=True)),
            float32="float32",
        )
        batch = {
            "sample_ids": ["sample-a"],
            "group_ids": ["url-a"],
            "model_inputs": {
                "input_ids": TensorExample([[10, 11, 1]]),
                "attention_mask": TensorExample([[1, 1, 1]]),
                "labels": TensorExample([[8, 9, 1, -100, -100]]),
            },
        }
        return state, torch_module, ModelExample(state), batch

    def test_forward_uses_original_labels_and_eval_no_grad(self):
        state, torch_module, model, batch = self.examples()
        original = batch["model_inputs"]["labels"].tolist()
        with patch.dict(sys.modules, {"torch": torch_module}):
            report = check_forward_batch(model, batch)
        self.assertEqual(report["logits_shape"], [1, 5, 16])
        self.assertEqual(report["loss"], 2.5)
        self.assertEqual(report["valid_target_token_count"], 3)
        self.assertEqual(report["masked_label_count"], 2)
        self.assertEqual(report["decoder_input_ids_preview"], [0, 8, 9, 1, 0])
        self.assertEqual(model.call_kwargs["labels"].tolist(), original)
        self.assertEqual(batch["model_inputs"]["labels"].tolist(), original)
        self.assertNotIn("decoder_input_ids", model.call_kwargs)
        self.assertNotIn("sample_ids", model.call_kwargs)
        self.assertIs(model.call_kwargs["use_cache"], False)
        self.assertIs(report["training_memory_verified"], False)
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(state["no_grad"])

    def test_non_finite_loss_and_logits_are_rejected(self):
        for kind in ["loss", "logits"]:
            state, torch_module, model, batch = self.examples()
            if kind == "loss":
                model.output.loss.values = float("nan")
            else:
                model.output.logits.finite = False
            with patch.dict(sys.modules, {"torch": torch_module}):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    check_forward_batch(model, batch)

    def test_wrong_logits_shape_is_rejected(self):
        state, torch_module, model, batch = self.examples()
        model.output.logits.shape = (1, 4, 16)
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "Logits shape"):
                check_forward_batch(model, batch)

    def test_wrong_decoder_shift_is_rejected_before_forward(self):
        state, torch_module, model, batch = self.examples()
        model.decoder_override = TensorExample([[8, 9, 1, 0, 0]])
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "right shift"):
                check_forward_batch(model, batch)
        self.assertIsNone(model.call_kwargs)

    def test_pre_shifted_decoder_inputs_and_all_masked_labels_are_rejected(self):
        state, torch_module, model, batch = self.examples()
        batch["model_inputs"]["decoder_input_ids"] = TensorExample([[0, 8, 9, 1, 0]])
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "original labels only"):
                check_forward_batch(model, batch)
        del batch["model_inputs"]["decoder_input_ids"]
        batch["model_inputs"]["labels"] = TensorExample([[-100] * 5])
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "non-masked target"):
                check_forward_batch(model, batch)
        self.assertIsNone(model.call_kwargs)

    def test_missing_scalar_loss_and_gradient_graph_are_rejected(self):
        for kind in ["none", "vector", "gradient"]:
            state, torch_module, model, batch = self.examples()
            if kind == "none":
                model.output.loss = None
            elif kind == "vector":
                model.output.loss.shape = (1,)
            else:
                model.output.loss.requires_grad = True
            with patch.dict(sys.modules, {"torch": torch_module}):
                with self.assertRaises(ValueError):
                    check_forward_batch(model, batch)

    def test_model_loader_uses_pinned_revision_cuda_and_fp32(self):
        state, torch_module, model, batch = self.examples()
        loaded = Mock()
        loaded.to.return_value = loaded
        auto = Mock()
        auto.from_pretrained.return_value = loaded
        modules = {"torch": torch_module, "transformers": types.SimpleNamespace(AutoModelForSeq2SeqLM=auto)}
        with patch.dict(sys.modules, modules):
            result = load_vit5_model()
        config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
        auto.from_pretrained.assert_called_once_with(config["model"]["id"], revision=config["model"]["revision"])
        loaded.to.assert_called_once_with(device="cuda:0", dtype="float32")
        loaded.eval.assert_called_once()
        self.assertIs(result, loaded)

    def test_loader_rejects_missing_gpu_and_unapproved_precision(self):
        state, torch_module, model, batch = self.examples()
        auto = Mock()
        modules = {"torch": torch_module, "transformers": types.SimpleNamespace(AutoModelForSeq2SeqLM=auto)}
        torch_module.cuda.is_available.return_value = False
        with patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError):
                load_vit5_model()
        config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
        config["precision"] = "fp16"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, "fp32"):
                    load_vit5_model(path)
        auto.from_pretrained.assert_not_called()


class OptimizationTests(unittest.TestCase):
    def examples(self):
        state, torch_module, forward_model, batch = ForwardTests().examples()
        parameter = Mock()
        parameter.device = "cuda:0"
        parameter.dtype = "float32"
        parameter.finite = True
        parameter.shape = (3,)
        parameter.numel.return_value = 3
        parameter.grad.finite = True
        parameter.grad.shape = (3,)
        parameter.grad.detach.return_value.abs.return_value.max.return_value.item.return_value = 0.3
        after = Mock()
        after.finite = True
        after.shape = (3,)
        difference = Mock()
        difference.abs.return_value.max.return_value.item.return_value = 0.00005
        after.__sub__ = Mock(return_value=difference)
        parameter.detach.return_value = after
        loss = Mock()
        loss.shape = ()
        loss.finite = True
        loss.requires_grad = True
        loss.item.return_value = 2.5
        model = Mock()
        model.parameters.side_effect = [iter([parameter])]
        model.named_parameters.return_value = [("layer_norm.weight", parameter)]
        model.return_value.loss = loss
        optimizer = Mock()
        events = Mock()
        events.attach_mock(model.train, "train")
        events.attach_mock(optimizer.zero_grad, "zero_grad")
        events.attach_mock(model, "forward")
        events.attach_mock(loss.backward, "backward")
        events.attach_mock(optimizer.step, "step")
        return torch_module, model, batch, optimizer, parameter, loss, events

    def test_single_step_order_original_labels_and_update_report(self):
        torch_module, model, batch, optimizer, parameter, loss, events = self.examples()
        with patch.dict(sys.modules, {"torch": torch_module}):
            report = check_optimization_step(model, batch, optimizer)
        operations = []
        for call in events.mock_calls:
            if call[0] in ["train", "zero_grad", "forward", "backward", "step"]:
                operations.append(call[0])
        self.assertEqual(operations, ["train", "zero_grad", "forward", "backward", "step"])
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        optimizer.step.assert_called_once_with()
        self.assertEqual(model.call_args.kwargs["labels"].tolist(), [[8, 9, 1, -100, -100]])
        self.assertNotIn("decoder_input_ids", model.call_args.kwargs)
        self.assertIs(model.call_args.kwargs["use_cache"], False)
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["gradient_max_abs"], 0.3)
        self.assertGreater(report["parameter_change_max_abs"], 0)
        self.assertFalse(report["gradient_accumulation_applied"])

    def test_invalid_loss_stops_before_backward_or_step(self):
        for kind in ["nan", "no_graph", "missing"]:
            torch_module, model, batch, optimizer, parameter, loss, events = self.examples()
            if kind == "nan":
                loss.item.return_value = float("nan")
            elif kind == "no_graph":
                loss.requires_grad = False
            else:
                model.return_value.loss = None
            with patch.dict(sys.modules, {"torch": torch_module}):
                with self.assertRaises(ValueError):
                    check_optimization_step(model, batch, optimizer)
            loss.backward.assert_not_called()
            optimizer.step.assert_not_called()

    def test_invalid_or_missing_gradients_prevent_update(self):
        for kind in ["infinite", "zero", "missing"]:
            torch_module, model, batch, optimizer, parameter, loss, events = self.examples()
            if kind == "infinite":
                parameter.grad.finite = False
            elif kind == "zero":
                parameter.grad.detach.return_value.abs.return_value.max.return_value.item.return_value = 0
            else:
                parameter.grad = None
            with patch.dict(sys.modules, {"torch": torch_module}):
                with self.assertRaises(ValueError):
                    check_optimization_step(model, batch, optimizer)
            loss.backward.assert_called_once_with()
            optimizer.step.assert_not_called()

    def test_unchanged_parameter_is_rejected_after_step(self):
        torch_module, model, batch, optimizer, parameter, loss, events = self.examples()
        difference = parameter.detach.return_value.__sub__.return_value
        difference.abs.return_value.max.return_value.item.return_value = 0
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "did not change"):
                check_optimization_step(model, batch, optimizer)
        optimizer.step.assert_called_once_with()

    def test_optimizer_uses_approved_configuration(self):
        torch_module = types.SimpleNamespace(optim=types.SimpleNamespace(AdamW=Mock()))
        model = Mock()
        with patch.dict(sys.modules, {"torch": torch_module}):
            optimizer = create_optimizer(model)
        torch_module.optim.AdamW.assert_called_once_with(
            model.parameters.return_value,
            lr=0.00005,
            weight_decay=0.01,
            foreach=False,
        )
        self.assertIs(optimizer, torch_module.optim.AdamW.return_value)


class SmallSampleTests(unittest.TestCase):
    def batches(self):
        batches = []
        for index in range(16):
            batches.append({
                "sample_ids": ["sample-" + str(index)],
                "model_inputs": {
                    "input_ids": TensorExample([[10, 11, 1]]),
                    "labels": TensorExample([[8, 9, 1]]),
                },
            })
        return batches

    def test_selection_is_repeatable_distinct_and_not_first_16(self):
        dataset = Mock()
        dataset.__len__ = Mock(return_value=2000)
        selected_indices = []
        with patch("src.tokenization.create_data_loader", return_value=["batch"]) as loader:
            for repetition in range(2):
                result = create_small_sample_batches(dataset, "tokenizer", seed=42)
                selected_indices.append(dataset.select.call_args.args[0])
                self.assertEqual(result, ["batch"])
            loader.assert_called_with(
                dataset.select.return_value, "tokenizer", batch_size=1, shuffle=False, seed=42,
            )
        self.assertEqual(selected_indices[0], selected_indices[1])
        self.assertEqual(len(set(selected_indices[0])), 16)
        self.assertNotEqual(selected_indices[0], list(range(16)))

    def test_measurement_weights_loss_by_valid_target_tokens(self):
        reports = [
            {"loss": 2.0, "valid_target_token_count": 1},
            {"loss": 4.0, "valid_target_token_count": 3},
        ]
        with patch("src.train.check_forward_batch", side_effect=reports):
            loss = measure_small_sample_loss(Mock(), ["batch-a", "batch-b"])
        self.assertEqual(loss, 3.5)

    def test_three_passes_reuse_same_samples_and_keep_only_scalar_losses(self):
        batches = self.batches()
        model = Mock()
        with patch("src.train.measure_small_sample_loss", side_effect=[8.0, 6.0, 5.0, 4.0]):
            with patch("src.train.check_optimization_step", return_value={"loss_before_update": 7.0}) as step:
                with patch("builtins.print"):
                    report = run_small_sample_check(model, batches, Mock())
        self.assertEqual(step.call_count, 48)
        for index, call in enumerate(step.call_args_list):
            self.assertIs(call.args[1], batches[index % 16])
        self.assertEqual(report["optimizer_steps"], 48)
        self.assertEqual(report["loss_decrease_percent"], 50.0)
        self.assertTrue(report["loss_decreased"])
        self.assertFalse(report["validation_used"])
        self.assertFalse(report["test_content_read"])
        self.assertEqual(len(report["history"]), 3)
        model.train.assert_called_once_with()

    def test_no_decrease_is_reported_without_claiming_success(self):
        with patch("src.train.measure_small_sample_loss", side_effect=[8.0, 8.5, 9.0, 9.5]):
            with patch("src.train.check_optimization_step", return_value={"loss_before_update": 9.0}):
                with patch("builtins.print"):
                    report = run_small_sample_check(Mock(), self.batches(), Mock())
        self.assertFalse(report["loss_decreased"])
        self.assertLess(report["loss_decrease_percent"], 0)

    def test_duplicate_samples_are_rejected_before_measurement_or_learning(self):
        batches = self.batches()
        batches[-1]["sample_ids"] = batches[0]["sample_ids"]
        with patch("src.train.measure_small_sample_loss") as measure:
            with patch("src.train.check_optimization_step") as step:
                with self.assertRaisesRegex(ValueError, "distinct"):
                    run_small_sample_check(Mock(), batches, Mock())
        measure.assert_not_called()
        step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
