import copy
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.train import check_forward_batch, load_vit5_model


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


if __name__ == "__main__":
    unittest.main()
