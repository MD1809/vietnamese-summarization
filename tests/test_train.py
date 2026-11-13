import copy
import json
import math
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.train import (
    check_forward_batch,
    capture_random_state,
    collect_run_source,
    create_optimizer,
    create_training_run,
    load_vit5_model,
    load_training_checkpoint,
    restore_random_state,
    train_accumulated_group,
    validate_model,
    save_training_checkpoint,
    capture_train_loader_state,
    restore_train_loader_state,
    save_last_checkpoint,
    load_last_checkpoint,
    update_best_checkpoint,
    load_best_checkpoint,
    train_epoch,
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

    def __getitem__(self, index):
        return TensorExample(self.values[index], shape=(len(self.values[index]),))


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


class ValidationTests(unittest.TestCase):
    def examples(self):
        state, torch_module, model, batch = ForwardTests().examples()
        weight = TensorExample([[1.0, 2.0]])
        weight.device = "cuda:0"
        weight.dtype = "torch.float32"
        weight.detach = Mock(return_value=weight)
        weight.cpu = Mock(return_value=weight)
        model.parameters = Mock(side_effect=self.parameters_for(weight))
        model.named_parameters = Mock(return_value=[("weight", weight)])

        def train():
            model.training = True

        def equal(left, right):
            return left.tolist() == right.tolist()

        model.train = Mock(side_effect=train)
        torch_module.equal = equal
        return torch_module, model, batch, weight

    def parameters_for(self, weight):
        def parameters():
            return iter([weight])
        return parameters

    def test_real_forward_path_uses_eval_no_grad_and_preserves_weights(self):
        torch_module, model, batch, weight = self.examples()
        with patch.dict(sys.modules, {"torch": torch_module}):
            report = validate_model(model, [batch], ["sample-a"])
        self.assertEqual(report["validation_loss"], 2.5)
        self.assertEqual(report["valid_target_token_count"], 3)
        self.assertEqual(report["parameter_tensor_count_checked"], 1)
        self.assertTrue(report["weights_unchanged_check_passed"])
        self.assertTrue(report["model_training_after_validation"])
        self.assertFalse(report["backward_run"])
        self.assertFalse(report["optimizer_step_run"])
        model.train.assert_called_once_with()
        weight.cpu.assert_called()

    def test_aggregation_uses_token_counts_instead_of_batch_count(self):
        torch_module, model, batch, weight = self.examples()
        reports = [
            {"loss": 2.0, "valid_target_token_count": 1, "sample_ids": ["a"]},
            {"loss": 4.0, "valid_target_token_count": 3, "sample_ids": ["b"]},
        ]
        with patch.dict(sys.modules, {"torch": torch_module}):
            with patch("src.train.check_forward_batch", side_effect=reports):
                report = validate_model(model, [batch, batch], ["a", "b"])
        self.assertEqual(report["validation_loss"], 3.5)
        self.assertEqual(report["weighted_loss_sum"], 14.0)
        self.assertEqual(report["valid_target_token_count"], 4)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["batch_count"], 2)

    def test_missing_repeated_or_reordered_samples_are_rejected(self):
        for expected in [["sample-a", "missing"], ["other"]]:
            torch_module, model, batch, weight = self.examples()
            with patch.dict(sys.modules, {"torch": torch_module}):
                with self.assertRaisesRegex(ValueError, "IDs or order"):
                    validate_model(model, [batch], expected)
            self.assertTrue(model.training)
        torch_module, model, batch, weight = self.examples()
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "IDs or order"):
                validate_model(model, [batch, batch], ["sample-a"])

    def test_changed_weight_is_detected_and_train_mode_is_restored(self):
        torch_module, model, batch, weight = self.examples()

        def changed_forward(model, batch):
            weight.values[0][0] = 3.0
            return {"loss": 2.5, "valid_target_token_count": 3, "sample_ids": ["sample-a"]}

        with patch.dict(sys.modules, {"torch": torch_module}):
            with patch("src.train.check_forward_batch", side_effect=changed_forward):
                with self.assertRaisesRegex(ValueError, "changed model weights"):
                    validate_model(model, [batch], ["sample-a"])
        self.assertTrue(model.training)

    def test_nan_loss_restores_train_mode_and_stops(self):
        torch_module, model, batch, weight = self.examples()
        model.output.loss.values = float("nan")
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                validate_model(model, [batch], ["sample-a"])
        self.assertTrue(model.training)


class CheckpointTests(unittest.TestCase):
    def tensor(self, value):
        tensor = Mock()
        tensor.cpu.return_value = tensor
        tensor.detach.return_value = tensor
        tensor.item.return_value = value
        return tensor

    def examples(self):
        config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
        optimizer_state = {
            "param_groups": [{"lr": 0.00005, "params": [0]}],
            "state": {0: {
                "step": self.tensor(1),
                "exp_avg": self.tensor(0.2),
                "exp_avg_sq": self.tensor(0.03),
            }},
        }
        saved = {
            "train_config": config,
            "progress": {"completed_epochs": 1, "global_step": 1},
            "optimizer_state_dict": optimizer_state,
            "random_state": {"example": "random-state"},
        }
        torch_module = types.SimpleNamespace(
            load=Mock(return_value=saved), save=Mock(),
            cuda=types.SimpleNamespace(is_available=Mock(return_value=True)),
            float32="float32", equal=Mock(return_value=True),
        )
        model = Mock()
        model.to.return_value = model
        parameter = types.SimpleNamespace(device="cuda:0", dtype="torch.float32")

        def parameters():
            return iter([parameter])

        model.parameters.side_effect = parameters
        auto_model = Mock()
        auto_model.from_pretrained.return_value = model
        auto_tokenizer = Mock()
        tokenizer = auto_tokenizer.from_pretrained.return_value
        optimizer = Mock()
        optimizer.state_dict.return_value = optimizer_state
        modules = {
            "torch": torch_module,
            "transformers": types.SimpleNamespace(
                AutoModelForSeq2SeqLM=auto_model, AutoTokenizer=auto_tokenizer,
            ),
        }
        return modules, saved, model, tokenizer, optimizer

    def test_save_contains_optimizer_progress_config_rng_and_refuses_overwrite(self):
        modules, saved, model, tokenizer, optimizer = self.examples()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "last"
            with patch.dict(sys.modules, modules):
                with patch("src.train.capture_random_state", return_value=saved["random_state"]):
                    save_training_checkpoint(model, tokenizer, optimizer, target, saved["progress"])
                    with self.assertRaises(FileExistsError):
                        save_training_checkpoint(model, tokenizer, optimizer, target, saved["progress"])
        model.save_pretrained.assert_called_once_with(target, safe_serialization=True)
        tokenizer.save_pretrained.assert_called_once_with(target)
        payload, state_path = modules["torch"].save.call_args.args
        self.assertEqual(payload, saved)
        self.assertEqual(state_path, target / "training_state.pt")

    def test_load_is_local_restores_optimizer_counters_and_rng(self):
        modules, saved, model, tokenizer, optimizer = self.examples()
        with patch.dict(sys.modules, modules):
            with patch("src.train.create_optimizer", return_value=optimizer):
                with patch("src.train.restore_random_state") as restore:
                    result = load_training_checkpoint("example/last")
        self.assertIs(result[0], model)
        self.assertIs(result[1], tokenizer)
        self.assertIs(result[2], optimizer)
        self.assertEqual(result[3], saved["progress"])
        self.assertEqual(result[4]["optimizer_step_values"], [1])
        self.assertEqual(result[4]["optimizer_state_tensor_count_checked"], 3)
        optimizer.load_state_dict.assert_called_once_with(saved["optimizer_state_dict"])
        restore.assert_called_once_with(saved["random_state"])
        modules["torch"].load.assert_called_once_with(
            Path("example/last/training_state.pt"), map_location="cpu", weights_only=True,
        )
        modules["transformers"].AutoTokenizer.from_pretrained.assert_called_once_with(
            Path("example/last"), use_fast=False, local_files_only=True,
        )

    def test_load_rejects_changed_configuration_before_loading_model(self):
        modules, saved, model, tokenizer, optimizer = self.examples()
        saved["train_config"]["learning_rate"] = 0.001
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                load_training_checkpoint("example/last")
        modules["transformers"].AutoModelForSeq2SeqLM.from_pretrained.assert_not_called()

    def test_load_rejects_optimizer_tensor_mismatch(self):
        modules, saved, model, tokenizer, optimizer = self.examples()
        modules["torch"].equal.return_value = False
        with patch.dict(sys.modules, modules):
            with patch("src.train.create_optimizer", return_value=optimizer):
                with patch("src.train.restore_random_state") as restore:
                    with self.assertRaisesRegex(ValueError, "state tensor was not restored"):
                        load_training_checkpoint("example/last")
        restore.assert_not_called()


    def test_random_state_restores_python_numpy_cpu_and_cuda(self):
        import random

        python_state = random.getstate()
        numpy_values = types.SimpleNamespace(tolist=Mock(return_value=[1, 2, 3]))
        numpy_module = types.SimpleNamespace(
            random=types.SimpleNamespace(
                get_state=Mock(return_value=("MT19937", numpy_values, 4, 0, 0.0)),
                set_state=Mock(),
            ),
            array=Mock(return_value="numpy-keys"), uint32="uint32",
        )
        torch_module = types.SimpleNamespace(
            get_rng_state=Mock(return_value="cpu-state"), set_rng_state=Mock(),
            equal=Mock(return_value=True),
            cuda=types.SimpleNamespace(
                get_rng_state_all=Mock(return_value=["cuda-state"]),
                set_rng_state_all=Mock(), device_count=Mock(return_value=1),
            ),
        )
        with patch.dict(sys.modules, {"torch": torch_module, "numpy": numpy_module}):
            saved = capture_random_state()
            random.random()
            restore_random_state(saved)
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(saved["numpy"]["keys"], [1, 2, 3])
        numpy_module.random.set_state.assert_called_once_with(("MT19937", "numpy-keys", 4, 0, 0.0))
        torch_module.set_rng_state.assert_called_once_with("cpu-state")
        torch_module.cuda.set_rng_state_all.assert_called_once_with(["cuda-state"])


class RunSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git(["init", "-b", "main"])
        self.git(["remote", "add", "origin", "https://github.com/MD1809/vietnamese-summarization.git"])
        (self.root / "src").mkdir()
        (self.root / "src" / "train.py").write_text("base\n", encoding="utf-8")
        self.git(["add", "src/train.py"])
        self.git(["-c", "user.name=Run Test", "-c", "user.email=run-test@example.invalid", "commit", "-m", "initial"])
        self.git(["update-ref", "refs/remotes/origin/main", "HEAD"])

    def git(self, arguments):
        return subprocess.run(
            ["git"] + arguments, cwd=self.root, capture_output=True, check=True,
        ).stdout

    def test_clean_source_records_actual_commit_and_has_no_patch(self):
        source, patch_content, snapshots = collect_run_source(self.root)
        self.assertEqual(source["git_commit"], self.git(["rev-parse", "HEAD"]).decode().strip())
        self.assertFalse(source["has_uncommitted_changes"])
        self.assertTrue(source["commit_on_fetched_origin_main"])
        self.assertEqual(patch_content, b"")
        self.assertEqual(snapshots, {})

    def test_dirty_source_is_rejected_by_default_and_fully_captured_for_check(self):
        (self.root / "src" / "train.py").write_text("changed\n", encoding="utf-8")
        (self.root / "src" / "new.py").write_text("new\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "uncommitted changes"):
            collect_run_source(self.root)
        source, patch_content, snapshots = collect_run_source(self.root, allow_uncommitted=True)
        self.assertTrue(source["has_uncommitted_changes"])
        self.assertIn(b"+changed", patch_content)
        self.assertEqual(snapshots, {"src/new.py": (self.root / "src" / "new.py").read_bytes()})

    def test_unpublished_commit_is_rejected(self):
        (self.root / "src" / "train.py").write_text("changed\n", encoding="utf-8")
        self.git(["add", "src/train.py"])
        self.git(["-c", "user.name=Run Test", "-c", "user.email=run-test@example.invalid", "commit", "-m", "local only"])
        with self.assertRaisesRegex(ValueError, "push and fetch"):
            collect_run_source(self.root)


class TrainingRunTests(unittest.TestCase):
    def test_profile_and_uncommitted_source_are_saved_and_read_back_without_training(self):
        config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
        source = {"git_commit": "example-sha", "has_uncommitted_changes": True}
        torch_module = types.SimpleNamespace(
            version=types.SimpleNamespace(cuda="13.0"),
            cuda=types.SimpleNamespace(
                is_available=Mock(return_value=True), device_count=Mock(return_value=1),
                get_device_properties=Mock(return_value=types.SimpleNamespace(name="Test GPU", total_memory=16*1024**3)),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive_root = root / "drive"
            manifest_root = drive_root / "data" / "manifests"
            manifest_root.mkdir(parents=True)
            for key in ["subset_manifest", "processed_manifest"]:
                (drive_root / config["paths"][key]).write_text("{}", encoding="utf-8")
            result = (config, {"train": [1, 2], "validation": [3]}, {}, "existing-fingerprint")
            with patch.dict(sys.modules, {"torch": torch_module}):
                with patch("src.tokenization.load_tokenization_source", return_value=result):
                    with patch("src.train.collect_run_source", return_value=(source, b"patch", {"src/new.py": b"new source"})):
                        with patch("importlib.metadata.version", return_value="example-version"):
                            with patch("src.utils.load_generation_config", return_value={"do_sample": False}):
                                report = create_training_run(root, drive_root, allow_uncommitted=True)
            output_dir = Path(report["paths"]["output_dir"])
            saved = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            self.assertEqual((output_dir / "source_changes.patch").read_bytes(), b"patch")
            self.assertEqual((output_dir / "source_snapshot/src/new.py").read_bytes(), b"new source")
            self.assertEqual(report["data"]["train_sample_count"], 2)
            self.assertEqual(report["train_config"], config)
            self.assertEqual(report["run_status"], "setup_only")
            self.assertFalse(report["data"]["test_content_read"])
            self.assertFalse(Path(report["paths"]["checkpoint_dir"]).exists())


class TrainingDeviceFixture:
    def examples(self):
        parameter = types.SimpleNamespace(
            device=types.SimpleNamespace(type="cuda"), dtype="float32",
        )
        model = Mock()

        def parameters():
            return iter([parameter])

        model.parameters.side_effect = parameters
        cuda = types.SimpleNamespace(
            synchronize=Mock(),
        )
        return types.SimpleNamespace(cuda=cuda, float32="float32"), model


class AccumulationTests(unittest.TestCase):
    def examples(self, counts):
        torch_module, model, batch, optimizer, parameter, loss, events = OptimizationTests().examples()

        def parameters():
            return iter([parameter])

        model.parameters.side_effect = parameters
        parameter.grad.detach.return_value.norm.return_value.item.return_value = 0.5
        clip = Mock()
        clip.return_value.item.return_value = 2.0
        torch_module.nn = types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=clip))
        events.attach_mock(clip, "clip")
        batches = []
        losses = []
        for index, count in enumerate(counts):
            current = copy.deepcopy(batch)
            current["sample_ids"] = ["sample-" + str(index)]
            current["model_inputs"]["labels"] = TensorExample([[8]*count + [-100]])
            batches.append(current)
            current_loss = Mock()
            current_loss.shape = ()
            current_loss.finite = True
            current_loss.requires_grad = True
            current_loss.item.return_value = 2.0 + index
            current_loss.__mul__ = Mock(return_value=Mock())
            events.attach_mock(current_loss.__mul__.return_value.backward, "backward_" + str(index))
            losses.append(current_loss)
        outputs = []
        for current_loss in losses:
            outputs.append(types.SimpleNamespace(loss=current_loss))
        model.side_effect = outputs
        return torch_module, model, batches, optimizer, parameter, losses, events

    def test_group_uses_token_weights_and_only_one_update_after_all_backwards(self):
        torch_module, model, batches, optimizer, parameter, losses, events = self.examples([1, 3])
        with patch.dict(sys.modules, {"torch": torch_module}):
            report = train_accumulated_group(model, batches, optimizer)
        losses[0].__mul__.assert_called_once_with(0.25)
        losses[1].__mul__.assert_called_once_with(0.75)
        self.assertEqual(report["token_weighted_group_loss"], 2.75)
        operations = []
        for call in events.mock_calls:
            if call[0] in ["zero_grad", "backward_0", "backward_1", "clip", "step"]:
                operations.append(call[0])
        self.assertEqual(operations, ["zero_grad", "backward_0", "backward_1", "clip", "step"])
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        optimizer.step.assert_called_once_with()
        self.assertEqual(report["gradient_norm_after_clipping"], 0.5)
        self.assertEqual(torch_module.nn.utils.clip_grad_norm_.call_args.kwargs["max_norm"], 1.0)

    def test_single_batch_tail_is_not_divided_by_eight(self):
        torch_module, model, batches, optimizer, parameter, losses, events = self.examples([3])
        with patch.dict(sys.modules, {"torch": torch_module}):
            report = train_accumulated_group(model, batches, optimizer)
        losses[0].__mul__.assert_called_once_with(1.0)
        self.assertEqual(report["loss_weights"], [1.0])
        self.assertEqual(report["optimizer_steps"], 1)

    def test_nonfinite_loss_or_gradient_does_not_update_weights(self):
        for kind in ["loss", "gradient"]:
            torch_module, model, batches, optimizer, parameter, losses, events = self.examples([3])
            if kind == "loss":
                losses[0].item.return_value = float("nan")
            else:
                parameter.grad.finite = False
            with patch.dict(sys.modules, {"torch": torch_module}):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    train_accumulated_group(model, batches, optimizer)
            optimizer.step.assert_not_called()

    def test_invalid_norm_after_clipping_blocks_optimizer_step(self):
        torch_module, model, batches, optimizer, parameter, losses, events = self.examples([3])
        parameter.grad.detach.return_value.norm.return_value.item.return_value = 2.0
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "after clipping"):
                train_accumulated_group(model, batches, optimizer)
        optimizer.step.assert_not_called()


class EpochCheckpointTests(unittest.TestCase):
    def loader(self):
        generator = Mock()
        generator.get_state.return_value = "saved-generator-state"
        return types.SimpleNamespace(
            generator=generator, num_workers=0, batch_size=1, drop_last=False,
            sampler=types.SimpleNamespace(), dataset=[{"id": "a"}, {"id": "b"}],
        )

    def test_loader_identity_and_generator_are_restored_without_iteration(self):
        loader = self.loader()
        saved = capture_train_loader_state(loader)
        loader.generator.get_state.return_value = "different-state"
        torch_module = types.SimpleNamespace(equal=Mock(return_value=True))
        with patch.dict(sys.modules, {"torch": torch_module}):
            restore_train_loader_state(loader, saved)
        loader.generator.set_state.assert_called_once_with("saved-generator-state")
        torch_module.equal.assert_called_once_with("different-state", "saved-generator-state")

    def test_changed_training_sample_order_is_rejected_before_setting_rng(self):
        loader = self.loader()
        saved = capture_train_loader_state(loader)
        loader.dataset.reverse()
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError, "sample_ids"):
                restore_train_loader_state(loader, saved)
        loader.generator.set_state.assert_not_called()

    def test_failed_pending_write_preserves_existing_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / "last"
            last.mkdir()
            (last / "existing.txt").write_text("previous checkpoint", encoding="utf-8")
            with patch.dict(sys.modules, {"torch": types.SimpleNamespace()}):
                with patch("src.train.save_training_checkpoint", side_effect=OSError("write interrupted")):
                    with self.assertRaisesRegex(OSError, "write interrupted"):
                        save_last_checkpoint(Mock(), Mock(), Mock(), root, {"completed_epochs": 1, "global_step": 3}, self.loader())
            self.assertEqual((last / "existing.txt").read_text(encoding="utf-8"), "previous checkpoint")

    def test_complete_pending_replaces_last_and_cleans_only_backup(self):
        loader = self.loader()
        progress = {"completed_epochs": 2, "global_step": 6}
        saved = {
            "progress": progress,
            "train_config": json.loads(Path("configs/train.json").read_text(encoding="utf-8")),
            "train_loader_state": capture_train_loader_state(loader),
        }

        def write_pending(model, tokenizer, optimizer, target, progress, config_path, train_loader):
            target.mkdir()
            for name in ["config.json", "model.safetensors", "tokenizer_config.json", "spiece.model", "training_state.pt"]:
                (target / name).write_bytes(b"checkpoint file")

        torch_module = types.SimpleNamespace(load=Mock(return_value=saved), equal=Mock(return_value=True))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last").mkdir()
            (root / "last" / "old.txt").write_bytes(b"old")
            (root / "unrelated.txt").write_bytes(b"keep")
            with patch.dict(sys.modules, {"torch": torch_module}):
                with patch("src.train.save_training_checkpoint", side_effect=write_pending):
                    report = save_last_checkpoint(Mock(), Mock(), Mock(), root, progress, loader)
            self.assertTrue((root / "last" / "training_state.pt").is_file())
            self.assertFalse((root / "last_previous").exists())
            self.assertFalse((root / "last_pending").exists())
            self.assertEqual((root / "unrelated.txt").read_bytes(), b"keep")
            self.assertEqual(report["saved_progress"], progress)

    def test_loader_checkpoint_requires_loader_on_resume(self):
        modules, saved, model, tokenizer, optimizer = CheckpointTests().examples()
        saved["train_loader_state"] = {"example": "state"}
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "requires the training DataLoader"):
                load_training_checkpoint("example/last")
        modules["transformers"].AutoModelForSeq2SeqLM.from_pretrained.assert_not_called()

    def test_resume_can_find_previous_checkpoint_after_interrupted_directory_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "last_previous"
            previous.mkdir()
            model = Mock()
            result = (model, "tokenizer", "optimizer", {"completed_epochs": 1, "global_step": 3}, {})
            with patch("src.train.load_training_checkpoint", return_value=result) as load:
                restored = load_last_checkpoint(root, self.loader())
            self.assertEqual(load.call_args.args[0], previous)
            self.assertEqual(restored[-1]["next_epoch"], 2)
            model.train.assert_called_once_with()


class BestCheckpointTests(unittest.TestCase):
    def examples(self):
        model = Mock()
        tokenizer = Mock()
        version = {"value": b"model-one"}

        def save_model(path, safe_serialization):
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(version["value"])

        def save_tokenizer(path):
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (path / "spiece.model").write_bytes(b"tokenizer")

        model.save_pretrained.side_effect = save_model
        tokenizer.save_pretrained.side_effect = save_tokenizer
        return model, tokenizer, version

    def validation(self, loss):
        return {
            "validation_loss": loss,
            "sample_ids": ["validation-a", "validation-b"],
            "sample_count": 2,
            "valid_target_token_count": 7,
            "complete_validation_in_order_check_passed": True,
            "weights_unchanged_check_passed": True,
        }

    def test_first_save_and_strict_improvement_keep_correct_epoch_and_weights(self):
        model, tokenizer, version = self.examples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress, first = update_best_checkpoint(model, tokenizer, root, {"completed_epochs": 1, "global_step": 3}, self.validation(2.0))
            self.assertTrue(first["updated"])
            self.assertEqual(progress["best_epoch"], 1)
            version["value"] = b"model-two"
            progress["completed_epochs"] = 2
            progress["global_step"] = 6
            progress, second = update_best_checkpoint(model, tokenizer, root, progress, self.validation(1.0))
            self.assertTrue(second["updated"])
            self.assertEqual(progress["best_validation_loss"], 1.0)
            self.assertEqual(progress["best_epoch"], 2)
            self.assertEqual(progress["best_global_step"], 6)
            self.assertEqual((root / "best/model.safetensors").read_bytes(), b"model-two")
            self.assertFalse((root / "best_previous").exists())
            self.assertFalse((root / "best_pending").exists())

    def test_equal_or_higher_loss_preserves_files_and_original_best_progress(self):
        model, tokenizer, version = self.examples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress, first = update_best_checkpoint(model, tokenizer, root, {"completed_epochs": 1, "global_step": 3}, self.validation(2.0))
            original_metadata = (root / "best/selection.json").read_bytes()
            version["value"] = b"must-not-be-saved"
            progress["completed_epochs"] = 2
            progress["global_step"] = 6
            for loss in [2.0, 3.0]:
                progress, result = update_best_checkpoint(model, tokenizer, root, progress, self.validation(loss))
                self.assertFalse(result["updated"])
                self.assertEqual(progress["best_epoch"], 1)
                self.assertEqual(progress["best_global_step"], 3)
                self.assertEqual((root / "best/selection.json").read_bytes(), original_metadata)
                self.assertEqual((root / "best/model.safetensors").read_bytes(), b"model-one")
        self.assertEqual(model.save_pretrained.call_count, 1)

    def test_nonfinite_or_negative_loss_is_rejected_before_saving(self):
        model, tokenizer, version = self.examples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for loss in [float("nan"), float("inf"), -1.0]:
                with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                    update_best_checkpoint(model, tokenizer, root, {"completed_epochs": 1, "global_step": 3}, self.validation(loss))
            self.assertFalse((root / "best").exists())
        model.save_pretrained.assert_not_called()

    def test_different_validation_split_is_rejected(self):
        model, tokenizer, version = self.examples()
        with tempfile.TemporaryDirectory() as directory:
            progress, first = update_best_checkpoint(model, tokenizer, directory, {"completed_epochs": 1, "global_step": 3}, self.validation(2.0))
            report = self.validation(1.0)
            report["sample_ids"] = ["other-a", "other-b"]
            with self.assertRaisesRegex(ValueError, "same configuration and validation split"):
                update_best_checkpoint(model, tokenizer, directory, progress, report)

    def test_failed_write_preserves_previous_best(self):
        model, tokenizer, version = self.examples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress, first = update_best_checkpoint(model, tokenizer, root, {"completed_epochs": 1, "global_step": 3}, self.validation(2.0))
            model.save_pretrained.side_effect = OSError("write interrupted")
            with self.assertRaisesRegex(OSError, "write interrupted"):
                update_best_checkpoint(model, tokenizer, root, progress, self.validation(1.0))
            saved = json.loads((root / "best/selection.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["validation_loss"], 2.0)
            self.assertEqual((root / "best/model.safetensors").read_bytes(), b"model-one")

    def test_best_loads_locally_in_fp32_for_evaluation(self):
        model, tokenizer, version = self.examples()
        modules, saved, loaded_model, loaded_tokenizer, optimizer = CheckpointTests().examples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            update_best_checkpoint(model, tokenizer, root, {"completed_epochs": 1, "global_step": 3}, self.validation(2.0))
            with patch.dict(sys.modules, modules):
                result = load_best_checkpoint(root)
            self.assertIs(result[0], loaded_model)
            self.assertIs(result[1], loaded_tokenizer)
            self.assertEqual(result[2]["epoch"], 1)
            loaded_model.to.assert_called_once_with(device="cuda:0", dtype="float32")
            loaded_model.eval.assert_called_once_with()
            modules["transformers"].AutoModelForSeq2SeqLM.from_pretrained.assert_called_once_with(root / "best", local_files_only=True)


class FullTrainingTests(unittest.TestCase):
    def loader(self, count):
        batches = []
        for index in range(count):
            batches.append({"sample_ids": ["train-" + str(index)]})
        loader = Mock()
        loader.dataset = []
        for batch in batches:
            loader.dataset.append({"id": batch["sample_ids"][0]})

        def iterator():
            return iter(batches)

        loader.__iter__ = Mock(side_effect=iterator)
        loader.__len__ = Mock(return_value=count)
        loader.generator = Mock()
        loader.num_workers = 0
        loader.batch_size = 1
        loader.drop_last = False
        return loader

    def test_epoch_streams_full_group_and_tail_and_counts_tokens_correctly(self):
        loader = self.loader(10)
        optimizer = types.SimpleNamespace(state={})
        torch_module, model = TrainingDeviceFixture().examples()
        sizes = []

        def train_group(model, group, optimizer, config_path):
            sizes.append(len(group))
            step = Mock()
            step.item.return_value = len(sizes)
            optimizer.state["example"] = {"step": step}
            ids = []
            for batch in group:
                ids.extend(batch["sample_ids"])
            return {"sample_ids": ids, "total_valid_target_tokens": len(group), "token_weighted_group_loss": float(len(sizes))}

        with patch.dict(sys.modules, {"torch": torch_module}):
            with patch("src.train.train_accumulated_group", side_effect=train_group):
                report = train_epoch(model, loader, optimizer)
        self.assertEqual(sizes, [8, 2])
        self.assertEqual(report["global_step"], 2)
        self.assertEqual(report["train_sample_count"], 10)
        self.assertEqual(report["train_valid_target_tokens"], 10)
        self.assertEqual(report["train_loss"], 1.2)


if __name__ == "__main__":
    unittest.main()
