import sys
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from src.inference import load_summarizer, summarize_text
from src.utils import load_generation_config


class InferenceTests(unittest.TestCase):
    def examples(self, full_length=3, used_length=3, device_type="cpu"):
        device = types.SimpleNamespace(type=device_type)
        model = Mock()
        model.parameters.return_value = iter([types.SimpleNamespace(device=device)])
        source_ids = Mock()
        source_ids.tolist.return_value = [[7] * used_length]
        source_ids.to.return_value = source_ids
        mask = Mock()
        mask.to.return_value = mask
        generated = Mock()
        generated.tolist.return_value = [[0, 8, 1]]
        model.generate.return_value = generated
        tokenizer = Mock()
        tokenizer.side_effect = [
            {"input_ids": [7] * full_length},
            {"input_ids": source_ids, "attention_mask": mask},
        ]
        tokenizer.decode.return_value = "  Bản tóm tắt.  "
        context = MagicMock()
        torch_module = types.SimpleNamespace(
            no_grad=Mock(return_value=context),
            cuda=types.SimpleNamespace(synchronize=Mock()),
        )
        return model, tokenizer, torch_module, context

    def test_normalization_generation_without_labels_and_decode(self):
        model, tokenizer, torch_module, context = self.examples()
        with patch.dict(sys.modules, {"torch": torch_module}):
            report = summarize_text("  Tôi\n\thọc.  ", model, tokenizer)
        self.assertEqual(report["normalized_text"], "Tôi học.")
        self.assertEqual(report["summary"], "Bản tóm tắt.")
        self.assertFalse(report["was_truncated"])
        self.assertEqual(tokenizer.call_args_list[0].args, ("Tôi học.",))
        self.assertEqual(tokenizer.call_args_list[1].kwargs["max_length"], 1024)
        kwargs = model.generate.call_args.kwargs
        self.assertNotIn("labels", kwargs)
        self.assertIs(kwargs["do_sample"], False)
        self.assertEqual(kwargs["num_beams"], 4)
        self.assertEqual(kwargs["max_new_tokens"], 128)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 0)
        model.eval.assert_called_once_with()
        context.__enter__.assert_called_once()
        context.__exit__.assert_called_once()
        tokenizer.decode.assert_called_once_with([0, 8, 1], skip_special_tokens=True)

    def test_no_repeat_ngram_setting_reaches_generation_and_report(self):
        model, tokenizer, torch_module, context = self.examples()
        generation = {"num_beams": 1, "max_new_tokens": 128, "do_sample": False, "no_repeat_ngram_size": 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.json"
            path.write_text(json.dumps(generation), encoding="utf-8")
            with patch.dict(sys.modules, {"torch": torch_module}):
                result = summarize_text("Nguồn.", model, tokenizer, generation_config_path=path)
        self.assertEqual(model.generate.call_args.kwargs["no_repeat_ngram_size"], 4)
        self.assertEqual(result["generation_config"], generation)

    def test_no_repeat_ngram_requires_a_nonnegative_integer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.json"
            for value in [-1, 1.5, True, "4", None]:
                config = {"num_beams": 1, "max_new_tokens": 128, "do_sample": False, "no_repeat_ngram_size": value}
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        load_generation_config(path)
            for value in [0, 4]:
                config["no_repeat_ngram_size"] = value
                path.write_text(json.dumps(config), encoding="utf-8")
                self.assertEqual(load_generation_config(path), config)

    def test_truncation_counts_include_the_full_and_used_source(self):
        model, tokenizer, torch_module, context = self.examples(1300, 1024)
        with patch.dict(sys.modules, {"torch": torch_module}):
            report = summarize_text("Văn bản dài.", model, tokenizer)
        self.assertEqual(report["input_token_count"], 1300)
        self.assertEqual(report["used_input_token_count"], 1024)
        self.assertTrue(report["was_truncated"])

    def test_empty_input_is_rejected_before_tokenization_or_generation(self):
        model, tokenizer, torch_module, context = self.examples()
        with patch.dict(sys.modules, {"torch": torch_module}):
            with self.assertRaisesRegex(ValueError, "rỗng"):
                summarize_text(" \n\t ", model, tokenizer)
        tokenizer.assert_not_called()
        model.generate.assert_not_called()

    def test_cuda_is_synchronized_for_timing(self):
        model, tokenizer, torch_module, context = self.examples(device_type="cuda")
        with patch.dict(sys.modules, {"torch": torch_module}):
            report = summarize_text("Một văn bản.", model, tokenizer)
        self.assertEqual(torch_module.cuda.synchronize.call_count, 2)
        self.assertGreaterEqual(report["elapsed_seconds"], 0)

    def test_missing_checkpoint_does_not_fall_back_to_pretrained_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.train.load_best_checkpoint") as load:
                with self.assertRaises(FileNotFoundError):
                    load_summarizer(directory)
        load.assert_not_called()

    def test_loader_uses_existing_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            best = Path(directory) / "best"
            best.mkdir()
            for name in ["config.json", "model.safetensors", "spiece.model", "tokenizer_config.json", "selection.json"]:
                (best / name).write_bytes(b"checkpoint file")
            with patch("src.train.load_best_checkpoint", return_value=("model", "tokenizer", "selection")) as load:
                result = load_summarizer(directory)
        load.assert_called_once_with(directory, "configs/train.json")
        self.assertEqual(result, ("model", "tokenizer", "selection"))


if __name__ == "__main__":
    unittest.main()
