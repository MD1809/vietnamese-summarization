import unittest
from unittest.mock import Mock, call, patch

from app.gradio_app import DemoSummarizer


class DemoSummarizerTests(unittest.TestCase):
    def test_repeated_requests_reuse_loaded_model_and_tokenizer(self):
        model = Mock()
        tokenizer = Mock()
        selection = {"best_epoch": 2}
        with patch("app.gradio_app.load_summarizer") as load:
            with patch("app.gradio_app.summarize_text") as summarize:
                load.return_value = (model, tokenizer, selection)
                summarize.side_effect = [
                    {"summary": "Kết quả thứ nhất."},
                    {"summary": "Kết quả thứ hai."},
                ]
                handler = DemoSummarizer("checkpoints/run")
                first = handler.summarize("Văn bản thứ nhất.")
                second = handler.summarize("Văn bản thứ hai.")
        load.assert_called_once_with("checkpoints/run", "configs/train.json")
        self.assertEqual(first, "Kết quả thứ nhất.")
        self.assertEqual(second, "Kết quả thứ hai.")
        self.assertEqual(summarize.call_args_list, [
            call("Văn bản thứ nhất.", model, tokenizer,
                 config_path="configs/train.json",
                 generation_config_path="configs/generation.json"),
            call("Văn bản thứ hai.", model, tokenizer,
                 config_path="configs/train.json",
                 generation_config_path="configs/generation.json"),
        ])
        self.assertIs(handler.selection, selection)

    def test_explicit_configuration_paths_reach_shared_inference(self):
        with patch("app.gradio_app.load_summarizer") as load:
            with patch("app.gradio_app.summarize_text") as summarize:
                load.return_value = (Mock(), Mock(), {})
                summarize.return_value = {"summary": "Kết quả."}
                handler = DemoSummarizer("run", "/repo/train.json", "/repo/generation.json")
                handler.summarize("Văn bản.")
        load.assert_called_once_with("run", "/repo/train.json")
        self.assertEqual(summarize.call_args.kwargs, {
            "config_path": "/repo/train.json",
            "generation_config_path": "/repo/generation.json",
        })
