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
                    {"summary": "Kết quả thứ nhất.", "was_truncated": False},
                    {"summary": "Kết quả thứ hai.", "was_truncated": False},
                ]
                handler = DemoSummarizer("checkpoints/run")
                first = handler.summarize("Văn bản thứ nhất.")
                second = handler.summarize("Văn bản thứ hai.")
        load.assert_called_once_with("checkpoints/run", "configs/train.json")
        self.assertEqual(first, ("Kết quả thứ nhất.", ""))
        self.assertEqual(second, ("Kết quả thứ hai.", ""))
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
                summarize.return_value = {"summary": "Kết quả.", "was_truncated": False}
                handler = DemoSummarizer("run", "/repo/train.json", "/repo/generation.json")
                handler.summarize("Văn bản.")
        load.assert_called_once_with("run", "/repo/train.json")
        self.assertEqual(summarize.call_args.kwargs, {
            "config_path": "/repo/train.json",
            "generation_config_path": "/repo/generation.json",
        })

    def test_blank_inputs_clear_result_without_calling_inference(self):
        with patch("app.gradio_app.load_summarizer") as load:
            with patch("app.gradio_app.summarize_text") as summarize:
                load.return_value = (Mock(), Mock(), {})
                handler = DemoSummarizer("run")
                for text in [None, "", " ", "\n\t  ", "\u00a0\u2003"]:
                    with self.subTest(text=text):
                        self.assertEqual(handler.summarize(text), (
                            "", "Vui lòng nhập văn bản cần tóm tắt."
                        ))
                summarize.assert_not_called()

    def test_truncation_notice_uses_inference_counts_and_clears_on_next_input(self):
        with patch("app.gradio_app.load_summarizer") as load:
            with patch("app.gradio_app.summarize_text") as summarize:
                load.return_value = (Mock(), Mock(), {})
                summarize.side_effect = [
                    {
                        "summary": "Tóm tắt phần đầu.",
                        "was_truncated": True,
                        "input_token_count": 1500,
                        "used_input_token_count": 1024,
                        "max_source_length": 1024,
                    },
                    {"summary": "Tóm tắt ngắn.", "was_truncated": False},
                ]
                handler = DemoSummarizer("run")
                summary, message = handler.summarize("Văn bản dài.")
                self.assertEqual(summary, "Tóm tắt phần đầu.")
                self.assertIn("1500 token", message)
                self.assertIn("1024 token đầu", message)
                self.assertIn("Phần còn lại không được dùng", message)
                self.assertEqual(handler.summarize("Văn bản ngắn."), (
                    "Tóm tắt ngắn.", ""
                ))
        load.assert_called_once()

    def test_blank_input_clears_previous_summary(self):
        with patch("app.gradio_app.load_summarizer") as load:
            with patch("app.gradio_app.summarize_text") as summarize:
                load.return_value = (Mock(), Mock(), {})
                summarize.return_value = {"summary": "Kết quả cũ.", "was_truncated": False}
                handler = DemoSummarizer("run")
                self.assertEqual(handler.summarize("Có nội dung."), ("Kết quả cũ.", ""))
                self.assertEqual(handler.summarize("  "), (
                    "", "Vui lòng nhập văn bản cần tóm tắt."
                ))
                summarize.assert_called_once()

    def test_loading_errors_are_readable_and_preserve_original_cause(self):
        examples = [
            (FileNotFoundError("missing model.safetensors"), "Thiếu file"),
            (ValueError("metadata mismatch"), "không khớp"),
            (ImportError("missing transformers"), "Thiếu thư viện"),
            (OSError("cannot read weights"), "Không đọc được checkpoint"),
            (RuntimeError("Best checkpoint check requires CUDA and fp32"), "GPU"),
            (RuntimeError("CUDA out of memory"), "bộ nhớ GPU"),
        ]
        for original, expected in examples:
            with self.subTest(error=original):
                with patch("app.gradio_app.load_summarizer", side_effect=original):
                    with self.assertRaises(RuntimeError) as caught:
                        DemoSummarizer("run")
                self.assertIn(expected, str(caught.exception))
                self.assertIn(str(original), str(caught.exception))
                self.assertIs(caught.exception.__cause__, original)
