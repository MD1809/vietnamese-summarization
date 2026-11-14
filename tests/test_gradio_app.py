import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

from app.gradio_app import DemoSummarizer, DemoExamples
from app.demo_data import select_completed_run, load_validation_examples


class DemoSummarizerTests(unittest.TestCase):
    def test_saved_textrank_parameters_are_used_for_demo(self):
        settings = {"damping": 0.9, "tolerance": 0.00001, "max_iterations": 150}
        with patch("app.gradio_app.load_summarizer", return_value=(Mock(), Mock(), {})):
            with patch("app.gradio_app.summarize_textrank") as extract:
                with patch("app.gradio_app.summarize_text") as generate:
                    extract.return_value = {
                        "summary": "extracted", "used_input_ids": [1], "textrank_config": settings,
                        "summary_was_truncated": False, "converged": True,
                    }
                    generate.return_value = {"summary": "generated", "used_input_ids": [1], "was_truncated": False}
                    handler = DemoSummarizer("run", textrank_config=settings)
                    self.assertEqual(handler.compare("source"), ("extracted", "generated", ""))
                    self.assertEqual(extract.call_args.kwargs, {
                        "config_path": "configs/train.json",
                        "damping": 0.9, "tolerance": 0.00001, "max_iterations": 150,
                    })

    def test_textrank_variant_mismatch_does_not_generate_or_display_comparison(self):
        settings = {"damping": 0.85, "tolerance": 0.000001, "max_iterations": 100}
        with patch("app.gradio_app.load_summarizer", return_value=(Mock(), Mock(), {})):
            with patch("app.gradio_app.summarize_textrank", return_value={"textrank_config": {"damping": 0.9}}):
                with patch("app.gradio_app.summarize_text") as generate:
                    handler = DemoSummarizer("run", textrank_config=settings)
                    with self.assertRaises(RuntimeError):
                        handler.compare("source")
                    generate.assert_not_called()

    def test_comparison_reuses_model_tokenizer_and_configuration(self):
        with patch("app.gradio_app.load_summarizer") as load:
            with patch("app.gradio_app.summarize_textrank") as extract:
                with patch("app.gradio_app.summarize_text") as generate:
                    model = Mock()
                    tokenizer = Mock()
                    load.return_value = (model, tokenizer, {})
                    extract.return_value = {
                        "summary": "extracted", "used_input_ids": [2, 3, 1],
                        "summary_was_truncated": False, "converged": True,
                    }
                    generate.return_value = {
                        "summary": "generated", "used_input_ids": [2, 3, 1], "was_truncated": False,
                    }
                    handler = DemoSummarizer("run", "/runtime/train.json", "/runtime/generation.json")
                    for text in ["source one", "source two"]:
                        self.assertEqual(handler.compare(text), ("extracted", "generated", ""))
                    load.assert_called_once()
                    for index, text in enumerate(["source one", "source two"]):
                        self.assertEqual(extract.call_args_list[index], call(
                            text, tokenizer, config_path="/runtime/train.json",
                        ))
                        self.assertEqual(generate.call_args_list[index], call(
                            text, model, tokenizer, config_path="/runtime/train.json",
                            generation_config_path="/runtime/generation.json",
                        ))

    def test_comparison_blank_input_clears_both_without_running_algorithms(self):
        with patch("app.gradio_app.load_summarizer", return_value=(Mock(), Mock(), {})):
            with patch("app.gradio_app.summarize_textrank") as extract:
                with patch("app.gradio_app.summarize_text") as generate:
                    handler = DemoSummarizer("run")
                    for text in [None, "", " \n\t "]:
                        self.assertEqual(handler.compare(text), ("", "", "Vui lòng nhập văn bản cần tóm tắt."))
                    extract.assert_not_called()
                    generate.assert_not_called()

    def test_comparison_rejects_different_source_tokens(self):
        with patch("app.gradio_app.load_summarizer", return_value=(Mock(), Mock(), {})):
            with patch("app.gradio_app.summarize_textrank", return_value={"used_input_ids": [2, 1]}):
                with patch("app.gradio_app.summarize_text", return_value={"used_input_ids": [3, 1]}):
                    handler = DemoSummarizer("run")
                    with self.assertRaisesRegex(RuntimeError, "TextRank"):
                        handler.compare("source")

    def test_comparison_discloses_source_cut_sentence_cut_and_nonconvergence(self):
        with patch("app.gradio_app.load_summarizer", return_value=(Mock(), Mock(), {})):
            with patch("app.gradio_app.summarize_textrank", return_value={
                "summary": "extracted", "used_input_ids": [2, 1], "summary_was_truncated": True,
                "max_summary_tokens": 128, "converged": False,
            }):
                with patch("app.gradio_app.summarize_text", return_value={
                    "summary": "generated", "used_input_ids": [2, 1], "was_truncated": True,
                    "input_token_count": 1500, "used_input_token_count": 1024, "max_source_length": 1024,
                }):
                    handler = DemoSummarizer("run")
                    first, second, notice = handler.compare("source")
                    self.assertEqual((first, second), ("extracted", "generated"))
                    for expected in ["1500 token", "1024 token đầu", "128 token", "trước khi hội tụ"]:
                        self.assertIn(expected, notice)

    def test_reference_is_only_returned_for_exact_selected_document(self):
        summarizer = Mock()
        summarizer.compare.return_value = ("extracted", "generated", "")
        handler = DemoExamples(summarizer, [{"id": "v1", "document": "source", "summary": "reference"}])
        self.assertEqual(handler.select("v1")[:4], ("source", "reference", "", ""))
        self.assertEqual(handler.summarize("source", "v1"), ("reference", "extracted", "generated", ""))
        for text, identifier in [("edited", "v1"), ("source", ""), ("source", "unknown")]:
            result = handler.summarize(text, identifier)
            self.assertEqual(result[:3], ("", "extracted", "generated"))
        self.assertEqual(handler.edit()[:3], ("", "", ""))
        self.assertEqual(handler.select("")[:4], ("", "", "", ""))
        for request in summarizer.compare.call_args_list:
            self.assertEqual(len(request.args), 1)
            self.assertNotIn("reference", request.args)

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


class DemoDataTests(unittest.TestCase):
    def make_run(self, root, run_id):
        run = root / "outputs" / run_id
        run.mkdir(parents=True)
        config = {"num_epochs": 10, "paths": {"processed_manifest": "data/manifests/processed.json"}}
        profile = {"run_id": run_id, "train_config": config}
        summary = {
            "run_id": run_id, "run_status": "completed", "completed_epochs": 10,
            "best_epoch": 2, "best_global_step": 500, "best_validation_loss": 2.1,
        }
        (run / "config.json").write_text(json.dumps(profile), encoding="utf-8")
        (run / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        best = root / "checkpoints" / run_id / "best"
        best.mkdir(parents=True)
        for name in ["config.json", "model.safetensors", "spiece.model", "tokenizer_config.json"]:
            (best / name).write_bytes(b"fixture")
        (best / "selection.json").write_text(json.dumps({
            "epoch": 2, "global_step": 500, "validation_loss": 2.1, "train_config": config,
        }), encoding="utf-8")
        return profile, best

    def test_latest_completed_is_selected_and_missing_best_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, "vit5_finetuning_20261106_000000_000000")
            profile, best = self.make_run(root, "vit5_finetuning_20261111_000000_000000")
            (root / "outputs" / "vit5_finetuning_20261112_000000_000000").mkdir()
            selected, summary, checkpoint = select_completed_run(root)
            self.assertEqual(selected, profile)
            self.assertEqual(checkpoint, best.parent)
            (best / "model.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                select_completed_run(root)

    def test_explicit_incomplete_run_and_mismatched_best_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, best = self.make_run(root, "vit5_finetuning_20261111_000000_000000")
            summary_path = root / "outputs" / profile["run_id"] / "training_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["best_epoch"] = 3
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                select_completed_run(root)
            summary["run_status"] = "interrupted"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                select_completed_run(root, profile["run_id"])

    def test_validation_references_are_verified_and_test_is_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, best = self.make_run(root, "vit5_finetuning_20261111_000000_000000")
            validation = root / "data" / "processed" / "validation.jsonl"
            validation.parent.mkdir(parents=True)
            rows = [{"id": "v1", "document": "source", "summary": "reference"}]
            content = (json.dumps(rows[0]) + "\n").encode()
            validation.write_bytes(content)
            manifest = {"files": {"validation": {
                "path": "data/processed/validation.jsonl", "pair_count": 1,
                "byte_count": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            }}}
            run = root / "outputs" / profile["run_id"]
            (run / "processed.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_validation_examples(root, profile), rows)
            validation.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_validation_examples(root, profile)

    def test_early_stopped_run_is_completed_for_explicit_and_automatic_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, best = self.make_run(root, "vit5_finetuning_20261114_083823_007233")
            run = root / "outputs" / profile["run_id"]
            profile["train_config"]["early_stopping"] = {"patience": 3, "min_delta": 0.0}
            (run / "config.json").write_text(json.dumps(profile), encoding="utf-8")
            selection_path = best / "selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["train_config"] = profile["train_config"]
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            summary_path = run / "training_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update({
                "completed_epochs": 5, "stopped_early": True, "stop_reason": "early_stopping",
                "early_stopping": {"best_loss": 2.1, "epochs_without_improvement": 3},
            })
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            for run_id in ["", profile["run_id"]]:
                selected, actual_summary, checkpoint = select_completed_run(root, run_id)
                self.assertEqual(selected, profile)
                self.assertEqual(actual_summary, summary)
                self.assertEqual(checkpoint, best.parent)

            for changes in [
                {"stopped_early": False}, {"stop_reason": "max_epochs"},
                {"early_stopping": {"epochs_without_improvement": 2}},
                {"early_stopping": None},
                {"completed_epochs": 0}, {"completed_epochs": 11},
            ]:
                with self.subTest(changes=changes):
                    summary_path.write_text(json.dumps({**summary, **changes}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        select_completed_run(root, profile["run_id"])
