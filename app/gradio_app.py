from pathlib import Path

from src.inference import load_summarizer, summarize_text
from src.metrics import score_summary
from src.preprocessing import normalize_text
from src.textRank import summarize_textrank


SAMPLE_METRIC_HEADERS = [
    "Phương pháp",
    "Số từ",
    "Tỷ lệ nén",
    "Thời gian (s)",
    "ROUGE-1 (%)",
    "ROUGE-2 (%)",
    "ROUGE-L (%)",
]

TEST_METRIC_HEADERS = [
    "Mô hình",
    "ROUGE-1 (%)",
    "ROUGE-2 (%)",
    "ROUGE-L (%)",
    "Lặp câu (%)",
]


def count_words(text):
    if not isinstance(text, str):
        return 0
    normalized = normalize_text(text)
    if normalized == "":
        return 0
    return len(normalized.split())


def format_sample_metrics(
    source_text,
    textrank_summary,
    vit5_summary,
    reference="",
    textrank_seconds=None,
    vit5_seconds=None,
):
    source_words = count_words(source_text)
    if source_words == 0 or (textrank_summary == "" and vit5_summary == ""):
        return []
    has_reference = isinstance(reference, str) and normalize_text(reference) != ""
    rows = []
    for label, summary, elapsed in [
        ("TextRank", textrank_summary, textrank_seconds),
        ("ViT5", vit5_summary, vit5_seconds),
    ]:
        words = count_words(summary)
        ratio = f"{(words / source_words * 100):.1f}%"
        elapsed_str = f"{elapsed:.2f}" if isinstance(elapsed, (int, float)) else "—"
        if has_reference and summary != "":
            scores = score_summary(reference, summary)
            r1 = f"{scores['rouge1']['f1'] * 100:.2f}"
            r2 = f"{scores['rouge2']['f1'] * 100:.2f}"
            rl = f"{scores['rougeL']['f1'] * 100:.2f}"
        else:
            r1 = r2 = rl = "—"
        rows.append([label, words, ratio, elapsed_str, r1, r2, rl])
    return rows


def build_overview_rows(metrics):
    if not isinstance(metrics, dict) or "results" not in metrics:
        return []
    rows = []
    for key, label in [
        ("textrank", "TextRank (Extractive)"),
        ("vit5", "ViT5-Base (Abstractive)"),
    ]:
        entry = metrics["results"].get(key)
        if not isinstance(entry, dict):
            continue
        f1 = entry.get("mean_f1_percent", {})
        rep = entry.get("sentence_repetition")
        rep_str = (
            f"{rep['repeated_sample_percent']:.2f}%"
            if isinstance(rep, dict) and "repeated_sample_percent" in rep
            else "—"
        )
        rows.append([
            label,
            f"{f1.get('rouge1', 0.0):.2f}",
            f"{f1.get('rouge2', 0.0):.2f}",
            f"{f1.get('rougeL', 0.0):.2f}",
            rep_str,
        ])
    return rows


def source_notice(result):
    if not result["was_truncated"]:
        return ""
    message = (
        "Văn bản có " + str(result["input_token_count"]) + " token. "
        "Chỉ sử dụng " + str(result["used_input_token_count"])
        + " token đầu (giới hạn " + str(result["max_source_length"])
        + " token). Phần còn lại không được dùng để tóm tắt."
    )
    return message


class DemoSummarizer:
    def __init__(
        self,
        checkpoint_run_dir,
        config_path="configs/train.json",
        generation_config_path="configs/generation.json",
        textrank_config=None,
    ):
        self.config_path = config_path
        self.generation_config_path = generation_config_path
        self.textrank_config = None
        self.last_timing = {}
        if textrank_config is not None:
            self.textrank_config = textrank_config.copy()
        try:
            self.model, self.tokenizer, self.selection = load_summarizer(
                checkpoint_run_dir,
                config_path,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "Lỗi: Thiếu file checkpoint hoặc cấu hình. "
                "Hãy kiểm tra kết nối Drive và đường dẫn thư mục best. "
                "Chi tiết: " + str(error)
            ) from error
        except ValueError as error:
            raise RuntimeError(
                "Lỗi: Checkpoint hoặc cấu hình không hợp lệ hay không khớp nhau. "
                "Hãy dùng đúng checkpoint best và cấu hình của lần huấn luyện. "
                "Chi tiết: " + str(error)
            ) from error
        except ImportError as error:
            raise RuntimeError(
                "Lỗi: Thiếu thư viện để tải model. "
                "Hãy chạy cell cài thư viện và kiểm tra lại môi trường. "
                "Chi tiết: " + str(error)
            ) from error
        except OSError as error:
            raise RuntimeError(
                "Lỗi: Không đọc được checkpoint. "
                "Hãy kiểm tra quyền truy cập Drive và các file trong thư mục best. "
                "Chi tiết: " + str(error)
            ) from error
        except RuntimeError as error:
            raise RuntimeError(
                "Lỗi: Không tải được model để chạy trên GPU. "
                "Hãy kiểm tra GPU, bộ nhớ GPU và tính toàn vẹn của checkpoint. "
                "Chi tiết: " + str(error)
            ) from error

    def summarize(self, text):
        if text is None:
            return "", "Vui lòng nhập văn bản cần tóm tắt."
        if normalize_text(text) == "":
            return "", "Vui lòng nhập văn bản cần tóm tắt."
        result = summarize_text(
            text,
            self.model,
            self.tokenizer,
            config_path=self.config_path,
            generation_config_path=self.generation_config_path,
        )
        message = source_notice(result)
        return result["summary"], message

    def compare(self, text):
        self.last_timing = {}
        if text is None:
            return "", "", "Vui lòng nhập văn bản cần tóm tắt."
        if normalize_text(text) == "":
            return "", "", "Vui lòng nhập văn bản cần tóm tắt."
        if self.textrank_config is None:
            textrank_result = summarize_textrank(
                text,
                self.tokenizer,
                config_path=self.config_path,
            )
        else:
            textrank_result = summarize_textrank(
                text,
                self.tokenizer,
                config_path=self.config_path,
                damping=self.textrank_config["damping"],
                tolerance=self.textrank_config["tolerance"],
                max_iterations=self.textrank_config["max_iterations"],
            )
            if textrank_result["textrank_config"] != self.textrank_config:
                raise RuntimeError("Lỗi: Thuật toán TextRank hiện tại khác bản đã đánh giá.")
        vit5_result = summarize_text(
            text,
            self.model,
            self.tokenizer,
            config_path=self.config_path,
            generation_config_path=self.generation_config_path,
        )
        if textrank_result["used_input_ids"] != vit5_result["used_input_ids"]:
            raise RuntimeError("Lỗi: TextRank và ViT5 không dùng cùng phần nguồn.")
        notices = []
        message = source_notice(vit5_result)
        if message != "":
            notices.append(message)
        if textrank_result["summary_was_truncated"]:
            notices.append(
                "TextRank không có câu nguyên vẹn nào vừa giới hạn "
                + str(textrank_result["max_summary_tokens"])
                + " token; kết quả là phần đầu của câu được xếp hạng cao nhất."
            )
        if not textrank_result["converged"]:
            notices.append("TextRank đã đạt giới hạn vòng lặp trước khi hội tụ.")
        message = "\n".join(notices)
        self.last_timing = {
            "textrank_seconds": textrank_result.get("elapsed_seconds"),
            "vit5_seconds": vit5_result.get("elapsed_seconds"),
        }
        return textrank_result["summary"], vit5_result["summary"], message


class DemoExamples:
    def __init__(self, summarizer, examples):
        self.summarizer = summarizer
        self.examples = {}
        self.choices = [("Văn bản tự nhập", "")]
        for index, example in enumerate(examples, 1):
            self.examples[example["id"]] = example
            self.choices.append(("Mẫu validation " + str(index) + " — " + example["id"], example["id"]))

    def select(self, example_id):
        if example_id not in self.examples:
            return "", "", "", "", "Không có bản tóm tắt tham chiếu cho văn bản tự nhập."
        example = self.examples[example_id]
        return example["document"], example["summary"], "", "", "Mẫu từ tập validation WikiLingua tiếng Việt."

    def edit(self):
        return "", "", "", "Văn bản đã sửa. Bản tham chiếu chỉ áp dụng khi nội dung khớp mẫu đã chọn."

    def summarize(self, text, example_id):
        reference = ""
        example = self.examples.get(example_id)
        if example is not None and text == example["document"]:
            reference = example["summary"]
        textrank_summary, vit5_summary, message = self.summarizer.compare(text)
        if reference == "":
            notice = "Không có bản tóm tắt tham chiếu cho văn bản tự nhập."
            if message != "":
                notice = message + "\n" + notice
            message = notice
        return reference, textrank_summary, vit5_summary, message

    def summarize_with_metrics(self, text, example_id):
        reference, textrank_summary, vit5_summary, message = self.summarize(text, example_id)
        timing = getattr(self.summarizer, "last_timing", {}) or {}
        metrics_rows = format_sample_metrics(
            text,
            textrank_summary,
            vit5_summary,
            reference=reference,
            textrank_seconds=timing.get("textrank_seconds"),
            vit5_seconds=timing.get("vit5_seconds"),
        )
        return reference, textrank_summary, vit5_summary, metrics_rows, message

    def select_with_metrics(self, example_id):
        source, reference, textrank_summary, vit5_summary, message = self.select(example_id)
        return source, reference, textrank_summary, vit5_summary, [], message

    def edit_with_metrics(self):
        reference, textrank_summary, vit5_summary, message = self.edit()
        return reference, textrank_summary, vit5_summary, [], message

    def clear(self):
        return "", "", "", "", "", [], "Không có bản tóm tắt tham chiếu cho văn bản tự nhập."


def create_demo(summarizer, examples=None, run_id="", overview=None):
    import gradio as gr  # type: ignore[import-not-found]

    if examples is None:
        examples = []
    if overview is None:
        overview = {}
    handler = DemoExamples(summarizer, examples)
    eval_metrics = overview.get("metrics") or {}
    loss_plot_path = overview.get("loss_plot_path")

    with gr.Blocks(title="Tóm tắt văn bản tiếng Việt") as demo:
        gr.Markdown("# Tóm tắt văn bản tiếng Việt — TextRank vs ViT5")
        gr.Markdown(
            "Chọn mẫu WikiLingua để đối chiếu với bản tham chiếu, hoặc nhập văn bản riêng. "
            "Bản tham chiếu chỉ dùng để so sánh và tính ROUGE."
        )
        if run_id != "":
            gr.Markdown(f"Model: `{run_id}`")

        choice = gr.Dropdown(label="Chọn mẫu validation hoặc văn bản tự nhập", choices=handler.choices, value="")
        source = gr.Textbox(label="Văn bản cần tóm tắt", lines=8)
        with gr.Row():
            submit = gr.Button("Tóm tắt văn bản", variant="primary", scale=4)
            clear_btn = gr.Button("Xóa", variant="secondary", scale=1)
        with gr.Row():
            textrank_summary = gr.Textbox(label="TextRank — tóm tắt trích xuất", lines=6, interactive=False)
            vit5_summary = gr.Textbox(label="ViT5 — tóm tắt sinh văn bản", lines=6, interactive=False)
        sample_metrics = gr.Dataframe(
            headers=SAMPLE_METRIC_HEADERS,
            value=[],
            label="So sánh chỉ số trực tiếp cho mẫu hiện tại",
            interactive=False,
        )
        reference = gr.Textbox(label="Bản tóm tắt tham chiếu — WikiLingua", lines=4, interactive=False)
        message = gr.Textbox(
            label="Thông báo", lines=2, interactive=False,
            value="Không có bản tóm tắt tham chiếu cho văn bản tự nhập.",
        )

        overview_rows = build_overview_rows(eval_metrics)
        has_plot = loss_plot_path is not None and Path(loss_plot_path).is_file()
        if overview_rows or has_plot:
            gr.Markdown("---\n## Tổng quan huấn luyện & Kết quả đánh giá trên tập Test")
            with gr.Row():
                if overview_rows:
                    gr.Dataframe(
                        headers=TEST_METRIC_HEADERS,
                        value=overview_rows,
                        label="Kết quả tổng hợp trên toàn bộ tập Test",
                        interactive=False,
                    )
                if has_plot:
                    gr.Image(
                        value=str(loss_plot_path),
                        label="Biểu đồ Training & Validation Loss",
                        interactive=False,
                    )

        choice.change(
            fn=handler.select_with_metrics,
            inputs=choice,
            outputs=[source, reference, textrank_summary, vit5_summary, sample_metrics, message],
        )
        source.input(
            fn=handler.edit_with_metrics,
            inputs=None,
            outputs=[reference, textrank_summary, vit5_summary, sample_metrics, message],
        )
        submit.click(
            fn=handler.summarize_with_metrics,
            inputs=[source, choice],
            outputs=[reference, textrank_summary, vit5_summary, sample_metrics, message],
            concurrency_limit=1,
        )
        clear_btn.click(
            fn=handler.clear,
            inputs=None,
            outputs=[choice, source, reference, textrank_summary, vit5_summary, sample_metrics, message],
        )
    demo.queue(default_concurrency_limit=1)
    return demo
