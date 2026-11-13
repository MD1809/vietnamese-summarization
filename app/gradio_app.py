from src.inference import load_summarizer, summarize_text
from src.preprocessing import normalize_text
from src.textRank import summarize_textrank


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


def create_demo(summarizer, examples=None, run_id=""):
    import gradio as gr

    if examples is None:
        examples = []
    handler = DemoExamples(summarizer, examples)
    with gr.Blocks(title="Tóm tắt văn bản tiếng Việt") as demo:
        gr.Markdown("# Tóm tắt văn bản tiếng Việt")
        gr.Markdown(
            "Chọn mẫu WikiLingua để đối chiếu với bản tham chiếu, hoặc nhập văn bản riêng. "
            "Bản tham chiếu chỉ dùng để so sánh, không được đưa vào mô hình khi sinh tóm tắt."
        )
        if run_id != "":
            gr.Markdown("Model fine-tune: `" + run_id + "` · checkpoint **best**")
        choice = gr.Dropdown(label="Chọn mẫu validation hoặc văn bản tự nhập", choices=handler.choices, value="")
        source = gr.Textbox(label="Văn bản cần tóm tắt", lines=10)
        submit = gr.Button("Tóm tắt văn bản", variant="primary")
        with gr.Row():
            textrank_summary = gr.Textbox(label="TextRank — tóm tắt trích xuất", lines=6, interactive=False)
            vit5_summary = gr.Textbox(label="ViT5 — tóm tắt sinh văn bản", lines=6, interactive=False)
        reference = gr.Textbox(label="Bản tóm tắt tham chiếu — WikiLingua", lines=5, interactive=False)
        message = gr.Textbox(
            label="Thông báo", lines=2, interactive=False,
            value="Không có bản tóm tắt tham chiếu cho văn bản tự nhập.",
        )
        choice.change(
            fn=handler.select,
            inputs=choice,
            outputs=[source, reference, textrank_summary, vit5_summary, message],
        )
        source.input(
            fn=handler.edit,
            inputs=None,
            outputs=[reference, textrank_summary, vit5_summary, message],
        )
        submit.click(
            fn=handler.summarize,
            inputs=[source, choice],
            outputs=[reference, textrank_summary, vit5_summary, message],
            concurrency_limit=1,
        )
    demo.queue(default_concurrency_limit=1)
    return demo
