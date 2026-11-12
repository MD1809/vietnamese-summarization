from src.inference import load_summarizer, summarize_text
from src.preprocessing import normalize_text


class DemoSummarizer:
    def __init__(
        self,
        checkpoint_run_dir,
        config_path="configs/train.json",
        generation_config_path="configs/generation.json",
    ):
        self.config_path = config_path
        self.generation_config_path = generation_config_path
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
        message = ""
        if result["was_truncated"]:
            message = (
                "Văn bản có " + str(result["input_token_count"]) + " token. "
                "Chỉ sử dụng " + str(result["used_input_token_count"])
                + " token đầu (giới hạn " + str(result["max_source_length"])
                + " token). Phần còn lại không được dùng để tóm tắt."
            )
        return result["summary"], message


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
            return "", "", "", "Không có bản tóm tắt tham chiếu cho văn bản tự nhập."
        example = self.examples[example_id]
        return example["document"], example["summary"], "", "Mẫu từ tập validation WikiLingua tiếng Việt."

    def edit(self):
        return "", "", "Văn bản đã sửa. Bản tham chiếu chỉ áp dụng khi nội dung khớp mẫu đã chọn."

    def summarize(self, text, example_id):
        reference = ""
        example = self.examples.get(example_id)
        if example is not None and text == example["document"]:
            reference = example["summary"]
        summary, message = self.summarizer.summarize(text)
        if reference == "":
            notice = "Không có bản tóm tắt tham chiếu cho văn bản tự nhập."
            if message != "":
                notice = message + "\n" + notice
            message = notice
        return reference, summary, message


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
        submit = gr.Button("Tóm tắt", variant="primary")
        with gr.Row():
            reference = gr.Textbox(label="Bản tóm tắt tham chiếu — WikiLingua", lines=6, interactive=False)
            summary = gr.Textbox(label="Bản tóm tắt do ViT5 sinh", lines=6, interactive=False)
        message = gr.Textbox(
            label="Thông báo", lines=2, interactive=False,
            value="Không có bản tóm tắt tham chiếu cho văn bản tự nhập.",
        )
        choice.change(fn=handler.select, inputs=choice, outputs=[source, reference, summary, message])
        source.input(fn=handler.edit, inputs=None, outputs=[reference, summary, message])
        submit.click(
            fn=handler.summarize,
            inputs=[source, choice],
            outputs=[reference, summary, message],
            concurrency_limit=1,
        )
    demo.queue(default_concurrency_limit=1)
    return demo
