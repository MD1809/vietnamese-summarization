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


def create_demo(summarizer):
    import gradio as gr

    with gr.Blocks(title="Tóm tắt văn bản tiếng Việt") as demo:
        gr.Markdown("# Tóm tắt văn bản tiếng Việt")
        source = gr.Textbox(label="Văn bản cần tóm tắt", lines=10)
        submit = gr.Button("Tóm tắt", variant="primary")
        summary = gr.Textbox(label="Kết quả tóm tắt", lines=5, interactive=False)
        message = gr.Textbox(label="Thông báo", lines=2, interactive=False)
        submit.click(
            fn=summarizer.summarize,
            inputs=source,
            outputs=[summary, message],
            concurrency_limit=1,
        )
    demo.queue(default_concurrency_limit=1)
    return demo
