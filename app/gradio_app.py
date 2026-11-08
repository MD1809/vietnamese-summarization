from src.inference import load_summarizer, summarize_text


class DemoSummarizer:
    def __init__(
        self,
        checkpoint_run_dir,
        config_path="configs/train.json",
        generation_config_path="configs/generation.json",
    ):
        self.config_path = config_path
        self.generation_config_path = generation_config_path
        self.model, self.tokenizer, self.selection = load_summarizer(
            checkpoint_run_dir,
            config_path,
        )

    def summarize(self, text):
        result = summarize_text(
            text,
            self.model,
            self.tokenizer,
            config_path=self.config_path,
            generation_config_path=self.generation_config_path,
        )
        return result["summary"]


def create_demo(summarizer):
    import gradio as gr

    with gr.Blocks(title="Tóm tắt văn bản tiếng Việt") as demo:
        gr.Markdown("# Tóm tắt văn bản tiếng Việt")
        source = gr.Textbox(label="Văn bản cần tóm tắt", lines=10)
        submit = gr.Button("Tóm tắt", variant="primary")
        summary = gr.Textbox(label="Kết quả tóm tắt", lines=5, interactive=False)
        submit.click(
            fn=summarizer.summarize,
            inputs=source,
            outputs=summary,
            concurrency_limit=1,
        )
    demo.queue(default_concurrency_limit=1)
    return demo
