import hashlib
import json
import math
from pathlib import Path

from src.utils import load_generation_config


def select_completed_run(drive_root, run_id=""):
    root = Path(drive_root).resolve()
    if run_id != "" and (Path(run_id).name != run_id or run_id in [".", ".."]):
        raise ValueError("Lỗi: Run ID không được chứa đường dẫn.")
    if run_id == "":
        for directory in sorted((root / "outputs").glob("vit5_finetuning_*"), reverse=True):
            summary_path = directory / "training_summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("run_status") == "completed":
                run_id = directory.name
                break
    if run_id == "":
        raise FileNotFoundError("Lỗi: Chưa có run hoàn tất với training_summary.json trên Drive.")
    output_dir = root / "outputs" / run_id
    profile = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    if profile["run_id"] != run_id or summary["run_id"] != run_id:
        raise ValueError("Lỗi: Hồ sơ và summary không thuộc run đã chọn.")
    if summary["run_status"] != "completed":
        raise ValueError("Lỗi: Run đang huấn luyện dở, chưa dùng làm demo chính.")
    if summary["completed_epochs"] != profile["train_config"]["num_epochs"]:
        raise ValueError("Lỗi: Số epoch hoàn thành không khớp hồ sơ run.")
    checkpoint_dir = root / "checkpoints" / run_id
    for name in ["config.json", "model.safetensors", "spiece.model", "tokenizer_config.json", "selection.json"]:
        path = checkpoint_dir / "best" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError("Lỗi: Best của run đã chọn thiếu file: " + str(path))
    selection = json.loads((checkpoint_dir / "best" / "selection.json").read_text(encoding="utf-8"))
    for key, summary_key in [
        ("epoch", "best_epoch"), ("global_step", "best_global_step"),
        ("validation_loss", "best_validation_loss"),
    ]:
        if selection[key] != summary[summary_key]:
            raise ValueError("Lỗi: Best không khớp summary của run đã chọn.")
    if selection["train_config"] != profile["train_config"]:
        raise ValueError("Lỗi: Cấu hình best không khớp hồ sơ run.")
    return profile, summary, checkpoint_dir


def load_validation_examples(drive_root, profile):
    root = Path(drive_root).resolve()
    config = profile["train_config"]
    run_dir = root / "outputs" / profile["run_id"]
    manifest_path = run_dir / Path(config["paths"]["processed_manifest"]).name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["files"]["validation"]
    path = (root / metadata["path"]).resolve()
    path.relative_to(root)
    content = path.read_bytes()
    if len(content) != metadata["byte_count"] or hashlib.sha256(content).hexdigest() != metadata["sha256"]:
        raise ValueError("Lỗi: Dữ liệu validation không khớp manifest của run.")
    examples = []
    seen = set()
    for line in content.decode("utf-8").splitlines():
        row = json.loads(line)
        if row["id"] in seen or not row["document"].strip() or not row["summary"].strip():
            raise ValueError("Lỗi: Mẫu validation trùng ID hoặc thiếu nội dung.")
        seen.add(row["id"])
        examples.append({"id": row["id"], "document": row["document"], "summary": row["summary"]})
    if len(examples) != metadata["pair_count"]:
        raise ValueError("Lỗi: Số mẫu validation không khớp manifest.")
    return examples


def load_demo_evaluation(drive_root, profile, summary, evaluation_id=""):
    root = Path(drive_root).resolve()
    output_dir = root / "outputs" / profile["run_id"]
    if evaluation_id != "" and (Path(evaluation_id).name != evaluation_id or evaluation_id in [".", ".."]):
        raise ValueError("Lỗi: Evaluation ID không được chứa đường dẫn.")
    evaluation_dir = None
    if evaluation_id != "":
        evaluation_dir = output_dir / evaluation_id
    else:
        directories = sorted(output_dir.glob("test_evaluation_*"), reverse=True)
        for directory in directories:
            metrics_path = directory / "metrics.json"
            if not metrics_path.is_file():
                continue
            candidate = json.loads(metrics_path.read_text(encoding="utf-8"))
            if candidate.get("run_status") == "completed":
                evaluation_dir = directory
                break
    if evaluation_dir is None:
        raise FileNotFoundError(
            "Lỗi: Run " + profile["run_id"]
            + " chưa có kết quả test hoàn tất. Hãy chạy notebook evaluation trước khi demo."
        )
    required = [
        "metrics.json", "decoding_selection.json", "generation.json",
        "decoding_comparison.json", "textrank_validation.json",
    ]
    for name in required:
        if not (evaluation_dir / name).is_file():
            raise FileNotFoundError("Lỗi: Lượt đánh giá đã chọn thiếu file: " + str(evaluation_dir / name))
    metrics = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    decision = json.loads((evaluation_dir / "decoding_selection.json").read_text(encoding="utf-8"))
    generation = load_generation_config(evaluation_dir / "generation.json")
    comparison = json.loads((evaluation_dir / "decoding_comparison.json").read_text(encoding="utf-8"))
    textrank_report = json.loads((evaluation_dir / "textrank_validation.json").read_text(encoding="utf-8"))
    if metrics["run_status"] != "completed":
        raise ValueError("Lỗi: Lượt đánh giá chưa hoàn tất.")
    if metrics["run_id"] != profile["run_id"] or decision["run_id"] != profile["run_id"]:
        raise ValueError("Lỗi: Kết quả đánh giá thuộc run khác.")
    if metrics["train_config"] != profile["train_config"]:
        raise ValueError("Lỗi: Cấu hình đánh giá khác cấu hình run.")
    for key in ["best_epoch", "best_global_step", "best_validation_loss"]:
        if metrics[key] != summary[key] or comparison[key] != summary[key]:
            raise ValueError("Lỗi: Best của kết quả đánh giá không khớp run.")
    if metrics["decision"] != decision:
        raise ValueError("Lỗi: Quyết định trong metrics không khớp file quyết định.")
    if generation != decision["generation_config"] or generation != metrics["generation_config"]:
        raise ValueError("Lỗi: Cấu hình sinh không khớp kết quả đã đánh giá.")
    method_beams = {"greedy": 1, "beam": 4}
    method = decision["selected_method"]
    if method not in method_beams or generation["num_beams"] != method_beams[method]:
        raise ValueError("Lỗi: Phương án decoding không khớp số beam.")
    if generation["max_new_tokens"] != profile["train_config"]["max_target_length"]:
        raise ValueError("Lỗi: Ngân sách sinh không khớp run.")
    for report in [comparison, textrank_report]:
        if report["checkpoint_run_id"] != profile["run_id"]:
            raise ValueError("Lỗi: Báo cáo validation thuộc run khác.")
        if report["comparison_id"] != decision["comparison_id"]:
            raise ValueError("Lỗi: Comparison ID không khớp quyết định.")
        if report["train_config"] != profile["train_config"]:
            raise ValueError("Lỗi: Cấu hình validation không khớp run.")
    if comparison["candidates"][method]["generation_config"] != generation:
        raise ValueError("Lỗi: Cấu hình sinh khác phương án validation đã chọn.")
    textrank_config = decision["textrank_config"]
    if textrank_config != metrics["textrank_config"] or textrank_config != textrank_report["textrank_config"]:
        raise ValueError("Lỗi: Tham số TextRank không khớp kết quả đã đánh giá.")
    damping = textrank_config["damping"]
    tolerance = textrank_config["tolerance"]
    max_iterations = textrank_config["max_iterations"]
    if type(damping) not in [int, float] or not math.isfinite(damping) or damping <= 0 or damping >= 1:
        raise ValueError("Lỗi: damping của TextRank không hợp lệ.")
    if type(tolerance) not in [int, float] or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Lỗi: tolerance của TextRank không hợp lệ.")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Lỗi: max_iterations của TextRank không hợp lệ.")
    source_root = Path(__file__).resolve().parent.parent
    for name in ["src/textRank.py", "src/inference.py", "src/preprocessing.py"]:
        content = (source_root / name).read_bytes()
        content = content.replace(b"\r\n", b"\n")
        current_hash = hashlib.sha256(content).hexdigest()
        if current_hash != metrics["source"]["runtime_source_sha256"][name]:
            raise ValueError("Lỗi: Mã thuật toán đã đổi sau đánh giá: " + name + ". Hãy đánh giá lại trước khi demo.")
    return metrics, decision, generation, evaluation_dir
