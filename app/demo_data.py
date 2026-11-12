import hashlib
import json
from pathlib import Path


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
