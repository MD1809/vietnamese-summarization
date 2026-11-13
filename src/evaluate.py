import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from src.data import _prepared_file_path, _split_manifest_fingerprint
from src.inference import load_summarizer, summarize_text
from src.metrics import get_rouge_settings, score_summaries, measure_sentence_repetition
from src.tokenization import load_tokenization_source
from src.utils import load_generation_config, set_seed


def choose_decoding(candidates):
    scores = {}
    counts = []
    for method in ["greedy", "beam"]:
        score = candidates[method]["mean_f1"]["rougeL"]
        if not math.isfinite(score) or score < 0 or score > 1:
            raise ValueError("Lỗi: Điểm ROUGE-L không hợp lệ.")
        scores[method] = score
        counts.append(candidates[method]["sample_count"])
    if counts[0] <= 0 or counts[0] != counts[1]:
        raise ValueError("Lỗi: Hai cách sinh phải được chấm cùng số mẫu.")
    if scores["beam"] > scores["greedy"]:
        return "beam"
    return "greedy"


def source_ids_fingerprint(ids):
    content = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def record_evaluation_source(project_root, extra_files=None):
    root = Path(project_root)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    files = [
        "src/evaluate.py",
        "src/inference.py",
        "src/metrics.py",
        "src/tokenization.py",
        "src/data.py",
        "src/preprocessing.py",
        "src/utils.py",
        "src/train.py",
    ]
    if extra_files is not None:
        files.extend(extra_files)
    hashes = {}
    for relative_path in files:
        content = (root / relative_path).read_bytes()
        hashes[relative_path] = hashlib.sha256(content).hexdigest()
    return {"git_commit": result.stdout.strip(), "runtime_source_sha256": hashes}


def run_decoding_comparison(
    project_root,
    drive_root,
    checkpoint_run_id,
    config_path="configs/train.json",
    generation_config_path="configs/generation.json",
):
    root = Path(drive_root).resolve()
    if Path(checkpoint_run_id).name != checkpoint_run_id or checkpoint_run_id in ["", ".", ".."]:
        raise ValueError("Lỗi: Run ID checkpoint không hợp lệ.")
    config, splits, source_files, subset_fingerprint = load_tokenization_source(
        root,
        config_path,
    )
    rows = splits["validation"]
    if len(rows) == 0:
        raise ValueError("Lỗi: Validation không có mẫu.")
    expected_ids = []
    for row in rows:
        expected_ids.append(row["id"])
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Lỗi: Validation có ID trùng.")
    base_generation = load_generation_config(generation_config_path)
    if base_generation["max_new_tokens"] != config["max_target_length"]:
        raise ValueError("Lỗi: Giới hạn sinh không khớp cấu hình đã chốt.")
    checkpoint_dir = root / "checkpoints" / checkpoint_run_id
    model, tokenizer, selection = load_summarizer(checkpoint_dir, config_path)
    if selection["validation_sample_ids"] != expected_ids:
        raise ValueError("Lỗi: Validation không khớp tập đã dùng chọn best.")
    source = record_evaluation_source(project_root)
    set_seed(config["seed"])
    created_at = datetime.now(timezone.utc)
    comparison_id = "decoding_comparison_" + created_at.strftime("%Y%m%d_%H%M%S_%f")
    output_dir = root / "outputs" / comparison_id
    output_dir.mkdir(parents=True, exist_ok=False)
    generation_configs = {}
    generation_paths = {}
    for method, beams in [("greedy", 1), ("beam", 4)]:
        candidate = base_generation.copy()
        candidate["num_beams"] = beams
        path = output_dir / (method + "_generation.json")
        path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        generation_configs[method] = candidate
        generation_paths[method] = path
        print("Chạy khởi động, không tính vào thời gian:", method)
        summarize_text(
            rows[0]["document"],
            model,
            tokenizer,
            config_path=config_path,
            generation_config_path=path,
        )
    candidates = {}
    all_records = {}
    expected_source_hashes = []
    for method in ["greedy", "beam"]:
        print("Bắt đầu:", method, "Số mẫu:", len(rows))
        records = []
        references = []
        predictions = []
        total_seconds = 0.0
        truncated_count = 0
        for index, row in enumerate(rows):
            result = summarize_text(
                row["document"],
                model,
                tokenizer,
                config_path=config_path,
                generation_config_path=generation_paths[method],
            )
            if result["generation_config"] != generation_configs[method]:
                raise ValueError("Lỗi: Tham số sinh thực tế không khớp.")
            source_hash = source_ids_fingerprint(result["used_input_ids"])
            if method == "greedy":
                expected_source_hashes.append(source_hash)
            elif source_hash != expected_source_hashes[index]:
                raise ValueError("Lỗi: Hai cách sinh không đọc cùng ID nguồn.")
            if result["used_input_token_count"] != len(result["used_input_ids"]):
                raise ValueError("Lỗi: Số token nguồn không khớp.")
            if result["used_input_token_count"] > config["max_source_length"]:
                raise ValueError("Lỗi: Nguồn vượt giới hạn.")
            generated_token_count = len(result["generated_ids"]) - 1
            if generated_token_count > base_generation["max_new_tokens"]:
                raise ValueError("Lỗi: Đầu ra vượt giới hạn sinh.")
            elapsed = result["elapsed_seconds"]
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Lỗi: Thời gian suy luận không hợp lệ.")
            total_seconds = total_seconds + elapsed
            if result["was_truncated"]:
                truncated_count = truncated_count + 1
            records.append({
                "id": row["id"],
                "group_id": row["group_id"],
                "document": row["document"],
                "reference": row["summary"],
                "prediction": result["summary"],
                "input_token_count": result["input_token_count"],
                "used_input_token_count": result["used_input_token_count"],
                "was_truncated": result["was_truncated"],
                "source_ids_sha256": source_hash,
                "generated_token_count": generated_token_count,
                "elapsed_seconds": elapsed,
            })
            references.append(row["summary"])
            predictions.append(result["summary"])
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(method, "Đã xử lý:", index + 1, "/", len(rows))
        scores = score_summaries(references, predictions)
        if scores["sample_count"] != len(rows):
            raise ValueError("Lỗi: Không chấm đủ validation.")
        prediction_path = output_dir / (method + "_predictions.jsonl")
        with prediction_path.open("w", encoding="utf-8") as file:
            for index, record in enumerate(records):
                record["scores"] = scores["per_sample"][index]
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        actual_ids = []
        for record in records:
            actual_ids.append(record["id"])
        if actual_ids != expected_ids:
            raise ValueError("Lỗi: Thứ tự hoặc số mẫu dự đoán không đúng.")
        candidates[method] = {
            "generation_config": generation_configs[method],
            "sample_count": len(records),
            "mean_f1": scores["mean_f1"],
            "mean_f1_percent": scores["mean_f1_percent"],
            "total_inference_seconds": total_seconds,
            "mean_inference_seconds": total_seconds / len(records),
            "truncated_source_count": truncated_count,
            "predictions_file": prediction_path.name,
            "sentence_repetition": measure_sentence_repetition(predictions),
        }
        all_records[method] = records
        print(method, "ROUGE F1 x 100:", scores["mean_f1_percent"])
        print(method, "Lặp câu:", candidates[method]["sentence_repetition"])
    proposed_method = choose_decoding(candidates)
    examples = []
    for index in range(min(3, len(rows))):
        examples.append({
            "id": rows[index]["id"],
            "document": rows[index]["document"],
            "reference": rows[index]["summary"],
            "greedy": all_records["greedy"][index],
            "beam": all_records["beam"][index],
        })
    examples_path = output_dir / "decoding_examples.json"
    examples_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "comparison_id": comparison_id,
        "created_at_utc": created_at.isoformat(),
        "status": "awaiting_user_confirmation",
        "output_dir": str(output_dir),
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_dir": str(checkpoint_dir / "best"),
        "best_epoch": selection["epoch"],
        "best_global_step": selection["global_step"],
        "best_validation_loss": selection["validation_loss"],
        "train_config": config,
        "source": source,
        "library_versions": {
            "torch": version("torch"),
            "transformers": version("transformers"),
            "rouge-score": version("rouge-score"),
        },
        "subset_manifest_sha256": subset_fingerprint,
        "verified_validation_file": source_files["validation"],
        "validation_sample_ids": expected_ids,
        "same_source_ids_check_passed": True,
        "warmup_runs_per_method": 1,
        "timing_scope": "sum of per-sample normalization, tokenization, generation and decode; excludes model loading, warmup, ROUGE and file writing",
        "metric_settings": get_rouge_settings(),
        "candidates": candidates,
        "selection_rule": "highest mean ROUGE-L F1; exact tie selects greedy",
        "proposed_method": proposed_method,
        "proposed_generation_config": generation_configs[proposed_method],
        "generation_config_updated": False,
        "example_policy": "first three validation samples in fixed manifest order",
        "examples_file": examples_path.name,
        "test_content_read": False,
    }
    report_path = output_dir / "decoding_comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if json.loads(report_path.read_text(encoding="utf-8")) != report:
        raise RuntimeError("Lỗi: Báo cáo đọc lại không khớp.")
    print("Đã lưu báo cáo:", report_path)
    print("Đề xuất:", proposed_method, "Chưa cập nhật generation.json.")
    return report


def load_fixed_test_source(drive_root, approved):
    config = approved["train_config"]
    subset = approved["subset"]
    processed = approved["processed"]
    for manifest in [subset, processed]:
        if manifest["dataset"] != config["dataset"] or manifest["seed"] != config["seed"]:
            raise ValueError("Lỗi: Manifest test không khớp dữ liệu hoặc seed đã chốt.")
    if processed["subset_manifest_sha256"] != _split_manifest_fingerprint(subset):
        raise ValueError("Lỗi: Manifest dữ liệu không khớp tập con cố định.")
    metadata = processed["files"]["test"]
    if metadata["path"] != config["paths"]["test_file"]:
        raise ValueError("Lỗi: Đường dẫn test không khớp cấu hình.")
    if processed["test_lock"]["locked"] is not True or processed["test_lock"]["sha256"] != metadata["sha256"]:
        raise ValueError("Lỗi: Dấu toàn vẹn test không khớp manifest đã khóa.")
    path = _prepared_file_path(drive_root, metadata["path"])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != metadata["sha256"] or len(content) != metadata["byte_count"]:
        raise ValueError("Lỗi: Nội dung hoặc kích thước file test đã thay đổi.")
    rows = []
    for line in content.decode("utf-8").splitlines():
        row = json.loads(line)
        for key in ["id", "group_id", "document", "summary"]:
            if not isinstance(row.get(key), str) or row[key].strip() == "":
                raise ValueError("Lỗi: Mẫu test thiếu trường hợp lệ: " + key)
        rows.append(row)
    expected = subset["splits"]["test"]
    if len(rows) == 0 or len(rows) != metadata["pair_count"] or len(rows) != expected["pair_count"]:
        raise ValueError("Lỗi: Số mẫu test không khớp manifest.")
    ids = []
    group_ids = set()
    for row in rows:
        ids.append(row["id"])
        group_ids.add(row["group_id"])
    if ids != sorted(expected["sample_ids"]) or len(set(ids)) != len(ids):
        raise ValueError("Lỗi: ID hoặc thứ tự test không khớp bộ cố định.")
    if group_ids != set(expected["group_ids"]):
        raise ValueError("Lỗi: Nhóm bài test không khớp manifest.")
    for split_name in ["train", "validation"]:
        other = subset["splits"][split_name]
        if set(ids).intersection(other["sample_ids"]) or group_ids.intersection(other["group_ids"]):
            raise ValueError("Lỗi: Test giao ID hoặc nhóm với " + split_name)
    return rows, metadata
