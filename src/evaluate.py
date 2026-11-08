import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from src.baseline import summarize_lead_three
from src.data import _prepared_file_path, _split_manifest_fingerprint
from src.inference import load_summarizer, summarize_text
from src.metrics import get_rouge_settings, score_summaries
from src.tokenization import load_tokenization_source
from src.utils import load_generation_config, load_train_config, set_seed


DECODING_RELEASE_COMMIT = "5cf7db98d3823ef2beb3368c9d0accc283e64c68"
DECODING_REPORT_DIR = "reports/decoding_comparison/decoding_comparison_20261108_175213_059002"


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
        }
        all_records[method] = records
        print(method, "ROUGE F1 x 100:", scores["mean_f1_percent"])
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


def verify_test_release(project_root, checkpoint_run_id):
    root = Path(project_root).resolve()
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    if remote.stdout.strip() != "https://github.com/MD1809/vietnamese-summarization.git":
        raise ValueError("Lỗi: Repository không đúng dự án.")
    subprocess.run(["git", "fetch", "origin"], cwd=root, capture_output=True, text=True, check=True)
    for target in ["HEAD", "origin/main"]:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", DECODING_RELEASE_COMMIT, target],
            cwd=root, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("Lỗi: Cần push G6.4 và checkout commit 5cf7db9 trước khi đọc test.")
    config = load_train_config(root / "configs/train.json")
    generation = load_generation_config(root / "configs/generation.json")
    paths = {
        "train_config": "configs/train.json",
        "generation_config": "configs/generation.json",
        "subset": config["paths"]["subset_manifest"],
        "processed": config["paths"]["processed_manifest"],
        "decision": DECODING_REPORT_DIR + "/decoding_selection.json",
        "comparison": DECODING_REPORT_DIR + "/decoding_comparison.json",
    }
    approved = {}
    for name, relative_path in paths.items():
        path = _prepared_file_path(root, relative_path)
        current = json.loads(path.read_text(encoding="utf-8"))
        release = subprocess.run(
            ["git", "show", DECODING_RELEASE_COMMIT + ":" + relative_path],
            cwd=root, capture_output=True, text=True, check=True,
        )
        if current != json.loads(release.stdout):
            raise ValueError("Lỗi: File đã thay đổi sau khi chốt G6.4: " + relative_path)
        approved[name] = current
    decision = approved["decision"]
    if decision["decision"] != "approved_by_user" or not decision["generation_config_updated"]:
        raise ValueError("Lỗi: Cấu hình sinh chưa được duyệt.")
    if decision["selected_generation_config"] != generation:
        raise ValueError("Lỗi: Cấu hình sinh không khớp quyết định đã duyệt.")
    if decision["checkpoint_run_id"] != checkpoint_run_id:
        raise ValueError("Lỗi: Run checkpoint không khớp quyết định G6.4.")
    if decision["comparison_id"] != approved["comparison"]["comparison_id"]:
        raise ValueError("Lỗi: Quyết định không khớp lượt so sánh validation.")
    return approved


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


def run_test_evaluation(project_root, drive_root, checkpoint_run_id):
    approved = verify_test_release(project_root, checkpoint_run_id)
    config = approved["train_config"]
    generation = approved["generation_config"]
    config_path = Path(project_root).resolve() / "configs/train.json"
    generation_path = Path(project_root).resolve() / "configs/generation.json"
    root = Path(drive_root).resolve()
    checkpoint_dir = root / "checkpoints" / checkpoint_run_id
    model, tokenizer, selection = load_summarizer(checkpoint_dir, config_path)
    for selection_key, report_key in [("epoch", "best_epoch"), ("global_step", "best_global_step"), ("validation_loss", "best_validation_loss")]:
        if selection[selection_key] != approved["comparison"][report_key]:
            raise ValueError("Lỗi: Best checkpoint không khớp lượt validation đã chốt.")
    rows, test_metadata = load_fixed_test_source(root, approved)
    set_seed(config["seed"])
    source = record_evaluation_source(project_root, extra_files=["src/baseline.py"])
    created_at = datetime.now(timezone.utc)
    evaluation_id = "test_evaluation_" + created_at.strftime("%Y%m%d_%H%M%S_%f")
    output_dir = root / "outputs" / evaluation_id
    output_dir.mkdir(parents=True, exist_ok=False)
    print("Khởi động baseline và ViT5, không tính vào thời gian.")
    summarize_lead_three(rows[0]["document"], tokenizer, config_path=config_path)
    summarize_text(rows[0]["document"], model, tokenizer, config_path=config_path, generation_config_path=generation_path)
    records = []
    references = []
    predictions = {"baseline": [], "vit5": []}
    total_seconds = {"baseline": 0.0, "vit5": 0.0}
    truncated_count = 0
    for index, row in enumerate(rows):
        baseline = summarize_lead_three(row["document"], tokenizer, config_path=config_path)
        vit5 = summarize_text(row["document"], model, tokenizer, config_path=config_path, generation_config_path=generation_path)
        if baseline["used_input_ids"] != vit5["used_input_ids"]:
            raise ValueError("Lỗi: Baseline và ViT5 không đọc cùng ID nguồn test.")
        if baseline["was_truncated"] != vit5["was_truncated"] or baseline["input_token_count"] != vit5["input_token_count"]:
            raise ValueError("Lỗi: Thông tin cắt nguồn không nhất quán.")
        source_count = len(vit5["used_input_ids"])
        if source_count > config["max_source_length"] or source_count != vit5["used_input_token_count"] or source_count != baseline["used_input_token_count"]:
            raise ValueError("Lỗi: Giới hạn token nguồn test không đúng.")
        if vit5["generation_config"] != generation:
            raise ValueError("Lỗi: Tham số sinh đã đổi trong khi đánh giá test.")
        generated_count = len(vit5["generated_ids"]) - 1
        if generated_count > generation["max_new_tokens"] or baseline["summary_token_count"] > config["max_target_length"]:
            raise ValueError("Lỗi: Tóm tắt vượt ngân sách đầu ra đã chốt.")
        for method, result in [("baseline", baseline), ("vit5", vit5)]:
            elapsed = result["elapsed_seconds"]
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Lỗi: Thời gian đánh giá test không hợp lệ.")
            total_seconds[method] = total_seconds[method] + elapsed
            predictions[method].append(result["summary"])
        if vit5["was_truncated"]:
            truncated_count = truncated_count + 1
        references.append(row["summary"])
        records.append({
            "id": row["id"],
            "group_id": row["group_id"],
            "document": row["document"],
            "reference": row["summary"],
            "input_token_count": vit5["input_token_count"],
            "used_input_token_count": source_count,
            "was_truncated": vit5["was_truncated"],
            "source_ids_sha256": source_ids_fingerprint(vit5["used_input_ids"]),
            "baseline": {
                "prediction": baseline["summary"],
                "summary_token_count": baseline["summary_token_count"],
                "summary_was_truncated": baseline["summary_was_truncated"],
                "selected_sentence_count": baseline["selected_sentence_count"],
                "elapsed_seconds": baseline["elapsed_seconds"],
            },
            "vit5": {
                "prediction": vit5["summary"],
                "generated_token_count": generated_count,
                "elapsed_seconds": vit5["elapsed_seconds"],
            },
        })
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print("Test đã xử lý:", index + 1, "/", len(rows))
    results = {}
    comparison_rows = []
    for method in ["baseline", "vit5"]:
        scores = score_summaries(references, predictions[method])
        if scores["sample_count"] != len(rows) or len(scores["per_sample"]) != len(rows):
            raise ValueError("Lỗi: Chưa chấm đủ mẫu test.")
        for index, record in enumerate(records):
            record[method]["scores"] = scores["per_sample"][index]
        results[method] = {
            "sample_count": len(rows),
            "mean_f1": scores["mean_f1"],
            "mean_f1_percent": scores["mean_f1_percent"],
            "total_inference_seconds": total_seconds[method],
            "mean_inference_seconds": total_seconds[method] / len(rows),
        }
        comparison_rows.append({
            "method": method,
            "sample_count": len(rows),
            "rouge1_f1_percent": scores["mean_f1_percent"]["rouge1"],
            "rouge2_f1_percent": scores["mean_f1_percent"]["rouge2"],
            "rougeL_f1_percent": scores["mean_f1_percent"]["rougeL"],
            "total_inference_seconds": total_seconds[method],
            "mean_inference_seconds": total_seconds[method] / len(rows),
            "truncated_source_count": truncated_count,
            "truncated_source_percent": truncated_count / len(rows) * 100,
        })
    test_ids = []
    for record in records:
        test_ids.append(record["id"])
    if test_ids != sorted(approved["subset"]["splits"]["test"]["sample_ids"]):
        raise ValueError("Lỗi: Chưa dự đoán đúng thứ tự hoặc đủ mẫu test.")
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    report = {
        "evaluation_id": evaluation_id,
        "created_at_utc": created_at.isoformat(),
        "status": "completed",
        "output_dir": str(output_dir),
        "decoding_release_commit": DECODING_RELEASE_COMMIT,
        "decoding_approval": approved["decision"],
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_dir": str(checkpoint_dir / "best"),
        "best_epoch": selection["epoch"],
        "best_global_step": selection["global_step"],
        "best_validation_loss": selection["validation_loss"],
        "train_config": config,
        "generation_config": generation,
        "metric_settings": get_rouge_settings(),
        "source": source,
        "library_versions": {"torch": version("torch"), "transformers": version("transformers"), "rouge-score": version("rouge-score")},
        "verified_test_file": test_metadata,
        "subset_manifest_sha256": approved["processed"]["subset_manifest_sha256"],
        "test_sample_ids": test_ids,
        "same_source_ids_check_passed": True,
        "test_lock_metadata_preserved": True,
        "truncated_source_count": truncated_count,
        "truncated_source_percent": truncated_count / len(rows) * 100,
        "output_budget_definition": {"baseline": "128 content tokens without special tokens", "vit5": "128 new tokens including EOS when emitted; excludes decoder start"},
        "warmup_runs_per_method": 1,
        "timing_scope": "sum of per-sample processing; excludes loading, warmup, ROUGE and file writing; baseline runs on CPU, ViT5 on CUDA",
        "results": results,
        "predictions_file": prediction_path.name,
        "comparison_file": csv_path.name,
        "test_content_read": True,
        "test_used_for_tuning": False,
        "training_performed": False,
        "generation_config_updated": False,
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if json.loads(report_path.read_text(encoding="utf-8")) != report:
        raise RuntimeError("Lỗi: Báo cáo test đọc lại không khớp.")
    print("Đã lưu kết quả test:", output_dir)
    return report
