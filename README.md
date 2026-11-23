# Tóm tắt văn bản tiếng Việt

Xây dựng và so sánh hệ thống tóm tắt văn bản tiếng Việt theo hai hướng tiếp cận:
- **Tóm tắt trích xuất (Extractive):** Thuật toán đồ thị **TextRank** tự cài đặt.
- **Tóm tắt sinh văn bản (Abstractive):** Mô hình Transformer **ViT5-Base** (`VietAI/vit5-base`) được fine-tune bằng vòng lặp PyTorch trên bộ dữ liệu **WikiLingua tiếng Việt**, kèm giao diện demo so sánh trực tiếp bằng **Gradio**.



## 1. Cấu trúc dự án

Dự án tách bạch **mã nguồn (GitHub)**, **dữ liệu & checkpoint nặng (Google Drive)** và **môi trường chạy GPU (Google Colab)**:

```text
vietnamese-summarization/
├── configs/
│   ├── train.json                 # Siêu tham số huấn luyện ViT5
│   └── generation.json            # Tham số sinh văn bản (Beam Search & chống lặp)
├── src/
│   ├── preprocessing.py           # Chuẩn hóa Unicode NFC và khoảng trắng
│   ├── data.py                    # Tải, làm sạch, chia tập theo nhóm bài gốc (group_id)
│   ├── tokenization.py            # Tokenize và DataLoader cho ViT5
│   ├── train.py                   # Vòng lặp train AMP FP16, validation, early stopping, checkpoint
│   ├── inference.py               # Nạp checkpoint best và sinh tóm tắt ViT5
│   ├── textRank.py                # Cài đặt thuật toán TextRank trích xuất
│   ├── metrics.py                 # Chấm ROUGE-1/2/L tiếng Việt (Unicode) và đo lặp câu
│   ├── evaluate.py                # So sánh decoding trên validation và kiểm tra tập test cố định
│   └── utils.py                   # Đọc/kiểm tra cấu hình và cố định seed
├── app/
│   ├── demo_data.py               # Kiểm tra toàn vẹn và nạp run/lượt đánh giá từ Drive
│   └── gradio_app.py              # Giao diện web Gradio so sánh TextRank vs ViT5
├── notebooks/
│   ├── vietnamese_summarization_data_preparation.ipynb  # Chuẩn bị dữ liệu lên Drive
│   ├── vietnamese_summarization_train.ipynb             # Huấn luyện / resume ViT5 trên Colab T4
│   ├── vietnamese_summarization_evaluation.ipynb        # Đánh giá validation & test
│   └── vietnamese_summarization_demo.ipynb              # Khởi chạy demo Gradio
├── data/manifests/                # Biên bản nguồn, làm sạch và chia tập cố định (không chứa text thô)
├── reports/training_runs/         # Báo cáo huấn luyện, biểu đồ loss và kết quả đánh giá
├── tests/                         # Bộ kiểm thử đơn vị (unittest)
└── requirements.txt
```

**Cấu trúc thư mục lưu trữ trên Google Drive** (không đưa lên Git vì dung lượng lớn):

```text
MyDrive/vietnamese-summarization/
├── data/
│   ├── raw/
│   ├── processed/full/            # train.jsonl, validation.jsonl, test.jsonl
│   └── manifests/full/
├── checkpoints/<run_id>/          # best/ và last/
└── outputs/<run_id>/              # Log huấn luyện, kết quả validation và test
```

---

## 2. Dữ liệu và cấu hình huấn luyện

### Dữ liệu (`esdurmus/wiki_lingua` — tiếng Việt)
Dữ liệu được chuẩn hóa Unicode NFC, loại bỏ mẫu rỗng/trùng lặp và chia theo **nhóm bài viết gốc (`group_id` là URL bài viết cùng các nhóm trùng nguồn)** với `seed = 42` để đảm bảo không rò rỉ dữ liệu giữa các tập:

| Tập dữ liệu | Số cặp (document, summary) | Mục đích |
| :--- | :---: | :--- |
| **Train** | 15,601 | Huấn luyện ViT5 |
| **Validation** | 2,034 | Chọn checkpoint `best`, early stopping và chọn cấu hình decoding |
| **Test** | 1,942 | Đánh giá cuối cùng (khóa cố định theo SHA-256) |

### Cấu hình ViT5-Base (`configs/train.json` & `configs/generation.json`)
- **Mô hình nền:** `VietAI/vit5-base` (revision cố định)
- **Giới hạn độ dài:** `max_source_length = 1024`, `max_target_length = 256`
- **Huấn luyện:** Batch vật lý `1`, `gradient_accumulation_steps = 4` (batch hiệu dụng `4`), `AdamW` (`lr = 1e-5`, `weight_decay = 0.01`, tách nhóm không decay cho bias/norm), `linear` scheduler (`warmup_ratio = 0.05`), `max_grad_norm = 1.0`, AMP `fp16` (validation/inference dùng `fp32`), early stopping (`patience = 3`).
- **Sinh văn bản (Decoding):** Beam Search (`num_beams = 4`), `max_length = 256`, `do_sample = false`, `no_repeat_ngram_size = 3`, `repetition_penalty = 1.15`.

---

## 3. Kết quả thực nghiệm (lần chạy tốt nhất hiện tại)

Lần huấn luyện tốt nhất hiện tại (run ID: **`vit5_finetuning_20261120_170035_256281`**):
- Early stopping dừng ở **epoch 7**; checkpoint **`best` đạt ở epoch 4** (`global_step = 15,597`, `best_validation_loss = 1.7750`).
- Xem chi tiết: [training_summary.json](reports/training_runs/vit5_finetuning_20261120_170035_256281/training_summary.json) · [training_loss.png](reports/training_runs/vit5_finetuning_20261120_170035_256281/figures/training_loss.png).

### So sánh trên tập Validation (2,034 mẫu)

| Phương pháp | ROUGE-1 | ROUGE-2 | ROUGE-L | Tỷ lệ mẫu lặp câu |
| :--- | :---: | :---: | :---: | :---: |
| TextRank (baseline trích xuất) | 19.93 | 8.44 | 13.47 | 0.00% |
| ViT5 Greedy (`num_beams=1`) | 34.01 | 17.46 | 28.33 | 0.39% |
| **ViT5 Beam Search (`num_beams=4`)** | **36.64** | **19.46** | **29.82** | **0.05%** |

### Kết quả trên tập Test (1,942 mẫu)

| Mô hình | ROUGE-1 | ROUGE-2 | ROUGE-L | Tỷ lệ mẫu lặp câu |
| :--- | :---: | :---: | :---: | :---: |
| TextRank (Extractive) | 19.98 | 8.58 | 13.67 | 0.00% |
| **ViT5-Base (Abstractive — Beam 4)** | **37.17** | **19.77** | **30.27** | **0.00%** |

Báo cáo chi tiết tập test: [metrics.json](reports/training_runs/vit5_finetuning_20261120_170035_256281/test_evaluation_20261122_173555_400057/metrics.json) · [comparison.csv](reports/training_runs/vit5_finetuning_20261120_170035_256281/test_evaluation_20261122_173555_400057/comparison.csv).

---

## 4. Hướng dẫn cài đặt và chạy dự án

### Bước 1: Kết nối thư mục Google Drive chung
1. Mở liên kết thư mục `vietnamese-summarization` được chia sẻ trên Google Drive (mục **Shared with me**).
2. Chuột phải vào thư mục → chọn **Organize → Add shortcut to Drive** (Thêm lối tắt vào Drive) và đặt trực tiếp trong **My Drive** để có đường dẫn `MyDrive/vietnamese-summarization`.
   *(Giữ nguyên đường dẫn này giúp chạy toàn bộ notebook Colab mà không cần sửa code.)*

### Bước 2: Chạy các Notebook trên Google Colab (theo thứ tự luồng xử lý)

| Thứ tự | Notebook | Mục đích | Runtime |
| :---: | :--- | :--- | :---: |
| 1 | [vietnamese_summarization_data_preparation.ipynb](notebooks/vietnamese_summarization_data_preparation.ipynb) | Tải và tiền xử lý dữ liệu WikiLingua lên Drive (chỉ cần chạy 1 lần) | CPU |
| 2 | [vietnamese_summarization_train.ipynb](notebooks/vietnamese_summarization_train.ipynb) | Huấn luyện mới hoặc resume ViT5 từ checkpoint `last` | GPU T4 |
| 3 | [vietnamese_summarization_evaluation.ipynb](notebooks/vietnamese_summarization_evaluation.ipynb) | Đánh giá validation (chọn decoding) và chấm ROUGE tập test | GPU T4 |
| 4 | [vietnamese_summarization_demo.ipynb](notebooks/vietnamese_summarization_demo.ipynb) | Mở giao diện Gradio so sánh TextRank và ViT5 | GPU T4 |

> **Lưu ý khi chạy Colab:** Luôn chạy các cell theo đúng thứ tự từ trên xuống. Các notebook tự động kéo mã nguồn mới nhất từ nhánh `main` của GitHub. Nếu dữ liệu và checkpoint đã có sẵn trên Drive và bạn chỉ muốn chạy thử giao diện, có thể mở trực tiếp bước 4 ([vietnamese_summarization_demo.ipynb](notebooks/vietnamese_summarization_demo.ipynb)).
