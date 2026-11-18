# Checklist cấu hình ViT5 theo ví dụ VietAI

Mục tiêu: giữ vòng lặp PyTorch tự viết, đưa thông số về gần notebook VietAI, dùng batch vật lý 1 và tích lũy 4 bước để phù hợp T4. Tài liệu ghi lại các thay đổi đã triển khai trong code, cấu hình và notebook. Ô `[x]` xác nhận phần triển khai đã viết.

Mốc tham khảo: [notebook Hugging Face của VietAI](https://github.com/vietai/ViT5/blob/151850bb988f8364c5499fbde241a998a6695c9d/examples/finetune_huggingface_example.ipynb), commit `151850bb988f8364c5499fbde241a998a6695c9d`. Notebook không khóa phiên bản thư viện; optimizer grouping theo Transformers 4.57.6 của dự án.

## Thông số đích

| Thông số | Trước thay đổi | Đã cấu hình cho run mới |
| --- | --- | --- |
| Model | `VietAI/vit5-base`, khóa revision | Giữ nguyên |
| Nguồn / nhãn đích | 1024 / 128 token | 1024 / 256 token |
| Batch / accumulation | 1 / 8 | **1 / 4**, batch hiệu dụng 4 |
| Learning rate | `5e-5`, cố định | `1e-5`, linear scheduler |
| Warmup | Không có | 5% tổng bước cập nhật dự kiến |
| Optimizer | AdamW, một nhóm tham số | AdamW, tách nhóm có / không weight decay |
| Weight decay / max grad norm | 0.01 / 1.0 | Giữ nguyên giá trị |
| Precision huấn luyện | FP32 | FP16 mixed precision bằng AMP |
| Epoch tối đa | 10 | 30 |
| Validation / checkpoint | Mỗi epoch / best theo loss | Giữ nguyên |
| Early stopping | Patience 3, min delta 0 | Giữ nguyên; có thể dừng trước 30 epoch |
| Sinh văn bản làm mốc | Beam 4, 128 token mới | Greedy, `max_length=256`, không chặn lặp |
| Seed / dữ liệu / ROUGE | 42 / split hiện có / Unicode | Giữ nguyên để so sánh nội bộ |

Khác ví dụ VietAI: batch `1 × 4` thay cho `4 × 1`; giữ validation, early stopping và checkpoint best. Split dữ liệu và cách tính ROUGE chưa được xác nhận tương đương VietAI, nên không gọi đây là tái lập đầy đủ thí nghiệm.

## Task 1 — Bổ sung khả năng đọc cấu hình

- [x] Trong `src/utils.py`, thêm kiểm tra `warmup_ratio=0.05`, `lr_scheduler_type="linear"`; giữ khả năng đọc cấu hình run cũ.
- [x] Rà soát ràng buộc FP32 và giới hạn token trong `src/train.py`, `src/evaluate.py` cùng các notebook; cập nhật nơi cần hỗ trợ cấu hình mới.

Commit: `feat(config): support ViT5 warmup and scheduler settings`

## Task 2 — Sửa optimizer và bổ sung lịch learning rate

- [x] Trong `src/train.py`, tách nhóm tham số: áp dụng decay 0.01 cho weights phù hợp, không decay bias và normalization theo quy tắc Trainer của phiên bản đã chọn. Cấu hình cũ không có `optimizer_grouping` giữ nhóm legacy khi resume.
- [x] Tạo linear scheduler; tổng bước = `ceil(len(train_loader) / accumulation) × num_epochs`, warmup = `ceil(0.05 × tổng bước)`.
- [x] Gọi scheduler sau lần optimizer thực sự cập nhật; ghi LR vào log. Với AMP, không tăng scheduler khi optimizer bỏ bước do overflow.

Commit: `feat(train): add grouped AdamW and linear warmup scheduling`

## Task 3 — Thêm AMP và hỗ trợ resume đầy đủ

- [x] Trong `src/train.py`, dùng `autocast` và `GradScaler`; giữ trọng số chính FP32, không chỉ chuyển toàn bộ model sang FP16.
- [x] Giữ loss tích lũy theo token; unscale trước clipping, sau đó scaler step/update; xử lý nhóm cuối thiếu 4 mẫu. Warmup LR=0 không buộc trọng số phải thay đổi ở bước đầu.
- [x] Cho validation hoạt động với cấu hình mới; validation và inference dùng FP32, training dùng AMP.
- [x] Viết phần lưu/khôi phục scheduler, scaler và bộ đếm cập nhật cùng optimizer, RNG, DataLoader; nối nhánh resume trong notebook.

Commit: `feat(train): add AMP training and resumable scaler state`

## Task 4 — Áp dụng thông số mới

- [x] Sửa `configs/train.json` theo bảng: 1024/256, batch 1, accumulation 4, LR `1e-5`, 30 epoch, AMP FP16, warmup 0.05 và linear scheduler.
- [x] Cập nhật metadata quyết định token thành `pending_manual_verification`, `training_memory_verified=false`; dùng đường dẫn báo cáo mới `wikilingua_truncation_1024_256.json`.

Commit: `feat(config): adopt ViT5 settings with effective batch size four`

## Task 5 — Đồng bộ notebook huấn luyện

- [x] Trong notebook train, kết nối scheduler/scaler với vòng lặp, log, checkpoint và resume; giữ validation và early stopping hiện có. Xóa output cũ để chờ chạy cấu hình mới.
- [x] Giữ phần ghi commit nguồn, phiên bản thư viện, cấu hình, hash dữ liệu và lý do dừng; thêm log số nhóm/cập nhật/bỏ qua và peak VRAM.

Commit: `feat(notebook): wire AMP and scheduler into ViT5 training`

## Task 6 — Đồng bộ sinh văn bản, đánh giá và demo

- [x] Trong cấu hình, `src/utils.py`, `src/inference.py` và `src/evaluate.py`, hỗ trợ mốc greedy: `num_beams=1`, `do_sample=false`, `no_repeat_ngram_size=0`, `max_length=256`. Notebook không tự bật chặn lặp.
- [x] Không truyền đồng thời `max_length` và `max_new_tokens`; cập nhật kiểm tra giới hạn theo từng quy ước. Giữ khả năng đọc báo cáo cũ.
- [x] Demo đọc đúng checkpoint và generation config của lượt đánh giá, hỗ trợ hai quy ước độ dài; giữ ROUGE Unicode và ghi rõ test từng được xem ở các run trước. Source hash vẫn phải khớp, nên mã inference mới cần lượt đánh giá tương ứng.

Commit: `feat(eval): align decoding settings across evaluation and demo`

## Task 7 — Cập nhật tài liệu cấu hình

- [x] Cập nhật README với cấu hình mới, cách chạy thủ công và khác biệt so với VietAI; giữ kết quả lịch sử, chưa ghi best epoch/ROUGE/tài nguyên mới khi chưa chạy.

Commit: `docs: document ViT5 migration and manual verification steps`

Checklist được commit cùng tài liệu ở Task 7; chỉ force-add đúng file này vì `docs/` đang bị ignore, không thay `.gitignore`.

Thứ tự triển khai: **1 → 2 → 3 → 4 → 5 → 6 → 7**.

Commit triển khai: Task 1 `ae7d190`; Task 2 `ffaf26e`; Task 3 `d619014`; Task 4 `7502fe6`; Task 5 `cd5e082`; Task 6 `b0dd2b4`; Task 7 `3bd932b`.
