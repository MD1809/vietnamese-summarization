# Bộ kết quả đánh giá tóm tắt tiếng Việt

Evaluation: `test_evaluation_20261108_183331_879067`. Đánh giá cuối trên 200 mẫu test cố định; các file trong thư mục cùng mô tả một lượt đánh giá.

## 1. Các file

- [metrics.json](metrics.json): điểm trung bình, cấu hình, checkpoint, revision, hash dữ liệu/mã và các kiểm tra của lượt chạy.
- [comparison.csv](comparison.csv): hai hàng baseline/ViT5, ROUGE-1/2/L F1 x100, thời gian và tỷ lệ cắt nguồn.
- [predictions.jsonl](predictions.jsonl): 200 hàng theo ID cố định; mỗi hàng có nguồn, tham chiếu, cả hai dự đoán, điểm từng mẫu và thông tin cắt nguồn.
- [error_analysis.md](error_analysis.md): đối chiếu 10 mẫu chọn với seed 42, phần giữ đúng, lỗi và giới hạn kết luận.
- [figures/rouge_comparison.png](figures/rouge_comparison.png): biểu đồ cột lấy điểm từ metrics.json; đơn vị F1 x100, trục bắt đầu từ 0.

## 2. Kết quả

| Phương pháp | ROUGE-1 | ROUGE-2 | ROUGE-L | Thời gian tổng (giây) |
| --- | ---: | ---: | ---: | ---: |
| Baseline ba câu đầu | 21.5613 | 8.0438 | 14.4402 | 11.6174 |
| ViT5 beam 4 | 24.5303 | 12.1843 | 22.0434 | 293.9619 |

Điểm là trung bình F1 từng mẫu nhân 100. Có 24/200 nguồn bị cắt (12%). Baseline đo trên CPU, ViT5 trên CUDA; thời gian không gồm tải mô hình, lượt khởi động, chấm ROUGE và ghi file. Không dùng số đo này như so sánh tốc độ trên cùng loại thiết bị.

## 3. Mô hình, dữ liệu và quy tắc

- Mô hình/tokenizer gốc: `VietAI/vit5-base`; revision `2209a38d735ede63e88f5aa52bcdc11a05a37b85`.
- Trọng số đã fine-tune: `vit5_finetuning_20261106_161925_175323/best/`, best epoch 2/step 500; validation loss 2.110764804105354.
- Cấu hình đã chốt: `num_beams=4, do_sample=false, max_new_tokens=128`; seed 42, tiền tố rỗng.
- Dataset: `esdurmus/wiki_lingua`, tiếng Việt; revision `ea3db3510cbd34d0f8dc612419ae40e4732f3b40`.
- Dữ liệu học/validation/test: 2000/201/200 cặp, giữ nguyên nhóm bài theo bộ chia đã chốt. ID test nằm trong metrics.json và từng dòng predictions.jsonl.
- SHA-256 test.jsonl: `38d26a7b469f4dbc79e65ff4a31f034c69626d6b0e4d023a2d12945068eb6731`; số byte: 720594.
- Dấu vân tay manifest tập con: `0b4018e264f7a86e45010a230c4ebbe5969b8743772022af4d17ae3e37efad57`. Manifest khóa vẫn được giữ nguyên.
- Nguồn: chuẩn hóa NFC/khoảng trắng, tối đa 1024 token ViT5; baseline/model được kiểm tra đọc cùng ID token nguồn.
- Đầu ra: baseline lấy tối đa ba câu rồi cắt tối đa 128 token nội dung; ViT5 sinh tối đa 128 token mới, tính EOS khi sinh và không tính decoder start.
- ROUGE: NFC, chữ thường, giữ chữ/số Unicode, thay ký tự khác bằng khoảng trắng rồi tách; không stemming; dùng rouge1/rouge2/rougeL, không dùng rougeLsum.
- Phiên bản: torch `2.11.0+cu130`, transformers `4.57.6`, rouge-score `0.1.2`.

## 4. Nguồn gốc thực nghiệm

- Commit chốt decoding trước khi mở test: `5cf7db98d3823ef2beb3368c9d0accc283e64c68`. Phương án beam 4 được chọn bằng ROUGE-L validation cao hơn; hòa chính xác mới chọn greedy.
- [Bảng so sánh validation](../../decoding_comparison/decoding_comparison_20261108_175213_059002/decoding_comparison.json) và [quyết định đã duyệt](../../decoding_comparison/decoding_comparison_20261108_175213_059002/decoding_selection.json).
- Log của đúng run huấn luyện phục hồi: [history](../../recovery/vit5_finetuning_20261106_161925_175323/training_history.csv), [summary](../../recovery/vit5_finetuning_20261106_161925_175323/training_summary.json), [verification](../../recovery/vit5_finetuning_20261106_161925_175323/training_verification.json).
- Phần phân tích loss được cập nhật từ log của cùng run: [phân tích](../../recovery/vit5_finetuning_20261106_161925_175323/training_analysis.md), [biểu đồ](../../recovery/vit5_finetuning_20261106_161925_175323/figures/training_loss.png).
- Commit nền Colab là `5cf7db98d3823ef2beb3368c9d0accc283e64c68`. Mã G6.5 được nạp thủ công trong lượt chạy; SHA từng file runtime nằm trong metrics.json. Mã và kết quả G6.5 được lưu ở commit `884ad2feeec410e49b4396cae851c63766a898ff`.
- Phân tích G6.6 được lưu ở commit `bc24a3265e736a73e674b1e81b8e9995dbc52f44`. Mẫu được cố định trước khi phân tích, không chọn theo điểm.
- Sau khi xem test, các việc đã làm là phân tích và trình bày kết quả. Chưa sửa phương pháp, checkpoint, cấu hình sinh hoặc nhãn để tối ưu điểm test.

## 5. Hạn chế cần nêu khi báo cáo

- ViT5 cao hơn baseline trên ba chỉ số trung bình của bộ test này; ROUGE không bảo đảm mọi thông tin đều đúng hoặc mọi mẫu đều tốt hơn.
- Bộ 10 mẫu đọc định tính cho thấy lặp, thiếu ý và một số sai lệch hành động/thời điểm/quy trình; không suy tỷ lệ lỗi toàn bộ 200 mẫu từ bộ 10.
- Một tham chiếu trong bộ phân tích có metadata ảnh JSON. Chưa xác định nơi phát sinh hoặc đo mức ảnh hưởng; nhãn và điểm test gốc vẫn được giữ.
- Nguồn dài bị cắt có thể mất ý. Chưa xác định nguyên nhân của mọi lỗi hoặc thử phương án khắc phục trong lượt đánh giá này.

## 6. Vị trí lưu và cách đối chiếu

- Trong source: `reports/test_evaluation/test_evaluation_20261108_183331_879067/`. Các liên kết tương đối phía trên dùng cấu trúc source.
- Trên Drive: `MyDrive/vietnamese-summarization/outputs/test_evaluation_20261108_183331_879067/`. Bộ bàn giao gồm năm file ở gốc (README.md, metrics.json, comparison.csv, predictions.jsonl, error_analysis.md) và figures/rouge_comparison.png.
- Log huấn luyện trên Drive: `MyDrive/vietnamese-summarization/outputs/vit5_finetuning_20261106_161925_175323/`; checkpoint tại `MyDrive/vietnamese-summarization/checkpoints/vit5_finetuning_20261106_161925_175323/best/`.
- Kết quả chọn decoding trên Drive: `MyDrive/vietnamese-summarization/outputs/decoding_comparison_20261108_175213_059002/`. Quyết định sau duyệt được quản lý trong source ở liên kết phía trên.
- Trọng số không nằm trong GitHub; bản ZIP best đã sao lưu trên máy trong `checkpoints/vit5_finetuning_20261106_161925_175323/`.
- Khi đối chiếu, dùng ID/evaluation_id, hash và cấu hình trong metrics.json; lấy điểm chưa làm tròn từ JSON/CSV. Biểu đồ ROUGE ở thư mục này đã được vẽ từ metrics.json trong G6.7. Theo điều chỉnh G8.3, notebook huấn luyện chạy một run mới và dùng kết quả của run đó; báo cáo trong thư mục này là kết quả lịch sử được giữ riêng. Test của run mới chỉ chạy khi chủ động bật evaluate_test sau khi chốt validation, không thay các điểm ở đây.
