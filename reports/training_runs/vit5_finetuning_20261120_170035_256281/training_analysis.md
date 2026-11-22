# Phân tích kết quả huấn luyện

- Run ID: vit5_finetuning_20261120_170035_256281
- Nguồn: training_history.csv và training_summary.json của đúng run.
- Best epoch: 4
- Best global step: 15597
- Best validation loss: 1.7749543475951344
- Biểu đồ: [training_loss.png](figures/training_loss.png).

| Epoch | Train loss | Validation loss |
| --- | --- | --- |
| 1 | 2.564135 | 1.934431 |
| 2 | 1.998351 | 1.851933 |
| 3 | 1.833167 | 1.792151 |
| 4 | 1.707887 | 1.774954 |
| 5 | 1.607573 | 1.780484 |
| 6 | 1.520233 | 1.801117 |
| 7 | 1.444169 | 1.801340 |

Train loss giảm nhưng validation loss ở epoch cuối cao hơn mức tốt nhất. Đây là dấu hiệu học quá sát train; giữ checkpoint có validation loss nhỏ nhất.

Loss được tổng hợp theo token đích hợp lệ, bỏ qua nhãn padding -100. Loss thấp không bảo đảm mọi bản tóm tắt đều đúng hoặc không lặp; đọc kết quả ROUGE và phân tích lỗi để đánh giá đầy đủ hơn.

- Số liệu được tạo bởi luồng huấn luyện/resume của chính run này.
- Bảng trên được tính từ log đã lưu, không thay đổi trọng số hoặc dữ liệu test.
