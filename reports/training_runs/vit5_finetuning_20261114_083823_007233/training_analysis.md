# Phân tích kết quả huấn luyện

- Run ID: vit5_finetuning_20261114_083823_007233
- Nguồn: training_history.csv và training_summary.json của đúng run.
- Best epoch: 2
- Best global step: 3902
- Best validation loss: 1.9117193787786906
- Biểu đồ: [training_loss.png](figures/training_loss.png).

| Epoch | Train loss | Validation loss |
| --- | --- | --- |
| 1 | 2.160491 | 1.953316 |
| 2 | 1.823829 | 1.911719 |
| 3 | 1.603148 | 1.918429 |
| 4 | 1.426857 | 1.951531 |
| 5 | 1.276336 | 2.003837 |

Train loss giảm nhưng validation loss ở epoch cuối cao hơn mức tốt nhất. Đây là dấu hiệu học quá sát train; giữ checkpoint có validation loss nhỏ nhất.

Loss được tổng hợp theo token đích hợp lệ, bỏ qua nhãn padding -100. Loss thấp không bảo đảm mọi bản tóm tắt đều đúng hoặc không lặp; đọc kết quả ROUGE và phân tích lỗi để đánh giá đầy đủ hơn.

- Số liệu được tạo bởi luồng huấn luyện/resume của chính run này.
- Bảng trên được tính từ log đã lưu, không thay đổi trọng số hoặc dữ liệu test.
