# Phân tích kết quả huấn luyện

- Run ID: vit5_finetuning_20261106_161925_175323
- Nguồn: training_history.csv và training_summary.json của đúng run.
- Best epoch: 2
- Best global step: 500
- Best validation loss: 2.110764804105354
- Biểu đồ: [training_loss.png](figures/training_loss.png).

| Epoch | Train loss | Validation loss |
| --- | --- | --- |
| 1 | 2.398828 | 2.115569 |
| 2 | 1.888900 | 2.110765 |
| 3 | 1.565508 | 2.212426 |

Train loss giảm nhưng validation loss ở epoch cuối cao hơn mức tốt nhất. Đây là dấu hiệu học quá sát train; giữ checkpoint có validation loss nhỏ nhất.

Loss được tổng hợp theo token đích hợp lệ, bỏ qua nhãn padding -100. Loss thấp không bảo đảm mọi bản tóm tắt đều đúng hoặc không lặp; đọc kết quả ROUGE và phân tích lỗi để đánh giá đầy đủ hơn.

- Huấn luyện lại trong lần chạy notebook này: False
- Bảng trên được tính từ log đã lưu, không thay đổi trọng số hoặc dữ liệu test.
