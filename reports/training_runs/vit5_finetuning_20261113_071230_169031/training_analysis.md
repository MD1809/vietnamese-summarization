# Phân tích kết quả huấn luyện

- Run ID: vit5_finetuning_20261113_071230_169031
- Nguồn: training_history.csv và training_summary.json của đúng run.
- Best epoch: 2
- Best global step: 500
- Best validation loss: 2.1058625596091947
- Biểu đồ: [training_loss.png](figures/training_loss.png).

| Epoch | Train loss | Validation loss |
| --- | --- | --- |
| 1 | 2.398543 | 2.115507 |
| 2 | 1.887079 | 2.105863 |
| 3 | 1.563590 | 2.193034 |
| 4 | 1.312433 | 2.255782 |
| 5 | 1.107884 | 2.414947 |
| 6 | 0.930741 | 2.625401 |
| 7 | 0.793361 | 2.774798 |
| 8 | 0.689305 | 2.875101 |
| 9 | 0.592972 | 3.070986 |
| 10 | 0.520566 | 3.317988 |

Train loss giảm nhưng validation loss ở epoch cuối cao hơn mức tốt nhất. Đây là dấu hiệu học quá sát train; giữ checkpoint có validation loss nhỏ nhất.

Loss được tổng hợp theo token đích hợp lệ, bỏ qua nhãn padding -100. Loss thấp không bảo đảm mọi bản tóm tắt đều đúng hoặc không lặp; đọc kết quả ROUGE và phân tích lỗi để đánh giá đầy đủ hơn.

- Số liệu được tạo bởi luồng huấn luyện/resume của chính run này.
- Bảng trên được tính từ log đã lưu, không thay đổi trọng số hoặc dữ liệu test.
