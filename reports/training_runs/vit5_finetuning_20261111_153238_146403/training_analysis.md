# Phân tích kết quả huấn luyện

- Run ID: vit5_finetuning_20261111_153238_146403
- Nguồn: training_history.csv và training_summary.json của đúng run.
- Best epoch: 2
- Best global step: 500
- Best validation loss: 2.102667951223594
- Biểu đồ: [training_loss.png](figures/training_loss.png).

| Epoch | Train loss | Validation loss |
| --- | --- | --- |
| 1 | 2.399453 | 2.120815 |
| 2 | 1.891031 | 2.102668 |
| 3 | 1.567235 | 2.215740 |
| 4 | 1.310376 | 2.262777 |
| 5 | 1.104167 | 2.425660 |
| 6 | 0.932104 | 2.613600 |
| 7 | 0.793834 | 2.747355 |
| 8 | 0.679249 | 2.924537 |
| 9 | 0.587835 | 3.045973 |
| 10 | 0.517013 | 3.264683 |

Train loss giảm nhưng validation loss ở epoch cuối cao hơn mức tốt nhất. Đây là dấu hiệu học quá sát train; giữ checkpoint có validation loss nhỏ nhất.

Loss được tổng hợp theo token đích hợp lệ, bỏ qua nhãn padding -100. Loss thấp không bảo đảm mọi bản tóm tắt đều đúng hoặc không lặp; đọc kết quả ROUGE và phân tích lỗi để đánh giá đầy đủ hơn.

- Số liệu được tạo bởi luồng huấn luyện/resume của chính run này.
- Bảng trên được tính từ log đã lưu, không thay đổi trọng số hoặc dữ liệu test.
