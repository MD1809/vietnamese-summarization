# Phân tích lịch sử huấn luyện

- Run ID: `vit5_finetuning_20261105_181249_692559`.
- Nguồn số liệu: `training_history.csv`, đối chiếu với `training_summary.json` và `training_verification.json`.
- Dữ liệu: 2.000 mẫu train, 201 mẫu validation; 3 epoch, 750 optimizer update.
- Biểu đồ: [training_loss.png](figures/training_loss.png).

| Epoch | Global step | Train loss | Validation loss |
| --- | --- | --- | --- |
| 1 | 250 | 2.3996 | 2.1142 |
| 2 | 500 | 1.8862 | 2.1033 |
| 3 | 750 | 1.5655 | 2.1777 |

## Nhận xét

- Train loss giảm qua cả ba epoch, cho thấy mô hình tiếp tục khớp dữ liệu train tốt hơn theo phép đo trong lúc học.
- Validation loss giảm nhẹ từ epoch 1 sang epoch 2, rồi tăng ở epoch 3 trong khi train loss vẫn giảm. Đây là dấu hiệu bắt đầu overfitting trong lượt chạy này; ba mốc đo chưa đủ để kết luận mức độ nghiêm trọng hoặc xu hướng ở những epoch chưa chạy.
- Giữ `best` tại epoch 2 / step 500, validation loss đầy đủ là `2.103290012113482`. Giá trị này đã được đo lại sau khi tải checkpoint và khớp kết quả đã lưu.
- `last` ở epoch 3 / step 750 phục vụ tiếp tục học; checkpoint dùng để đánh giá là `best`. Không thay siêu tham số hoặc học thêm trong G5.7.

## Cách đọc biểu đồ

- Trục ngang là epoch, không phải từng optimizer step; mỗi đường có đúng ba mốc đo, không làm mượt hoặc thêm điểm giả.
- Cả hai loss được tổng hợp theo token đích hợp lệ, bỏ nhãn padding `-100`.
- Train loss là tổng hợp trong lúc trọng số thay đổi và dropout hoạt động. Validation loss được đo cuối epoch với trọng số cố định, `eval()` và `no_grad()`. Khoảng cách giữa hai đường không chỉ phụ thuộc overfitting; không coi chúng là hai phép đo hoàn toàn cùng điều kiện.
- Chưa sử dụng tập test. Loss này chưa thay cho đánh giá chất lượng tóm tắt bằng ROUGE và đọc ví dụ ở giai đoạn 6.
