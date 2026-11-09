# Tóm tắt văn bản tiếng Việt bằng Transformer

Xây dựng hệ thống tóm tắt sinh văn bản tiếng Việt bằng cách tự fine-tune mô hình **ViT5-Base** trên bộ dữ liệu **WikiLingua tiếng Việt**.

Dự án sử dụng PyTorch để huấn luyện trên Google Colab, đánh giá bằng ROUGE-1, ROUGE-2 và ROUGE-L, so sánh với baseline trích xuất và xây dựng giao diện demo bằng Gradio.

## Chạy demo từ phiên Colab mới

1. Mở `notebooks/vietnamese_summarization_demo.ipynb` trong Google Colab và chọn runtime T4 GPU. Dùng chức năng ngắt kết nối và xóa runtime cũ để bắt đầu phiên mới; chỉ xóa output không xóa các biến trong bộ nhớ.
2. Drive cần có `MyDrive/vietnamese-summarization/checkpoints/vit5_finetuning_20261106_161925_175323/best/`. Thư mục chứa model/tokenizer và `selection.json`; trọng số không nằm trong GitHub. Đây là checkpoint người thực hiện đã lưu trên Drive của mình, chưa cung cấp liên kết tải công khai.
3. Chạy cell 1–4 để kết nối Drive, lấy mã nền `c38254699f346b9c1205cb2b3d567fa04fbbe174`, cài thư viện và kiểm tra GPU. Nếu Colab yêu cầu khởi động lại sau khi cài, khởi động lại rồi chạy cell 1, 2 và tiếp tục từ cell 4.
4. Cell 5 import `app.gradio_app` trực tiếp từ repository; không chọn hoặc tải file mã thủ công. Cell 6 tải model/tokenizer một lần; output cần có `cuda:0`, best epoch 2, global step 500 và validation loss khoảng 2.110765.
5. Chạy cell 7 để mở Gradio, nhập văn bản và bấm **Tóm tắt**. Liên kết công khai chỉ dùng khi ứng dụng còn hoạt động trong phiên Colab. Cell 8 kiểm tra xử lý đầu vào; cell 9 lưu ba ví dụ dự phòng, không huấn luyện lại hoặc cập nhật trọng số.

Cấu hình trình diễn: nguồn tối đa 1024 token, tối đa 128 token mới, beam 4 và không lấy mẫu ngẫu nhiên. Văn bản vượt giới hạn chỉ được tóm tắt từ phần đầu; giao diện thông báo số token đã giữ. Model vẫn có thể lặp, thiếu ý hoặc sinh sai thông tin; xem phân tích tại `reports/test_evaluation/test_evaluation_20261108_183331_879067/error_analysis.md`.

## Kết quả dự phòng khi báo cáo

Sau khi chạy cell 9, lưu file tải về thành `reports/demo_examples.jsonl`. Bản sao trên Drive nằm trong `outputs/demo_fallback_<thời gian UTC>/demo_examples.jsonl`; cell in đường dẫn cụ thể. Giữ cùng notebook đã lưu output để đối chiếu lượt chạy.

Mỗi dòng JSON là một ví dụ gồm `example_id`, `source`, `summary`, `message` và `metadata` ghi checkpoint, cấu hình, phiên bản mã/thư viện, thời điểm sinh. Ba đầu vào tự soạn dùng để trình diễn; không coi chúng là tập đánh giá độc lập hoặc khẳng định chưa xuất hiện trong dữ liệu huấn luyện.

Khi GPU hoặc mạng không khả dụng, mở bản JSONL đã tải về trên máy bằng trình soạn thảo và trình bày `source` cùng `summary`. Nêu rõ đây là **kết quả đã lưu**, không phải suy luận trực tiếp. Giữ nguyên đầu ra thật kể cả khi có lặp hoặc thiếu ý. Phương án này không cần chạy Gradio, kết nối Drive hoặc tải model.
