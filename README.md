# Tóm tắt văn bản tiếng Việt bằng Transformer

Xây dựng hệ thống tóm tắt sinh văn bản tiếng Việt bằng cách tự fine-tune mô hình **ViT5-Base** trên bộ dữ liệu **WikiLingua tiếng Việt**.

Dự án sử dụng PyTorch để huấn luyện trên Google Colab, đánh giá bằng ROUGE-1, ROUGE-2 và ROUGE-L, so sánh với baseline trích xuất và xây dựng giao diện demo bằng Gradio.

## Chuẩn bị dữ liệu

Mở `notebooks/vietnamese_summarization_data_preparation.ipynb` trên Google Colab bằng runtime thường, không cần GPU. Dùng phiên mới và chạy lần lượt chín cell code từ trên xuống; các cell Markdown giải thích ngay trước từng phần. Nếu Colab yêu cầu khởi động lại sau khi cài thư viện, khởi động lại rồi chạy hai cell code đầu và tiếp tục từ cell import/cấu hình, không cần cài lại ngay.

Notebook lấy mã nền `b21dc1b3313021c88ff37a834fda5b3cdb83b2f3` từ GitHub, dùng cache WikiLingua trên Drive, chuẩn hóa/làm sạch, chia nhóm và khôi phục đúng ID tập con từ manifest đã chốt. Chỉ có một luồng chạy, không tải mã thủ công hoặc chọn chế độ tạo/đọc dữ liệu. Thống kê token trên train không tạo batch; tokenization và DataLoader phục vụ học thuộc notebook huấn luyện.

Đầu ra trên Drive: `MyDrive/vietnamese-summarization/data/processed/` chứa `train.jsonl`, `validation.jsonl`, `test.jsonl` với 2.000/201/200 mẫu; `data/manifests/` chứa metadata đi kèm. Nội dung file đã có phải khớp, nếu khác chương trình dừng thay vì ghi đè. Không chọn lại ID hoặc dùng test để điều chỉnh mô hình. Sau khi chạy, lưu notebook với output cuối gọn về `notebooks/`; không cần tải JSONL vào source hoặc thay các manifest đã chốt trong Git.

## Huấn luyện một run mới

Mở `notebooks/vietnamese_summarization_train.ipynb` trên Colab với GPU T4. Notebook có 13 cell code, chạy theo thứ tự: kết nối/lấy mã → tokenize/DataLoader → tải ViT5-Base và AdamW → vòng lặp epoch → validation và last/best → loss/phân tích → sinh tóm tắt → so sánh greedy/beam trên validation. Mã module nền là `b5cb961ed118f014d8fc4c5c5af607d429361949`; không tải file mã thủ công.

Mặc định `resume_run_id=""`: fine-tune mới từ ViT5-Base tiền huấn luyện, tạo run ID và thư mục riêng trên Drive. Không có cờ tắt huấn luyện hoặc đường dẫn gắn cứng kết quả run cũ. Cell 7 chứa vòng lặp epoch; `src.train.train_epoch` và `train_accumulated_group` thực hiện forward/loss/backward, tích lũy gradient, clipping và optimizer.step. Sau mỗi epoch chạy validation, chọn best theo loss và lưu last.

Đầu vào là JSONL/manifest trên Drive do notebook dữ liệu tạo. Cấu hình đang dùng: nguồn/đích 1024/128, batch 1, accumulation 8, AdamW learning rate 5e-5, tối đa 10 epoch. Muốn thử cấu hình khác, sửa `configs/train.json`, commit/push và cập nhật `target_commit` trước khi bắt đầu run mới; thay quy mô dữ liệu cần chuẩn bị manifest nhất quán. Không sửa số epoch/learning rate giữa một run đang resume.

Nếu Colab ngắt, nhập run ID đã lưu vào `resume_run_id`, dùng đúng commit/cấu hình rồi chạy lại từ đầu. Model, optimizer, tiến độ và RNG/DataLoader được khôi phục từ last ở cuối epoch; epoch đang dở được chạy lại. Run có source patch/snapshot phát triển cần khôi phục đúng source riêng, notebook sạch không tự bỏ qua kiểm tra tương thích.

Mọi kết quả lấy từ run vừa tạo hoặc resume: log, summary, phân tích loss và `sample_summary.json` nằm trong `Drive/outputs/<run_id>/`; trọng số ở `Drive/checkpoints/<run_id>/last/` và `best/`. So sánh validation ghi vào `Drive/outputs/decoding_comparison_<UTC>/` và ghi đúng checkpoint run ID. Để giữ kết quả run mới trong source, dùng `reports/training_runs/<run_id>/`, đặt PNG trong `figures/`; không ghi đè `reports/recovery/` hoặc báo cáo lịch sử.

Cell test cuối mặc định `evaluate_test=False`. Sau khi xem validation và chốt, điền `selected_method="greedy"` hoặc `"beam"` rồi mới bật test. Quyết định/cấu hình được ghi trước khi đọc test, baseline và ViT5 dùng cùng nguồn, kết quả thuộc đúng best của run mới và nằm trong `outputs/<run_id>/test_evaluation_<UTC>/`. Test từng được xem trong dự án, nên phải nêu điều đó khi báo cáo các lượt cải tiến; không dùng test để chọn siêu tham số.

Chạy toàn bộ luồng chính thực sự tốn thời gian huấn luyện và sinh trên validation. Nếu Colab yêu cầu khởi động lại sau cài thư viện, chạy hai cell code đầu rồi tiếp tục từ cell import/cấu hình. Kết quả báo cáo trước đây vẫn được giữ trong `reports/recovery/`, `reports/decoding_comparison/` và `reports/test_evaluation/`; notebook mới không đọc chúng để giả làm đầu ra của run vừa huấn luyện.

Lượt chạy đã kiểm chứng: `vit5_finetuning_20261111_153238_146403`, đủ 10 epoch/2.500 update, best epoch 2/step 500, validation loss 2.102667951223594. [Log và summary](reports/training_runs/vit5_finetuning_20261111_153238_146403/training_summary.json), [phân tích](reports/training_runs/vit5_finetuning_20261111_153238_146403/training_analysis.md) và [biểu đồ](reports/training_runs/vit5_finetuning_20261111_153238_146403/figures/training_loss.png). Notebook ghi validation ROUGE-L F1 x100: greedy 20.9959, beam 4 là 21.3983 trên 201 mẫu; test của run mới chưa chạy. Điểm test lịch sử không được gán cho run này.

## Đánh giá TextRank và ViT5 trong notebook riêng

Mở `notebooks/vietnamese_summarization_evaluation.ipynb` trong phiên Colab GPU mới, không cần chạy notebook train trước đó. Notebook có tám cell code. Để `evaluation_run_id=""` để chọn run hoàn tất mới nhất hoặc điền run cụ thể; cần config.json, training_summary.json, best và dữ liệu/manifest tương ứng trên Drive. Cấu hình lấy từ hồ sơ run, mã đánh giá lấy commit mới nhất đã push khi target_commit rỗng.

Chạy cell 1–5 để thiết lập và đo validation: ViT5 greedy/beam cùng TextRank. Đọc kết quả, rồi trong cell 6 chọn `selected_method="beam"` hoặc `"greedy"`, bật `evaluate_test=True` và chạy 6–8 để chấm test/xuất ZIP. Mặc định test tắt; tham chiếu chỉ được đưa vào chấm ROUGE sau khi có dự đoán. Không dùng điểm test để chỉnh thuật toán, và cần ghi rõ test đã từng được đánh giá trong dự án.

Nếu ngắt phiên sau validation: điền đúng run ID và Comparison ID ở cell 1 (hoặc dùng quy tắc tự chọn), chạy 1–4 rồi 6–8. Validation phải có đủ báo cáo TextRank/ViT5 do notebook này tạo; source/cấu hình đã đổi thì cần đo lại validation. Không cần cell tạm, không huấn luyện lại. Cell 5 tải ZIP validation để giải nén vào `reports/training_runs/<run_id>/validation/<comparison_id>/`. Cell 8 tải ZIP test, giải nén vào `reports/training_runs/<run_id>/<test_evaluation_id>/`, gồm dự đoán, ROUGE, comparison.csv, quyết định và hồ sơ nguồn. Giữ kết quả lịch sử riêng.

Theo lựa chọn tách trách nhiệm, người thực hiện sẽ tự bỏ cell code 11–13 và các tiêu đề tương ứng trong notebook train sau khi run đang chạy hoàn tất. Giữ cell 1–10, gồm validation loss chọn best/checkpoint/phân tích loss. Source bàn giao hiện chưa tự sửa notebook train trong task tạo evaluation này. Phần hướng dẫn train phía trên phản ánh bản trước khi tách; dùng notebook evaluation để đánh giá mới, không chạy chấm lại cả hai nơi.

## Chạy demo từ phiên Colab mới

1. Mở `notebooks/vietnamese_summarization_demo.ipynb` trong phiên Colab mới với T4 GPU. Notebook gồm sáu cell code, không cần upload mã hoặc tải file kết quả thủ công.
2. Cell 1 kết nối Drive. Để `demo_run_id=""` để chọn run mới nhất có `training_summary.json` ghi hoàn tất, theo thời gian tạo trong tên run; có thể điền một run ID hoàn tất cụ thể. Run đang huấn luyện dở chưa được chọn. Drive cần giữ `outputs/<run_id>/config.json`, `training_summary.json`, bản sao processed manifest, dữ liệu validation và `checkpoints/<run_id>/best/`.
3. Cell 2 lấy mã demo mới nhất đã push lên `origin/main` và cố định hash trong phiên. Cell 3 cài thư viện. Nếu Colab yêu cầu khởi động lại, chạy lại cell 1, 2 rồi tiếp tục từ cell 4.
4. Cell 4 kiểm tra run/best, đọc cấu hình train/generation từ hồ sơ run và đọc các mẫu validation đúng hash manifest. Cấu hình được lưu tạm ngoài repository, không thay configs của lần train đang dở. Không đọc test hoặc tải lại WikiLingua. Nếu best của run đã chọn bị thiếu hoặc không khớp, dừng với lỗi rõ ràng, không tự chuyển sang model cũ.
5. Cell 5 tải model/tokenizer một lần lên GPU; các lần bấm nút dùng lại model và tokenizer. Cell 6 mở Gradio: chọn mẫu validation hoặc nhập nguồn riêng, bấm **Tóm tắt văn bản** để chạy TextRank trên CPU và ViT5 bằng model đã tải. Hai kết quả đặt cạnh nhau (TextRank trái, ViT5 phải), bản tham chiếu đặt phía dưới nếu có. Hiển thị toàn bộ mẫu validation theo thứ tự file, không chọn mẫu dựa vào đầu ra. Văn bản tự nhập không có tham chiếu; sửa nguồn hoặc đổi mẫu sẽ xóa cả hai kết quả cũ. Bản tham chiếu không truyền vào hai phương pháp. Giao diện kiểm tra token nguồn chung, thông báo cắt nguồn/cắt câu TextRank nếu có, không tính ROUGE hoặc chạy Lead-3. Liên kết công khai chỉ hoạt động trong phiên Colab đang chạy. Sau khi cập nhật mã demo, dùng phiên mới và chạy lại cell 1–6, không thêm cell tạm.

Cấu hình trình diễn: nguồn tối đa 1024 token, tối đa 128 token mới, beam 4 và không lấy mẫu ngẫu nhiên. Văn bản vượt giới hạn chỉ được tóm tắt từ phần đầu; giao diện thông báo số token đã giữ. Model vẫn có thể lặp, thiếu ý hoặc sinh sai thông tin; xem phân tích tại `reports/test_evaluation/test_evaluation_20261108_183331_879067/error_analysis.md`.

## Kết quả dự phòng khi báo cáo

`reports/demo_examples.jsonl` giữ ba ví dụ lịch sử đã tạo trước khi dọn notebook; metadata ghi đúng run sinh chúng. Notebook demo mới chỉ mở ứng dụng, không tạo lại bộ ví dụ này. Sau khi chạy demo, tải notebook về `notebooks/vietnamese_summarization_demo.ipynb` nếu cần lưu trạng thái chạy.

Mỗi dòng JSON là một ví dụ gồm `example_id`, `source`, `summary`, `message` và `metadata` ghi checkpoint, cấu hình, phiên bản mã/thư viện, thời điểm sinh. Ba đầu vào tự soạn dùng để trình diễn; không coi chúng là tập đánh giá độc lập hoặc khẳng định chưa xuất hiện trong dữ liệu huấn luyện.

Khi GPU hoặc mạng không khả dụng, mở bản JSONL đã tải về trên máy bằng trình soạn thảo và trình bày `source` cùng `summary`. Nêu rõ đây là **kết quả đã lưu**, không phải suy luận trực tiếp. Giữ nguyên đầu ra thật kể cả khi có lặp hoặc thiếu ý. Phương án này không cần chạy Gradio, kết nối Drive hoặc tải model.
