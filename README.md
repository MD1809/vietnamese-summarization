# Tóm tắt văn bản tiếng Việt bằng Transformer

Xây dựng hệ thống tóm tắt sinh văn bản tiếng Việt bằng cách tự fine-tune mô hình **ViT5-Base** trên bộ dữ liệu **WikiLingua tiếng Việt**.

Dự án sử dụng PyTorch để huấn luyện trên Google Colab, đánh giá bằng ROUGE-1, ROUGE-2 và ROUGE-L, so sánh với baseline trích xuất và xây dựng giao diện demo bằng Gradio.

## Chuẩn bị dữ liệu

Mở `notebooks/vietnamese_summarization_data_preparation.ipynb` trên Google Colab bằng runtime thường, không cần GPU. Dùng phiên mới và chạy lần lượt chín cell code từ trên xuống; các cell Markdown giải thích ngay trước từng phần. Nếu Colab yêu cầu khởi động lại sau khi cài thư viện, khởi động lại rồi chạy hai cell code đầu và tiếp tục từ cell import/cấu hình, không cần cài lại ngay.

Notebook lấy mã nền `b21dc1b3313021c88ff37a834fda5b3cdb83b2f3` từ GitHub, dùng cache WikiLingua trên Drive, chuẩn hóa/làm sạch, chia nhóm và khôi phục đúng ID tập con từ manifest đã chốt. Chỉ có một luồng chạy, không tải mã thủ công hoặc chọn chế độ tạo/đọc dữ liệu. Thống kê token trên train không tạo batch; tokenization và DataLoader phục vụ học thuộc notebook huấn luyện.

Đầu ra trên Drive: `MyDrive/vietnamese-summarization/data/processed/` chứa `train.jsonl`, `validation.jsonl`, `test.jsonl` với 2.000/201/200 mẫu; `data/manifests/` chứa metadata đi kèm. Nội dung file đã có phải khớp, nếu khác chương trình dừng thay vì ghi đè. Không chọn lại ID hoặc dùng test để điều chỉnh mô hình. Sau khi chạy, lưu notebook với output cuối gọn về `notebooks/`; không cần tải JSONL vào source hoặc thay các manifest đã chốt trong Git.

## Huấn luyện một run mới

Mở `notebooks/vietnamese_summarization_train.ipynb` trên Colab với GPU T4. Notebook có 10 cell code: kết nối/lấy mã/cài thư viện → tokenize/DataLoader → tải ViT5-Base và AdamW → vòng lặp epoch/validation loss/last-best → log/biểu đồ/phân tích và tải ZIP. Sinh tóm tắt, so sánh decoding và chấm ROUGE nằm trong notebook evaluation riêng.

Mặc định `resume_run_id=""` tạo run mới từ ViT5-Base tiền huấn luyện. Cell 2 lấy commit mới nhất đã push lên `origin/main` khi `target_commit=""`, rồi cố định commit cho cả phiên; có thể điền hash để chọn phiên bản cụ thể. Muốn thử cấu hình khác, sửa `configs/train.json`, commit/push trước khi bắt đầu run mới. Không upload file mã thủ công.

Cell 7 chứa vòng lặp epoch; `src.train.train_epoch` và `train_accumulated_group` thực hiện forward/loss/backward, tích lũy gradient, clipping và optimizer.step. Sau mỗi epoch chạy validation loss, chọn best theo loss nhỏ nhất và lưu last để resume. Cấu hình hiện tại: nguồn/đích 1024/128, batch 1, accumulation 8, AdamW learning rate 5e-5 và 10 epoch. Dữ liệu JSONL/manifest trên Drive phải đúng tập đã chuẩn bị; không đổi dữ liệu hoặc siêu tham số giữa run đang resume.

Nếu Colab ngắt, nhập đúng run ID vào `resume_run_id` ở cell 1, để `target_commit=""` ở cell 2 tự lấy commit của hồ sơ run rồi chạy lại từ đầu. Khôi phục model, optimizer, tiến độ và RNG/DataLoader từ last cuối epoch; epoch đang dở được chạy lại. Các file checkpoint phải còn đầy đủ. Nếu thư viện yêu cầu khởi động lại sau cài đặt, chạy lại cell 1, 2 rồi tiếp tục từ cell 4.

Kết quả nằm ở `Drive/outputs/<run_id>/`; trọng số ở `Drive/checkpoints/<run_id>/best/` và `last/`. Cell 10 tải ZIP training_results gồm config.json, bản sao manifest, CSV/summary, phân tích và figures/training_loss.png; không có trọng số. Giải nén giữ nguyên cấu trúc vào `reports/training_runs/<run_id>/`. ZIP là bản sao metadata/kết quả, không thay thế best/last trên Drive; cần đủ dung lượng cho cả checkpoint hiện tại và bản tạm khi lưu.

Run mới đã đối chiếu: `vit5_finetuning_20261113_071230_169031`, 10 epoch/2500 update, best epoch 2/step 500, validation loss 2.1058625596091947. [Summary](reports/training_runs/vit5_finetuning_20261113_071230_169031/training_summary.json), [phân tích loss](reports/training_runs/vit5_finetuning_20261113_071230_169031/training_analysis.md) và [biểu đồ](reports/training_runs/vit5_finetuning_20261113_071230_169031/figures/training_loss.png) khớp notebook. Validation loss tăng từ epoch 3 trong khi train loss giảm; dùng best thay vì checkpoint cuối cho đánh giá/demo. Chưa có ROUGE TextRank/ViT5 của run mới; các báo cáo run cũ giữ riêng, không gán điểm lịch sử cho run này.

## Đánh giá TextRank và ViT5 trong notebook riêng

Mở `notebooks/vietnamese_summarization_evaluation.ipynb` trong phiên Colab GPU mới, không cần chạy notebook train trước đó. Notebook có tám cell code. Để `evaluation_run_id=""` để chọn run hoàn tất mới nhất hoặc điền run cụ thể; cần config.json, training_summary.json, best và dữ liệu/manifest tương ứng trên Drive. Cấu hình lấy từ hồ sơ run, mã đánh giá lấy commit mới nhất đã push khi target_commit rỗng.

Chạy cell 1–5 để thiết lập và đo validation: ViT5 greedy/beam cùng TextRank. Đọc kết quả, rồi trong cell 6 chọn `selected_method="beam"` hoặc `"greedy"`, bật `evaluate_test=True` và chạy 6–8 để chấm test/xuất ZIP. Mặc định test tắt; tham chiếu chỉ được đưa vào chấm ROUGE sau khi có dự đoán. Không dùng điểm test để chỉnh thuật toán, và cần ghi rõ test đã từng được đánh giá trong dự án.

Nếu ngắt phiên sau validation: điền đúng run ID và Comparison ID ở cell 1 (hoặc dùng quy tắc tự chọn), chạy 1–4 rồi 6–8. Validation phải có đủ báo cáo TextRank/ViT5 do notebook này tạo; source/cấu hình đã đổi thì cần đo lại validation. Không cần cell tạm, không huấn luyện lại. Cell 5 tải ZIP validation để giải nén vào `reports/training_runs/<run_id>/validation/<comparison_id>/`. Cell 8 tải ZIP test, giải nén vào `reports/training_runs/<run_id>/<test_evaluation_id>/`, gồm dự đoán, ROUGE, comparison.csv, quyết định và hồ sơ nguồn. Giữ kết quả lịch sử riêng.

Notebook train đã được người thực hiện rút gọn còn cell 1–10; các kiểm thử train và evaluation được tách theo đúng nhiệm vụ. Dùng notebook evaluation để đánh giá mới; validation loss chọn best vẫn được giữ trong vòng huấn luyện.

## Chạy demo từ phiên Colab mới

1. Mở `notebooks/vietnamese_summarization_demo.ipynb` trong phiên Colab mới với T4 GPU. Notebook gồm sáu cell code, không cần upload mã hoặc tải file kết quả thủ công.
2. Cell 1 kết nối Drive. Để `demo_run_id=""` để chọn run huấn luyện hoàn tất mới nhất; để `demo_evaluation_id=""` để chọn lượt test hoàn tất mới nhất trong chính run đó, hoặc điền ID cụ thể. Drive cần giữ hồ sơ/summary/manifest, dữ liệu validation, best và thư mục `outputs/<run_id>/test_evaluation_<UTC>/`. Run mới chưa có đánh giá hợp lệ thì dừng, không tự dùng lại model cũ.
3. Cell 2 lấy mã demo mới nhất đã push lên `origin/main` và cố định hash trong phiên. Cell 3 cài thư viện. Nếu Colab yêu cầu khởi động lại, chạy lại cell 1, 2 rồi tiếp tục từ cell 4.
4. Cell 4 đối chiếu best/run/config với metrics.json, decoding_selection.json, generation.json và báo cáo validation đi kèm của lượt test đã chọn. Cấu hình train lấy từ hồ sơ run; cấu hình sinh ViT5 và tham số TextRank lấy đúng kết quả đánh giá. Mã TextRank/inference/chuẩn hóa phải khớp hash trong báo cáo; nếu đã đổi thuật toán thì cần đánh giá lại. Cấu hình ghi tạm ngoài repository, không sửa hồ sơ train gốc. Chỉ đọc validation để chọn mẫu tham chiếu; không đọc predictions test hoặc chấm lại ROUGE. Cell in run ID, Evaluation ID và phương án sinh để kiểm tra trước khi tải model.
5. Cell 5 tải model/tokenizer một lần lên GPU; các lần bấm nút dùng lại model và tokenizer. Cell 6 mở Gradio: chọn mẫu validation hoặc nhập nguồn riêng, bấm **Tóm tắt văn bản** để chạy TextRank trên CPU và ViT5 bằng model đã tải. Hai kết quả đặt cạnh nhau (TextRank trái, ViT5 phải), bản tham chiếu đặt phía dưới nếu có. Hiển thị toàn bộ mẫu validation theo thứ tự file, không chọn mẫu dựa vào đầu ra. Văn bản tự nhập không có tham chiếu; sửa nguồn hoặc đổi mẫu sẽ xóa cả hai kết quả cũ. Bản tham chiếu không truyền vào hai phương pháp. Giao diện kiểm tra token nguồn chung, thông báo cắt nguồn/cắt câu TextRank nếu có, không tính ROUGE hoặc chạy Lead-3. Liên kết công khai chỉ hoạt động trong phiên Colab đang chạy. Sau khi cập nhật mã demo, dùng phiên mới và chạy lại cell 1–6, không thêm cell tạm.

Cấu hình trình diễn lấy từ lượt đánh giá tương ứng và được giữ ổn định trong phiên. Với run `vit5_finetuning_20261113_071230_169031`, lượt `test_evaluation_20261113_092101_564105` dùng greedy (`num_beams=1`), tối đa 128 token mới, không lấy mẫu; nguồn tối đa 1024 token. TextRank dùng đúng damping/ngưỡng/số vòng đã lưu, không dùng lại mặc định nếu chúng khác báo cáo. Có run hoặc lượt đánh giá mới thì mở phiên demo mới và chạy 1–6; không đổi model giữa các lần bấm. Văn bản dài có thể bị cắt và model vẫn có thể lặp, thiếu ý hoặc sai thông tin. Lần đồng bộ này chưa thêm bảng ROUGE vào giao diện.

## Kết quả dự phòng khi báo cáo

`reports/demo_examples.jsonl` giữ ba ví dụ lịch sử đã tạo trước khi dọn notebook; metadata ghi đúng run sinh chúng. Notebook demo mới chỉ mở ứng dụng, không tạo lại bộ ví dụ này. Sau khi chạy demo, tải notebook về `notebooks/vietnamese_summarization_demo.ipynb` nếu cần lưu trạng thái chạy.

Mỗi dòng JSON là một ví dụ gồm `example_id`, `source`, `summary`, `message` và `metadata` ghi checkpoint, cấu hình, phiên bản mã/thư viện, thời điểm sinh. Ba đầu vào tự soạn dùng để trình diễn; không coi chúng là tập đánh giá độc lập hoặc khẳng định chưa xuất hiện trong dữ liệu huấn luyện.

Khi GPU hoặc mạng không khả dụng, mở bản JSONL đã tải về trên máy bằng trình soạn thảo và trình bày `source` cùng `summary`. Nêu rõ đây là **kết quả đã lưu**, không phải suy luận trực tiếp. Giữ nguyên đầu ra thật kể cả khi có lặp hoặc thiếu ý. Phương án này không cần chạy Gradio, kết nối Drive hoặc tải model.
