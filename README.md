# Tóm tắt văn bản tiếng Việt bằng Transformer

Xây dựng hệ thống tóm tắt sinh văn bản tiếng Việt bằng cách tự fine-tune mô hình **ViT5-Base** trên bộ dữ liệu **WikiLingua tiếng Việt**.

Dự án sử dụng PyTorch để huấn luyện trên Google Colab, đánh giá bằng ROUGE-1, ROUGE-2 và ROUGE-L, so sánh với baseline trích xuất và xây dựng giao diện demo bằng Gradio.

## Cấu hình mới và trạng thái kiểm chứng

Cấu hình mới tham khảo [notebook VietAI/ViT5](https://github.com/vietai/ViT5/blob/151850bb988f8364c5499fbde241a998a6695c9d/examples/finetune_huggingface_example.ipynb), giữ vòng lặp PyTorch tự viết. [Checklist triển khai và kiểm tra thủ công](docs/vit5_configuration_checklist.md).

| Thông số | Run mới |
| --- | --- |
| Model / seed | ViT5-base, revision cố định / 42 |
| Nguồn / nhãn đích | 1024 / 256 token |
| Batch / accumulation | 1 / 4, batch hiệu dụng 4 |
| Optimizer | AdamW, nhóm không decay cho bias/normalization |
| LR / scheduler / warmup | `1e-5` / linear / 5% tổng bước dự kiến |
| Weight decay / clipping | 0.01 / max grad norm 1.0 |
| Precision | AMP FP16 khi train; trọng số chính, validation và inference FP32 |
| Epoch / early stopping | Tối đa 30 / patience 3, min delta 0 |
| Checkpoint | Best theo validation loss; last gồm optimizer, scheduler, scaler và RNG/DataLoader |
| Decoding làm mốc | Greedy, `max_length=256`, không sampling, không chặn lặp |

Đây chưa phải tái lập đầy đủ thí nghiệm VietAI: batch vật lý 1 thay vì 4; giữ validation, early stopping, split nhóm và ROUGE Unicode của dự án. `max_length` bao gồm decoder start và EOS; `max_new_tokens` của các báo cáo cũ chỉ tính token mới, gồm EOS. TextRank dùng ngân sách content-token bằng giới hạn nhãn, nên không coi hai ngân sách là đồng nhất tuyệt đối.

**Chưa chạy test, kiểm tra bộ nhớ T4, train hoặc evaluation cho cấu hình mới.** Chỉ rà soát mã và cú pháp/JSON. Metadata `pending_manual_verification` và các ô chưa đánh dấu trong checklist thể hiện việc còn chờ chạy thật; kết quả bên dưới thuộc run cũ.

## Chuẩn bị dữ liệu

Mở `notebooks/vietnamese_summarization_data_preparation.ipynb` trên Google Colab bằng runtime thường, không cần GPU. Dùng phiên mới và chạy lần lượt chín cell code từ trên xuống; các cell Markdown giải thích ngay trước từng phần. Nếu Colab yêu cầu khởi động lại sau khi cài thư viện, khởi động lại rồi chạy hai cell code đầu và tiếp tục từ cell import/cấu hình, không cần cài lại ngay.

Notebook lấy commit mới nhất trên `origin/main` khi `target_commit=""`, dùng cache WikiLingua trên Drive, chuẩn hóa/làm sạch và giữ toàn bộ các tập đã chia theo nhóm trong `wikilingua_splits.json`: **15.601 train / 2.034 validation / 1.942 test**. Không lấy tập con và không chia lại dữ liệu. Thống kê token chỉ dùng train; báo cáo cắt token dùng train/validation.

Đầu ra trên Drive: `MyDrive/vietnamese-summarization/data/processed/full/` chứa `train.jsonl`, `validation.jsonl`, `test.jsonl`; `data/manifests/full/` chứa `wikilingua_full.json`, manifest nội dung/hash, thống kê và báo cáo cắt token 1024/256 (`wikilingua_truncation_1024_256.json`). Đây là dữ liệu mặc định trong `configs/train.json`, không phụ thuộc run cũ. File đã có phải khớp nội dung; không ghi đè dữ liệu khác. Test chỉ dùng cho đánh giá cuối cùng.

Nơi lưu thống nhất: mã nguồn, notebook và cấu hình nằm trên GitHub; dữ liệu và kết quả chạy nằm trên Google Drive. Các đường dẫn tương đối trong `configs/train.json` được ghép với `drive_root`, không phải thư mục clone mã nguồn `/content/vietnamese-summarization`. Notebook tự tạo các thư mục cần thiết:

```text
MyDrive/vietnamese-summarization/
├── data/
│   ├── raw/
│   ├── processed/full/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── manifests/full/
├── checkpoints/<run_id>/
└── outputs/<run_id>/
```

Không cần chép JSONL, manifest đầy đủ hoặc trọng số vào repository. `.gitignore` loại các thư mục dữ liệu phát sinh này khỏi Git nếu chúng được tải về máy. Manifest nguồn/làm sạch/chia nhóm được giữ trong repository để đối chiếu; manifest đầy đủ và dữ liệu phát sinh nằm trên Drive.

## Huấn luyện một run mới

Mở `notebooks/vietnamese_summarization_train.ipynb` trên Colab với GPU T4. Notebook có 10 cell code: kết nối/lấy mã/cài thư viện → tokenize/DataLoader → tải ViT5-Base và AdamW → vòng lặp epoch/validation loss/last-best → log/biểu đồ/phân tích và tải ZIP. Sinh tóm tắt, so sánh decoding và chấm ROUGE nằm trong notebook evaluation riêng.

Trước run mới, push thay đổi và chạy notebook chuẩn bị dữ liệu trên Drive. Sau đó mở notebook train đã cập nhật và chạy từ đầu. Output cũ trong các notebook đã được xóa; báo cáo lịch sử trong `reports/` được giữ nguyên. Notebook evaluation giữ mặc định `evaluate_test=False`.

Mặc định `resume_run_id=""` tạo run mới từ ViT5-Base tiền huấn luyện. Cell 2 lấy commit mới nhất đã push lên `origin/main` khi `target_commit=""`, rồi cố định commit cho cả phiên; có thể điền hash để chọn phiên bản cụ thể. Muốn thử cấu hình khác, sửa `configs/train.json`, commit/push trước khi bắt đầu run mới. Không upload file mã thủ công.

Cell 7 chứa vòng lặp epoch; `src.train.train_epoch` và `train_accumulated_group` thực hiện forward/loss/backward, tích lũy gradient, clipping và optimizer.step. Sau mỗi epoch chạy validation loss, chọn best theo loss nhỏ nhất và lưu last để resume. Cấu hình hiện tại: nguồn/đích 1024/256, batch 1, accumulation 4, AdamW learning rate 1e-5, warmup 5%, linear scheduler, AMP FP16 và tối đa 30 epoch. Dữ liệu JSONL/manifest trên Drive phải đúng tập đã chuẩn bị; không đổi dữ liệu hoặc siêu tham số giữa run đang resume.

Early stopping cho run mới được cấu hình trong `configs/train.json`: `early_stopping.patience=3`, `min_delta=0.0`. Số 30 epoch là giới hạn tối đa; dừng sau 3 epoch liên tiếp mà validation loss không giảm. Với `min_delta` lớn hơn 0, loss phải giảm hơn ngưỡng này so với mốc cải thiện đáng kể gần nhất để đặt lại bộ đếm; checkpoint best vẫn luôn chọn loss nhỏ nhất. Lịch sử và last được lưu trước khi dừng, summary ghi `stopped_early` và `stop_reason`; đánh giá/sinh tóm tắt tiếp tục dùng best. Run cũ vẫn dùng commit/cấu hình gốc.

Nếu Colab ngắt, nhập đúng run ID vào `resume_run_id` ở cell 1, để `target_commit=""` ở cell 2 tự lấy commit của hồ sơ run rồi chạy lại từ đầu. Khôi phục model, optimizer, scheduler, AMP scaler, tiến độ và RNG/DataLoader từ last cuối epoch; epoch đang dở được chạy lại. Bộ đếm early stopping được lưu trong last; run đã đạt patience sẽ không chạy thêm epoch khi resume. Các file checkpoint phải còn đầy đủ. Nếu thư viện yêu cầu khởi động lại sau cài đặt, chạy lại cell 1, 2 rồi tiếp tục từ cell 4.

Kết quả nằm ở `Drive/outputs/<run_id>/`; trọng số ở `Drive/checkpoints/<run_id>/best/` và `last/`. Cell 10 tải ZIP training_results gồm config.json, bản sao manifest, CSV/summary, phân tích và figures/training_loss.png; không có trọng số. Giải nén giữ nguyên cấu trúc vào `reports/training_runs/<run_id>/`. ZIP là bản sao metadata/kết quả, không thay thế best/last trên Drive; cần đủ dung lượng cho cả checkpoint hiện tại và bản tạm khi lưu.

Run đầy đủ đã hoàn tất: `vit5_finetuning_20261114_083823_007233`, 15.601 train / 2.034 validation / 1.942 test. Early stopping dừng sau epoch 5, tổng 9.755 update; best ở epoch 2 / step 3.902, validation loss 1.9117193787786906. [Summary](reports/training_runs/vit5_finetuning_20261114_083823_007233/training_summary.json), [phân tích loss](reports/training_runs/vit5_finetuning_20261114_083823_007233/training_analysis.md) và [biểu đồ](reports/training_runs/vit5_finetuning_20261114_083823_007233/figures/training_loss.png). Run được tiếp tục từ checkpoint cuối epoch 3 sau khi đổi phiên; lịch sử đủ 5 epoch. JSONL dữ liệu và trọng số vẫn nằm trên Drive; báo cáo trong repository không thay thế checkpoint.

Validation `decoding_comparison_20261114_184049_880794` chọn beam 4, `no_repeat_ngram_size=4`, tối đa 128 token. ROUGE-1/2/L F1 × 100: greedy 31.45/16.16/27.19; beam 33.90/18.42/28.79; TextRank 24.22/9.54/16.12. [Báo cáo validation](reports/training_runs/vit5_finetuning_20261114_083823_007233/validation/decoding_comparison_20261114_184049_880794/decoding_comparison.json). Test `test_evaluation_20261115_011027_113129` đã hoàn tất đủ 1.942 mẫu: ViT5 beam 33.59/18.22/28.75; TextRank 24.26/9.61/16.29. [Điểm test](reports/training_runs/vit5_finetuning_20261114_083823_007233/test_evaluation_20261115_011027_113129/metrics.json), [bảng so sánh](reports/training_runs/vit5_finetuning_20261114_083823_007233/test_evaluation_20261115_011027_113129/comparison.csv). Cấu hình sinh được chọn trên validation trước test; metadata ghi rõ test từng được xem trong các thử nghiệm của dự án.

Run tập con trước đó: `vit5_finetuning_20261113_071230_169031`, 10 epoch/2.500 update, best epoch 2/step 500, validation loss 2.1058625596091947. Các báo cáo lịch sử được giữ riêng; không so sánh trực tiếp loss hoặc ROUGE với run đầy đủ vì tập đánh giá khác nhau.

## Đánh giá TextRank và ViT5 trong notebook riêng

Mở `notebooks/vietnamese_summarization_evaluation.ipynb` trong phiên Colab GPU mới, không cần chạy notebook train trước đó. Notebook có tám cell code. Để `evaluation_run_id=""` để chọn run hoàn tất mới nhất hoặc điền run cụ thể; cần config.json, training_summary.json, best và dữ liệu/manifest tương ứng trên Drive. Cấu hình lấy từ hồ sơ run, mã đánh giá lấy commit mới nhất đã push khi target_commit rỗng.

Chạy cell 1–5 để thiết lập và đo validation: ViT5 greedy/beam cùng TextRank. Đọc kết quả, rồi trong cell 6 chọn `selected_method="beam"` hoặc `"greedy"`, bật `evaluate_test=True` và chạy 6–8 để chấm test/xuất ZIP. Mặc định test tắt; tham chiếu chỉ được đưa vào chấm ROUGE sau khi có dự đoán. Không dùng điểm test để chỉnh thuật toán, và cần ghi rõ test đã từng được đánh giá trong dự án.

Nếu ngắt phiên sau validation: điền đúng run ID và Comparison ID ở cell 1 (hoặc dùng quy tắc tự chọn), chạy 1–4 rồi 6–8. Validation phải có đủ báo cáo TextRank/ViT5 do notebook này tạo; source/cấu hình đã đổi thì cần đo lại validation. Không cần cell tạm, không huấn luyện lại. Cell 5 tải ZIP validation để giải nén vào `reports/training_runs/<run_id>/validation/<comparison_id>/`. Cell 8 tải ZIP test, giải nén vào `reports/training_runs/<run_id>/<test_evaluation_id>/`, gồm dự đoán, ROUGE, comparison.csv, quyết định và hồ sơ nguồn. Giữ kết quả lịch sử riêng.

Notebook train đã được người thực hiện rút gọn còn cell 1–10; các kiểm thử train và evaluation được tách theo đúng nhiệm vụ. Dùng notebook evaluation để đánh giá mới; validation loss chọn best vẫn được giữ trong vòng huấn luyện.

## Chọn decoding và kiểm tra thủ công

Notebook evaluation đọc cấu hình sinh của chính run đã chọn, không tự bật chặn lặp. Run mới có mốc greedy không chặn lặp; cell validation so thêm beam 4. Nếu muốn thử `no_repeat_ngram_size=4`, sửa cấu hình tạm ở cell 4, chạy lại validation và đọc ví dụ trước khi chốt. Quy tắc đếm lặp câu chỉ phát hiện câu trùng nguyên văn sau chuẩn hóa, không phát hiện mọi kiểu lặp ngữ nghĩa.

Thứ tự chạy thủ công sau khi push các commit mới:

1. Mở phiên mới, chạy notebook chuẩn bị dữ liệu; đọc báo cáo truncation 1024/256 trên train/validation.
2. Mở Colab T4, chạy notebook train với `resume_run_id=""`. Theo dõi loss hữu hạn, peak VRAM, LR và số bước AMP bỏ qua; kiểm tra resume đúng last nếu cần. Warmup theo tổng bước của 30 epoch, không tính lại khi early stopping hoặc resume.
3. Chạy evaluation trên validation, giữ `evaluate_test=False`. Chốt greedy/beam và chặn lặp nếu có rồi mới bật test. Test đã từng được xem ở các thử nghiệm trước, không coi là tập hoàn toàn chưa từng quan sát.
4. Chạy demo bằng run/lượt đánh giá mới. Source hash của mã inference phải khớp báo cáo; các báo cáo cũ vẫn đọc được nhưng demo sẽ từ chối nếu mã thuật toán khác.
5. So run mới với run cũ trên cùng split và cùng decoding nếu muốn đánh giá riêng tác động của huấn luyện. Chưa có ROUGE hoặc best epoch mới để báo cáo.

`global_step` đếm cập nhật optimizer thành công. Nhóm AMP bị tràn gradient vẫn được tính vào số nhóm đã xử lý, nhưng không tăng scheduler/global step. CSV có LR cho bước tiếp theo, số cập nhật/thử/bỏ qua và peak VRAM từng epoch. Không resume checkpoint cũ bằng cấu hình mới; run cũ cần commit/cấu hình gốc. Báo cáo/checkpoint/hồ sơ cũ được giữ nguyên.

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
