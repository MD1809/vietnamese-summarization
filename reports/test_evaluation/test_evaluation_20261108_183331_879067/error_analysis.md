# Phân tích chất lượng và lỗi tóm tắt — G6.6

Ngày phân tích: 09/11/2026. Evaluation: `test_evaluation_20261108_183331_879067`. Best: epoch 2/step 500 của run `vit5_finetuning_20261106_161925_175323`; beam 4, do_sample=false, tối đa 128 token mới. Giữ nguyên kết quả và cấu hình G6.5.

Dữ liệu đối chiếu: [predictions.jsonl](predictions.jsonl), [metrics.json](metrics.json), [comparison.csv](comparison.csv). Nguồn, tham chiếu và hai dự đoán nguyên vẹn của mỗi ID nằm trong JSONL; các đoạn dẫn dưới đây không thay đổi dữ liệu chấm.

## 1. Chọn mẫu và phạm vi

- Lấy đúng 10 vị trí bằng Python 3.13: `random.Random(42).sample(range(200), 10)`, trên thứ tự dòng của predictions.jsonl đã đánh giá.
- Chỉ số tính từ 0: `[163, 28, 6, 189, 70, 62, 57, 35, 188, 26]`. Giữ nguyên thứ tự lấy mẫu và các ID dưới đây, không chọn lại theo điểm hoặc nội dung.
- Không sinh lại, huấn luyện, đổi decoding, sửa tham chiếu hoặc tính lại bảng điểm test đã công bố.
- Đối chiếu độ trung thành và bao phủ so với nguồn, dùng tham chiếu làm thêm một căn cứ. Tham chiếu có thể có nhiễu hoặc nêu ý không được diễn đạt rõ trong nguồn; không tự coi mọi khác biệt với tham chiếu là lỗi model.
- Với các mẫu sức khỏe, chỉ phân tích quan hệ giữa các văn bản, không thẩm định hay khuyến nghị y khoa.
- Các câu dự đoán lặp liên tiếp được trình bày gộp và ghi đúng số lần để dễ đọc. JSONL giữ nguyên mọi lần lặp.

## 2. Tóm tắt 10 mẫu

Điểm dưới đây là ROUGE-L F1 x100, dùng để đối chiếu; kết luận định tính dựa vào nội dung nguồn và đầu ra.

| Mẫu | Chỉ số dòng (từ 0) | Nội dung | Token nguồn đầy đủ/thực dùng | Baseline | ViT5 | Nhận xét chính |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 163 | Tết tóc kiểu Pháp | 285/285 | 14.81 | 34.92 | Sai hành động; lặp câu; thiếu ý |
| 2 | 28 | Biện pháp bảo vệ trong một văn bản sức khỏe | 464/464 | 10.75 | 7.41 | Lặp câu; thiếu ý/điều kiện |
| 3 | 6 | Thưởng thức dưa Kiwano | 205/205 | 10.00 | 20.51 | Giữ được một phần ý; lặp câu; thiếu lựa chọn |
| 4 | 189 | Tập luyện và thay đổi vóc dáng | 1294/1024 | 11.43 | 22.56 | Nguồn bị cắt; lặp nhiều; thiếu ý |
| 5 | 70 | Thực hiện mục tiêu dài hạn | 880/880 | 11.90 | 14.55 | Giữ được hai ý; gần lặp ý; thiếu bao phủ |
| 6 | 62 | Cấp quyền vị trí cho Chrome trên iPhone | 101/101 | 9.72 | 5.23 | Sai/thiếu quy trình; lặp nhiều; nhãn có metadata |
| 7 | 57 | Chuẩn bị cơ thể trước sinh | 940/940 | 7.84 | 20.93 | Sai mốc thời gian; lặp câu; thiếu ý |
| 8 | 35 | Trao đổi với bác sĩ về sỏi a-mi-đan | 305/305 | 20.75 | 30.77 | Giữ được hành động chính; lặp câu; bỏ các hướng khác |
| 9 | 188 | Chọn ớt chuông để đông lạnh | 119/119 | 10.00 | 35.29 | Ý chọn quả đúng một phần; lặp câu; mất điều kiện |
| 10 | 26 | Vẽ chú mèo | 118/118 | 20.69 | 8.70 | Lặp toàn đầu ra; thiếu nhiều bước |

## 3. Đối chiếu từng mẫu

### Mẫu 1 — Tết tóc kiểu Pháp

**ID:** `vi_cfb776d6a63a6a09d40fb0d1165cdf48a8e28bffba81d90b6c7ad62ee8b67b6c`. Chỉ số dòng 163; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Nếu bạn muốn tết thấp thì chỉ cần nhớ túm hết phần tóc phía trên tai để làm phần tóc phía trên của bím tóc.

> Thêm một cây kẹp bắt mắt hoặc nhiều kẹp dọc theo bím tóc là một cách hay để bím tóc trông xinh xắn hơn.

**Tham chiếu:** Lấy phần tóc phía trên của bím tóc. Giữ bím tóc bằng chai xịt tạo kiểu (không bắt buộc). Tô điểm cho tóc (không bắt buộc).

**Baseline:** “Với bím tóc kiểu Pháp truyền thống, đây sẽ là phần tóc phía trước, ở gần trán và đỉnh đầu.” “Bạn không cần phải tết tóc kiểu Pháp ngay trên đỉnh đầu.” “Đây là cách dễ nhất để học nhưng trên thực tế bạn có thể tết tóc kiểu Pháp từ bất kỳ chỗ nào.”

**ViT5:** “Cắt bỏ phần tóc phía trên trán và đỉnh đầu.” “Dùng chai xịt tạo kiểu hoặc gel xịt tóc.” *(lặp liên tiếp 3 lần)*

- **Phần giữ đúng:** ViT5 giữ được ý dùng chai xịt/gel để giữ bím tóc, gần với mục thứ hai của tham chiếu.
- **Lỗi/hạn chế và bằng chứng:** ViT5 đổi thao tác lấy/túm tóc thành “Cắt bỏ phần tóc phía trên trán và đỉnh đầu.” Đây là hành động không được nguồn hỗ trợ và khác thao tác của tham chiếu. Câu dùng chai xịt/gel lặp nguyên văn 3 lần; phần trang trí và tính không bắt buộc không được giữ.
- **Đối chiếu baseline và cách diễn giải:** Baseline giữ đúng các mô tả về vị trí bắt đầu tết nhưng chỉ lấy đoạn đầu; không đến được phần giữ nếp và trang trí. ViT5 có ROUGE-L cao hơn nhưng chứa lỗi hành động rõ ràng, nên điểm cao hơn không đồng nghĩa trung thành hơn ở mọi chi tiết.

### Mẫu 2 — Biện pháp bảo vệ trong một văn bản sức khỏe

**ID:** `vi_23d1c54815ae67e5de4fef1136c51af002af73ae36576be4b9c8a0d8574a23ae`. Chỉ số dòng 28; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Bao cao su nam sẽ giúp ngừa thai, tránh lây nhiễm bệnh, và giúp cho dương vật không bị dính kinh nguyệt.

> Hãy nhớ rằng loại cốc này không có chức năng tránh thai.

**Tham chiếu:** Sử dụng biện pháp tránh thai. Mang loại cốc mềm giúp ngăn kinh nguyệt chảy ra ngoài. Tinh trùng có thể sống sót trong vòng 72 giờ.

**Baseline:** “Trong thời gian có kinh nguyệt, bạn sẽ có nguy cơ cao mắc các bệnh lây truyền qua đường tình dục và viêm nhiễm vùng chậu.” “Hơn nữa, bạn vẫn có thể mang thai trong thời gian đèn đỏ.” “Dưới đây là một số biện pháp phòng ngừa hiệu quả: Màng ngăn âm đạo làm hai nhiệm vụ ngăn ngừa thai và ức chế kinh nguyệt.”

**ViT5:** “Tránh các bệnh lây truyền qua đường tình dục.” “Sử dụng bao cao su nữ.” “Sử dụng bao cao su nam.” “Sử dụng bao cao su nữ.”

- **Phần giữ đúng:** ViT5 nhận ra chủ đề biện pháp bảo vệ và nêu hai loại bao cao su có trong nguồn.
- **Lỗi/hạn chế và bằng chứng:** Câu “Sử dụng bao cao su nữ.” xuất hiện 2 lần. Không giữ phần cốc mềm cùng giới hạn chức năng của nó, và không giữ mục về thời gian mà tham chiếu nêu. Không coi việc đổi thứ tự hai loại bao cao su là lỗi trình tự bắt buộc, vì nguồn trình bày các lựa chọn chứ không phải chuỗi thao tác.
- **Đối chiếu baseline và cách diễn giải:** Baseline giữ phần mở đầu về nguy cơ và một biện pháp đầu tiên, cũng bỏ nhiều ý về sau. Trong mẫu này ViT5 thấp hơn baseline về ROUGE-L; nhận xét chỉ dựa vào mức bao phủ và trùng khớp của dữ liệu, không xác nhận tính đúng y khoa của nguồn.

### Mẫu 3 — Thưởng thức dưa Kiwano

**ID:** `vi_053dc7811331a6de2ef712dfea6a19f7132b4fee0f0b3f438bb8497633ef1b4d`. Chỉ số dòng 6; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Bạn có thể cắn từng miếng nhỏ rồi lựa bỏ hạt và nhổ ra ngoài hoặc cắn một miếng lớn và nhai luôn cả hạt.

> Nếu thích, bạn cũng có thể múc hạt quả Kiwano ra bát và dùng muỗng để thưởng thức.

**Tham chiếu:** Cầm một nửa quả đặt ở ngang tầm miệng. Thưởng thức. Cân nhắc việc múc hạt ra ngoài.

**Baseline:** “Bóp quả dưa từ dưới một cách từ từ nhưng chắc chắn.” “Mỗi một bọc nhỏ màu xanh sẽ chứa một hạt gần giống như hạt của quả dưa chuột.” “Khi bóp nhẹ, hạt quả sẽ dễ dàng trồi lên bề mặt.”

**ViT5:** “Bóp một bọc nhỏ màu xanh xung quanh hạt dưa Kiwano.” “Thưởng thức hạt dưa Kiwano.” *(lặp liên tiếp 2 lần)*

- **Phần giữ đúng:** ViT5 giữ được chủ đề Kiwano và ý thưởng thức phần hạt/quả, ngắn hơn đoạn diễn giải của baseline.
- **Lỗi/hạn chế và bằng chứng:** “Thưởng thức hạt dưa Kiwano.” lặp 2 lần; tùy chọn múc ra bát/dùng muỗng bị bỏ. “Bóp một bọc nhỏ màu xanh” diễn đạt khác đoạn nguồn “Bóp quả dưa từ dưới”; chỉ ghi nhận sự thay đổi đối tượng mô tả, chưa gán một lỗi thực tế chắc chắn chỉ từ cách diễn đạt này.
- **Đối chiếu baseline và cách diễn giải:** Baseline chép đúng phần bóp quả và túi hạt nhưng bỏ các cách thưởng thức ở phần sau. Tham chiếu có mục cầm nửa quả ngang miệng chưa được nguồn mô tả trực tiếp; không quy toàn bộ chênh lệch với mục này thành lỗi bịa hoặc thiếu do model.

### Mẫu 4 — Tập luyện và thay đổi vóc dáng

**ID:** `vi_f34cbb7e7d7c67b18a0749ddf01f2b1e8fb3dde6ec11b2bac92d30ca9398faef`. Chỉ số dòng 189; nguồn bị cắt.

**Nguồn, trích bằng chứng:**

> Để giảm lượng mỡ cơ thể và tăng cơ, hãy tập nhiều bài tập tăng nhịp tim hay tập thể dục nhịp điệu.

> Một điều nữa bạn nên chú ý là tập các bài tập cho lưng.

**Tham chiếu:** Hãy chuẩn bị cơ thể bạn cho quá trình tăng cơ bắp và giảm béo. Tăng các bài tập cardio. Hãy xen kẽ nghỉ ngơi giữa những lần vận động cường độ cao. Hãy tập nhiều bài tập cho các bộ phận khác nhau của cơ thể. Làm rắn chắc các cơ quanh hông, đùi, bụng và ngực bằng các bài tập cơ bắp.

**Baseline:** “Các thay đổi tích cực của cơ thể bạn không chỉ là một cơ thể đẹp hơn mà còn là tinh thần tốt hơn.” “Hãy luôn thoải mái với bản thân để có thể thấy kết quả rõ ràng nhất.” “Ngủ.”

**ViT5:** “Hãy thoải mái với bản thân.” “Tập thể dục.” “Tập các bài tập cho lưng.” *(lặp liên tiếp 10 lần)*

- **Phần giữ đúng:** ViT5 giữ được ý thoải mái với bản thân và tập luyện; câu về lưng có căn cứ trong nguồn.
- **Lỗi/hạn chế và bằng chứng:** “Tập các bài tập cho lưng.” lặp 10 lần. Các ý cardio, xen kẽ vận động/nghỉ và tập nhiều nhóm cơ trong nguồn/tham chiếu không được tóm lược đủ.
- **Đối chiếu baseline và cách diễn giải:** Baseline chỉ giữ hai câu mở đầu và “Ngủ.”, nên bỏ gần toàn bộ phần tập luyện. ViT5 có ROUGE-L cao hơn nhưng đầu ra bị chi phối bởi một câu lặp. Nguồn 1294 token chỉ dùng 1024: có rủi ro mất phần cuối, nhưng artifact chỉ lưu dấu vân tay ID nguồn, chưa lưu bản decode phần thực đọc. Vì vậy chưa định vị chính xác câu nào bị cắt và không khẳng định cắt là nguyên nhân của các lỗi vừa nêu.

### Mẫu 5 — Thực hiện mục tiêu dài hạn

**ID:** `vi_5b778dda0d4c87b970390b544ef06e6b823cf1f76a861db255dc44ccf565ba37`. Chỉ số dòng 70; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Cách tốt nhất để đảm bảo tiến độ là tạo thời gian biểu thực hiện mục tiêu hàng ngày.

> Một trong những cách tốt nhất để luôn có động lực là theo dõi tiến độ và kiểm tra thường xuyên.

**Tham chiếu:** Tạo môi trường phù hợp. Làm việc. Thực hiện mục tiêu một cách nhất quán. Luôn có động lực. Theo dõi tiến độ.

**Baseline:** “Dù mục tiêu của bạn là gì thì luôn tồn tại môi trường tốt hơn để đạt được mục tiêu.” “Làm tất cả những gì có thể để chắc chắn rằng mọi người và môi trường xung quanh không tạo ra trở ngại.” “Ví dụ, nếu bạn học ngành y, bạn phải học lâu hơn và tập trung vào công việc.”

**ViT5:** “Xác định môi trường tốt nhất để đạt được mục tiêu.” “Xác định thời gian biểu để hoàn thành mục tiêu phụ.” “Tạo thời gian biểu để hoàn thành mục tiêu phụ.”

- **Phần giữ đúng:** ViT5 giữ được hai ý có căn cứ: môi trường phù hợp và thời gian biểu để thực hiện mục tiêu.
- **Lỗi/hạn chế và bằng chứng:** “Xác định thời gian biểu để hoàn thành mục tiêu phụ.” và “Tạo thời gian biểu để hoàn thành mục tiêu phụ.” gần lặp ý dù không trùng nguyên văn. Đầu ra bỏ phần theo dõi tiến độ, duy trì động lực và tính nhất quán mà nguồn/tham chiếu nêu.
- **Đối chiếu baseline và cách diễn giải:** Baseline tập trung vào môi trường và ví dụ người học ngành y; ViT5 bao phủ thêm lịch thực hiện, nhưng chưa bao phủ các ý còn lại. Đây là mẫu có phần diễn đạt đúng đáng ghi nhận, không phải một bản tóm tắt đầy đủ và không lặp.

### Mẫu 6 — Cấp quyền vị trí cho Chrome trên iPhone

**ID:** `vi_4c96cab9906017773102bf1bd6817526faa803ad6763185ac4b5d40717f11eba`. Chỉ số dòng 62; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Nhấn vào ứng dụng màu xám có hình bánh răng trên đó.

> Google Chrome có thể truy cập vị trí của điện thoại khi bạn đang sử dụng ứng dụng, chứ không phải khi ứng dụng bị đóng

**Tham chiếu, trích phần chỉ dẫn:** Nhấn Location (Vị trí). Nhấn While Using the App (Trong khi sử dụng ứng dụng).

Phần chỉ dẫn còn nêu mở Settings trên iPhone rồi chọn Chrome. Nhãn gốc chứa hai khối JSON metadata ảnh với các khóa smallUrl, bigUrl, smallWidth và licensing; không chép lại toàn bộ khối này ở đây. Bản nguyên vẹn vẫn nằm trong predictions.jsonl và đã được dùng để chấm.

**Baseline:** “Nhấn vào ứng dụng màu xám có hình bánh răng trên đó.” “Bạn thường sẽ tìm thấy mục Settings (Cài đặt) trên màn hình chính.” “Bạn sẽ tìm thấy tùy chọn này trong danh sách các ứng dụng gần cuối trang Settings (Cài đặt).”

**ViT5:** “Mở Google Chrome.” “Nhấp vào Settings (Cài đặt).” *(lặp liên tiếp 10 lần)*

- **Phần giữ đúng:** ViT5 giữ được các tên Chrome và Settings, nhưng chỉ giữ đúng tên riêng chưa đủ để bảo toàn cách thao tác.
- **Lỗi/hạn chế và bằng chứng:** ViT5 bắt đầu bằng “Mở Google Chrome.” rồi lặp “Nhấp vào Settings (Cài đặt).” 10 lần. Đầu ra không giữ điểm bắt đầu trong Settings hệ thống và chuỗi chọn Chrome/Location/While Using the App mà tham chiếu mô tả; điều kiện quyền chỉ khi đang dùng ứng dụng trong nguồn bị bỏ.
- **Đối chiếu baseline và cách diễn giải:** Baseline giữ được phần mở Settings nhưng cũng không tới điều kiện quyền vị trí. Nhãn có nhiễu metadata khiến việc diễn giải ROUGE kém rõ ràng hơn; đồng thời model có lỗi quy trình/lặp độc lập, nên không quy toàn bộ điểm thấp cho nhiễu. Chưa xác định metadata phát sinh từ dữ liệu gốc hay bước xử lý nào; không sửa nhãn và tính lại test trong task này.

### Mẫu 7 — Chuẩn bị cơ thể trước sinh

**ID:** `vi_49a2e9dec0bf8b7e9c0a0a87777b8d2a178cd0b65ded1224189601e42bbd58ee`. Chỉ số dòng 57; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Tuy nhiên, bạn nên tăng thêm lượng calo mà bạn tiêu thụ trong suốt quá trình mang thai.

> Nếu có thể, hãy ngủ giữa những cơn co thắt tử cung.

**Tham chiếu:** Giữ gìn sức khỏe trong khi mang bầu. Kiểm soát cân nặng trong khi mang thai. Tuân thủ tháp dinh dưỡng khi mang bầu. Nghỉ ngơi giữa các cơn co thắt tử cung.

**Baseline:** “Giữ cho cơ thể ở trạng thái khỏe mạnh nhất trước khi chuyển dạ có thể giảm nguy cơ gặp phải tình trạng chuyển dạ kéo dài.” “Chuẩn bị cho cơ thể trước những gì sắp diễn ra có thể giúp bạn chuyển dạ dễ dàng và nhanh hơn khi tới lúc sinh nở.” “Đi bộ hoặc bơi thường xuyên, kết hợp với vài động tác chuyên biệt nữa.”

**ViT5:** “Lên kế hoạch cho một cuộc chuyển dạ hoàn hảo trước khi chuyển dạ.” “Tăng lượng calo tiêu thụ trong suốt quá trình chuyển dạ.” *(lặp liên tiếp 2 lần)* “Ngủ giữa các cơn co thắt tử cung.” “Thư giãn giữa các cơn co thắt tử cung.”

- **Phần giữ đúng:** ViT5 giữ được các ý ngủ và thư giãn giữa các cơn co thắt, có trong đoạn cuối nguồn.
- **Lỗi/hạn chế và bằng chứng:** Nguồn đặt việc tăng lượng calo trong “quá trình mang thai”, còn model viết “Tăng lượng calo tiêu thụ trong suốt quá trình chuyển dạ.” và lặp câu đó 2 lần. Đây là thay đổi mốc thời gian của hướng dẫn so với chính nguồn. Phần kiểm soát cân nặng, dinh dưỡng và luyện tập không được giữ đủ.
- **Đối chiếu baseline và cách diễn giải:** Baseline giữ phần sức khỏe/tập luyện mở đầu, không bao phủ phần nghỉ giữa các cơn co thắt. ViT5 cao hơn ROUGE-L nhưng vẫn có lỗi mốc thời gian; đối chiếu này không phải lời khuyên sức khỏe hoặc thẩm định y khoa.

### Mẫu 8 — Trao đổi với bác sĩ về sỏi a-mi-đan

**ID:** `vi_2b412a0143b130bb431a66c223517065416d34766fa72b85cfc4c4d1cf9b976c`. Chỉ số dòng 35; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Bạn cũng có thể hỏi bác sĩ về việc lấy sỏi a-mi-đan.

> Các mô a-mi-đan sâu có thể được loại bỏ bằng tia laser.

**Tham chiếu:** Nói chuyện với bác sĩ về thủ thuật cắt a-mi-đan. Cân nhắc dùng một liệu trình kháng sinh để điều trị tình trạng sỏi nghiêm trọng hoặc dai dẳng. Hỏi về phương pháp điều trị laser.

**Baseline:** “Thủ thuật cắt a-mi-đan tương đối đơn giản và hiệu quả, độ rủi ro tương đối thấp và thời gian hồi phục ngắn.” “Tình trạng đau họng và chảy máu nhẹ thường là mối lo phổ biến nhất.” “Nếu bác sĩ lo ngại về tiền sử bệnh, độ tuổi hoặc các yếu tố khác ở bạn, có thể họ sẽ khuyên bạn dùng phương pháp khác.”

**ViT5:** “Hỏi bác sĩ về thủ thuật cắt a-mi-đan.” “Hỏi bác sĩ về việc lấy sỏi a-mi-đan.” *(lặp liên tiếp 3 lần)*

- **Phần giữ đúng:** Câu “Hỏi bác sĩ về thủ thuật cắt a-mi-đan.” giữ được mục chính đầu tiên của tham chiếu; việc hỏi về lấy sỏi cũng có trong nguồn.
- **Lỗi/hạn chế và bằng chứng:** “Hỏi bác sĩ về việc lấy sỏi a-mi-đan.” lặp 3 lần. Các phần riêng về kháng sinh và laser trong nguồn/tham chiếu bị bỏ; các điều kiện và lưu ý của nguồn cũng không được tóm lược.
- **Đối chiếu baseline và cách diễn giải:** Baseline giữ mô tả thủ thuật, rủi ro và cân nhắc của bác sĩ ở đoạn đầu nhưng bỏ các hướng ở phần sau. ViT5 tóm được một hành động chính ngắn gọn hơn, song không có độ bao phủ tốt trên toàn đoạn. Không suy ra hiệu quả hay chỉ định điều trị từ phép so sánh văn bản này.

### Mẫu 9 — Chọn ớt chuông để đông lạnh

**ID:** `vi_f0cc3df13880ec194c408cc80d4a9fbbb715ff2579cc4d54643cea835a2d74e0`. Chỉ số dòng 188; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Những quả ớt chuông mà bạn chọn phải có màu xanh đậm và có độ cứng.

> Nếu không thể đông lạnh ớt chuông ngay, bạn sẽ tạm thời cho vào tủ lạnh và cố gắng đông lạnh trong cùng ngày.

**Tham chiếu:** Chọn ớt chuông xanh chín.

**Baseline:** “Những quả ớt chuông mà bạn chọn phải có màu xanh đậm và có độ cứng.” “Chọn ớt chuông càng tươi càng tốt.” “Ớt chuông được thu hoạch ngay tại vườn là hiệu quả nhất, nhưng bạn cũng có thể mua ở chợ hoặc siêu thị và chọn những quả ớt chuông còn nguyên vẹn.”

**ViT5:** “Chọn ớt chuông tươi.” “Đông lạnh ớt chuông.” *(lặp liên tiếp 2 lần)*

- **Phần giữ đúng:** “Chọn ớt chuông tươi.” giữ đúng ý chọn độ tươi, có trong nguồn, và diễn đạt gọn hơn baseline.
- **Lỗi/hạn chế và bằng chứng:** “Đông lạnh ớt chuông.” lặp 2 lần; màu xanh đậm/độ cứng và điều kiện bảo quản trong ngày không được giữ. Hành động đông lạnh có trong nguồn nên không gán nhãn bịa thông tin cho riêng hành động này.
- **Đối chiếu baseline và cách diễn giải:** Baseline giữ màu xanh đậm, độ cứng và độ tươi, nhưng dài so với tham chiếu chỉ một câu. ViT5 có ROUGE-L cao hơn, đồng thời mất một số tiêu chí chọn quả mà baseline giữ được.

### Mẫu 10 — Vẽ chú mèo

**ID:** `vi_212044bcc1e1c5f9ec61f9edbb984e2eac71d96ff9f4ce2ad4739d8e04d38a51`. Chỉ số dòng 26; nguồn không bị cắt.

**Nguồn, trích bằng chứng:**

> Vẽ thêm đuôi cho chú mèo.

> Bạn có thể dùng bút chì màu, bút sáp màu, bút lông hoặc màu nước.

**Tham chiếu:** Phác thảo những nét chính. Phác họa những đặc điểm cơ bản của gương mặt. Thêm nhiều đặc điểm nữa cho phần đầu. Phác thảo những hình thuôn dài và đường tròn tượng trưng cho phần đùi, chân và bàn chân. Vẽ những đặc điểm chính của chú mèo. Xóa các đường phác thảo và thêm nhiều chi tiết hơn. Tô màu chú mèo.

**Baseline:** “Vẽ phác một hình tròn để làm đầu, một hình chữ nhật với nét cong ở gần đầu tượng trưng cho phần thân và một hình thuôn dài cho vùng đùi.” “Vẽ thêm các nét hướng dẫn cho vùng miệng, tai và mặt.” “Cố gắng vẽ phần mõm hơi ngắn và vuông.”

**ViT5:** “Vẽ các nét hướng dẫn trên mặt.” *(lặp liên tiếp 4 lần)*

- **Phần giữ đúng:** ViT5 giữ được thao tác vẽ nét hướng dẫn trên mặt, có trong nguồn.
- **Lỗi/hạn chế và bằng chứng:** Đầu ra chỉ có “Vẽ các nét hướng dẫn trên mặt.” lặp 4 lần. Không giữ các nét chính của đầu/thân/đùi, chi tiết mũi/đuôi/lông và phần tô màu được nguồn hỗ trợ.
- **Đối chiếu baseline và cách diễn giải:** Baseline giữ phần dựng các khối chính, nét mặt và mõm nên bao phủ nhiều hơn ViT5 ở mẫu này. Tham chiếu có mục xóa đường phác thảo nhưng nguồn không nêu trực tiếp thao tác đó; không lấy riêng việc thiếu mục này làm bằng chứng model sai. Không quan sát được đảo một chuỗi bước hoàn chỉnh: model đã bỏ hầu hết bước, nên nhãn chính là thiếu ý/lặp.

## 4. Tổng hợp và giới hạn kết luận

- **Phần làm được:** các đầu ra có những ý bám nguồn; mẫu 5 giữ môi trường và lịch thực hiện, mẫu 8 giữ việc hỏi bác sĩ, mẫu 9 giữ ý chọn độ tươi. Không mẫu nào trong bộ 10 này đồng thời đầy đủ ý, trung thành và không lặp; không thay mẫu để tìm một ví dụ đẹp toàn diện.
- **Lặp:** 9/10 mẫu lặp ít nhất một câu nguyên văn. Mẫu 5 còn lại gần lặp cùng ý thời gian biểu bằng hai cách viết. Đây là quan sát trên 10 mẫu, không phải tỷ lệ lỗi của toàn bộ 200 test.
- **Thiếu ý/điều kiện:** cả 10 mẫu bỏ ít nhất một ý hoặc điều kiện có căn cứ trong nguồn. Các phần đúng chưa đủ để xem đầu ra là bản hướng dẫn thay thế toàn bộ nguồn.
- **Sai hoặc không được nguồn hỗ trợ:** mẫu 1 thêm hành động cắt tóc; mẫu 7 chuyển thời điểm của hướng dẫn từ mang thai sang chuyển dạ. Mẫu 6 không giữ quy trình cấp quyền được nguồn/tham chiếu mô tả. Các nhãn này có thể cùng xuất hiện với lặp và thiếu ý, không phải các nhóm loại trừ nhau.
- **Cắt nguồn:** chỉ mẫu 4 bị cắt (1294 còn 1024 token). Chưa có bản decode đầu vào thực dùng để định vị từng câu ngoài giới hạn. Chín mẫu không bị cắt vẫn lặp hoặc gần lặp, nên không quy hiện tượng lặp chỉ cho cắt nguồn. Chưa có thí nghiệm xác định nguyên nhân của lỗi lặp.
- **Nhãn nhiễu:** mẫu 6 có metadata ảnh JSON trong tham chiếu. Quy tắc metric vẫn giữ các chữ/số trong đó, nên điểm cũng chịu ảnh hưởng của phần không phải nội dung tóm tắt. Chưa đo mức ảnh hưởng, chưa xác định nơi phát sinh; giữ nguyên nhãn và bảng test. Mẫu 3 và 10 còn cho thấy không nên suy mọi tiêu đề tham chiếu thành một hành động được nguồn diễn đạt rõ.
- **ROUGE và nội dung:** ViT5 có ROUGE-L cao hơn baseline ở 7/10 mẫu, thấp hơn ở 3/10. Mẫu 1 vẫn có lỗi hành động dù điểm cao hơn; mẫu 4 có điểm cao hơn nhưng lặp nhiều. Các thống kê 10 mẫu này không thay bảng trung bình 200 mẫu: baseline ROUGE-L 14.4402, ViT5 22.0434.
- **Quy trình sau phân tích:** giữ nguyên checkpoint, beam 4, dữ liệu và điểm test. Nếu muốn thử chống lặp, thêm dữ liệu hoặc làm sạch nhãn trong tương lai, cần công bố đó là phương án mới sau khi đã xem test và xác nhận lại thiết kế đánh giá; task này chưa thực hiện các thay đổi đó.
