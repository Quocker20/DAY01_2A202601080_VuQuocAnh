# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> *Khi tăng dần chỉ số temperature, phản hồi trả về chuyển từ tính chính xác, khuôn mẫu sang sáng tạo hơn, nhưng cũng ngẫu nhiên và rời rạc hơn. Ở mức temperature = 1.2 đã bắt đầu có dấu hiệu kém mạch lạc, và ở mức cuối là 1.8 ngữ pháp và logic bị phá vỡ và trả nên kém mạch lạc rõ rệt.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Tôi sẽ để temperature của trợ lý soạn thảo hợp đồng pháp lý ở chính xác temperature = 0.0, tuy nhiên nếu có sự kiểm soát của con người hoặc các trợ lý khác thì mức tối đa có thể là temperature = 0.2. Tôi sẽ để temperature của trợ lý viết slogan quảng cáo ở mức 0.7 và lên tới tối đa 1.0 nếu có sự kiểm soát của người hoặc trợ lý khác. Sự khác biệt ở đây là tính chất của mục tiêu được đưa ra, việc soạn thảo hợp đồng sẽ cần độ chính xác, và mọi sai lầm đều sinh ra hậu quả lớn, và không cần sự sáng tạo. Còn việc đặt temperature của trợ lý slogan quảng cáo ở mức khoảng 0.7 sẽ đảm bảo có một độ sáng tạo nhất định mà chưa đến mức kém mạch lạc.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *20.000 người dùng × 2 lần/ngày × 500 token đầu ra = 20.000.000 token/ngày (= 20.000 nghìn token). GPT-4o: 20.000 × $0.010 = $200/ngày. GPT-4o-mini: 20.000 × $0.0006 = $12/ngày. Chênh lệch: model lớn tốn thêm khoảng $188/ngày (~gấp 16 lần). Model lớn xứng đáng khi task đòi hỏi suy luận phức tạp hoặc độ chính xác cao (vd. tư vấn pháp lý, phân tích tài chính) — sai sót ở đây tốn kém hơn nhiều so với phần chi phí API chênh lệch. Model nhỏ là lựa chọn đúng cho workload khối lượng lớn, tác vụ đơn giản/lặp lại (vd. trả lời câu hỏi thường gặp, gợi ý nhanh) — nơi tốc độ và chi phí quan trọng hơn độ tinh tế của câu trả lời.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Phản hồi của "nhà thơ" không dùng thuật ngữ kỹ thuật, giọng văn nhẹ nhàng, giàu cảm xúc. Phản hồi của "kỹ sư senior" đi thẳng vào định nghĩa chính xác, có xu hướng kèm ví dụ/thuật ngữ. Cùng một câu hỏi nhưng hai giọng văn, cách mở đầu và mức độ kỹ thuật khác hẳn nhau. Điều này cho thấy system prompt điều khiển được giọng văn, phong cách diễn đạt (ẩn dụ vs. chính xác) và mức độ thuật ngữ sử dụng — chứ không thay đổi nội dung cốt lõi (khái niệm machine learning) mà model truyền tải.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Đoạn văn được chọn (163 từ): "Trí tuệ nhân tạo đang dần trở thành một phần quen thuộc trong đời sống hiện đại. Công nghệ này giúp con người tìm kiếm thông tin, dịch ngôn ngữ, phân tích dữ liệu và tự động hóa nhiều công việc lặp lại. Trong giáo dục, học sinh có thể dùng trợ lý AI để giải thích khái niệm khó, gợi ý cách học và luyện tập theo năng lực cá nhân. Tuy nhiên, người dùng không nên tin hoàn toàn vào mọi câu trả lời vì mô hình vẫn có thể tạo ra thông tin sai hoặc thiếu ngữ cảnh. Khi sử dụng AI, chúng ta cần kiểm tra nguồn, bảo vệ dữ liệu cá nhân và tự chịu trách nhiệm cho quyết định cuối cùng. Nếu được áp dụng đúng cách, AI không thay thế hoàn toàn con người mà đóng vai trò như một công cụ hỗ trợ, giúp tiết kiệm thời gian và mở rộng khả năng sáng tạo." Với model `gpt-4o`, `count_tokens` đếm được tổng cộng **189 token**, trong khi cách ước lượng `số từ / 0.75` cho kết quả `163 / 0.75 ≈ 217,33 token`. Ước lượng thô cao hơn `(217,33 - 189) / 189 × 100 ≈ 15%`. Vì vậy, với đoạn tiếng Việt này, dùng công thức thô sẽ **dự toán thừa** ngân sách API do công thức giả định trung bình mỗi token chỉ tương ứng 0,75 từ, trong khi bộ mã hóa của `gpt-4o` xử lý các từ và cụm tiếng Việt trong đoạn hiệu quả hơn giả định đó.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Chatbot văn bản (a) hưởng lợi nhiều nhất từ streaming: người dùng thấy chữ xuất hiện ngay lập tức thay vì nhìn màn hình trống chờ toàn bộ câu trả lời, giúp cảm giác "phản hồi nhanh" dù tổng thời gian xử lý không đổi. Trợ lý giọng nói (b) hưởng lợi ít hơn vì hệ thống text-to-speech thường cần một cụm câu hoàn chỉnh mới đọc tự nhiên được — stream từng token nhỏ có thể khiến giọng đọc bị ngắt quãng hoặc phải đệm lại (buffer) trước khi phát, làm giảm bớt lợi ích so với chatbot văn bản. Pipeline dịch tài liệu chạy ngầm ban đêm (c) hầu như không cần streaming: không có người xem trực tiếp đầu ra, hệ thống chỉ quan tâm kết quả cuối cùng khi job hoàn tất, nên gọi API theo kiểu chờ phản hồi đầy đủ (non-streaming) là đủ và đơn giản hơn.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Với delay cố định, nếu hàng nghìn client đều gặp lỗi cùng lúc (vd. server quá tải), tất cả sẽ retry lại sau đúng cùng một khoảng thời gian — tạo ra một đợt "sóng" request đồng loạt mới, khiến server tiếp tục quá tải và có thể lặp lại chu kỳ lỗi. Exponential backoff giãn dần thời gian chờ giữa các lần thử (base_delay × 2^attempt), giúp giảm áp lực lên server theo thời gian và cho hệ thống cơ hội hồi phục trước khi nhận thêm request retry. Tuy nhiên, vấn đề còn sót lại là: nếu mọi client đều dùng đúng công thức backoff giống nhau, chúng vẫn retry đồng bộ ở các mốc thời gian giống nhau (vd. tất cả cùng thử lại ở giây thứ 0.1, rồi 0.2, rồi 0.4...), gây ra hiện tượng "thundering herd" theo từng đợt. Kỹ thuật jitter thêm một khoảng trễ ngẫu nhiên vào mỗi lần retry để rải đều thời điểm các client thử lại, phá vỡ sự đồng bộ đó và giúp tải lên server được phân tán mượt mà hơn.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *System prompt đã dùng: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." Thứ nhất, nếu xóa cụm **"thân thiện"**, giọng văn của trợ lý sẽ trở nên trung tính/khô khan hơn — mất đi sự khích lệ, xưng hô gần gũi, và có thể trả lời cộc lốc như một tài liệu tra cứu thay vì một người hướng dẫn. Thứ hai, nếu xóa cụm **"trả lời ngắn gọn bằng tiếng Việt"**, trợ lý có thể trả lời dài dòng hơn nhiều (do không còn ràng buộc độ dài) và không còn chắc chắn dùng tiếng Việt — dễ lẫn tiếng Anh hoặc trộn ngôn ngữ khi người dùng hỏi bằng thuật ngữ tiếng Anh.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Ví dụ: ở lượt 1, người dùng nói "Mình đang học Python, không rành thuật toán." Từ lượt 2 đến lượt 5, cuộc trò chuyện chuyển sang các chủ đề khác (hỏi về cú pháp, thư viện, debug lỗi...). Đến lượt 6, người dùng hỏi "Giải thích cái này đơn giản hơn được không?" — nhưng vì history chỉ giữ 4 lượt gần nhất (8 message), thông tin "không rành thuật toán" ở lượt 1 đã bị cắt khỏi ngữ cảnh, khiến trợ lý không còn biết cần giải thích ở mức cơ bản, có thể trả lời lại vẫn dùng thuật ngữ phức tạp như ban đầu. Cách khắc phục: thay vì xóa hẳn các lượt cũ, tóm tắt chúng thành 1–2 câu ngắn (vd. "Người dùng là người mới học Python, không rành thuật toán") và giữ bản tóm tắt này như một phần cố định trong system prompt hoặc một message riêng ở đầu history — cập nhật bản tóm tắt mỗi khi có lượt bị cắt, thay vì để thông tin quan trọng biến mất hoàn toàn.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
