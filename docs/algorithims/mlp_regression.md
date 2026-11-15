# Multilayer Perceptron (MLP) trong bài lab dự đoán giá nhà

Nhóm dự kiến sử dụng mạng nơ-ron truyền thẳng nhiều tầng (Multilayer Perceptron - MLP) làm mô hình trọng tâm của bài thực hành theo yêu cầu của đề bài. Tài liệu này trình bày phương án kiến trúc, cách huấn luyện, theo dõi quá trình học và kế hoạch kiểm tra cho bài lab.

## 1. Mục đích và lý do lựa chọn

MLP là một mạng nơ-ron nhân tạo gồm một lớp đầu vào, một hoặc nhiều lớp ẩn và một lớp đầu ra. Khác với các mô hình tuyến tính đơn giản, MLP có khả năng xấp xỉ các hàm phi tuyến phức tạp nhờ vào các hàm kích hoạt (activation function) phi tuyến tính ở các lớp ẩn. Điều này phù hợp với thị trường nhà đất, nơi giá trị bất động sản thường tương tác phi tuyến với vị trí, thu nhập khu vực và quy mô phòng ốc.

Với một mẫu dữ liệu có vector đặc trưng đầu vào $x \in \mathbb{R}^d$, mạng MLP có 2 lớp ẩn thực hiện các phép tính tuần tự qua từng lớp:

- Lớp ẩn thứ nhất (32 nơ-ron, kích hoạt ReLU):
  $$z^{[1]} = W^{[1]} x + b^{[1]}, \qquad a^{[1]} = \text{ReLU}(z^{[1]}) = \max(0, z^{[1]})$$

- Lớp ẩn thứ hai (16 nơ-ron, kích hoạt ReLU):
  $$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}, \qquad a^{[2]} = \text{ReLU}(z^{[2]}) = \max(0, z^{[2]})$$

- Lớp đầu ra (1 nơ-ron, kích hoạt Linear):
  $$\hat{y} = W^{[3]} a^{[2]} + b^{[3]}$$

Trong đó, $W^{[1]}, W^{[2]}, W^{[3]}$ là các ma trận trọng số, $b^{[1]}, b^{[2]}, b^{[3]}$ là các vector bias, và $\hat{y}$ là giá nhà dự đoán theo đơn vị USD. Vì đây là bài toán hồi quy, lớp đầu ra sử dụng hàm kích hoạt tuyến tính để có thể dự đoán một giá trị thực liên tục mà không bị giới hạn trong khoảng $[0, 1]$.

Ưu điểm của mô hình là có thể tự động học các biểu diễn đặc trưng trừu tượng mà không cần tạo thủ công quá nhiều biến tương tác. Tuy nhiên, hạn chế của mạng MLP là rất nhạy cảm với thang đo của dữ liệu đầu vào (cần phải chuẩn hóa z-score cẩn thận), có nhiều siêu tham số cần quản lý và có nguy cơ quá khớp (overfitting) nếu mạng học vẹt các mẫu trên tập Train.

## 2. Dữ liệu đầu vào và kết quả đầu ra

- **Đầu vào:** Ma trận đặc trưng hai chiều kích thước $N \times d$, trong đó $N$ là số lượng mẫu nhà và $d$ là số chiều sau khi đã điền giá trị thiếu, mã hóa one-hot biến phân loại và chuẩn hóa z-score toàn bộ biến số. Dữ liệu phải là số thực hữu hạn, không chứa giá trị NaN.
- **Đầu ra:** Vector giá trị dự đoán $\hat{y}$ kích thước $N \times 1$ có đơn vị là USD, tương ứng với mức giá nhà trung vị dự kiến cho từng khu vực.
- **Theo dõi tiến trình:** Quá trình huấn luyện cần ghi nhận lịch sử hàm mất mát (loss) và sai số tuyệt đối trung bình (MAE) qua từng epoch trên cả tập Train và tập Validation để vẽ đường học (learning curve).

## 3. Dữ liệu sử dụng

Nhóm sử dụng tập dữ liệu California Housing đã được chia độc lập ngay từ đầu:

- **Tập Train (80%):** Dùng để tính toán lan truyền xuôi (forward pass), tính gradient qua lan truyền ngược (backpropagation) và cập nhật các trọng số $W$ và bias $b$ của mạng.
- **Tập Validation (10%):** Dùng làm tập kiểm tra độc lập trong suốt 100 epochs huấn luyện. Nhóm sẽ theo dõi khoảng cách giữa loss trên Train và loss trên Validation để phát hiện sớm dấu hiệu học vẹt (overfitting).
- **Tập Test (10%):** Hoàn toàn không được sử dụng trong quá trình huấn luyện và điều chỉnh. Tập Test chỉ được nạp một lần duy nhất ở bước cuối cùng để đo đạc và so sánh công bằng với các mô hình baseline.

Số chiều đầu vào $d$ được lấy động từ kích thước ma trận sau khi đi qua bộ tiền xử lý trên tập Train, đảm bảo mô hình linh hoạt nếu sau này có sự thay đổi về số lượng đặc trưng mã hóa.

## 4. Phương pháp huấn luyện và đánh giá

### Hàm mục tiêu

Mô hình được huấn luyện bằng cách tối thiểu hóa hàm mất mát sai số bình phương trung bình (Mean Squared Error - MSE):

$$J(W, b) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

Trong đó $y_i$ là giá nhà thực tế và $\hat{y}_i$ là giá nhà do mạng nơ-ron dự đoán. Hàm MSE phạt rất nặng các sai số lớn do có phép bình phương, giúp mạng tập trung thu hẹp các lỗi dự đoán sai lệch nhiều. Đồng thời, nhóm theo dõi thêm chỉ số MAE trong quá trình học để có cái nhìn trực quan về mức chênh lệch trung bình theo USD:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

### Cách tối ưu

Nhóm dự kiến sử dụng thuật toán tối ưu **Adam (Adaptive Moment Estimation)**. Đây là giải pháp kết hợp ưu điểm của phương pháp quán tính (Momentum) và RMSprop, tự động điều chỉnh tốc độ học cho từng trọng số dựa trên việc ước lượng moment bậc một (giá trị trung bình gradient) và moment bậc hai (phương sai không định tâm):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Thuật toán Adam giúp quá trình hội tụ diễn ra ổn định và ít nhạy cảm hơn với việc khởi tạo trọng số ban đầu so với Gradient Descent cơ bản.

### Cấu hình và tiêu chí đánh giá

Theo đúng yêu cầu của đề bài thực hành, cấu hình dự kiến cho mạng MLP bao gồm:
- **Tốc độ học (Learning rate):** $\alpha = 0.001$.
- **Số lượng epochs:** 100 epochs.
- **Kích thước batch (Batch size):** 32 mẫu mỗi bước cập nhật.
- **Tập kiểm tra theo dõi:** Sử dụng tập Validation độc lập để tính toán `val_loss` và `val_mae` tại mỗi epoch.

Tiêu chí đánh giá chất lượng học tập của mô hình gồm:
1. Đường mất mát trên tập Train phải giảm dần và ổn định theo thời gian.
2. Đường mất mát trên tập Validation phải theo sát đường Train, không xuất hiện hiện tượng phân kỳ quá mức (dấu hiệu của overfitting).
3. Đánh giá cuối cùng trên tập Test bằng 4 chỉ số: MSE, RMSE, MAE và $R^2$.

## 5. Kế hoạch kiểm tra

Nhóm dự kiến thực hiện các bước kiểm tra tính đúng đắn trước khi nghiệm thu mô hình:
- **Kiểm tra tương thích kích thước:** Số chiều đầu vào của lớp Dense đầu tiên phải khớp hoàn toàn với số cột đầu ra của bộ tiền xử lý.
- **Kiểm tra tính hợp lệ của đầu ra:** Vector dự đoán phải có cùng số dòng với tập dữ liệu đầu vào, không được chứa giá trị NaN, giá trị vô cực hoặc các giá trị phi lý.
- **Kiểm tra tính nhất quán sau khi lưu trữ:** Mô hình sau khi huấn luyện sẽ được lưu thành tệp `results/saved_models/mlp_house_price_model.keras`. Khi nạp lại từ tệp này, mô hình phải cho ra kết quả dự đoán giống hệt như lúc vừa huấn luyện xong trên cùng một tập dữ liệu đầu vào.
- **Thử nghiệm chạy ngắn (Smoke test):** Chạy thử nghiệm với 1 epoch trên một tập dữ liệu mẫu nhỏ để xác nhận luồng lan truyền xuôi, tính mất mát và cập nhật trọng số hoạt động trơn tru trước khi chạy đủ 100 epochs.

## 6. Kế hoạch đánh giá và giới hạn

Mỗi lần chạy thực nghiệm sẽ được ghi nhận lại toàn bộ lịch sử loss và các chỉ số đo lường. Nhóm sẽ so sánh kết quả của mạng MLP với ba mô hình baseline trên cùng một tập Test duy nhất.

Một số giới hạn cần lưu ý:
- **Đặc thù dữ liệu dạng bảng:** Trong nhiều bài toán thực tế với dữ liệu dạng bảng có kích thước vừa phải, các mô hình học máy dạng cây (như Random Forest) thường hoạt động rất hiệu quả và có thể cho kết quả tương đương hoặc nhỉnh hơn mạng MLP. Nhóm sẽ ghi nhận khách quan nếu kết quả thực nghiệm cho thấy baseline tốt hơn và giải thích nguyên nhân dựa trên đặc điểm cấu trúc dữ liệu.
- **Ảnh hưởng của trần giá:** Dữ liệu có mức trần 500.001 USD. Mạng MLP có xu hướng dự đoán mức giá trung bình của vùng lân cận nên có thể gặp khó khăn khi dự đoán chính xác các khu vực có giá trị nhà rất cao bị làm tròn trần.
- **Khả năng dự đoán giá âm:** Hàm kích hoạt Linear ở lớp cuối có thể xuất ra giá trị âm nếu gặp các mẫu có đặc trưng ngoại lai bất thường. Nhóm sẽ kiểm tra xem có mẫu nào bị dự đoán âm hay không để có biện pháp xử lý trong báo cáo.
