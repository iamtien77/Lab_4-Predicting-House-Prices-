# Linear Regression trong bài lab dự đoán giá nhà

Nhóm dự kiến sử dụng mô hình Hồi quy tuyến tính (Linear Regression) làm mô hình baseline cơ sở đầu tiên theo yêu cầu của Task 6 trong đề bài. Tài liệu này trình bày phương án tính toán, phương pháp giải và kế hoạch kiểm tra cho mô hình trong bài lab.

## 1. Mục đích và lý do lựa chọn

Linear Regression là mô hình học máy có giám sát cổ điển và đơn giản nhất trong các bài toán hồi quy. Mô hình giả định rằng có một mối quan hệ tuyến tính giữa các đặc trưng đầu vào $x$ và giá trị mục tiêu liên tục $y$. Việc đưa Linear Regression vào bài lab giúp nhóm xây dựng một mốc chuẩn tối thiểu (baseline). Nếu một mô hình phức tạp hơn như mạng nơ-ron MLP không vượt qua được mốc này, điều đó cho thấy kiến trúc mạng hoặc quá trình tiền xử lý đang gặp vấn đề.

Với một mẫu dữ liệu có vector đặc trưng $x = (x_1, x_2, \dots, x_d)^T$, mô hình tính toán giá nhà dự đoán $\hat{y}$ bằng tổ hợp tuyến tính của các đặc trưng:

$$\hat{y} = w^T x + b = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d$$

Trong đó:
- $w = (w_1, w_2, \dots, w_d)^T$ là vector trọng số tương ứng với từng đặc trưng.
- $b = w_0$ là hệ số điều chỉnh (bias / intercept).
- $\hat{y}$ là giá nhà trung vị dự đoán theo đơn vị USD.

Ưu điểm lớn nhất của mô hình là thời gian huấn luyện cực kỳ nhanh, tiêu tốn ít bộ nhớ và có khả năng giải thích cao. Nhóm có thể quan sát trực tiếp dấu và độ lớn của từng trọng số $w_j$ để hiểu đặc trưng nào (ví dụ: thu nhập trung vị hay số phòng) có ảnh hưởng thuận chiều hoặc nghịch chiều đến giá nhà. 

Tuy nhiên, hạn chế cốt lõi của mô hình là không thể tự nắm bắt các mối quan hệ phi tuyến tính hoặc các tương tác phức tạp giữa các biến (chẳng hạn mối tương tác giữa tọa độ địa lý kinh độ - vĩ độ để tạo thành vùng ven biển hay vùng trung tâm).

## 2. Dữ liệu đầu vào và kết quả đầu ra

- **Đầu vào:** Ma trận đặc trưng số $X$ có kích thước $N \times d$, là kết quả sau khi đi qua cùng một bộ tiền xử lý trên tập Train (đã xử lý giá trị khuyết thiếu và chuẩn hóa thang đo).
- **Đầu ra:** Vector giá nhà dự đoán $\hat{y}$ kích thước $N \times 1$ có đơn vị USD.
- **Hệ số đã học:** Vector trọng số $w$ và hệ số chặn $b$ được lưu lại để phục vụ phân tích mức độ ảnh hưởng của từng đặc trưng.

## 3. Dữ liệu sử dụng

Mô hình được huấn luyện hoàn toàn trên tập Train (80%). Sau khi học được các trọng số $w$ và bias $b$, mô hình sẽ được lưu lại. 

Tập Validation (10%) được dùng để kiểm tra tính ổn định của mô hình và đối chiếu sơ bộ. Tập Test (10%) được giữ nguyên vẹn để đánh giá độc lập một lần duy nhất ở bước cuối cùng cùng với mạng MLP và hai mô hình cây.

## 4. Phương pháp huấn luyện và đánh giá

### Hàm mục tiêu

Mô hình tìm các hệ số $w$ và $b$ bằng phương pháp bình phương tối thiểu (Ordinary Least Squares - OLS), hướng đến việc tối thiểu hóa tổng bình phương phần dư (Residual Sum of Squares - RSS) giữa giá trị thực tế $y_i$ và giá trị dự đoán $\hat{y}_i$:

$$J(w, b) = \frac{1}{N} \sum_{i=1}^N (y_i - (w^T x_i + b))^2$$

Biểu thức này hoàn toàn tương đồng với hàm mất mát Mean Squared Error (MSE) được sử dụng để tối ưu mạng MLP, tạo cơ sở toán học đồng nhất để so sánh trực tiếp hai mô hình.

### Cách tối ưu

Khác với mạng MLP cần tối ưu qua nhiều epochs bằng thuật toán lặp (Adam), Linear Regression trong bài thực hành có thể được giải quyết bằng nghiệm giải tích dạng đóng (Normal Equation) hoặc thông qua kỹ thuật phân rã ma trận (chẳng hạn như phân tích giá trị kỳ dị SVD hoặc phân rã QR):

$$\hat{w} = (X^T X)^{-1} X^T y$$

Phương pháp giải này tìm ra điểm tối ưu toàn cục chỉ sau một lần tính toán ma trận, không cần phải chọn tốc độ học hay số vòng lặp.

### Cấu hình và tiêu chí đánh giá

- **Cấu hình dự kiến:** Mô hình sử dụng hồi quy tuyến tính chuẩn có tính thêm hệ số chặn (fit intercept = True).
- **Tiêu chí đánh giá:** Đo lường 4 chỉ số trên cùng tập Test:
  1. Mean Squared Error (MSE) theo đơn vị USD&sup2;.
  2. Root Mean Squared Error (RMSE) theo đơn vị USD.
  3. Mean Absolute Error (MAE) theo đơn vị USD.
  4. Hệ số xác định ($R^2$ Score) để biết mô hình giải thích được bao nhiêu phần trăm biến thiên của giá nhà.

## 5. Kế hoạch kiểm tra

Nhóm dự kiến thực hiện các bước kiểm tra sau:
- **Kiểm tra hiện tượng đa cộng tuyến:** Một số đặc trưng trong bộ dữ liệu California Housing có mối tương quan mạnh với nhau (chẳng hạn `total_rooms`, `total_bedrooms` và `households`). Nhóm sẽ kiểm tra xem ma trận hiệp phương sai có bị suy biến hay không để đảm bảo việc nghịch đảo ma trận diễn ra ổn định.
- **Kiểm tra phân phối phần dư (Residuals):** Phần dư $e_i = y_i - \hat{y}_i$ cần được vẽ đồ thị phân tán để xem xét liệu sai số có phân bổ đồng đều quanh mức 0 hay có xu hướng lệch ở các vùng giá cao.
- **Kiểm tra tính nhất quán:** Dự đoán của mô hình trước và sau khi lưu thành file `results/saved_models/baseline_models.joblib` phải trùng khớp nhau tuyệt đối.

## 6. Kế hoạch đánh giá và giới hạn

Linear Regression đóng vai trò thước đo nền tảng (benchmark) trong bài lab. Kết quả của mô hình này sẽ được đặt cạnh kết quả của Decision Tree, Random Forest và MLP trong cùng một bảng tổng hợp.

Một số giới hạn cần lưu ý trong bài thực hành:
- **Hạn chế về mặt ranh giới tuyến tính:** Do giá nhà phụ thuộc lớn vào vị trí địa lý mang tính phi tuyến (khu vực giáp biển, khu vực đồi núi), mô hình tuyến tính dự kiến sẽ gặp khó khăn khi mô tả ranh giới giá theo tọa độ kinh độ và vĩ độ.
- **Nguy cơ dự đoán giá âm:** Với các căn nhà có đặc trưng thu nhập rất thấp và tuổi đời cũ, đường hồi quy tuyến tính có thể kéo giá dự đoán xuống dưới 0. Nhóm sẽ ghi nhận tỷ lệ các mẫu bị dự đoán âm nếu có trong báo cáo.
