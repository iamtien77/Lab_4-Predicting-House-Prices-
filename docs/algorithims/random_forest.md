# Random Forest Regression trong bài lab dự đoán giá nhà

Nhóm dự kiến sử dụng mô hình Rừng ngẫu nhiên (Random Forest Regressor) làm mô hình baseline thứ ba theo yêu cầu của Task 6 trong đề bài. Tài liệu này trình bày nguyên lý học tập kết hợp (Ensemble Learning), phương pháp giảm phương sai và kế hoạch kiểm tra cho mô hình trong bài lab.

## 1. Mục đích và lý do lựa chọn

Cây quyết định đơn lẻ tuy dễ hiểu nhưng có phương sai lớn và rất nhạy cảm với dữ liệu huấn luyện. Random Forest giải quyết triệt để nhược điểm này bằng cách áp dụng kỹ thuật **kết hợp mô hình (Ensemble Learning)** theo phương pháp **Bagging (Bootstrap Aggregation)** kết hợp với việc chọn ngẫu nhiên tập con các đặc trưng. Mô hình xây dựng một tập hợp gồm nhiều cây quyết định độc lập, sau đó lấy trung bình kết quả dự đoán của tất cả các cây.

Với một tập hợp gồm $B$ cây quyết định $\{T_1, T_2, \dots, T_B\}$, mức giá dự đoán $\hat{y}$ cho một mẫu dữ liệu $x$ được tính bằng trung bình cộng dự đoán của từng cây:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x)$$

Theo lý thuyết xác suất thống kê, nếu các cây dự đoán có phương sai là $\sigma^2$ và hệ số tương quan dương giữa các cây là $\rho$, thì phương sai của dự đoán kết hợp sẽ giảm xuống:

$$\text{Var}(\hat{y}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

Bằng cách chọn ngẫu nhiên tập con các đặc trưng tại mỗi lần phân nhánh, Random Forest làm giảm hệ số tương quan $\rho$ giữa các cây, từ đó giúp phương sai tổng thể giảm đáng kể so với một cây quyết định đơn lẻ.

Ưu điểm của Random Forest là độ ổn định rất cao, ít bị quá khớp hơn hẳn cây đơn lẻ, và thường mang lại hiệu năng thuộc hàng tốt nhất trong số các phương pháp học máy truyền thống đối với dữ liệu dạng bảng (tabular data). Hơn nữa, mô hình cung cấp thước đo mức độ quan trọng của đặc trưng (Feature Importance) giúp nhóm hiểu rõ yếu tố nào đóng vai trò then chốt quyết định giá nhà.

Hạn chế của mô hình là tốn nhiều thời gian huấn luyện và bộ nhớ hơn so với cây đơn lẻ hoặc hồi quy tuyến tính, đồng thời việc diễn giải cấu trúc của 100 cây kết hợp không còn trực quan như một cây đơn lẻ.

## 2. Dữ liệu đầu vào và kết quả đầu ra

- **Đầu vào:** Ma trận đặc trưng số $X$ có kích thước $N \times d$, được xử lý qua cùng một pipeline tiền xử lý trên tập Train.
- **Đầu ra:** Vector giá nhà dự đoán $\hat{y}$ có đơn vị là USD. Kết quả dự đoán trung bình từ nhiều cây sẽ tạo ra một hàm xấp xỉ mượt mà hơn nhiều so với dự đoán bậc thang của cây đơn lẻ.
- **Thông tin phụ:** Điểm số đánh giá mức độ quan trọng của từng đặc trưng (Feature Importances) được trích xuất để phân tích định tính.

## 3. Dữ liệu sử dụng

- **Tập Train (80%):** Dùng để lấy các mẫu con Bootstrap và huấn luyện độc lập từng cây quyết định trong rừng.
- **Tập Validation (10%):** Dùng để đối chiếu sơ bộ về sai số dự đoán và theo dõi mức độ ổn định khi tăng số lượng cây.
- **Tập Test (10%):** Dùng để đánh giá độc lập một lần duy nhất ở bước nghiệm thu cuối cùng, làm cơ sở đối chiếu trực tiếp với mạng MLP.

## 4. Phương pháp huấn luyện và đánh giá

### Hàm mục tiêu và giải thuật

Quá trình xây dựng Random Forest gồm hai cơ chế ngẫu nhiên hóa cốt lõi:

1. **Bootstrap Aggregation:** Với mỗi cây $b = 1, \dots, B$, một tập dữ liệu con cùng kích thước được tạo ra bằng cách lấy mẫu có hoàn lại (sampling with replacement) từ tập Train ban đầu.
2. **Feature Subsampling:** Tại mỗi nút phân nhánh của từng cây, thay vì quét qua toàn bộ $d$ đặc trưng, thuật toán chỉ chọn ngẫu nhiên một tập con gồm $m$ đặc trưng (thường là $m = \sqrt{d}$ hoặc $m = d/3$). Sau đó, thuật toán mới tìm điểm cắt tối ưu trên tập con này dựa trên tiêu chí giảm thiểu sai số bình phương trung bình (MSE Reduction).

### Cấu hình và tiêu chí đánh giá

Cấu hình dự kiến cho mô hình Random Forest trong bài lab gồm:
- **Số lượng cây (n_estimators):** 100 cây quyết định, vừa đủ để mô hình hội tụ ổn định mà không tốn quá nhiều thời gian tính toán trên máy cá nhân.
- **Hạt giống ngẫu nhiên (random_state):** Cố định `seed = 42` để đảm bảo việc lấy mẫu Bootstrap và chọn ngẫu nhiên đặc trưng hoàn toàn có thể tái lập.
- **Tiêu chí đánh giá:** Tính toán 4 chỉ số MSE, RMSE, MAE và $R^2$ trên cùng tập Test để so sánh với các mô hình còn lại.

## 5. Kế hoạch kiểm tra

Nhóm dự kiến thực hiện các bước kiểm tra sau:
- **Trích xuất độ quan trọng của đặc trưng:** Sau khi huấn luyện, nhóm sẽ in bảng hoặc vẽ biểu đồ độ quan trọng của từng đặc trưng. Dự kiến các đặc trưng như thu nhập trung vị (`median_income`) và vị trí địa lý sẽ có tỷ trọng ảnh hưởng cao nhất đến giá nhà.
- **Kiểm tra tính hợp lệ của đầu ra:** Vector dự đoán phải là số thực hữu hạn, không chứa NaN hay giá trị âm phi lý.
- **Kiểm tra tính nhất quán:** Mô hình được lưu trữ trong `results/saved_models/baseline_models.joblib` khi nạp lại phải đưa ra kết quả dự đoán hoàn toàn trùng khớp trên cùng một tập dữ liệu đầu vào.

## 6. Kế hoạch đánh giá và giới hạn

Random Forest được xem là đối thủ đối chứng mạnh nhất đối với mạng nơ-ron MLP trong bài thực hành này.

Một số điểm cần lưu ý và giới hạn:
- **Ưu thế của mô hình cây trên dữ liệu bảng:** Theo nhiều nghiên cứu và kinh nghiệm thực hành trong học máy, các mô hình dạng cây kết hợp (như Random Forest hoặc Gradient Boosting) thường thích ứng rất tự nhiên với dữ liệu dạng bảng có các biến phân loại và giá trị khuyết thiếu. Do đó, nếu kết quả thực nghiệm trên tập Test cho thấy Random Forest đạt $R^2$ cao hơn hoặc MAE thấp hơn so với mạng MLP, điều đó hoàn toàn bình thường. Nhóm sẽ ghi nhận kết quả này một cách trung thực và phân tích lý do dựa trên đặc điểm của từng loại thuật toán.
- **Không có khả năng ngoại suy xu hướng:** Tương tự như cây quyết định, Random Forest không thể dự đoán được các giá trị nhà vượt quá mức giá cao nhất đã gặp trong tập huấn luyện. Đối với các căn nhà có giá trị rất cao trên 500.001 USD, mô hình sẽ có xu hướng dự đoán thấp hơn giá trị thực.
