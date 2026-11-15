# Tài liệu thiết kế thuật toán trong bài lab dự đoán giá nhà

Thư mục này chứa các tài liệu thiết kế chi tiết cho từng phương pháp hồi quy được áp dụng trong bài thực hành dự đoán giá nhà California Housing. Mục đích của các tài liệu này là xác định rõ cơ sở toán học, cách thức huấn luyện và tiêu chí đánh giá của từng thuật toán trước khi bước vào triển khai thực tế.

## Danh sách các thuật toán

Trong bài lab này, nhóm dự kiến thử nghiệm và so sánh bốn mô hình hồi quy:

| Thuật toán | Vai trò trong bài lab | Tài liệu thiết kế chi tiết |
| :--- | :--- | :--- |
| **Multilayer Perceptron (MLP)** | Mô hình trọng tâm của bài thực hành (mạng nơ-ron Keras Sequential) | [mlp_regression.md](mlp_regression.md) |
| **Linear Regression** | Mô hình baseline cơ sở, đại diện cho phương pháp tuyến tính truyền thống | [linear_regression.md](linear_regression.md) |
| **Decision Tree Regression** | Mô hình baseline phi tuyến dạng cây đơn lẻ | [decision_tree.md](decision_tree.md) |
| **Random Forest Regression** | Mô hình baseline học kết hợp (Ensemble Bagging) từ nhiều cây | [random_forest.md](random_forest.md) |

## Nguyên tắc thử nghiệm và đối chiếu công bằng

Để việc so sánh giữa các mô hình đảm bảo tính khoa học và khách quan, nhóm áp dụng các nguyên tắc thống nhất sau:

1. **Dùng chung một cách chia dữ liệu:** Toàn bộ bốn mô hình đều được huấn luyện trên cùng một tập Train (80%) và được đánh giá trên cùng một tập Test độc lập (10%) với cùng giá trị seed ngẫu nhiên (`42`).
2. **Dùng chung dữ liệu đầu vào:** Các mô hình đều nhận ma trận đặc trưng đầu vào đã được xử lý bởi cùng một bộ tiền xử lý (imputation + one-hot encoding + standard scaling) được học duy nhất từ tập Train.
3. **Đánh giá đa chiều:** Sử dụng bốn chỉ số thống nhất gồm Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Squared Error (MSE) và Hệ số xác định ($R^2$).
4. **Không thiên vị:** Nhóm ghi nhận trung thực kết quả đo lường thực tế trên tập Test. Nếu các mô hình dạng cây cho kết quả tốt hơn mạng nơ-ron MLP, nhóm sẽ phân tích nguyên nhân dựa trên đặc điểm của dữ liệu dạng bảng.
