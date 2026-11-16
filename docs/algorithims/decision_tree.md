# Decision Tree Regression trong bài lab dự đoán giá nhà

Nhóm dự kiến sử dụng mô hình Cây quyết định hồi quy (Decision Tree Regressor) làm mô hình baseline phi tuyến tính đầu tiên theo yêu cầu của Task 6 trong đề bài. Tài liệu này trình bày nguyên lý phân vùng không gian đặc trưng, cách huấn luyện và kế hoạch kiểm tra cho mô hình trong bài lab.

## 1. Mục đích và lý do lựa chọn

Decision Tree Regressor là mô hình học máy phi tuyến tính hoạt động theo cơ chế chia để trị. Khác với mô hình tuyến tính cố gắng vẽ một siêu phẳng phẳng qua toàn bộ không gian, Cây quyết định chia không gian đặc trưng nhiều chiều thành các vùng siêu chữ nhật độc lập và không giao nhau $R_1, R_2, \dots, R_M$. 

Với mỗi mẫu dữ liệu đầu vào $x$ rơi vào vùng $R_m$, mô hình sẽ dự đoán giá nhà bằng giá trị trung bình cộng của các mẫu thuộc tập huấn luyện nằm trong vùng đó:

$$\hat{y} = \sum_{m=1}^M c_m \cdot I(x \in R_m), \qquad c_m = \text{mean}(y_i \mid x_i \in R_m)$$

Trong đó $I(\cdot)$ là hàm chỉ thị (bằng 1 nếu $x \in R_m$ và bằng 0 nếu ngược lại), còn $c_m$ là hằng số biểu thị mức giá trung bình của vùng lá $m$.

Ưu điểm của Cây quyết định là rất trực quan, dễ hình dung qua sơ đồ cây và có khả năng tự động học các tương tác phi tuyến tính dạng điều kiện phân nhánh (chẳng hạn: "nếu vị trí gần biển và thu nhập cao thì giá cao"). Ngoài ra, cây quyết định không bị ảnh hưởng bởi việc các đặc trưng có thang đo khác nhau (dù nhóm vẫn áp dụng chung bộ tiền xử lý đã chuẩn hóa để giữ tính đồng nhất trong toàn bộ pipeline).

Hạn chế lớn nhất của cây quyết định đơn lẻ là rất dễ bị **quá khớp (overfitting)** nếu để cây phát triển tự do không kiểm soát. Cây có thể học thuộc lòng từng mẫu cá biệt trên tập Train, dẫn đến việc sai số trên tập kiểm tra thực tế tăng vọt. Do đó, việc giới hạn độ sâu tối đa của cây là bắt buộc.

## 2. Dữ liệu đầu vào và kết quả đầu ra

- **Đầu vào:** Ma trận đặc trưng số sau khi đi qua bộ tiền xử lý chung trên tập Train.
- **Đầu ra:** Vector giá nhà dự đoán $\hat{y}$ theo đơn vị USD. Kết quả dự đoán của cây quyết định có đặc tính gián đoạn dạng bậc thang (step-wise constant), nghĩa là các mẫu rơi vào cùng một nút lá sẽ nhận giá trị dự đoán hoàn toàn giống nhau.
- **Cấu trúc cây:** Cấu trúc phân nhánh và độ sâu của cây được lưu lại để phục vụ việc xem xét các đặc trưng được chọn để chia ở các nút gốc.

## 3. Dữ liệu sử dụng

- **Tập Train (80%):** Dùng để thuật toán tìm kiếm các điểm cắt (splits) tối ưu và xây dựng cây.
- **Tập Validation (10%):** Dùng để đánh giá sơ bộ độ sâu của cây, giúp nhóm quan sát sự chênh lệch sai số giữa Train và Validation để tránh chọn độ sâu quá lớn gây học vẹt.
- **Tập Test (10%):** Dùng để đánh giá độc lập một lần duy nhất ở bước nghiệm thu cuối cùng.

## 4. Phương pháp huấn luyện và đánh giá

### Hàm mục tiêu

Thuật toán xây dựng cây bằng phương pháp phân chia nhị phân đệ quy (Recursive Binary Splitting) theo hướng tham lam (Greedy). Tại mỗi nút, thuật toán quét qua tất cả các đặc trưng $j$ và tất cả các điểm cắt có thể $s$ để chia tập dữ liệu hiện tại $D$ thành hai nhánh con:

$$D_L(j, s) = \{x \mid x_j \le s\}, \qquad D_R(j, s) = \{x \mid x_j > s\}$$

Mục tiêu là tìm cặp $(j, s)$ sao cho mức độ suy giảm sai số bình phương trung bình (MSE Reduction) là lớn nhất:

$$\Delta \text{MSE} = \text{MSE}(D) - \left[ \frac{|D_L|}{|D|} \text{MSE}(D_L) + \frac{|D_R|}{|D|} \text{MSE}(D_R) \right]$$

Trong đó $\text{MSE}(D) = \frac{1}{|D|} \sum_{i \in D} (y_i - \bar{y}_D)^2$ đo mức độ phân tán của giá nhà xung quanh giá trị trung bình tại nút đó. Quá trình phân nhánh dừng lại khi đạt tới giới hạn độ sâu được thiết lập hoặc khi số lượng mẫu tại nút con nhỏ hơn ngưỡng tối thiểu.

### Cách tối ưu

Vì việc tìm cây quyết định tối ưu toàn cục là bài toán NP-đầy, thuật toán sử dụng phương pháp tìm kiếm cục bộ tham lam (như thuật toán CART). Tại mỗi bước, thuật toán chỉ quan tâm đến điểm cắt tốt nhất cho nút hiện tại mà không nhìn trước các bước chia trong tương lai.

### Cấu hình và tiêu chí đánh giá

Để đảm bảo mô hình không bị quá khớp nghiêm trọng trong bài lab, nhóm dự kiến cấu hình các siêu tham số như sau:
- **Độ sâu tối đa (max_depth):** Giới hạn ở mức vừa phải (dự kiến `max_depth = 5`) để cây duy trì tính tổng quát hóa tốt.
- **Hạt giống ngẫu nhiên (random_state):** Cố định `seed = 42` để hỗ trợ tái lập kết quả trong cùng môi trường dữ liệu và thư viện.
- **Tiêu chí đánh giá:** Tính toán đầy đủ 4 chỉ số MSE, RMSE, MAE và $R^2$ trên cùng tập Test.

## 5. Kế hoạch kiểm tra

Nhóm dự kiến thực hiện các bước kiểm tra sau:
- **Kiểm tra hiện tượng quá khớp:** So sánh R² trên Train và Validation. Khoảng cách lớn có thể là dấu hiệu overfitting; ghi nhận hạn chế và chỉ đề xuất thử cấu hình khác bằng Validation trong một thí nghiệm được duyệt. Không tự thay `max_depth=5` dựa trên Test.
- **Kiểm tra tính hợp lệ của dự đoán:** Đảm bảo toàn bộ giá trị dự đoán đều nằm trong miền giá trị dương hợp lý (không bị âm do bản chất giá trị lá $c_m$ là trung bình của các giá trị thực tế không âm).
- **Kiểm tra tính nhất quán:** CLI lưu `results/saved_models/baseline_models.joblib`; notebook dùng artifact dưới `notebooks/artifacts/` và hiển thị bảng/biểu đồ trong notebook. Dự đoán trước/sau khi lưu và nạp phải khớp trong sai số số học cho phép.

## 6. Kế hoạch đánh giá và giới hạn

Decision Tree Regressor đóng vai trò cầu nối trung gian giữa mô hình tuyến tính đơn giản và các mô hình phức tạp hơn.

Một số giới hạn cần lưu ý:
- **Dự đoán gián đoạn (Step-wise):** Cây quyết định không thể dự đoán một đường cong mượt mà. Nếu giá trị nhà thực tế có tính liên tục mượt theo thu nhập, cây chỉ có thể ước lượng theo từng nấc giá cố định.
- **Kém linh hoạt khi ngoại suy:** Nếu một căn nhà ở tập Test có đặc trưng nằm ngoài khoảng giá trị của tập Train, cây quyết định chỉ có thể gán nó vào lá gần nhất và dự đoán mức giá trung bình của lá đó, không thể ngoại suy tăng dần theo xu hướng như Linear Regression hay MLP.
- **Độ ổn định thấp:** Những thay đổi nhỏ trong dữ liệu huấn luyện có thể làm cấu trúc cây thay đổi đáng kể. Random Forest ở bước tiếp theo hướng tới giảm sự nhạy cảm này, không bảo đảm loại bỏ hoàn toàn.
