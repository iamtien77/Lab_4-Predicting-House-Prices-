# Lab 4 — Dự đoán giá nhà bằng MLP Regression

Dự án xây dựng mô hình MLP để dự đoán giá nhà và so sánh với Linear Regression, Decision Tree, Random Forest. Thiết kế chi tiết được trình bày trong [docs/design.md](docs/design.md).

Code và notebook hiện mới có phần khung hướng dẫn, chưa triển khai đầy đủ việc huấn luyện và đánh giá. Vì vậy, chưa có kết quả thực nghiệm để kết luận mô hình nào tốt hơn.

## 1. Bài toán và yêu cầu của đề

Bài toán regression dự đoán `median_house_value` từ dữ liệu California Housing trong `data/raw/housing.csv`. Chương trình đọc file trên máy và báo lỗi nếu thiếu file hoặc các cột không đúng yêu cầu.

Đề gốc ở `docs/Practice exercise 4.pdf` dùng Boston Housing và minh họa cách chia train/test theo tỷ lệ 80/20. Thiết kế này chọn California Housing và chia **80/10/10** để có tập validation riêng. Báo cáo cần giải thích rõ sự điều chỉnh này.

MLP dùng kiến trúc `32 ReLU → 16 ReLU → 1 linear`, Adam, MSE loss, theo dõi MAE và huấn luyện 100 epochs với batch size 32. Learning rate được chọn là 0.001; đề không ghi cụ thể giá trị này. Kết quả được so sánh với ba baseline bằng MSE, MAE, R² và thêm RMSE để dễ diễn giải sai số theo USD.

## 2. Đặc điểm dữ liệu cần lưu ý

- CSV có 20.640 dòng và 10 cột, gồm chín đặc trưng cùng target `median_house_value`.
- Cột `total_bedrooms` thiếu 207 giá trị. Cột `ocean_proximity` là biến phân loại có năm nhóm.
- Target lớn nhất là **500001 USD**, xuất hiện ở **965/20640** dòng. Đây là dấu hiệu dữ liệu bị giới hạn trên; cần giữ các mẫu này và nêu rõ khi phân tích kết quả.
- Nhóm `ISLAND` chỉ có **năm dòng**. Một tập có thể không chứa nhóm này sau khi chia, nên encoder cần xử lý được nhóm chưa gặp trong train. Số mẫu quá ít cũng chưa đủ để đánh giá riêng nhóm một cách đáng tin cậy.
- Target biểu thị giá nhà trung vị của một khu vực, không phải giá giao dịch của từng căn nhà.

Target được giữ nguyên đơn vị **USD**. MAE và RMSE có đơn vị USD, MSE có đơn vị USD², còn R² không có đơn vị. Nếu thử biến đổi target về sau, cần đưa dự đoán về đơn vị gốc trước khi đánh giá.

## 3. Các bước xử lý và huấn luyện

1. `load_housing_data(filepath=None)` đọc CSV, kiểm tra dữ liệu và trả `(df, target_col)`. Nếu thiếu target, chương trình báo lỗi thay vì tự điền giá trị.
2. Tách X/y rồi gọi `split_data_8_1_1`. Hàm trả sáu tập chưa tiền xử lý: X_train, X_val, X_test, y_train, y_val, y_test. Với seed 42 và dữ liệu này, số dòng dự kiến là 16.512/2.064/2.064. Khi dùng `qcut`, cần kiểm tra phân vị trùng và nhóm có quá ít mẫu.
3. Khám phá dữ liệu train để tìm hiểu phân bố và lựa chọn đặc trưng. Giữ test cho lần đánh giá cuối.
4. `fit_preprocessor(X_train)` tạo bộ tiền xử lý bằng `ColumnTransformer`. Biến số được điền bằng median rồi chuẩn hóa với StandardScaler. Biến phân loại được điền bằng giá trị phổ biến nhất rồi mã hóa bằng `OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)`.
5. `transform_features(preprocessor, X)` biến đổi dữ liệu thành dense ndarray theo các cột đã học từ train. Dùng lại bộ tiền xử lý cho validation, test và dữ liệu cần dự đoán. Số chiều đầu vào MLP lấy từ kết quả biến đổi.
6. Huấn luyện các baseline trước, sau đó đến MLP. Cả bốn mô hình dùng cùng cách chia dữ liệu và tiền xử lý. Decision Tree dùng `max_depth=5`, Random Forest dùng `n_estimators=100`.
7. Sau khi chọn cấu hình bằng validation, đánh giá cả bốn mô hình trên cùng test. Lập bảng chỉ số, vẽ biểu đồ giá thực tế so với dự đoán, đường loss/MAE và biểu đồ so sánh mô hình.
8. Dùng ba mẫu từ test để minh họa dự đoán. Dữ liệu đầu vào phải đi qua bộ tiền xử lý đã fit trước khi đưa vào mô hình.

`HousingFeatureEngineer` dùng để thử tạo thêm đặc trưng và mặc định chưa bật. Chỉ bổ sung bước này sau khi có baseline và có kế hoạch so sánh trên validation.

## 4. Các file kết quả và cách sử dụng

| File hoặc thư mục | Nội dung |
| --- | --- |
| `data/processed/train.csv`, `val.csv`, `test.csv` | Các tập dữ liệu có cả target, chưa điền giá trị thiếu, mã hóa hay chuẩn hóa; giữ NaN và không lưu index |
| `results/saved_models/preprocessor.joblib` | Toàn bộ bộ tiền xử lý đã fit trên train, gồm imputer, encoder và scaler |
| `results/saved_models/mlp_house_price_model.keras` | Mô hình MLP đã huấn luyện |
| `results/saved_models/baseline_models.joblib` | Ba mô hình baseline đã huấn luyện |
| `results/metrics/model_performance_summary.csv` | Các cột `Model`, `MSE`, `RMSE`, `MAE`, `R2_Score` trên cùng test |
| `results/figures/` | Các biểu đồ huấn luyện và đánh giá |

`save_processed_splits` lưu dữ liệu chưa tiền xử lý dưới dạng CSV. Notebook tiếp theo đọc CSV, tách target, rồi gọi `load_preprocessor` và `transform_features`. Quy trình này dùng CSV thay cho mảng `.npy` đã chuẩn hóa và lưu cả bộ tiền xử lý thay vì chỉ lưu scaler.

Khi nạp lại mô hình, cần dùng đúng bộ tiền xử lý của lần huấn luyện đó. Kiểm thử phải kiểm tra cả quá trình tiền xử lý và dự đoán để bảo đảm kết quả trước và sau khi lưu khớp nhau trong sai số cho phép.

## 5. Vai trò của các phần trong dự án

| File hoặc thư mục | Nhiệm vụ |
| --- | --- |
| `src/config.py`, `src/data/` | Quản lý cấu hình, đọc CSV, chia tập và tiền xử lý |
| `src/features/` | Tạo đặc trưng bổ sung khi cần thử nghiệm |
| `src/models/` | Xây dựng baseline và MLP; lưu, nạp mô hình cùng bộ tiền xử lý |
| `src/evaluation/`, `src/utils/`, `main.py` | Tính chỉ số, vẽ biểu đồ, ghi log và điều phối các bước |
| `notebooks/01_eda_and_preprocessing.ipynb` | Khám phá dữ liệu train và giải thích cách tiền xử lý |
| `notebooks/02_mlp_model_training.ipynb` | Huấn luyện baseline trước MLP và lưu mô hình |
| `notebooks/03_baseline_and_comparison.ipynb` | Nạp mô hình, đánh giá trên test và phân tích kết quả |
| `tests/` | Kiểm tra dữ liệu, mô hình, chỉ số và kết quả sau khi lưu/nạp |

Notebook và `main.py` gọi chung các hàm trong `src/` để thống nhất cách xử lý.

## 6. Môi trường chạy và các bước kiểm tra

`requirements.txt` liệt kê các thư viện học máy. Jupyter và pytest chưa được khai báo trong file này; cần thống nhất môi trường và phiên bản trước khi triển khai.

Sau khi hoàn thiện code và môi trường, chạy `python main.py`, các notebook theo thứ tự 01 → 02 → 03 và `pytest tests/ -v`. Các test mới có phần khung và đang được đánh dấu bỏ qua, nên chưa thể dùng chúng để xác nhận chương trình hoạt động đúng.

Những việc cần kiểm tra:

- [ ] Đọc đúng dữ liệu và báo lỗi rõ khi thiếu file, sai cột hoặc target không hợp lệ.
- [ ] Chia dữ liệu lặp lại được với cùng seed; các tập không trùng mẫu và X/y khớp từng dòng.
- [ ] Chỉ fit tiền xử lý trên train; xử lý được giá trị thiếu, nhóm mới và thay đổi thứ tự cột.
- [ ] Baseline và MLP trả dự đoán đúng kích thước trên cùng dữ liệu.
- [ ] Kết quả dự đoán khớp nhau trước và sau khi lưu/nạp.
- [ ] Các chỉ số đúng đơn vị; test chỉ dùng để đánh giá cuối.
- [ ] Notebook chạy theo thứ tự và có nhận xét dựa trên kết quả thực nghiệm.

## 7. Thứ tự triển khai

Bắt đầu bằng việc thống nhất dữ liệu đầu vào và các file cần lưu. Tiếp theo, hoàn thiện phần đọc dữ liệu, chia tập và tiền xử lý; huấn luyện baseline rồi đến MLP. Cuối cùng, bổ sung lưu/nạp mô hình, đánh giá và viết báo cáo trong notebook.

Trong quá trình này, ưu tiên kiểm tra data leakage, cách mã hóa biến phân loại và sự nhất quán giữa huấn luyện với dự đoán.
