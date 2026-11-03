# Lab 4 — MLP Regression dự đoán giá nhà

**Trạng thái: skeleton thiết kế, chưa triển khai logic huấn luyện/đánh giá.** Các hàm và test chủ yếu là khung hướng dẫn; notebook chứa Markdown và comment. Chưa có kết quả thực nghiệm để kết luận mô hình nào tốt hơn. Các lệnh bên dưới là quy trình dự kiến sau khi triển khai, không phải bằng chứng hệ thống đã chạy.

## 1. Phạm vi và đối chiếu đề bài

Bài toán regression dự đoán `median_house_value` từ dữ liệu California tại `data/raw/housing.csv`. Nguồn local là nguồn duy nhất; thiếu file hoặc sai schema phải báo lỗi, không tự tải hay đổi sang dataset khác.

Đề gốc nằm ở `docs/Practice exercise 4.pdf`. Task 2 của PDF dùng Boston Housing và ví dụ chia train/test 80/20, đồng thời dùng test làm validation. Thiết kế nội bộ trong tài liệu phân công chọn California Housing và chia **80/10/10** để có validation riêng. Đây là lựa chọn của dự án, không phải tỷ lệ được PDF yêu cầu. Khi nộp bài cần trình bày rõ khác biệt này.

Phần giữ theo đề: MLP Keras `32 ReLU → 16 ReLU → 1 linear`, Adam (learning rate 0.001), MSE loss, MAE khi huấn luyện, 100 epochs và batch size 32; so sánh với Linear Regression, Decision Tree, Random Forest. Đánh giá MSE, MAE, R²; bổ sung RMSE để dễ diễn giải.

## 2. Dữ liệu và giới hạn đã kiểm tra

- CSV local có 20.640 hàng, 10 cột: 9 đặc trưng và target `median_house_value`.
- `total_bedrooms` có 207 giá trị thiếu; `ocean_proximity` là categorical có 5 nhóm.
- Giá trị target lớn nhất là **500001 USD**, xuất hiện **965/20640** hàng. Đây là dấu hiệu giới hạn trên cần nêu trong báo cáo; không tự loại các hàng này.
- Nhóm `ISLAND` chỉ có **5 hàng**. Chia ngẫu nhiên có thể khiến một tập không có nhóm này; encoder phải xử lý nhóm chưa thấy. Không suy ra hiệu năng đáng tin cậy cho nhóm này từ vài quan sát.
- Target là giá trị nhà trung vị của khu vực trong dataset, không phải giá giao dịch một căn nhà cụ thể. Kết quả chỉ phản ánh phân phối và cách chia của bộ dữ liệu.

Target giữ nguyên đơn vị **USD** trong phần cốt lõi, không log/scale target. MAE và RMSE có đơn vị USD; MSE là USD²; R² không có đơn vị. Nếu thêm biến đổi target về sau, cần quyết định riêng và inverse-transform trước đánh giá.

## 3. Luồng dữ liệu dự kiến

1. `load_housing_data(filepath=None)` đọc CSV local, kiểm schema và trả `(df, target_col)`. Không impute target thiếu.
2. Tách X/y, dùng `split_data_8_1_1` trả 6 đối tượng pandas **chưa tiền xử lý**: X_train, X_val, X_test, y_train, y_val, y_test. Seed 42; dự kiến 16.512/2.064/2.064 hàng với dữ liệu hiện tại. Chính sách qcut phải xử lý phân vị trùng hoặc nhóm quá ít rõ ràng.
3. EDA phục vụ quyết định mô hình thực hiện trên train sau split. Không dùng test để chọn đặc trưng hay tinh chỉnh.
4. `fit_preprocessor(X_train)` fit một `ColumnTransformer`: numeric dùng median imputation rồi StandardScaler; categorical dùng most-frequent imputation rồi `OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)`.
5. `transform_features(preprocessor, X)` trả dense ndarray theo schema/thứ tự cột đã học; không fit lại trên validation, test hoặc mẫu suy luận. Chiều vào MLP lấy từ số cột sau transform.
6. Huấn luyện baseline trước để kiểm toàn tuyến, sau đó MLP với validation riêng. Cả bốn mô hình dùng cùng split và preprocessing. Decision Tree dự kiến max_depth=5, Random Forest n_estimators=100.
7. Sau khi chốt lựa chọn bằng validation, đánh giá cả bốn mô hình trên test một lần trong giai đoạn báo cáo cuối. Xuất bảng metrics, actual-vs-predicted, loss/MAE curves và biểu đồ so sánh; phân tích kết quả thực tế, không định trước mô hình thắng.
8. Demo dùng 3 mẫu giữ lại từ test để minh họa suy luận; đây không phải 3 căn nhà mới ngoài dataset. Suy luận raw input phải dùng đầy đủ preprocessor đã fit rồi mới predict.

Feature engineering (`HousingFeatureEngineer`) là phần mở rộng **tắt mặc định**, chưa nằm trong pipeline cốt lõi. Chỉ bật sau baseline và khi có kế hoạch đánh giá trên validation.

## 4. Hợp đồng artifact

| Artifact dự kiến                                      | Nội dung                                                                                     |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `data/processed/train.csv`, `val.csv`, `test.csv` | Raw split có cả target; giữ NaN đặc trưng, chưa impute/encode/scale; không lưu index |
| `results/saved_models/preprocessor.joblib`            | Toàn bộ fitted ColumnTransformer: imputer, encoder/categories và scaler                    |
| `results/saved_models/mlp_house_price_model.keras`    | MLP đã huấn luyện                                                                         |
| `results/saved_models/baseline_models.joblib`         | Ba baseline đã fit                                                                          |
| `results/metrics/model_performance_summary.csv`       | Model, MSE, RMSE, MAE, R2_Score trên cùng test                                              |
| `results/figures/`                                    | Các biểu đồ kết quả sau thực nghiệm                                                   |

`save_processed_splits` lưu raw CSV; notebook sau đọc các CSV này, tách target rồi gọi `load_preprocessor` và `transform_features`. Không có hợp đồng `.npy` đã scale và không lưu scaler riêng. Việc lưu/tải phải giữ schema, đơn vị target và khả năng tái tạo cùng dự đoán; test persistence cần kiểm cả chuỗi preprocessing → model.

## 5. Cấu trúc và trách nhiệm

| Khu vực                                         | Vai trò                                                    | Phụ trách             |
| ------------------------------------------------ | ----------------------------------------------------------- | ----------------------- |
| `src/config.py`, `src/data/`                 | Cấu hình, local loader, split, preprocessing và raw CSV  | Huyền & My             |
| `src/features/`                                | Feature engineering optional, tắt mặc định              | Huyền & My             |
| `src/models/`                                  | Baselines, MLP, persistence model và preprocessor          | Tường & Trí          |
| `src/evaluation/`, `src/utils/`, `main.py` | Metrics, biểu đồ, logging và điều phối               | Tiến & Thịnh          |
| `notebooks/01_eda_and_preprocessing.ipynb`     | Diễn giải dữ liệu và preprocessing                     | Huyền & My             |
| `notebooks/02_mlp_model_training.ipynb`        | Baseline trước, rồi MLP và lưu models                  | Tường & Trí          |
| `notebooks/03_baseline_and_comparison.ipynb`   | Nạp models, đánh giá test và so sánh                  | Tiến & Thịnh          |
| `tests/`                                       | Kiểm hợp đồng dữ liệu, model, metrics và persistence | Theo module phụ trách |

Notebook gọi chung logic trong `src/`; không xây một pipeline riêng. `main.py` điều phối cùng các module. Giữ cấu trúc hiện tại, chưa cần thêm tầng hay framework.

## 6. Môi trường và xác minh dự kiến

`requirements.txt` hiện liệt kê dependency ML; chưa xác minh môi trường chạy và chưa bổ sung dependency trong đợt chỉnh skeleton. README trước có hướng dẫn pytest/Jupyter nhưng hai công cụ này chưa được khai báo; cần thống nhất môi trường phát triển trước khi triển khai.

Sau khi logic và môi trường được hoàn thiện, các điểm chạy dự kiến là `python main.py`, notebook theo thứ tự 01 → 02 → 03 và `pytest tests/ -v`. Hiện tại chạy file không chứng minh pipeline hoạt động; test placeholder không chứng minh tính đúng.

Checklist chưa nghiệm thu:

- [ ] Loader/schema local, target hợp lệ và lỗi có thông tin.
- [ ] Split tái lập, không giao nhau, X/y thẳng hàng; xử lý qcut edge cases.
- [ ] Imputer/encoder/scaler chỉ học từ train; NaN, category mới, thứ tự feature được kiểm.
- [ ] Baseline và MLP fit/predict đúng shape trên cùng dữ liệu.
- [ ] Lưu/tải toàn chuỗi cho cùng dự đoán.
- [ ] Metrics đúng đơn vị, test chỉ dùng cho đánh giá cuối, không định trước thứ hạng.
- [ ] Notebook chạy tuần tự và có kết quả/nhận xét thực nghiệm thực tế.

## 7. Thứ tự triển khai sau khi được duyệt

Chốt schema và hợp đồng artifact → loader/split/preprocessor → baseline → MLP → persistence và đánh giá → notebook/báo cáo. Ưu tiên kiểm leakage, categorical encoding và tính nhất quán giữa train/inference; chưa thêm feature engineering phức tạp.
