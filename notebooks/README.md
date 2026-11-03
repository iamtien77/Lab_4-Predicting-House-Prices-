# Notebooks — Lab 4

**Trạng thái: skeleton, chưa có logic thực thi hoặc kết quả huấn luyện.** Các code cell chỉ chứa comment hướng dẫn. Notebook sẽ gọi các module trong `src/` sau khi triển khai, không sao chép implementation.

| Notebook | Trách nhiệm | Phụ trách |
| --- | --- | --- |
| `01_eda_and_preprocessing.ipynb` | CSV California local → split raw → EDA train → fit preprocessor trên train → lưu raw CSV và preprocessor | Huyền & My |
| `02_mlp_model_training.ipynb` | Đọc raw train/val và preprocessor → transform → baseline trước → MLP → lưu models | Tường & Trí |
| `03_baseline_and_comparison.ipynb` | Đọc raw test và artifacts → đánh giá cuối cùng → so sánh và diễn giải | Tiến & Thịnh |

Quy ước chung:

- California local, target `median_house_value` giữ nguyên USD. Không fallback sang Boston. PDF dùng Boston/80:20; California/80:10:10 là thiết kế nội bộ cần giải thích khi nộp.
- Train/val/test dự kiến 16.512/2.064/2.064 hàng. EDA để chọn đặc trưng chỉ dùng train.
- `fit_preprocessor(X_train)` tạo fitted ColumnTransformer; `transform_features(preprocessor, X)` trả dense ndarray. Numeric: median + StandardScaler; categorical: most_frequent + OneHotEncoder với unknown ignore, sparse_output=False, drop=None.
- `save_processed_splits` lưu `train.csv`, `val.csv`, `test.csv` gồm target, giữ NaN đặc trưng và chưa scale. Không dùng `.npy`.
- `save_preprocessor`/`load_preprocessor` lưu/tải toàn bộ fitted transformer tại `results/saved_models/preprocessor.joblib`, không chỉ scaler.
- Chiều input MLP lấy từ dữ liệu sau transform. `HousingFeatureEngineer` là optional và tắt mặc định.
- Test dùng một lần ở notebook03 sau khi chốt mô hình bằng validation; notebook02 không đánh giá test. Báo cáo MAE/RMSE bằng USD, MSE bằng USD² và R² không đơn vị.
- Dữ liệu có 207 NaN ở total_bedrooms; target tối đa 500001 USD ở 965/20640 hàng; ISLAND chỉ 5 hàng. Ghi rõ giới hạn, không tự loại quan sát.

Sau khi triển khai và xác minh môi trường, chạy notebook theo thứ tự 01 → 02 → 03. Jupyter/pytest chưa được khai báo trong requirements hiện tại; cần thống nhất dependency trước bước triển khai.
