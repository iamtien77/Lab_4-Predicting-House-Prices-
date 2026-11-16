# Hướng dẫn sử dụng notebook — Lab 4

Ba notebook trình bày lần lượt các bước chuẩn bị dữ liệu, huấn luyện và đánh giá mô hình. Các ô code gọi hàm trong `src/` để dùng chung cách xử lý với chương trình chính. Phiên bản PyTorch chưa chạy xác minh vì môi trường thiếu `torch`; code không phải bằng chứng huấn luyện thành công.

## Thứ tự thực hiện

| Notebook | Nội dung |
| --- | --- |
| `01_eda_and_preprocessing.ipynb` | Đọc CSV, chia tập, khám phá dữ liệu train, fit bộ tiền xử lý và lưu dữ liệu |
| `02_mlp_model_training.ipynb` | Đọc train/validation, nạp bộ tiền xử lý, huấn luyện baseline trước MLP và lưu mô hình |
| `03_baseline_and_comparison.ipynb` | Đọc test, nạp mô hình, đánh giá và giải thích kết quả so sánh |

## Dữ liệu và tiền xử lý

Dùng California Housing từ `data/raw/housing.csv`, với target `median_house_value` giữ nguyên đơn vị USD. Đề gốc minh họa Boston Housing và tỷ lệ 80/20; báo cáo cần giải thích lựa chọn California Housing và tỷ lệ 80/10/10.

Với 20.640 mẫu, train/validation/test dự kiến có 16.512/2.064/2.064 dòng. Việc khám phá dữ liệu để chọn đặc trưng chỉ dùng train.

`fit_preprocessor(X_train)` tạo `ColumnTransformer` đã fit trên train. Biến số được điền bằng median rồi chuẩn hóa bằng StandardScaler. Biến phân loại được điền bằng giá trị phổ biến nhất rồi mã hóa bằng `OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)`. Hàm `transform_features(preprocessor, X)` dùng bộ này để trả dense ndarray cho từng tập.

`save_processed_splits` lưu `train.csv`, `val.csv`, `test.csv` gồm đặc trưng và target chưa tiền xử lý, vẫn giữ NaN. Các notebook đọc CSV rồi mới biến đổi dữ liệu; quy trình không dùng `.npy`.

`save_preprocessor` và `load_preprocessor` lưu/nạp toàn bộ bộ tiền xử lý tại `results/saved_models/preprocessor.joblib`. Mô hình phải đi kèm đúng bộ tiền xử lý của lần huấn luyện đó.

## Huấn luyện và đánh giá

Huấn luyện baseline trước MLP. Số chiều đầu vào MLP lấy từ dữ liệu sau tiền xử lý. `HousingFeatureEngineer` là phần mở rộng và mặc định chưa bật.

Notebook 02 dùng PyTorch CPU và seed chung, target scaling chỉ fit trên train. Mạng chạy đủ 100 epochs mặc định, khôi phục best validation MSE checkpoint, không early stopping. RMSE là metric chính; MSE và RMSE cho cùng thứ hạng checkpoint. History dictionary lưu metrics USD²/USD trên snapshot cuối epoch, tính đều theo từng mẫu; objective training dùng target chuẩn hóa. Notebook 03 đánh giá test cuối cùng, MAE/RMSE theo USD, MSE theo USD², R² không có đơn vị.

Checkpoint `notebooks/artifacts/mlp_house_price_model.pt` lưu weights, target statistics, kiến trúc và best epoch. Loader nạp CPU rồi `predict_mlp` inverse-transform về USD; không nạp `.keras` cũ, cần chạy lại notebook 02 để tạo `.pt`.

Dữ liệu có 207 giá trị thiếu ở `total_bedrooms`, 965 dòng đạt target lớn nhất 500001 USD và chỉ năm dòng thuộc nhóm `ISLAND`. Cần nêu rõ các giới hạn này khi nhận xét, thay vì tự loại mẫu khỏi dữ liệu.

## Chuẩn bị môi trường

`requirements.txt` đã khai báo PyTorch, JupyterLab, nbclient và pytest. Khi có đủ dependencies, chạy notebook theo thứ tự 01 → 02 → 03. Chưa có xác nhận notebook hoặc PyTorch tests chạy thành công trong môi trường hiện tại.
