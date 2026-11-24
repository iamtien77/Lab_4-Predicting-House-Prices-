# 04 — Tiền xử lý và feature engineering

Giai đoạn này chuyển raw features thành ma trận số dùng chung cho baseline và MLP. Mục tiêu là xử lý missing values, biểu diễn categorical features và điều chỉnh thang đo numeric features; mọi statistics chỉ được học từ train.

## 1. Input, output và điều kiện bắt đầu

Input của `fit_preprocessor()` là `X_train`: DataFrame không rỗng có đúng 8 numeric features và `ocean_proximity`. Target đã tách riêng. Numeric features được phép thiếu nhưng không chứa infinity; categorical phải là chuỗi hoặc missing. Cột hoàn toàn thiếu trên train bị từ chối vì không có thông tin tính giá trị điền khuyết.

Output của hàm là fitted `ColumnTransformer`. Sau đó `transform_features()` trả dense NumPy array có shape `(N, D)` và toàn bộ giá trị hữu hạn; `D` lấy từ `get_feature_names_out()`, không hardcode.

| Đầu ra | Vai trò |
| --- | --- |
| Train features | Fit ba baseline và MLP |
| Validation features | Theo dõi và đánh giá cấu hình/checkpoint |
| Test features | Đánh giá cuối sau khi training đã cố định |
| Fitted preprocessor | Giữ imputation, scaling và vocabulary để inference |
| Feature names | Diễn giải cột và kiểm tra input dimension |

CSV `data/processed/{train,val,test}.csv` trong run vẫn là **raw splits**, giữ NaN và target USD. Chúng không phải transformed matrix.

## 2. Nhánh numeric: cách làm và lý do

Pipeline trong [preprocessing.py](../../src/data/preprocessing.py) là `SimpleImputer(strategy="median")` → `HousingFeatureEngineer` → `StandardScaler`.

1. Imputer tính median của từng feature trên train và thay ô thiếu bằng median tương ứng. Median ít bị kéo bởi một số giá trị rất lớn hơn mean, nhưng không giữ thông tin về cơ chế missing.
2. Imputer xuất DataFrame để feature engineering truy cập cột bằng tên. `HousingFeatureEngineer` tạo bản sao, giữ thứ tự cột đã học và không dùng target.
3. Scaler học mean và population standard deviation trên numeric features của train sau biến đổi. Với feature không hằng:

   $$x'_{ij}=\frac{x_{ij}-\mu_{j,train}}{\sigma_{j,train}}.$$

   Feature hằng được xử lý với scale bằng 1. Validation/test có mean khác 0 hoặc standard deviation khác 1 là điều có thể xảy ra; không fit lại scaler để ép chúng về 0/1.

Scaling giúp MLP tối ưu trên các chiều có thang đo gần nhau. Cây không cần scaling để hoạt động, nhưng project giữ chung biểu diễn để nhất quán. Standardization không làm dữ liệu bất kỳ trở thành phân phối chuẩn và không loại outlier.

## 3. Nhánh categorical

`SimpleImputer(strategy="most_frequent")` học mode trên train để điền missing, sau đó `OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)` tạo các cột 0/1. Nhánh này không qua StandardScaler.

Run tham chiếu có thứ tự nhãn `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`. Ví dụ `INLAND` thành `[0, 1, 0, 0, 0]`; nhãn chưa thấy thành `[0, 0, 0, 0, 0]`. Unknown category không gây lỗi nhưng model chưa có quan sát train trực tiếp cho ý nghĩa nhãn mới.

`drop=None` giữ mọi cột one-hot. Khi Linear Regression có intercept, các cột này phụ thuộc tuyến tính; không diễn giải từng hệ số như tác động độc lập duy nhất. Dự đoán vẫn được đánh giá trên validation.

## 4. Feature engineering tùy chọn

[HousingFeatureEngineer](../../src/features/feature_engineering.py) có hai tùy chọn mặc định tắt. `--create-interactions` tạo ba ratios, không tạo mọi tích chéo giữa features.

| Feature | Công thức | Ý nghĩa dự kiến |
| --- | --- | --- |
| `rooms_per_household` | `total_rooms / households` | Quy mô phòng trên hộ |
| `bedrooms_ratio` | `total_bedrooms / total_rooms` | Tỷ lệ phòng ngủ |
| `people_per_household` | `population / households` | Số người trên hộ |

Ratios tính **sau imputation, trước log1p**. Nếu bật ratios, `households` và `total_rooms` phải dương. Không tự thay mẫu số 0 bằng một số nhỏ vì có thể tạo feature cực lớn và che lỗi dữ liệu.

`--apply-log` áp dụng $\log(1+x)$ lên `total_rooms`, `total_bedrooms`, `population`, `households`; input phải lớn hơn −1. Log nén các giá trị lớn, nhưng có cải thiện model hay không phải được kiểm tra trên validation.

Ví dụ minh họa, không phải kết quả một mẫu thực nghiệm: với 100 phòng, 20 phòng ngủ, 80 người và 10 hộ, ratios là 10; 0,2; 8. Nếu bật log, `total_rooms` thành `log(101)` còn ratio phòng/hộ vẫn bằng 10 trước scaling. Không chia hai giá trị đã log để tính ratio.

Với 5 nhãn categorical trong train, mặc định có `8 + 5 = 13` chiều; bật ratios có `11 + 5 = 16`. Nếu vocabulary train khác, số chiều cũng khác. Hai tùy chọn tắt thì transformer vẫn kiểm tra dữ liệu và giữ numeric features gốc.

## 5. Thứ tự gọi và chống leakage

```python
# Sau khi đã có raw splits:
preprocessor = fit_preprocessor(
    X_train, apply_log=False, create_interactions=False
)
train_features = transform_features(preprocessor, X_train)
val_features = transform_features(preprocessor, X_val)
# Chỉ sau khi training và cấu hình đã cố định:
test_features = transform_features(preprocessor, X_test)
input_dim = train_features.shape[1]
```

Khi inference, nạp preprocessor và gọi `transform_features()`; không fit lại trên mẫu mới. Median numeric, mode categorical, mean/scale và one-hot vocabulary phải giữ nguyên. Hàm transform kiểm tra fitted state, schema, shape và finite values; sai tên cột hoặc thiếu feature cần sửa ở nguồn thay vì cắt cột cho vừa model.

## 6. Kết quả thực tế theo 5W1H

Nguồn là [manifest run 20261116T113320345877Z_6c8931e5](../../runs/20261116T113320345877Z_6c8931e5/manifest.json), hoàn tất ngày 16/11/2026. Đây là đối chiếu artifacts đã có, không huấn luyện lại.

| Thành phần | Kết quả và diễn giải |
| --- | --- |
| **What** | Manifest ghi 13 feature names; `apply_log=false`, `create_interactions=false`. Từ split sizes và hợp đồng transform, shape suy ra là train `(16512, 13)`, validation/test `(2064, 13)`; không phải ma trận được lưu riêng. |
| **Why** | Biểu diễn số đáp ứng input contract của models. Imputation tránh phải bỏ mẫu vì missing features. Run không có ablation để quy riêng mức cải thiện RMSE cho từng bước. |
| **Who** | Baseline và MLP dùng cùng ma trận; người inference dùng lại fitted preprocessor. |
| **When** | Fit sau split, trước training; test transform sau khi cấu hình cố định. |
| **Where** | Manifest chứa names/flags; artifact tại [preprocessor.joblib](../../runs/20261116T113320345877Z_6c8931e5/results/saved_models/preprocessor.joblib). |
| **How** | Đối chiếu manifest và code; kiểm tra shape/finite do `transform_features()` thực hiện. [test_data.py](../../tests/test_data.py) và [test_features.py](../../tests/test_features.py) mô tả các kiểm tra; chưa chạy lại tests trong lần viết tài liệu. |

**Kết luận:** đã có biểu diễn thống nhất và fitted artifact để bàn giao cho models. Ratios/log chưa bật nên không gán metrics run này cho các tùy chọn đó.

**Hạn chế và hành động:** median/mode làm mất dấu hiệu missing; unknown category và nhóm hiếm có thể dự đoán kém. Muốn thử ratios/log cần run riêng, giữ split/seed, đối chiếu validation và không chọn cấu hình theo test đã xem.

## 7. Tiêu chí hoàn thành và bàn giao

- Target đã tách khỏi features; preprocessor chỉ fit train.
- Tất cả ma trận đúng shape và hữu hạn; input dimension lấy động.
- Flags được ghi cùng run; preprocessor được lưu cùng model.
- Inference dùng cùng schema và transformer.

Bàn giao features và target USD cho [baseline](05_baseline_models.md). [MLP](06_mlp_training.md) sẽ bổ sung target scaling nội bộ.

[Quay lại mục lục](README.md)
