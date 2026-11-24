# 05 — Xây dựng các baseline

Baseline tạo mốc để biết MLP có mang lại lợi ích trên cùng bài toán hay không. Project huấn luyện Linear Regression, Decision Tree và Random Forest theo thứ tự này trước MLP. Random Forest là đối chứng phi tuyến mạnh hơn, không hàm ý mọi baseline đều đơn giản hoặc dễ giải thích như nhau. Implementation hiện tại không có DummyRegressor.

## 1. Input, output và nguyên tắc so sánh

Input là `train_features` hữu hạn từ giai đoạn 04 và `y_train` theo USD, cùng số dòng/thứ tự. Baseline không chuẩn hóa target; output của mỗi model là vector prediction USD `(N,)`.

Tất cả model sử dụng chung split 80/10/10, seed 42 và cùng 13 features trong run tham chiếu. Chỉ fit train. Validation phục vụ đối chiếu; test chỉ được đánh giá sau khi cấu hình cố định. Không so sánh model train trên features bật ratios với model khác ở một run tùy ý rồi quy toàn bộ chênh lệch cho thuật toán.

## 2. Linear Regression: mốc quan hệ tuyến tính

Mô hình dự đoán:

$$\hat y=b+\sum_{j=1}^{D}w_jx_j.$$

OLS tìm hệ số tối thiểu hóa tổng squared errors trên train; `fit_intercept=True` cho phép học bias. Implementation gọi scikit-learn `LinearRegression`, không tự lập trình gradient descent cho baseline này.

Ưu điểm là mối liên hệ giữa feature đã biến đổi và prediction tương đối trực tiếp; làm mốc kiểm tra liệu quan hệ tuyến tính đã đủ hay chưa. Hệ số numeric liên quan đến một đơn vị feature chuẩn hóa, không phải một đơn vị raw feature. Với one-hot đầy đủ và intercept, các cột phụ thuộc tuyến tính nên hệ số riêng không có diễn giải nhân quả hoặc độc lập duy nhất.

Hạn chế: khó biểu diễn ranh giới địa lý và tương tác phi tuyến nếu features không mã hóa chúng; nhạy với outlier theo squared error; có thể dự đoán âm hoặc ngoại suy lớn. Không clipping prediction để làm đẹp metrics.

## 3. Decision Tree: chia không gian thành các vùng

`DecisionTreeRegressor(max_depth=5, random_state=42)` chọn feature và threshold để giảm squared error, lặp lại trên các nhánh đến điều kiện dừng. Prediction tại một leaf là trung bình target của các mẫu train trong leaf đó.

Độ sâu giới hạn 5 tạo tối đa 32 leaves nếu cây phát triển đầy đủ. Đây là giới hạn lý thuyết, không phải số leaves đã đo trong artifact. Giới hạn độ sâu giúp kiểm soát độ phức tạp và hỗ trợ đọc quy tắc, nhưng có thể không đủ biểu diễn các quan hệ chi tiết.

Cây tạo hàm dự đoán dạng từng vùng hằng; các mẫu khác nhau cùng leaf có prediction giống nhau. Không suy ra overfitting chỉ vì cây có nhiều nhánh, hoặc underfitting chỉ vì validation kém: cần đối chiếu train/validation và cấu hình.

## 4. Random Forest: tổng hợp nhiều cây

`RandomForestRegressor(n_estimators=100, random_state=42, max_features=1.0)` huấn luyện nhiều cây với bootstrap theo default hiện hành của estimator trong run. Prediction cuối là trung bình:

$$\hat y(x)=\frac{1}{100}\sum_{t=1}^{100}\hat y_t(x).$$

Bootstrap làm các cây thấy tập mẫu khác nhau. **`max_features=1.0` xét toàn bộ features tại mỗi split**, vì vậy không mô tả run này như sử dụng feature subsampling. Code không đặt `max_depth` cho forest; không lấy `DT_MAX_DEPTH=5` của cây đơn áp sang forest.

Trung bình nhiều cây có thể giảm dao động so với cây đơn và biểu diễn quan hệ phi tuyến. Đánh đổi là chi phí fit, bộ nhớ và khả năng giải thích tổng thể. Run này không đo thời gian riêng từng model nên không gán con số tốc độ hoặc bộ nhớ. Các cây dự đoán từ giá trị trong train, nên khó ngoại suy xu hướng giá ra ngoài miền target đã học.

## 5. Quy trình thực hiện trong code

[BaselineRegressionSuite](../../src/models/baseline_models.py) giữ dictionary models theo thứ tự Linear Regression → Decision Tree → Random Forest.

```python
suite = BaselineRegressionSuite(
    random_state=42, dt_max_depth=5,
    rf_n_estimators=100, fit_intercept=True
)
suite.fit_all(train_features, y_train)
validation_predictions = suite.predict_all(val_features)
```

`fit_all()` kiểm tra X/y rồi fit lần lượt; trạng thái fitted chỉ được đánh dấu khi hoàn tất. `predict_all()` từ chối gọi trước fit, kiểm tra fitted state từng estimator và xác nhận vector prediction đúng shape, hữu hạn. `get_model(name)` lấy model bằng tên và báo `KeyError` nếu sai tên.

Trong [main.py](../../main.py), baselines được lưu ngay sau fit, rồi MLP được train. Bảng validation tổng hợp được tính sau training MLP để so sánh đủ bốn model. Artifact [baseline_models.joblib](../../runs/20261116T113320345877Z_6c8931e5/results/saved_models/baseline_models.joblib) chứa ba models của cùng run.

## 6. Kết quả validation và cách diễn giải

Nguồn: [validation_summary.csv](../../runs/20261116T113320345877Z_6c8931e5/results/metrics/validation_summary.csv), run `20261116T113320345877Z_6c8931e5`, ngày 16/11/2026, validation 2.064 mẫu. Số hiển thị đã làm tròn; CSV giữ độ chính xác gốc.

| Model | MSE (USD²) | RMSE (USD) | MAE (USD) | R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 5.483.637.211,76 | 74.051,58 | 51.760,14 | 0,5855 |
| Decision Tree | 5.158.492.192,23 | 71.822,64 | 50.444,72 | 0,6101 |
| Random Forest | 2.547.708.533,51 | 50.474,83 | 32.535,17 | 0,8074 |

**Quan sát:** Random Forest có RMSE thấp nhất trong ba baseline, thấp hơn Linear Regression khoảng 23.576,75 USD, tương đương 31,84%; thấp hơn cây đơn khoảng 21.347,81 USD, tương đương 29,72%. Decision Tree hơn Linear Regression khoảng 2.228,94 USD RMSE.

**Ý nghĩa:** dữ liệu và cấu hình này cho thấy đối chứng phi tuyến forest tạo mốc mạnh. Kết quả phù hợp với khả năng mô hình hóa nhiều vùng và tổng hợp cây; bảng metrics không chứng minh riêng bootstrap là nguyên nhân hay mọi cấu hình forest đều tốt hơn tuyến tính.

**Hạn chế:** một split, một seed, target bị capping và nhóm hiếm làm kết luận chưa đại diện cho mọi vùng hay thời điểm. Không có train metrics baseline trong bảng này để kết luận dứt khoát underfit/overfit. R² là chất lượng tương đối so với dự đoán trung bình trên tập đánh giá, không phải “80,74% dự đoán chính xác”.

**Hành động:** dùng các mốc validation trên để đánh giá MLP ở giai đoạn 06. Nếu nghiên cứu thêm độ sâu/số cây/features, dùng validation hoặc một quy trình cross-validation được thiết kế trước; không tối ưu theo test hiện tại đã xem. [Giai đoạn 07](07_evaluation_and_error_analysis.md) trình bày biểu đồ test actual-vs-predicted, residuals và so sánh model; các hình test không được dùng làm cơ sở tuning ở giai đoạn baseline.

## 7. So sánh trade-offs và failure modes

| Model | Lý do có trong bài lab | Failure mode cần theo dõi | Cách đánh giá phù hợp |
| --- | --- | --- | --- |
| Linear Regression | Mốc tuyến tính, ít cấu hình | Phi tuyến chưa được biểu diễn; ảnh hưởng outlier; prediction âm | RMSE/MAE validation và residuals theo mức giá |
| Decision Tree | Đối chứng quy tắc phi tuyến | Độ sâu 5 có thể quá thô; nhạy thay đổi mẫu | So sánh train/validation khi thử độ sâu; giữ test cuối |
| Random Forest | Đối chứng ensemble mạnh | Chi phí lớn hơn; không ngoại suy tốt; khó diễn giải tổng thể | Metrics validation, ổn định nhiều seeds và tài nguyên nếu mở rộng |

Cả ba cần numeric features hữu hạn, target đúng đơn vị và các mẫu train đại diện cho miền dự đoán. Việc chạy thành công không thay thế kiểm tra chất lượng dữ liệu.

## 8. Kết quả theo 5W1H và bàn giao

| Thành phần | Kết quả thực tế |
| --- | --- |
| **What** | Ba baseline đã có artifact; forest đạt validation RMSE 50.474,83 USD, thấp nhất trong ba cấu hình. |
| **Why** | Tạo mốc kiểm chứng giá trị của MLP; chênh lệch chỉ có ý nghĩa trong cùng dữ liệu, features và run. |
| **Who** | Người huấn luyện dùng đối chiếu MLP; người đọc báo cáo dùng kiểm tra tính công bằng. |
| **When** | Fit trước MLP trong run ngày 16/11/2026, seed 42; cấu hình từ [manifest](../../runs/20261116T113320345877Z_6c8931e5/manifest.json). |
| **Where** | CSV validation và `baseline_models.joblib` được liên kết ở trên; source tại `src/models/baseline_models.py`. |
| **How** | Đối chiếu CSV với cấu hình manifest và code. [test_models.py](../../tests/test_models.py) có kiểm tra fit/predict/round-trip; lần viết tài liệu không chạy training hoặc tests lại. |

**Kết luận:** giai đoạn đã tạo ba mốc có thể so sánh với MLP, chưa chứng minh model tốt nhất ngoài dataset. Bàn giao suite đã fit, predictions validation và artifact; giữ test cho [giai đoạn đánh giá cuối](07_evaluation_and_error_analysis.md).

[Tiếp: huấn luyện MLP](06_mlp_training.md) · [Quay lại mục lục](README.md)
