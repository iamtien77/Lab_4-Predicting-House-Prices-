# 07 — Đánh giá và phân tích sai số

## 1. Phạm vi và trình tự

Chương này dùng run **20261116T113320345877Z_6c8931e5**, seed 42, 13 features, MLP 100 epochs; các bảng và 9 hình bên dưới đều thuộc cùng run. Learning curves được phân tích riêng ở [chương 06](06_mlp_training.md). Đây là phân tích kết quả cố định, không phải bước chọn lại model theo test.

**Input:** bốn mô hình đã fit, fitted preprocessor, X/y validation và test. **Output:** bảng metrics theo split, scatter actual–predicted, histogram residuals và kết luận có giới hạn.

1. Fit preprocessing và models chỉ trên train.
2. Theo dõi MLP và chọn best checkpoint theo validation MSE.
3. Sau khi cố định training, transform test bằng preprocessor đã fit.
4. Dự đoán USD; kiểm tra số mẫu, thứ tự và giá trị hữu hạn.
5. Tính MSE, RMSE, MAE, R²; lưu kết quả và vẽ.
6. Phân tích sai số để mô tả giới hạn. Nếu phát triển thêm sau khi đã xem test, cần thiết kế holdout phù hợp cho đánh giá tiếp.

## 2. Kết quả validation và test

Validation có 2.064 mẫu, test cũng có 2.064 mẫu. Bảng làm tròn để đọc; CSV là nguồn giá trị đầy đủ.

| Model | Validation RMSE (USD) | Validation MAE (USD) | Validation R² |
| --- | ---: | ---: | ---: |
| Random Forest | 50.474,83 | 32.535,17 | 0,8074 |
| MLP PyTorch | 55.288,39 | 36.551,04 | 0,7689 |
| Decision Tree | 71.822,64 | 50.444,72 | 0,6101 |
| Linear Regression | 74.051,58 | 51.760,14 | 0,5855 |

Nguồn: [validation_summary.csv](../../runs/20261116T113320345877Z_6c8931e5/results/metrics/validation_summary.csv).

| Model | Test MSE (USD²) | Test RMSE (USD) | Test MAE (USD) | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 2.297.281.701,04 | 47.929,97 | 31.298,94 | 0,8255 |
| MLP PyTorch | 2.642.317.401,20 | 51.403,48 | 35.366,58 | 0,7993 |
| Linear Regression | 4.446.723.192,43 | 66.683,76 | 49.481,55 | 0,6622 |
| Decision Tree | 4.908.901.183,75 | 70.063,55 | 49.788,76 | 0,6271 |

Nguồn: [model_performance_summary.csv](../../runs/20261116T113320345877Z_6c8931e5/results/metrics/model_performance_summary.csv). Hàm `summarize_models()` sort R² giảm dần. Với cùng y, thứ hạng MSE/RMSE và R² tương ứng nhau; MAE có thể xếp khác.

Random Forest có RMSE thấp nhất trên cả hai split trong run này. Linear Regression và Decision Tree đổi thứ tự giữa validation và test, nhắc rằng một split không đại diện mọi tình huống. Test RMSE của MLP thấp hơn validation không tự chứng minh leakage; hai tập gồm các mẫu khác nhau. Không suy ra mức cải thiện có ý nghĩa thống kê khi chưa có phân tích độ bất định.

## 3. Biểu đồ so sánh metrics

![So sánh bốn metrics test](../../runs/20261116T113320345877Z_6c8931e5/results/figures/model_comparison_bar.png)

**Cách đọc:** bốn panel riêng cho MSE (USD²), RMSE/MAE (USD), R² (không đơn vị). Trục X là model; ba error metrics càng thấp càng tốt, R² càng cao càng tốt. Nhãn số trên hình làm tròn dạng scientific notation.

- **Quan sát:** Random Forest dẫn đầu cả bốn metrics; MLP đứng sau. Linear Regression và Decision Tree có MAE gần nhau nhưng RMSE khác rõ hơn.
- **Ý nghĩa:** lỗi lớn tác động đến RMSE mạnh hơn MAE; MLP không mặc nhiên tốt hơn baseline trên dữ liệu bảng này.
- **Hạn chế:** không có error bars hoặc nhiều lần lặp; không so độ cao cột giữa hai panel có đơn vị khác nhau.
- **Hành động:** báo kèm cấu hình, split và run; không dùng thứ hạng test để dò lại model.

## 4. Cách đọc các hình đánh giá từng mô hình

**Actual vs predicted:** X là target thực tế, Y là dự đoán USD; đường đỏ đứt là y=x. Điểm trên đường là dự đoán cao, dưới đường là dự đoán thấp. Cột điểm thẳng đứng ở target 500.001 phản ánh nhiều nhãn cùng giá trị capped.

**Residual histogram:** residual = actual − predicted; X là sai số USD, Y là số mẫu, 40 bins. Residual dương là dự đoán thấp, âm là dự đoán cao; đường đỏ tại 0. Hàm hiện tạo histogram duy nhất, dù docstring có nhắc scatter residual theo prediction.

Các hình có trục tự co giãn: không so sánh độ rộng theo pixel hoặc chiều cao bin giữa models như một phép đo thống kê. Histogram gần hình chuông không đủ chứng minh phân phối chuẩn; không đủ kết luận heteroscedasticity nếu chưa xét residual theo fitted values.

### 4.1. Linear Regression

![Actual vs predicted Linear Regression](../../runs/20261116T113320345877Z_6c8931e5/results/figures/linear_regression_actual_vs_predicted.png)

- **Quan sát:** Điểm phân tán rộng, nhiều mẫu giá cao nằm dưới đường lý tưởng; có dự đoán âm và dự đoán trên 500.001 USD.
- **Ý nghĩa:** Cấu trúc tuyến tính chưa mô tả đầy đủ quan hệ trong bộ dữ liệu; R² test 0,6622 phản ánh giới hạn của cấu hình này.
- **Hạn chế:** Không suy ra mọi mô hình tuyến tính đều kém; feature engineering khác có thể cho kết quả khác.
- **Hành động:** Giữ làm mốc; thử features trên validation trong thí nghiệm sau, không clipping dự đoán để làm đẹp kết quả.

![Residual histogram Linear Regression](../../runs/20261116T113320345877Z_6c8931e5/results/figures/linear_regression_residuals.png)

- **Quan sát:** Phần dư tập trung gần 0 nhưng đuôi phải kéo dài tới khoảng 300.000 USD; có cả đuôi âm.
- **Ý nghĩa:** Một số mẫu bị dự đoán thấp đáng kể; MAE và RMSE cần đọc cùng nhau.
- **Hạn chế:** histogram không gắn được lỗi với từng feature và không chứng minh mô hình không bias chỉ vì đỉnh gần 0.
- **Hành động:** đối chiếu absolute errors và cỡ mẫu theo nhóm; giữ lại các mẫu sai nhiều khi báo cáo, không loại để làm giảm metrics.

### 4.2. Decision Tree

![Actual vs predicted Decision Tree](../../runs/20261116T113320345877Z_6c8931e5/results/figures/decision_tree_actual_vs_predicted.png)

- **Quan sát:** Các điểm tạo nhiều dải ngang; nhiều giá thực tế khác nhau nhận cùng một dự đoán. Có các mẫu giá cao bị dự đoán thấp xa đường chéo.
- **Ý nghĩa:** Mỗi lá trả một giá trị cố định; depth=5 giới hạn số vùng chia. Hình phù hợp với hạn chế biểu diễn thô của cây nông.
- **Hạn chế:** Dải ngang là cơ chế cây, không tự chứng minh overfitting; cần train/validation để đánh giá bias–variance.
- **Hành động:** Nếu thử depth khác, chỉ chọn bằng validation; không đổi depth sau khi xem test hiện tại.

![Residual histogram Decision Tree](../../runs/20261116T113320345877Z_6c8931e5/results/figures/decision_tree_residuals.png)

- **Quan sát:** Phần dư có khối lớn quanh 0 và đuôi ở cả hai phía, đuôi dương vượt 300.000 USD.
- **Ý nghĩa:** RMSE test 70.063,55 USD cao nhất trong bốn cấu hình; một số sai số lớn ảnh hưởng mạnh đến MSE.
- **Hạn chế:** histogram không gắn được lỗi với từng feature và không chứng minh mô hình không bias chỉ vì đỉnh gần 0.
- **Hành động:** đối chiếu absolute errors và cỡ mẫu theo nhóm; giữ lại các mẫu sai nhiều khi báo cáo, không loại để làm giảm metrics.

### 4.3. Random Forest

![Actual vs predicted Random Forest](../../runs/20261116T113320345877Z_6c8931e5/results/figures/random_forest_actual_vs_predicted.png)

- **Quan sát:** Các điểm ở vùng giá thấp và trung bình tập trung tương đối sát đường chéo; vẫn có mẫu lệch xa và nhiều dự đoán dưới trần tại cột target 500.001.
- **Ý nghĩa:** Trung bình nhiều cây cho dự đoán ít bị chia thành dải ngang rõ như cây đơn; metrics tổng thể tốt nhất trong run này.
- **Hạn chế:** Không đồng nghĩa dự đoán tốt mọi nhóm hoặc ngoại suy được vùng mới; capping vẫn ảnh hưởng diễn giải.
- **Hành động:** Báo cả sai số lớn và kết quả theo nhóm; đánh giá nhiều seeds/spatial holdout là hướng sau.

![Residual histogram Random Forest](../../runs/20261116T113320345877Z_6c8931e5/results/figures/random_forest_residuals.png)

- **Quan sát:** Histogram tập trung rõ gần 0, có đuôi âm và dương thưa hơn vùng trung tâm.
- **Ý nghĩa:** Phù hợp với MAE test 31.298,94 USD thấp nhất; vẫn có các trường hợp sai số hàng trăm nghìn USD.
- **Hạn chế:** histogram không gắn được lỗi với từng feature và không chứng minh mô hình không bias chỉ vì đỉnh gần 0.
- **Hành động:** đối chiếu absolute errors và cỡ mẫu theo nhóm; giữ lại các mẫu sai nhiều khi báo cáo, không loại để làm giảm metrics.

### 4.4. MLP PyTorch

![Actual vs predicted MLP PyTorch](../../runs/20261116T113320345877Z_6c8931e5/results/figures/mlp_pytorch_actual_vs_predicted.png)

- **Quan sát:** Nhiều điểm bám đường chéo; độ phân tán còn đáng kể ở vùng giá cao. Tại cột target 500.001, dự đoán trải từ thấp hơn đến vượt trần.
- **Ý nghĩa:** MLP học được quan hệ phi tuyến nhưng chưa vượt Random Forest trong cấu hình tham chiếu.
- **Hạn chế:** Một hình không chứng minh hội tụ hoặc độ ổn định qua seed; output linear không bảo đảm giá không âm trên mọi dữ liệu.
- **Hành động:** Đọc cùng learning curves ở chương 06; giữ checkpoint đã chọn bằng validation và không tuning theo scatter test.

![Residual histogram MLP PyTorch](../../runs/20261116T113320345877Z_6c8931e5/results/figures/mlp_pytorch_residuals.png)

- **Quan sát:** Đỉnh gần 0, hai đuôi vẫn hiện diện và đuôi dương kéo dài hơn vùng trung tâm.
- **Ý nghĩa:** MAE 35.366,58 USD và RMSE 51.403,48 USD cho thấy sai số lớn vẫn đóng góp đáng kể.
- **Hạn chế:** histogram không gắn được lỗi với từng feature và không chứng minh mô hình không bias chỉ vì đỉnh gần 0.
- **Hành động:** đối chiếu absolute errors và cỡ mẫu theo nhóm; giữ lại các mẫu sai nhiều khi báo cáo, không loại để làm giảm metrics.

## 5. Phân tích nhóm lỗi và giới hạn

Để phân tích theo nhóm, cần giữ cùng thứ tự giữa raw test, y và predictions. Tính absolute_error = |y − prediction|; sau đó group theo capped/uncapped hoặc ocean_proximity và báo cả count lẫn MAE. Nhóm không có mẫu phải ghi “không có mẫu”, không ghi MAE=0.

Notebook 03 có luồng phân tích nhóm riêng. Không lấy số liệu notebook gắn nhãn CLI. Các PNG CLI trên không phải hình phân tích nhóm; không có hình residual-vs-fitted hoặc feature importance trong bộ 15 PNG tham chiếu.

Các giới hạn chính gồm dữ liệu lịch sử, target trung vị khu vực, target capping, nhóm hiếm, random split một seed và test đã được quan sát trong phát triển. Kết quả không phải đánh giá định giá giao dịch nhà hiện tại. Prediction âm là failure mode cần ghi nhận, không tự clipping.

### Kết quả kiểm tra lại từ bundle CLI đã lưu

Ngày 24/11/2026, nạp model/preprocessor của run tham chiếu sau khi kiểm tra 25 hashes artifacts, dự đoán lại trên raw test rồi đối chiếu đủ bốn metrics của bốn models với CSV: đạt tolerance rtol=1e-5, atol=1e-3. Không fit hoặc tuning lại. Đây là phép kiểm chứng bổ sung từ CLI bundle, không lấy từ output notebook.

| Nhóm test | Số mẫu | MAE MLP (USD) |
| --- | ---: | ---: |
| Target = 500.001 | 91 | 63.516,00 |
| Target khác 500.001 | 1.973 | 34.068,26 |
| &lt;1H OCEAN | 920 | 37.526,12 |
| INLAND | 658 | 25.542,87 |
| NEAR BAY | 229 | 44.960,25 |
| NEAR OCEAN | 257 | 44.239,24 |

Hai dòng capped/uncapped là một cách chia nhóm; bốn dòng ocean_proximity là cách chia khác, không cộng gộp tất cả sáu dòng. Test không có mẫu ISLAND nên không có MAE cho nhóm này. Capped có MAE cao hơn uncapped trong run, nhưng không được quy toàn bộ chênh lệch cho riêng capping vì các nhóm còn khác nhau về giá và features.

Số prediction âm trên test: Linear Regression **5**, Decision Tree **0**, Random Forest **0**, MLP **0**. MLP không có prediction âm trong tập này không bảo đảm không âm ở mọi đầu vào. Mean residual lần lượt là 2.037,74; 1.353,91; 989,63; 2.851,54 USD: hơi dương trên test, nhưng trung bình nhỏ có thể che giấu sai số âm/dương lớn.

Source hiện tại chỉ khớp 10/18 hashes ghi trong run; xem phạm vi kiểm chứng ở [chương 09](09_execution_and_validation.md). Reload parity không chứng minh training lại source hiện tại tái tạo toàn bộ run.

## 6. Phân tích kết quả đạt được theo 5W1H

| Thành phần | Kết quả và ý nghĩa |
| --- | --- |
| What | Có bảng validation/test cho bốn model và 9 hình test; RF RMSE 47.929,97 USD, MLP 51.403,48 USD. |
| Why | Metrics và hình cho thấy độ lớn lỗi, hướng sai và các giới hạn mà một điểm số tổng không thể hiện hết. |
| Who | Người huấn luyện và người đọc báo cáo dùng bằng chứng để đánh giá các cấu hình trong phạm vi lab. |
| When | Artifacts ngày 16/11/2026; rà soát hình và CSV ngày 24/11/2026, sau khi training đã cố định. |
| Where | CSV và PNG trong results/metrics, results/figures của cùng run được dẫn ở trên. |
| How | Đọc số từ CSV, xem trực tiếp ảnh, đối chiếu công thức và quy ước dấu residual trong code. |

## 7. Tiêu chí hoàn thành và bàn giao

- Có đơn vị, split, số mẫu, run ID cho mọi bảng.
- Mỗi ảnh có cách đọc và nhận xét không vượt quá bằng chứng.
- Không dùng test để chọn epoch hoặc đổi cấu hình sau khi quan sát.
- Bàn giao bảng metrics, figures và giới hạn cùng model/preprocessor ở [chương 08](08_artifacts_and_inference.md).

Nguồn triển khai: [metrics](../../src/evaluation/metrics.py), [visualization](../../src/evaluation/visualization.py), [main.py](../../main.py), [notebook 03](../../notebooks/03_baseline_and_comparison.ipynb).

[Quay lại mục lục](README.md)
