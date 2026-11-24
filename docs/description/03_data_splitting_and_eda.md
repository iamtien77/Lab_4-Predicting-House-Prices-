# 03 — Chia tập và phân tích EDA

> Phần EDA đã được bổ sung chi tiết theo phê duyệt. Nhận xét dựa trên 5 hình và CSV train có sẵn của run tham chiếu `20261116T113320345877Z_6c8931e5`; đã đọc lại ngày 24/11/2026. Không chạy lại training hoặc tạo run mới.

## 1. Mục tiêu, input và output

Giai đoạn này giúp người học xác định dữ liệu dùng để huấn luyện, hiểu phân bố và quan hệ giữa các biến, rồi đưa ra giả thuyết cho preprocessing. EDA không chứng minh mô hình tốt và không thay thế đánh giá validation/test.

- **Input:** dữ liệu thô hợp lệ, 9 features và target `median_house_value` theo USD.
- **Output:** raw train/validation/test splits; thống kê mô tả train; 5 file hình EDA; nhận xét và các vấn đề bàn giao cho preprocessing.
- **Đơn vị quan sát:** khu dân cư, không phải từng căn nhà riêng lẻ.
- **Nguồn đối chiếu:** [manifest](../../runs/20261116T113320345877Z_6c8931e5/manifest.json), [train CSV](../../runs/20261116T113320345877Z_6c8931e5/data/processed/train.csv), [code visualization](../../src/evaluation/visualization.py).

## 2. Chia tập trước khi học thống kê

| Tập | Tỷ lệ | Số mẫu trong run tham chiếu | Vai trò |
| --- | ---: | ---: | --- |
| Train | 80% | 16.512 | EDA chi tiết, fit preprocessing và model |
| Validation | 10% | 2.064 | Theo dõi, đối chiếu cấu hình và chọn checkpoint |
| Test | 10% | 2.064 | Đánh giá cuối sau khi cố định mô hình |

[split_data_8_1_1](../../src/data/preprocessing.py) tách train/temp, rồi chia temp thành validation/test. Cấu hình tham chiếu dùng seed 42, phân tầng target bằng quantile bins: 10 bins ở lần tách đầu, 5 bins ở lần tách sau. Ranh giới phân vị trùng nhau có thể làm giảm số bins thực tế. Đây không phải phân tầng theo thu nhập.

Khi thực hiện cần kiểm tra tổng số mẫu được bảo toàn, index không chồng lấn giữa các tập và X/y được căn chỉnh. Kiểm tra index nên thực hiện trên kết quả split trong bộ nhớ; không suy ra độc lập chỉ từ kích thước ba CSV đã lưu. Trong lần cập nhật tài liệu này đã kiểm đếm kích thước CSV, chưa chạy lại toàn bộ kiểm thử split.

Thống kê tổng quan CSV ở [chương 02](02_data_understanding.md) phục vụ audit. EDA dùng để định hướng biến đổi features chỉ làm trên train. Không fit imputer/scaler hoặc chọn biến dựa vào test.

## 3. Quy trình EDA cho bài lab

1. Xác định run, tập dữ liệu, số mẫu và trạng thái raw/processed.
2. Kiểm tra schema, kiểu dữ liệu, missing values và duplicates.
3. Tính thống kê mô tả: count, mean, std, min, Q1, median, Q3, max.
4. Khảo sát từng biến số và target bằng histogram/boxplot.
5. Phân tích phân bố không gian bằng tọa độ.
6. Phân tích quan hệ tuyến tính giữa các biến số bằng Pearson.
7. Phân tích số lượng và phân bố target theo biến categorical.
8. Tổng hợp phát hiện, giới hạn và giả thuyết preprocessing cần kiểm nghiệm trên validation.

Đây là phạm vi EDA cơ bản phù hợp pipeline hiện tại, không phải mọi kỹ thuật EDA có thể có. Bộ 5 hình chưa có scatter riêng `median_income`–target, missingness heatmap, hoặc hình so sánh trước/sau log. Những hình này chỉ là hướng mở rộng nếu cần trả lời thêm câu hỏi; tài liệu không coi chúng là đầu ra đã được tạo.

## 4. Thống kê mô tả đã kiểm tra trên train

Train vẫn là dữ liệu thô trước imputation, gồm 16.512 dòng và 10 cột. Có **167 giá trị thiếu ở total_bedrooms**, các cột còn lại không thiếu; **0 dòng trùng hoàn toàn trong train**. Không dùng con số 207 missing của toàn CSV thay cho thống kê train.

Bảng dưới được tính lại từ train CSV; std là sample standard deviation của pandas `describe()`, không phải population std dùng cho target scaling trong MLP. Dấu phẩy phân tách hàng nghìn, dấu chấm chỉ phần thập phân.

| Biến | Count | Mean | Std | Min | Q1 | Median | Q3 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| longitude | 16,512 | -119.58 | 2.00 | -124.35 | -121.80 | -118.51 | -118.01 | -114.31 |
| latitude | 16,512 | 35.64 | 2.13 | 32.54 | 33.93 | 34.26 | 37.72 | 41.95 |
| housing_median_age | 16,512 | 28.60 | 12.58 | 1 | 18 | 29 | 37 | 52 |
| total_rooms | 16,512 | 2,634.87 | 2,188.82 | 2 | 1,444 | 2,127 | 3,142 | 37,937 |
| total_bedrooms | 16,345 | 537.00 | 420.57 | 1 | 296 | 433 | 646 | 6,445 |
| population | 16,512 | 1,424.60 | 1,113.65 | 3 | 787 | 1,167 | 1,724 | 28,566 |
| households | 16,512 | 498.65 | 382.77 | 1 | 280 | 408 | 604 | 6,082 |
| median_income | 16,512 | 3.88 | 1.92 | 0.50 | 2.56 | 3.54 | 4.76 | 15.00 |
| median_house_value | 16,512 | 206,954.90 | 115,524.63 | 14,999 | 119,600 | 179,700 | 264,700 | 500,001 |

`median_income` giữ nguyên thang đo trong CSV; không diễn giải các giá trị trên thành USD nguyên bản khi chưa đối chiếu quy ước nguồn. Các giá trị đã làm tròn nên không dùng bảng này thay cho dữ liệu khi tính toán.

Có thể tái kiểm tra bằng đoạn sau khi chạy Python từ thư mục gốc; đoạn này chỉ đọc dữ liệu:

```python
from pathlib import Path
import pandas as pd

run_dir = Path("runs/20261116T113320345877Z_6c8931e5")
train = pd.read_csv(run_dir / "data/processed/train.csv")
print(train.describe().T)
print(train.isna().sum())
print("Duplicates:", train.duplicated().sum())
print("Capped:", train["median_house_value"].eq(500001).sum())
print(train.groupby("ocean_proximity")["median_house_value"]
      .agg(["count", "median"]))
```

## 5. Đã nhận được những biểu đồ nào?

Có **5 file PNG**, không phải chỉ 5 ô biểu đồ: hình histogram gồm 9 ô; hình target và hình categorical mỗi hình gồm 2 ô. Các hình dưới đây đều lấy từ cùng run tham chiếu.

### 5.1. Histogram các cột số — eda_feature_distributions.png

![Histogram 9 cột số trên train](../../runs/20261116T113320345877Z_6c8931e5/results/figures/eda_feature_distributions.png)

**Mục đích và cách đọc:** mỗi ô biểu diễn một cột số, trục X là giá trị của cột, trục Y là số mẫu trong khoảng giá trị. Hàm dùng 35 bins và bỏ NaN riêng ở từng cột. Với `train_frame` truyền từ CLI, hình gồm 8 numeric features và cả target; không hiểu đây là 9 input features numeric.

- **Quan sát:** total_rooms, total_bedrooms, population và households lệch phải rõ, phần lớn mẫu ở vùng giá trị thấp và có đuôi kéo dài. Median_income cũng có đuôi phải. Longitude/latitude tập trung thành nhiều vùng; tuổi nhà có nhiều đỉnh và một cụm ở đầu trên gần 52.
- **Ý nghĩa:** thang đo giữa các features khác nhau; các biến tổng quy mô có giá trị cực lớn so với median. Đây là cơ sở xem xét scaling và thử biến đổi log cho các biến đếm.
- **Hạn chế:** histogram phụ thuộc số bins; hình không tự xác định điểm nào là dữ liệu sai. Việc bỏ NaN khi vẽ không phải đã xử lý missing. Đỉnh ở biên tuổi nhà chỉ là quan sát cần đối chiếu quy tắc thu thập.
- **Hành động đề xuất:** giữ baseline preprocessing; thử log1p theo tùy chọn hiện có và đối chiếu validation. Không tự xóa đuôi phân bố, không kỳ vọng StandardScaler làm phân phối trở thành Gaussian.

### 5.2. Scatter địa lý — eda_geographical_scatter.png

![Phân bố vị trí, giá và dân số trên train](../../runs/20261116T113320345877Z_6c8931e5/results/figures/eda_geographical_scatter.png)

**Mục đích và cách đọc:** X là longitude, Y là latitude; mỗi điểm là một khu dân cư. Màu theo colorbar biểu diễn giá nhà USD, xanh thấp hơn và đỏ cao hơn. Tham số diện tích điểm là `population / 100`, được clip trong [10, 300], với alpha 0.4.

- **Quan sát:** các điểm không phủ đều không gian. Có cụm màu đỏ ở dải phía tây quanh các vùng tọa độ (-122, 37–38) và (-118, 33–34); nhiều vùng nội địa có màu xanh hơn.
- **Ý nghĩa:** quan hệ giữa vị trí và giá có cấu trúc không gian; hai tọa độ có thể hữu ích khi được xét cùng nhau trong mô hình phi tuyến.
- **Hạn chế:** đây là scatter tọa độ, không có bản đồ nền hay ranh giới hành chính. Điểm chồng lấn và kích thước bị clip nên không thể đọc chính xác dân số từ diện tích. Hình không chứng minh vị trí gây ra giá cao hay mô hình sẽ dự đoán tốt vùng chưa thấy.
- **Hành động đề xuất:** giữ features địa lý trong pipeline hiện tại; ghi spatial holdout là hướng đánh giá sau này. Không loại tọa độ chỉ vì tương quan Pearson riêng lẻ với target thấp.

### 5.3. Heatmap tương quan Pearson — eda_correlation_heatmap.png

![Ma trận Pearson trên các cột số của train](../../runs/20261116T113320345877Z_6c8931e5/results/figures/eda_correlation_heatmap.png)

**Mục đích và cách đọc:** hàng/cột là biến số; mỗi ô là Pearson r, từ -1 đến 1. Đỏ thể hiện tương quan dương, xanh là âm, gần trắng là gần 0; đường chéo bằng 1. Số trên hình làm tròn hai chữ số thập phân. Categorical chưa được đưa vào ma trận.

- **Quan sát:** median_income–target khoảng **0.69**, lớn nhất về trị tuyệt đối trong các input numeric so với target trên hình. Total_bedrooms–households khoảng **0.98**, total_rooms–total_bedrooms khoảng **0.93**; longitude–latitude khoảng **-0.92**.
- **Ý nghĩa:** thu nhập có liên hệ tuyến tính đáng chú ý với target. Các biến tổng quy mô chia sẻ nhiều thông tin; hệ số của Linear Regression có thể khó diễn giải riêng khi features tương quan mạnh.
- **Hạn chế:** Pearson không đo đầy đủ quan hệ phi tuyến hoặc tương tác, không chứng minh nhân quả và không tương đương feature importance. Với missing values, pandas tính tương quan trên các cặp quan sát có đủ giá trị; số mẫu giữa các ô có thể khác nhau.
- **Hành động đề xuất:** dùng kết quả để đặt giả thuyết về ratios như rooms_per_household, rồi kiểm nghiệm trên validation. Không tự bỏ feature hoặc khẳng định ratios tốt hơn chỉ từ heatmap.

### 5.4. Histogram và boxplot target — eda_target_distribution.png

![Histogram và boxplot giá nhà trên train](../../runs/20261116T113320345877Z_6c8931e5/results/figures/eda_target_distribution.png)

**Mục đích và cách đọc:** bên trái là histogram 45 bins, X là USD, Y là số mẫu; đường đỏ đứt ở 500.001 USD. Bên phải là boxplot: hộp từ Q1 đến Q3, đường đỏ là median; whiskers theo quy tắc mặc định 1.5 IQR, các điểm ngoài whiskers được vẽ riêng. Whiskers không mặc định là min/max.

- **Quan sát:** phần lớn giá nằm thấp hơn vùng trần, phân bố có đuôi phải và cột tăng mạnh sát trần. Median là **179.700 USD**, mean **206.954,90 USD**; 50% mẫu giữa **119.600 và 264.700 USD**. Train có **787/16.512 mẫu** đúng 500.001 USD.
- **Ý nghĩa:** mean lớn hơn median phù hợp với quan sát lệch phải. Capping làm nhãn ở đầu trên không phản ánh mức giá thực tế vượt trần; cần diễn giải sai số ở nhóm này riêng.
- **Hạn chế:** chiều cao bin cuối bao gồm cả khoảng giá, không bằng chính xác số mẫu tại trần. Điểm ngoài whiskers không mặc nhiên là dữ liệu lỗi. Nhãn capped không cho biết giá thật phía trên trần.
- **Hành động đề xuất:** giữ nhãn và mẫu hiện có, báo cáo giới hạn capping; phân tích capped/uncapped ở bước đánh giá. Target scaling của MLP chỉ thay thang đo, không khắc phục capping.

### 5.5. Bar chart và boxplot theo ocean_proximity — eda_categorical_distribution.png

![Số mẫu và phân bố giá theo ocean_proximity trên train](../../runs/20261116T113320345877Z_6c8931e5/results/figures/eda_categorical_distribution.png)

**Mục đích và cách đọc:** bên trái là số mẫu từng nhóm, có nhãn số trên cột; bên phải là boxplot giá USD cùng nhóm. Đọc đồng thời cỡ mẫu và phân bố giá, không chỉ so sánh độ cao median.

| Nhóm | Số mẫu train | Median target (USD) |
| --- | ---: | ---: |
| &lt;1H OCEAN | 7.324 | 215.500 |
| INLAND | 5.226 | 108.300 |
| NEAR OCEAN | 2.127 | 229.100 |
| NEAR BAY | 1.830 | 232.800 |
| ISLAND | 5 | 414.700 |

- **Quan sát:** hai nhóm đầu chiếm phần lớn mẫu; ISLAND chỉ 5 mẫu. Median INLAND thấp hơn các nhóm còn lại; các nhóm có vùng giá chồng lấn đáng kể.
- **Ý nghĩa:** ocean_proximity cung cấp thông tin phân nhóm có thể hữu ích; one-hot encoding phù hợp với cách biểu diễn categorical đang dùng.
- **Hạn chế:** median cao của ISLAND chỉ mô tả 5 mẫu trong train, không đủ kết luận rộng về nhóm đảo. Chênh lệch giữa các nhóm chưa kiểm soát thu nhập, vị trí hoặc các biến khác nên không phải hiệu ứng nhân quả.
- **Hành động đề xuất:** giữ encoding có xử lý category chưa thấy; khi báo cáo sai số theo nhóm phải kèm số mẫu. Không tự cân bằng lại dữ liệu hoặc kết luận chất lượng model từ boxplot target.

## 6. Từ phát hiện EDA đến preprocessing

| Phát hiện có bằng chứng | Cách xử lý/giả thuyết liên quan | Điều còn phải kiểm chứng |
| --- | --- | --- |
| total_bedrooms thiếu 167 giá trị train | Median imputation chỉ fit train | Tính nhất quán transform và không leakage |
| Thang đo numeric khác nhau | StandardScaler trong pipeline hiện tại | Tác động chất lượng không suy ra chỉ từ EDA |
| Các biến quy mô lệch phải, tương quan mạnh | Thử log1p và ratios bằng tùy chọn hiện có | So sánh trên validation, giữ nguyên test |
| Categorical khác nhau và ISLAND rất hiếm | One-hot, unknown-ignore | Sai số theo nhóm cần cỡ mẫu đủ |
| Target có trần 500.001 USD | Giữ nhãn, nêu giới hạn và phân tích nhóm capped | Không biết giá thật vượt trần |
| Giá có cấu trúc không gian | Giữ tọa độ; đề xuất spatial holdout tương lai | Random split hiện tại chưa đo generalization theo vùng mới |

Xem [chương 04](04_preprocessing_and_features.md) để theo dõi cách biến đổi cụ thể. Các đề xuất thử nghiệm không đồng nghĩa thay đổi cấu hình mặc định trong lần cập nhật tài liệu này.

## 7. Thứ tự học tập và thứ tự thực thi

Trong báo cáo, EDA được đặt trước preprocessing/modeling để giải thích các lựa chọn. Trong [main.py](../../main.py), dữ liệu được split trước, rồi fit preprocessing và training; 5 EDA PNG được xuất ở đoạn sau đánh giá model nhưng vẫn chỉ dùng train_frame. Vì vậy không mô tả CLI như đã thực thi vẽ EDA trước training.

Notebook 01 trình bày EDA trong luồng học tập và giữ hình trong output notebook. Các hình nhúng trong chương này lấy từ CLI run đã chỉ định, không lấy từ notebook. Khi chạy với `--skip-plots`, CLI không tạo 5 PNG; không lấy hình run khác rồi gắn vào run đó.

## 8. Phân tích kết quả đạt được theo 5W1H

| Thành phần | Phân tích kết quả giai đoạn |
| --- | --- |
| **What — Đạt được gì?** | Đã kiểm đếm splits 16.512/2.064/2.064, mô tả train và xem 5 hình; nhận diện missing, lệch phải, tương quan, cấu trúc địa lý, capping và nhóm hiếm. |
| **Why — Có ý nghĩa gì?** | Cung cấp căn cứ giải thích preprocessing hiện tại và đặt giả thuyết log/ratios; chưa chứng minh cải thiện chất lượng dự đoán. |
| **Who — Ai sử dụng?** | Người chuẩn bị dữ liệu bàn giao cho người huấn luyện; người đọc báo cáo dùng hình và số liệu để kiểm tra lập luận. |
| **When — Khi nào?** | Artifacts thuộc run ngày 16/11/2026; đọc hình và tính lại thống kê CSV ngày 24/11/2026. EDA dùng train sau split; thời điểm xuất hình CLI nêu ở mục 7. |
| **Where — Bằng chứng ở đâu?** | Train CSV, manifest và 5 PNG thuộc run 20261116T113320345877Z_6c8931e5, được liên kết trong chương. |
| **How — Xác minh thế nào?** | Xem trực tiếp hình, đối chiếu hàm vẽ và tính lại describe/missing/duplicates/capped/group counts/medians từ train CSV; không huấn luyện lại hoặc thay artifacts. |

### Kết luận giai đoạn và bàn giao

Đã có mô tả EDA cơ bản theo phạm vi bài lab, bao gồm biểu đồ thực tế và ý nghĩa từng hình. Đầu ra bàn giao là raw splits, thống kê train và danh sách vấn đề cần xử lý. Không bàn giao quyết định xóa outlier hoặc chọn feature dựa riêng vào tương quan.

Giới hạn: chỉ một split tham chiếu; các thống kê là mô tả mẫu hiện có, không chứng minh nhân quả hoặc generalization. Lần cập nhật này chưa chạy lại kiểm thử độc lập index, chưa tái xác minh toàn bộ hashes manifest, chưa chạy lại training. Phần kết quả không được diễn giải thành chứng nhận toàn pipeline.

## 9. Checklist nghiệm thu EDA

- [x] Nêu run, nguồn train và kích thước ba tập.
- [x] Có thống kê mô tả, missing và duplicates trên train.
- [x] Mô tả đúng 5 file hình, trục, màu, kích thước điểm và thành phần boxplot.
- [x] Mỗi hình có quan sát, ý nghĩa, hạn chế và hành động đề xuất.
- [x] Có nhận xét định lượng về capping, nhóm hiếm và tương quan làm tròn trên hình.
- [x] Liên hệ phát hiện với preprocessing và ghi rõ giả thuyết cần validation.
- [x] Phân biệt CLI/notebook, EDA/model evaluation và train/toàn CSV.
- [ ] Nếu cần chứng nhận lại toàn run: kiểm tra hashes và chạy các kiểm thử split liên quan; ngoài phạm vi cập nhật tài liệu này.

## Nguồn nội bộ

- [Chương 02 — Data understanding](02_data_understanding.md)
- [Chương 04 — Preprocessing](04_preprocessing_and_features.md)
- [Code split/preprocessing](../../src/data/preprocessing.py)
- [Code visualization](../../src/evaluation/visualization.py)
- [CLI orchestration](../../main.py)
- [Notebook 01](../../notebooks/01_eda_and_preprocessing.ipynb)
- [Tests dữ liệu](../../tests/test_data.py)

[Quay lại mục lục](README.md)
