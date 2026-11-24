# 00 — Dữ liệu đầu vào

## 1. Mục tiêu và phạm vi

Chương này giúp người học xác định đúng dữ liệu đầu vào, hiểu đơn vị quan sát và cách đọc từng cột trước khi thực hiện regression. Đầu ra là data dictionary và thông tin nhận diện CSV; chưa chia tập, điền khuyết hoặc huấn luyện.

Nguồn do người dùng cung cấp: [California Housing Prices — Kaggle / camnugent](https://www.kaggle.com/datasets/camnugent/california-housing-prices). Đây là trang phân phối được dẫn nguồn. Trang không trả nội dung mô tả đọc được qua công cụ trong phiên; chưa xác minh license, năm khảo sát, nguồn thu thập gốc hoặc sự trùng khớp phiên bản với bản cục bộ. Các thống kê dưới đây lấy từ [housing.csv](../../data/raw/housing.csv), không gán cho mọi phiên bản trên Kaggle.

## 2. Nhận diện file và cách đọc

Kiểm đếm ngày **24/11/2026**: **20.640 dòng × 10 cột**, gồm 9 features và target. SHA-256:

```text
b214a65099c1dcc2ea7797935aff103fa0445220a366fb74beacb59486e548ff
```

Hash nhận diện đúng bản dữ liệu; không chứng minh tính chính xác hoặc nguồn gốc. Giữ nguyên CSV gốc để có thể đối chiếu.

Chạy từ thư mục gốc dự án:

```python
from src.data.loader import load_housing_data

df, target = load_housing_data()
X = df.drop(columns=target)
y = df[target]
print(df.shape)
print(df.head(1))
```

[Loader](../../src/data/loader.py) dùng pandas đọc CSV cục bộ, xác thực schema và trả DataFrame cùng tên target. Không tự tải Kaggle hoặc tự điền missing. X vẫn là features thô, chưa phải ma trận sẵn sàng cho MLP.

## 3. Đơn vị quan sát và data dictionary

Theo [README](../../README.md), mỗi dòng mô tả một khu dân cư. Các biến tổng số phòng, người và hộ là dữ liệu tổng hợp khu vực, không phải thuộc tính của một căn nhà cụ thể.

| Cột | Vai trò | Kiểu quan sát | Ý nghĩa, đơn vị và lưu ý |
| --- | --- | --- | --- |
| `longitude` | Numeric feature | float64 | Kinh độ, độ; tọa độ khu vực |
| `latitude` | Numeric feature | float64 | Vĩ độ, độ; dùng cùng kinh độ |
| `housing_median_age` | Numeric feature | float64 | Tuổi nhà trung vị, năm; không phải tuổi từng căn |
| `total_rooms` | Numeric feature | float64 | Tổng số phòng của khu vực, phòng |
| `total_bedrooms` | Numeric feature | float64 | Tổng số phòng ngủ, phòng; có 207 giá trị thiếu |
| `population` | Numeric feature | float64 | Dân số khu vực, người |
| `households` | Numeric feature | float64 | Số hộ, hộ; có thể làm mẫu số ratios |
| `median_income` | Numeric feature | float64 | Thu nhập trung vị theo thang dataset; chưa xác minh hệ số quy đổi, không gọi là USD nguyên bản |
| `median_house_value` | Target | float64 | Giá trị nhà trung vị, USD theo quy ước dự án |
| `ocean_proximity` | Categorical feature | str | Nhãn vị trí tương đối với biển/vịnh, không có thứ tự số tự nhiên |

Dtype chuỗi có thể là `str`, `string` hoặc `object` tùy pandas. Contract yêu cầu giá trị phân loại là chuỗi hoặc missing, không yêu cầu một tên dtype duy nhất. Biến đếm lưu dưới dạng float không có nghĩa mọi giá trị thập phân đều hợp lý về domain.

Năm nhãn quan sát là `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`. Không tự mã hóa 1–5 rồi coi khoảng cách giữa các mã có ý nghĩa địa lý.

## 4. Diễn giải một dòng thật

Dòng đầu CSV được đọc trực tiếp ngày 24/11/2026:

| Nhóm | Giá trị |
| --- | --- |
| Tọa độ | longitude = −122,23; latitude = 37,88 |
| Tuổi trung vị | 41 năm |
| Quy mô | 880 phòng; 129 phòng ngủ; 322 người; 126 hộ |
| Thu nhập | `median_income` = 8,3252 theo thang dataset |
| Vị trí | `NEAR BAY` |
| Target quan sát | 452.600 USD |

Đây là khu vực có 126 hộ, không phải một nhà có 880 phòng. Nếu bật ratios, `rooms_per_household = 880 / 126 ≈ 6,9841`; đây là tỷ số tổng hợp, không phải số phòng đo trực tiếp cho từng hộ. Target 452.600 USD là nhãn đã có, **không phải prediction**. Ví dụ này không cung cấp bằng chứng về độ chính xác của model.

## 5. Giới hạn và nội dung bàn giao

Missing `total_bedrooms` dẫn tới imputation train-only. Có 965 dòng target bằng 500.001 USD và nhóm ISLAND chỉ 5 dòng, cần thận trọng khi phân tích sai số và nhóm hiếm. Audit đầy đủ ở [chương 02](02_data_understanding.md); hình và phân tích EDA train ở [chương 03](03_data_splitting_and_eda.md).

Không suy ra giá thị trường hiện tại, năm thu thập hoặc tính đại diện ngoài vùng nghiên cứu từ tên dataset. Phù hợp bài lab không đồng nghĩa đủ điều kiện định giá giao dịch thực tế. Metadata Kaggle và quy đổi thu nhập còn là phần chưa xác minh.

## 6. Phân tích kết quả đạt được theo 5W1H

| Thành phần | Kết quả có bằng chứng |
| --- | --- |
| **What** | Xác nhận CSV 20.640 × 10, 9 features, target và dictionary đủ 10 cột. |
| **Why** | Target liên tục phù hợp regression; features cung cấp đầu vào dự đoán nhưng chưa chứng minh nhân quả. |
| **Who** | Người học đọc đúng đơn vị; người kiểm tra dùng schema/hash nhận diện dữ liệu. |
| **When** | Kiểm tra CSV ngày 24/11/2026; thời điểm khảo sát gốc chưa xác minh. |
| **Where** | CSV cục bộ, loader, README; Kaggle là địa chỉ do người dùng cung cấp. |
| **How** | Đọc shape/dtypes, dòng đầu và tính hash; chưa so sánh byte với bản Kaggle tải mới. |

**Kết luận:** đạt mục tiêu nhận diện và giải thích dữ liệu cục bộ; provenance bên ngoài mới đạt một phần. Bàn giao DataFrame thô và tên target cho audit, không bàn giao ma trận chuẩn hóa hoặc model.

### Checklist nghiệm thu

- [x] File, kích thước, hash và phạm vi số liệu rõ ràng.
- [x] Phân biệt feature/target và khu dân cư/căn nhà.
- [x] Ví dụ từ dữ liệu thật, không gọi nhãn là prediction.
- [x] Ghi rõ metadata và thang thu nhập chưa xác minh.

[Tiếp: Bài toán và metrics](01_problem_and_metrics.md) · [Mục lục](README.md)
