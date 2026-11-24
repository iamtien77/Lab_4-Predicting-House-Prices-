# 00 — Dữ liệu đầu vào

## 1. Mục tiêu và phạm vi

Chương này giúp người học xác định đúng dữ liệu đầu vào, hiểu đơn vị quan sát và cách đọc từng cột trước khi thực hiện regression. Đầu ra là data dictionary và thông tin nhận diện CSV; chưa chia tập, điền khuyết hoặc huấn luyện.

### Nguồn gốc học thuật và lịch sử thu thập dữ liệu (Dataset Provenance)

Tập dữ liệu California Housing có xuất xứ học thuật và lịch sử thu thập có thể truy vết rõ ràng qua chuỗi tài liệu sau:

- **Nguồn dữ liệu gốc (Primary Source):** Dữ liệu được trích xuất từ cuộc **Tổng điều tra Dân số và Nhà ở Hoa Kỳ năm 1990 (1990 U.S. Census)** do Cục Thống kê Dân số Hoa Kỳ (*U.S. Census Bureau*) thực hiện trên toàn bang California.
- **Nghiên cứu công bố gốc (Original Publication):** Tập dữ liệu được hai nhà nghiên cứu **R. Kelley Pace** và **Ronald Barry** thu thập, xử lý và công bố chính thức trong bài báo khoa học *"Sparse Spatial Autoregressions"*, đăng trên tạp chí *Statistics & Probability Letters* (Volume 33, Issue 3, ngày 5 tháng 5 năm 1997, trang 291–297; DOI: `10.1016/S0167-7152(96)00140-X`). Mục đích ban đầu của bài báo là xây dựng và đánh giá các thuật toán hồi quy tự tương quan không gian thưa trên tập dữ liệu địa lý lớn.
- **Đơn vị quan sát (Observation Unit):** Mỗi dòng trong bảng đại diện cho một **nhóm khối điều tra dân số (Census Block Group)** tại bang California năm 1990. Đây là đơn vị địa lý nhỏ nhất mà Cục Thống kê Dân số Hoa Kỳ công bố công khai dữ liệu mẫu (mỗi block group thông thường có từ 600 đến 3.000 dân, trung bình khoảng 1.425 người trong tập dữ liệu này). Toàn bộ bang California thời điểm 1990 có đúng 20.640 block groups, tương ứng với **20.640 dòng** trong tệp CSV.
- **Đóng góp của Aurélien Géron:** Trong giáo trình máy học kinh điển *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"* (O'Reilly Media, 2017/2019), tác giả Aurélien Géron đã trích xuất lại bộ dữ liệu từ kho lưu trữ StatLib, bổ sung thêm đặc trưng phân loại vị trí địa lý `ocean_proximity` (tính toán dựa trên tọa độ kinh độ/vĩ độ tới bờ biển) để tạo bài toán thực hành phong phú hơn.
- **Kênh phân phối Kaggle:** Bộ dữ liệu sau đó được chia sẻ trên [California Housing Prices — Kaggle / camnugent](https://www.kaggle.com/datasets/camnugent/california-housing-prices) phục vụ cộng đồng học tập.
- **Bản dữ liệu cục bộ thực nghiệm:** Tệp [data/raw/housing.csv](../../data/raw/housing.csv) gồm 20.640 dòng × 10 cột, mã băm SHA-256: `b214a65099c1dcc2ea7797935aff103fa0445220a366fb74beacb59486e548ff`.

## 2. Nhận diện file và cách đọc

Kiểm đếm ngày **24/11/2026**: **20.640 dòng × 10 cột**, gồm 9 features và target. SHA-256:

```text
b214a65099c1dcc2ea7797935aff103fa0445220a366fb74beacb59486e548ff
```

Hash nhận diện đúng bản dữ liệu; giữ nguyên CSV gốc để có thể đối chiếu.

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

Mỗi dòng mô tả một nhóm khối dân cư (Census Block Group). Các biến tổng số phòng, người và hộ là dữ liệu tổng hợp khu vực, không phải thuộc tính của một căn nhà cụ thể.

| Cột | Vai trò | Kiểu quan sát | Ý nghĩa, đơn vị và lưu ý |
| --- | --- | --- | --- |
| `longitude` | Numeric feature | float64 | Kinh độ địa lý (kinh độ tâm cụm dân cư, độ) |
| `latitude` | Numeric feature | float64 | Vĩ độ địa lý (vĩ độ tâm cụm dân cư, độ) |
| `housing_median_age` | Numeric feature | float64 | Tuổi nhà trung vị của các ngôi nhà trong khu vực (năm) |
| `total_rooms` | Numeric feature | float64 | Tổng số phòng của tất cả các ngôi nhà trong khu vực (phòng) |
| `total_bedrooms` | Numeric feature | float64 | Tổng số phòng ngủ trong khu vực (phòng; có 207 giá trị thiếu) |
| `population` | Numeric feature | float64 | Tổng số dân cư sinh sống trong khu vực (người) |
| `households` | Numeric feature | float64 | Tổng số hộ gia đình sinh sống trong khu vực (hộ) |
| `median_income` | Numeric feature | float64 | Thu nhập trung vị của các hộ gia đình trong nhóm khối, đo theo đơn vị chục nghìn USD ($10.000 USD) theo thời giá năm 1990 (ví dụ giá trị 8,3252 tương ứng với khoảng $83.252 USD) |
| `median_house_value` | Target | float64 | Giá trị nhà trung vị trong nhóm khối, đơn vị USD. Biểu mẫu khảo sát năm 1990 áp trần (capping) ở mức $500.001 USD (toàn bộ giá trị từ $500.001 trở lên đều được ghi nhận ở mốc này) |
| `ocean_proximity` | Categorical feature | str | Nhãn vị trí tương đối với bờ biển California do Aurélien Géron bổ sung dựa trên tọa độ không gian, gồm 5 nhóm: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND` |

Dtype chuỗi có thể là `str`, `string` hoặc `object` tùy pandas. Contract yêu cầu giá trị phân loại là chuỗi hoặc missing, không yêu cầu một tên dtype duy nhất. Biến đếm lưu dưới dạng float không có nghĩa mọi giá trị thập phân đều hợp lý về domain.

Năm nhãn quan sát là `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`. Không tự mã hóa 1–5 rồi coi khoảng cách giữa các mã có ý nghĩa địa lý.

## 4. Diễn giải một dòng thật

Dòng đầu CSV được đọc trực tiếp ngày 24/11/2026:

| Nhóm | Giá trị |
| --- | --- |
| Tọa độ | longitude = −122,23; latitude = 37,88 |
| Tuổi trung vị | 41 năm |
| Quy mô | 880 phòng; 129 phòng ngủ; 322 người; 126 hộ |
| Thu nhập | `median_income` = 8,3252 (~$83.252 USD theo thời giá 1990) |
| Vị trí | `NEAR BAY` |
| Target quan sát | 452.600 USD |

Đây là khu vực có 126 hộ, không phải một nhà có 880 phòng. Nếu bật ratios, `rooms_per_household = 880 / 126 ≈ 6,9841`; đây là tỷ số tổng hợp, không phải số phòng đo trực tiếp cho từng hộ. Target 452.600 USD là nhãn đã có, **không phải prediction**. Ví dụ này không cung cấp bằng chứng về độ chính xác của model.

## 5. Giới hạn và nội dung bàn giao

Dữ liệu phản ánh tình hình kinh tế - xã hội và thị trường nhà ở California tại thời điểm khảo sát năm 1990, do đó không suy diễn trực tiếp cho giá thị trường bất động sản hiện tại. Missing `total_bedrooms` (207 dòng) dẫn tới việc phải áp dụng imputation chỉ trên tập train. Có 965 dòng target chạm trần tại 500.001 USD và nhóm `ISLAND` chỉ có 5 dòng, cần đặc biệt lưu ý khi phân tích phần dư và đánh giá nhóm nhãn hiếm. Audit chi tiết ở [chương 02](02_data_understanding.md); phân tích phân bố và EDA tập train ở [chương 03](03_data_splitting_and_eda.md).

## 6. Phân tích kết quả đạt được theo 5W1H

| Thành phần | Kết quả có bằng chứng |
| --- | --- |
| **What** | Dữ liệu gồm 20.640 dòng × 10 cột (9 features và 1 target); mô tả đặc trưng nhân khẩu, kinh tế và vị trí địa lý của các cụm dân cư. |
| **Why** | Target liên tục phù hợp cho bài toán hồi quy (regression); cung cấp đầy đủ cả biến số và biến phân loại để thực hành pipeline chuẩn hóa. |
| **Who** | Thu thập bởi Cục Thống kê Dân số Hoa Kỳ (U.S. Census Bureau); công bố bởi Pace & Barry (1997); bổ sung đặc trưng bởi Aurélien Géron. |
| **When** | Thu thập trong cuộc Tổng điều tra Dân số Hoa Kỳ năm 1990 (1990 U.S. Census); công bố học thuật năm 1997; kiểm tra tệp cục bộ ngày 24/11/2026. |
| **Where** | Bang California, Hoa Kỳ (theo cấp độ Census Block Group); lưu trữ cục bộ tại `data/raw/housing.csv`. |
| **How** | Đọc bằng pandas thông qua `load_housing_data()`, xác thực schema, kiểu dữ liệu, kiểm tra mã băm SHA-256; không tải dữ liệu qua mạng khi thực thi. |

**Kết luận:** Đạt mục tiêu nhận diện và giải thích dữ liệu cục bộ với nguồn gốc học thuật và lịch sử thu thập rõ ràng, có căn cứ. Bàn giao DataFrame thô và tên target cho audit ở chương tiếp theo.

### Checklist nghiệm thu

- [x] Nguồn gốc xuất xứ (provenance), thời gian thu thập (1990) và nghiên cứu gốc (Pace & Barry, 1997) được xác định rõ ràng.
- [x] File, kích thước (20.640 × 10), hash SHA-256 và phạm vi dữ liệu minh bạch.
- [x] Phân biệt rõ đơn vị quan sát là nhóm khối dân cư (Census Block Group), không phải từng căn nhà riêng lẻ.
- [x] Giải thích chính xác hệ số đo lường thu nhập ($10.000 USD) và cơ chế trần khảo sát (Capping tại $500.001 USD).

[Tiếp: Bài toán và metrics](01_problem_and_metrics.md) · [Mục lục](README.md)
