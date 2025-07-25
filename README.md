# Left_Udjat: Phát Hiện Máy Bay Phản Lực Bằng Phương Pháp Học Chuyển Giao (Modern Jet Aircraft Detection Using Transfer Learning Methods)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLOv11](https://img.shields.io/badge/YOLOv11-latest-green.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![skills_f22_detection_gif](runs/detect/f22_detection_result4/skills_f22_edited_2.gif)

## 📖 Mục Lục
- [Giới Thiệu](#giới-thiệu)
- [Tính Năng Chính](#tính-năng-chính)
- [Cài Đặt](#cài-đặt)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Dữ Liệu](#dữ-liệu)
- [Kiến Trúc Mô Hình & Huấn Luyện](#kiến-trúc-mô-hình--huấn-luyện)
- [Sử Dụng](#sử-dụng)
- [Kết Quả](#kết-quả)
- [Mô Hình Đã Huấn Luyện Trước](#mô-hình-đã-huấn-luyện-trước)
- [Giấy Phép](#giấy-phép)
- [Lời Cảm Ơn](#lời-cảm-ơn)
- [Liên Hệ](#liên-hệ)

## 🚀 Giới Thiệu
Dự án **Left_Udjat** tập trung vào việc phát hiện đối tượng máy bay phản lực trong hình ảnh và video sử dụng mô hình YOLOv11 tiên tiến. Mục tiêu là xây dựng một hệ thống phát hiện mạnh mẽ và hiệu quả, có khả năng nhận diện các loại máy bay chiến đấu cụ thể như F22, F35, và J20.

## ✨ Tính Năng Chính
-   **Huấn luyện YOLOv11:** Hỗ trợ huấn luyện mô hình YOLOv11 trên bộ dữ liệu tùy chỉnh.
-   **Phát hiện đối tượng:** Thực hiện nhận diện máy bay phản lực trên hình ảnh và video.
-   **Xử lý video:** Cắt, điều chỉnh FPS và tốc độ phát lại video.
-   **Chuyển đổi Video sang GIF:** Dễ dàng chuyển đổi video kết quả nhận diện sang định dạng GIF để chia sẻ và trình bày.

## 🛠️ Cài Đặt

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/your-username/fighter_object_detection.git
    cd fighter_object_detection
    ```
    (Thay `https://github.com/your-username/fighter_object_detection.git` bằng URL repo của bạn)

2.  **Tạo và Kích hoạt Môi trường Ảo (Khuyến nghị):**
    ```bash
    python -m venv venv
    # Trên Windows
    .\venv\Scripts\activate
    # Trên macOS/Linux
    source venv/bin/activate
    ```

3.  **Cài đặt Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Lưu ý: Đảm bảo bạn có cài đặt PyTorch với hỗ trợ CUDA nếu muốn huấn luyện/nhận diện trên GPU.*

## 📂 Cấu Trúc Dự Án
```
fighter_object_detection/
├── data/
│   ├── interim/             # Dữ liệu tạm thời sau khi xử lý ban đầu
│   ├── processed/           # Dữ liệu đã được xử lý, sẵn sàng cho huấn luyện (YOLO format)
│   ├── processed_cache/     # Cache của dữ liệu đã xử lý
│   └── raw/                 # Dữ liệu gốc tải về
│       ├── 333.v1i.coco/
│       ├── Cats.v1i.coco/
│       ├── fight_10000_images/
│       └── yolov8veesles.v1i.coco/
├── runs/                    # Thư mục lưu trữ kết quả huấn luyện, phát hiện
│   └── detect/
│       ├── finetuned_test_report/
│       ├── yolov11_fighter_jets_kaggle_<timestamp>/ # Kết quả huấn luyện
│       └── f22_detection_result/                     # Kết quả nhận diện video
├── src/
│   ├── data/                # Scripts xử lý và chuẩn bị dữ liệu
│   │   ├── convert_coco_to_yolo.py
│   │   ├── deduplicate_dataset.py
│   │   ├── merge_coco.py
│   │   ├── split_dataset.py
│   │   └── video_editor.py  # Script chỉnh sửa video
│   ├── model/               # Scripts liên quan đến mô hình và huấn luyện
│   │   ├── detect.py        # Script nhận diện trên ảnh/video
│   │   ├── train_yolov11.py # Script huấn luyện mô hình
│   │   └── model.py
│   └── visualization/       # Scripts hỗ trợ trực quan hóa
│       └── convert_video_to_gif.py # Script chuyển đổi video sang GIF
├── fighter_jets_kaggle.yaml # Cấu hình dataset cho Kaggle
├── fighter_jets.yaml        # Cấu hình dataset chung
├── requirements.txt         # Danh sách các thư viện Python
└── README.md                # File README này
```

## 📊 Dữ Liệu
Dự án sử dụng kết hợp nhiều bộ dữ liệu máy bay phản lực, chủ yếu từ Roboflow Universe. Dữ liệu được hợp nhất và chuyển đổi sang định dạng YOLO để huấn luyện.

Các bộ dữ liệu nguồn bao gồm:
-   **`333.v1i.coco`**: Cung cấp bởi người dùng Roboflow, được cấp phép theo **CC BY 4.0**. Chứa 1150 hình ảnh huấn luyện với các lớp đa dạng như: `A10`, `A400M`, `AG600`, `AV8B`, `B1`, `B2`, `B52`, `Be200`, `C130`, `C17`, `C2`, `C5`, `E2`, `E7`, `EF2000`, `F117`.
    -   [Link Roboflow](https://universe.roboflow.com/aircraft-awe6z/333-wsq53)
-   **`fight_10000_images`**: Cung cấp bởi người dùng Roboflow, được cấp phép theo **MIT License**. Chứa khoảng 7070 hình ảnh huấn luyện với các máy bay chiến đấu.
    -   [Link Roboflow](https://universe.roboflow.com/missile-thingie/fighter-jet-detection)
-   Ngoài ra, các bộ dữ liệu khác như `Cats.v1i.coco` và `yolov8veesles.v1i.coco` cũng được sử dụng trong quá trình tiền xử lý dữ liệu.

Dữ liệu được xử lý cuối cùng cho huấn luyện được cấu hình trong `fighter_jets_kaggle.yaml` với các lớp chính:
-   `F22`
-   `F35`
-   `J20`

## 🧠 Kiến Trúc Mô Hình & Huấn Luyện
Dự án sử dụng **YOLOv11** làm kiến trúc mô hình chính cho tác vụ phát hiện đối tượng. Quá trình huấn luyện được thực hiện bằng cách fine-tuning mô hình `yolov11l` (YOLOv11 Large) trên bộ dữ liệu hợp nhất.

**Các tham số huấn luyện chính (trong `src/model/train_yolov11.py`):**
-   `epochs`: 200
-   `imgsz`: 1024
-   `batch`: 6
-   `device`: 0 (GPU đầu tiên)
-   `optimizer`: Adam
-   `patience`: 20 (Early stopping)
-   `Data Augmentation`: Hỗ trợ nhiều kỹ thuật tăng cường dữ liệu như `hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`, `copy_paste`.

## 🚀 Sử Dụng

### 1. Huấn Luyện Mô Hình
Để huấn luyện mô hình YOLOv11 trên bộ dữ liệu đã chuẩn bị:
```bash
python src/model/train_yolov11.py
```
Kết quả huấn luyện sẽ được lưu trong thư mục `runs/detect/yolov11_fighter_jets_kaggle_<timestamp>`.

### 2. Phát Hiện Đối Tượng Trên Video
Để chạy nhận diện trên một video cụ thể và lưu video kết quả với bounding box:
```bash
python src/model/detect.py
```
*Lưu ý: Các đường dẫn mô hình và video đầu vào đã được hardcode trong `src/model/detect.py`. Đảm bảo rằng bạn đã tải `best.pt` từ quá trình huấn luyện của mình vào đường dẫn được chỉ định.*

Video đầu ra sẽ được lưu vào thư mục `runs/detect/f22_detection_result/`.

### 3. Cắt & Chỉnh Sửa Video
Để cắt video, điều chỉnh FPS hoặc tốc độ phát lại:
```bash
python src/data/video_editor.py
```
*Lưu ý: Các tham số video đầu vào/đầu ra, thời gian cắt, FPS và tốc độ đã được hardcode trong `src/data/video_editor.py`. Bạn có thể thay đổi chúng trực tiếp trong file.*

### 4. Chuyển Đổi Video sang GIF
Để tạo file GIF từ video đã xử lý:
```bash
python src/visualization/convert_video_to_gif.py
```
*Lưu ý: Các đường dẫn video đầu vào/GIF đầu ra, FPS và kích thước GIF đã được hardcode trong `src/visualization/convert_video_to_gif.py`. Điều chỉnh các tham số này trong file để đạt được chất lượng GIF mong muốn.*

## 📈 Kết Quả
Các kết quả huấn luyện, đánh giá và nhận diện được lưu trữ trong thư mục `runs/detect/`.

-   **Báo cáo huấn luyện:** Trong `runs/detect/yolov11_fighter_jets_kaggle_<timestamp>/` bạn sẽ tìm thấy:
    -   Các biểu đồ hiệu suất (`BoxF1_curve.png`, `BoxP_curve.png`, `BoxPR_curve.png`, `BoxR_curve.png`)
    -   Ma trận nhầm lẫn (`confusion_matrix.png`, `confusion_matrix_normalized.png`)
    -   File `results.csv` chứa các chỉ số huấn luyện.
    -   Các checkpoint mô hình (`weights/best.pt`, `weights/last.pt`).

    **Ví dụ về biểu đồ kết quả:**
    ![BoxF1_curve](runs/detect/runs/yolov11_fighter_jets_kaggle_1753230213/BoxF1_curve.png)
    ![confusion_matrix_normalized](runs/detect/runs/yolov11_fighter_jets_kaggle_1753230213/confusion_matrix_normalized.png)
    *(Lưu ý: Các hình ảnh trên là ví dụ, đường dẫn có thể thay đổi tùy theo timestamp của lần chạy)*

-   **Kết quả nhận diện video:** Video đã được nhận diện và xử lý sẽ có sẵn trong `runs/detect/f22_detection_result/`.


-   **Dự đoán trên tập validation:**
    -   **Ảnh gốc (Labels):**
        ![val_batch0_labels](runs/detect/runs/detect/val/val_batch0_labels.jpg)
    -   **Ảnh dự đoán (Predictions):**
        ![val_batch0_pred](runs/detect/runs/detect/val/val_batch0_pred.jpg)

## 📦 Mô Hình Đã Huấn Luyện Trước
Dự án sử dụng mô hình **`yolov11l.pt`** làm điểm khởi đầu cho quá trình fine-tuning. Mô hình này được tải tự động bởi thư viện Ultralytics. Mô hình `best.pt` sau khi huấn luyện sẽ nằm trong thư mục kết quả của lần chạy tương ứng.

## 📄 Giấy Phép
Dự án này được cấp phép theo Giấy phép **MIT License**.
Các bộ dữ liệu nguồn có thể có giấy phép riêng, ví dụ: bộ dữ liệu `333.v1i.coco` được cấp phép theo **CC BY 4.0**.






