import torch
from ultralytics import YOLO
import argparse
import os

def detect_video(weights_path, source_video, project_name, run_name):
    """
    Chạy nhận diện trên video bằng mô hình YOLOv11 đã được huấn luyện.

    Args:
        weights_path (str): Đường dẫn đến file trọng số của mô hình (ví dụ: 'best.pt').
        source_video (str): Đường dẫn đến video đầu vào.
        project_name (str): Tên thư mục dự án để lưu kết quả.
        run_name (str): Tên cụ thể cho lần chạy này.
    """
    # Kiểm tra xem CUDA có khả dụng không và thiết lập thiết bị
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Đang sử dụng thiết bị: {device}")

    # Kiểm tra sự tồn tại của file trọng số và video
    if not os.path.exists(weights_path):
        print(f"Lỗi: Không tìm thấy file trọng số tại: {weights_path}")
        return
    if not os.path.exists(source_video):
        print(f"Lỗi: Không tìm thấy video tại: {source_video}")
        return

    # Tải mô hình YOLO đã huấn luyện
    model = YOLO(weights_path)
    model.to(device)

    print(f"Đã tải thành công mô hình từ: {weights_path}")
    print(f"Bắt đầu nhận diện trên video: {source_video}")

    # Chạy nhận diện trên video
    # save=True sẽ lưu video kết quả với bounding box
    # show=True sẽ hiển thị video trong một cửa sổ (có thể không hoạt động trên server)
    results = model.predict(
        source=source_video,
        show=False, # Đặt là False để tránh lỗi trên môi trường không có GUI
        save=True,
        stream=True, # Xử lý video theo luồng để tiết kiệm bộ nhớ
        project=project_name,
        name=run_name,
        conf=0.7 # Có thể điều chỉnh ngưỡng tin cậy ở đây
    )

    # Lặp qua generator để bắt đầu quá trình xử lý.
    # Thư viện sẽ tự động lưu video khi `save=True` và `stream=True`.
    for _ in results:
        pass

    output_path = os.path.join(project_name, run_name)
    print("Quá trình nhận diện hoàn tất.")
    print(f"Video kết quả đã được lưu tại thư mục: '{output_path}'")

if __name__ == '__main__':
    # Các tham số được định nghĩa sẵn trong code
    model_weights = "runs/detect/runs/yolov11_fighter_jets_kaggle_1753230213/weights/best.pt"
    input_video = r"data\video_test\skills_f22_edited.mp4"
    project_folder = "runs/detect"
    run_name_output = "f22_detection_result"

    # Gọi hàm nhận diện với các tham số đã định nghĩa
    detect_video(
        weights_path=model_weights,
        source_video=input_video,
        project_name=project_folder,
        run_name=run_name_output
    ) 