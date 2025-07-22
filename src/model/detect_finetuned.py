import torch
from ultralytics import YOLO
import os

def evaluate_finetuned_model_on_test_set():
    """
    Đánh giá mô hình đã được fine-tune (sử dụng file best.pt) trên tập dữ liệu test
    và tạo báo cáo hiệu suất.
    """
    # --- THAY ĐỔI ĐƯỜNG DẪN NÀY NẾU CẦN ---
    # Trỏ đến file best.pt từ lần huấn luyện của bạn
    # Ví dụ: 'runs/detect/yolov11_fighter_jets_.../weights/best.pt'
    # Dựa trên log trước đó của bạn, có vẻ đường dẫn này là hợp lý:
    path_to_best_model = r'runs\detect\best_yolov11l.pt'

    # Kiểm tra xem file model có tồn tại không
    if not os.path.exists(path_to_best_model):
        print(f"Lỗi: Không tìm thấy file model tại '{path_to_best_model}'")
        print("Vui lòng kiểm tra lại đường dẫn và tên thư mục của lần chạy.")
        return

    # Đường dẫn đến file cấu hình dataset cục bộ (quan trọng để tìm đúng tập test)
    data_yaml_path = 'fighter_jets.yaml'
    if not os.path.exists(data_yaml_path):
        print(f"Lỗi: Không tìm thấy file cấu hình dữ liệu tại '{data_yaml_path}'")
        print("Vui lòng đảm bảo file fighter_jets.yaml nằm ở thư mục gốc của dự án.")
        return

    # Thiết lập thiết bị (GPU hoặc CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")

    # Tải mô hình đã huấn luyện
    model = YOLO(path_to_best_model)
    model.to(device)
    
    print(f"Đã tải thành công mô hình từ: {path_to_best_model}")

    # Chạy đánh giá (validation) trên tập dữ liệu test
    # split='test' sẽ đảm bảo nó chỉ dùng tập test đã định nghĩa trong fighter_jets.yaml
    print("Bắt đầu đánh giá mô hình trên tập test để tạo báo cáo hiệu suất...")
    results = model.val(
        data=data_yaml_path, 
        split='test', 
        imgsz=1024, # Đảm bảo kích thước ảnh phù hợp với huấn luyện
        project='runs/detect', 
        name='finetuned_test_report' # Lưu báo cáo vào thư mục riêng
    )

    print("\n" + "="*50)
    print("ĐÁNH GIÁ TRÊN TẬP TEST HOÀN TẤT")
    print("="*50)
    print("Báo cáo hiệu suất chi tiết (metrics, plots, v.v.) đã được lưu tại:")
    print(f"runs/detect/finetuned_test_report/")
    print("Bạn có thể xem các file: results.csv, confusion_matrix.png, v.v.")
    print("\n" + "=== Các chỉ số chính ===")
    print(f"mAP50-95: {results.box.map}")
    print(f"mAP50: {results.box.map50}")
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")
    print("=========================")

if __name__ == '__main__':
    evaluate_finetuned_model_on_test_set() 