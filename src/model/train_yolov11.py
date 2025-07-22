import os
# QUAN TRỌNG: Đặt biến môi trường này TRƯỚC khi import torch_xla
# để tắt máy chủ đo lường (metric server) và tránh lỗi "Could not set metric server port".
os.environ['PT_XLA_DISABLE_SERVER'] = '1'

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import time
import glob
import torch

# Chỉ import khi chạy trên môi trường có XLA (như Kaggle TPU)
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

def train_yolov11_on_tpu():
    """
    Huấn luyện mô hình YOLOv11 trên Kaggle với TPU, sử dụng torch.compile.
    """
    # === KIỂM TRA MÔI TRƯỜNG TPU ===
    if not TPU_AVAILABLE:
        print("Lỗi: Thư viện torch_xla không khả dụng. Vui lòng chạy trên môi trường Kaggle TPU.")
        return
        
    device = xm.xla_device()
    print(f"Đã phát hiện và chọn thiết bị TPU: {device}")

    # === CẤU HÌNH ĐƯỜNG DẪN CHO KAGGLE ===
    kaggle_working_dir = '/kaggle/working'
    runs_dir = os.path.join(kaggle_working_dir, 'runs', 'detect')
    train_dataset_path = '/kaggle/input/yaml-data/fighter_jets_kaggle.yaml' 
    
    # === TỰ ĐỘNG TÌM CHECKPOINT GẦN NHẤT HOẶC DÙNG MODEL MẶC ĐỊNH ===
    # Model mặc định nếu không tìm thấy checkpoint nào
    model_path = "/kaggle/working/models/yolo11l.pt"

    if os.path.exists(runs_dir):
        list_of_runs = [d for d in glob.glob(os.path.join(runs_dir, '*')) if os.path.isdir(d)]
        if list_of_runs:
            latest_run = max(list_of_runs, key=os.path.getmtime)
            checkpoint_path = os.path.join(latest_run, 'weights', 'best.pt')
            if os.path.exists(checkpoint_path):
                model_path = checkpoint_path
                print(f"Đã tìm thấy checkpoint. Tiếp tục huấn luyện từ: {model_path}")
            else:
                print(f"Đã tìm thấy lần chạy '{latest_run}', nhưng không có file 'best.pt'. Sử dụng model mặc định: {model_path}")
        else:
            print(f"Không tìm thấy lần chạy nào trước đó. Sử dụng model mặc định: {model_path}")
    else:
        print(f"Thư mục '{runs_dir}' không tồn tại. Sử dụng model mặc định: {model_path}")

    # --- QUY TRÌNH HUẤN LUYỆN ĐƯỢC CẤU TRÚC LẠI ĐỂ ĐẢM BẢO TƯƠNG THÍCH ---

    # 1. Định nghĩa TẤT CẢ các tham số huấn luyện
    run_name = f"yolov11_tpu_kaggle_{int(time.time())}"
    args = dict(
        model=model_path,  # Vẫn cần thiết để trainer biết cấu trúc
        data=train_dataset_path, 
        cache=True,
        epochs=200,              
        imgsz=1024,              
        batch=12,                 
        workers=0,
        name=run_name,           
        device=device,
        amp=False,
        # === Tối ưu hóa & Lập lịch ===
        optimizer='Adam', lr0=0.001, lrf=0.01, weight_decay=0.0005, patience=20,
        # === Tăng cường dữ liệu (đã tắt bớt để gỡ lỗi hiệu năng) ===
        hsv_h=0.01, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, 
        scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, 
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        # === Lưu trữ & Xuất kết quả ===
        project=runs_dir, save_period=10, val=True,
    )

    # 2. Tải mô hình một cách chính xác bằng API cấp cao của YOLO
    print("Đang tải mô hình ban đầu bằng API của YOLO...")
    model_object = YOLO(model_path)
    model_object.to(device)
    print("Tải mô hình ban đầu thành công.")

    # 3. Khởi tạo Trainer với các tham số đã định nghĩa
    trainer = DetectionTrainer(overrides=args)

    # 4. "Tiêm" mô hình đã được tải đúng cách vào trainer
    #    Điều này sẽ ghi đè lên quy trình tạo model mặc định của trainer
    trainer.model = model_object.model  # .model là module nn.Module thực sự

    # 5. Biên dịch mô hình đã được "tiêm" vào trainer
    print("Bắt đầu biên dịch mô hình với torch.compile...")
    trainer.model = torch.compile(trainer.model, backend="openxla")
    print("Biên dịch mô hình hoàn tất!")

    # 6. Bắt đầu quá trình huấn luyện
    #    Trainer sẽ sử dụng mô hình đã được biên dịch của chúng ta
    trainer.train()

    print("Quá trình huấn luyện hoàn tất!")
    print(f"Kết quả huấn luyện được lưu tại: {os.path.join(runs_dir, run_name)}")

    # === ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (Không thay đổi) ===
    print("\n" + "="*50)
    print("BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP DỮ LIỆU TEST")
    print("="*50 + "\n")

    best_model_path = os.path.join(runs_dir, run_name, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        best_model.to(device)
        test_results = best_model.val(split='test', data=train_dataset_path, device=device)
    else:
        print(f"Không tìm thấy file best.pt tại {best_model_path} để đánh giá.")
    
    print("\n" + "="*50)
    print("ĐÁNH GIÁ TRÊN TẬP TEST HOÀN TẤT")
    print("="*50)

if __name__ == "__main__":
    train_yolov11_on_tpu()