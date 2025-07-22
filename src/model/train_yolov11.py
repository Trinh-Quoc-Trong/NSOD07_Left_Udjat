from ultralytics import YOLO
import time
import os
import glob

# Chỉ import khi chạy trên môi trường có XLA (như Kaggle TPU)
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

def train_yolov11_on_tpu():
    """
    Huấn luyện mô hình YOLOv11 trên Kaggle với TPU,
    có khả năng tự động tiếp tục từ checkpoint tốt nhất của lần chạy trước.
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

    # Tải mô hình YOLO và chuyển lên TPU ngay từ đầu
    model = YOLO(model_path)
    model.to(device)

    # Tạo một tên duy nhất cho lần chạy huấn luyện mới
    run_name = f"yolov11_tpu_kaggle_{int(time.time())}"

    # Bắt đầu huấn luyện mô hình với các tham số cho TPU
    results = model.train(
        # === Tham số chính ===
        data=train_dataset_path, 
        cache=True,              # QUAN TRỌNG: Tải dữ liệu vào RAM để tăng tốc độ I/O
        epochs=200,              
        imgsz=1024,              
        batch=12,                 
        workers=0,               # QUAN TRỌNG: Đặt workers=0 khi dùng TPU
        name=run_name,           
        device=device,           # Sử dụng thiết bị TPU đã được xác định
        amp=False,               # QUAN TRỌNG: Tắt AMP để tránh lỗi kiểm tra driver NVIDIA trên TPU

        # === Tối ưu hóa & Lập lịch tốc độ học ===
        optimizer='Adam',        
        lr0=0.001,               
        lrf=0.01,                
        weight_decay=0.0005,     
        patience=20,             

        # === Tăng cường dữ liệu (Data Augmentation) ===
        hsv_h=0.01,
        hsv_s=1.0,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.05,
        scale=0.05,
        shear=3.0,
        perspective=0.0001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0, # Mosaic thường được tắt trong các epoch cuối, tắt hẳn có thể ổn định hơn
        mixup=0.0,
        copy_paste=0.5,

        # === Lưu trữ & Xuất kết quả ===
        project=runs_dir, 
        save_period=10,      
        val=True,            
    )

    print("Quá trình huấn luyện hoàn tất!")
    print(f"Kết quả huấn luyện được lưu tại: {os.path.join(runs_dir, run_name)}")

    # === ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ===
    print("\n" + "="*50)
    print("BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP DỮ LIỆU TEST")
    print("="*50 + "\n")

    best_model_path = os.path.join(runs_dir, run_name, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        best_model.to(device) # QUAN TRỌNG: Chuyển mô hình đã tải lên TPU trước khi đánh giá
        
        test_results = best_model.val(split='test', data=train_dataset_path, device=device)
    else:
        print(f"Không tìm thấy file best.pt tại {best_model_path} để đánh giá.")
    
    print("\n" + "="*50)
    print("ĐÁNH GIÁ TRÊN TẬP TEST HOÀN TẤT")
    print("="*50)

if __name__ == "__main__":
    train_yolov11_on_tpu()