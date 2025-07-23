from ultralytics import YOLO
import time
import os
import glob

def train_yolov11():
    """
    Huấn luyện mô hình YOLOv11 trên Kaggle, có khả năng tự động tiếp tục
    từ checkpoint tốt nhất của lần chạy trước.
    """
    # === CẤU HÌNH ĐƯỜNG DẪN CHO KAGGLE ===
    # Thư mục làm việc trên Kaggle, nơi lưu trữ output
    runs_dir = '/kaggle/working/runs'

    # Đường dẫn đến file yaml cấu hình dataset.
    # *LƯU Ý*: Hãy đảm bảo file này tồn tại trong môi trường Kaggle của bạn.
    train_dataset_path = 'fighter_jets_kaggle.yaml' 
    
    # === TỰ ĐỘNG TÌM CHECKPOINT GẦN NHẤT ===
    model_path = "yolo11l.pt"  # Mô hình mặc định nếu không có checkpoint



    # Tải mô hình YOLOv11n hoặc checkpoint tốt nhất từ lần chạy trước.
    model = YOLO(model_path)

    # Tạo một tên duy nhất cho lần chạy huấn luyện mới
    run_name = f"yolov11_fighter_jets_kaggle_{int(time.time())}"

    # Bắt đầu huấn luyện mô hình với các tham số mới
    results = model.train(
        # === Tham số chính ===
        data=train_dataset_path, 
        epochs=200,              
        imgsz=1024,              
        batch=6,                 
        workers=2,               
        name=run_name,           
        device=0,                

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
        # mosaic=1.0,
        # mixup=0.0,
        copy_paste=0.5,

        # === Lưu trữ & Xuất kết quả ===
        project=runs_dir, # Thư mục gốc để lưu các kết quả huấn luyện trên Kaggle.
        save_period=10,      
        val=True,            
    )

    print("Quá trình huấn luyện hoàn tất!")
    print(f"Kết quả huấn luyện được lưu tại: {os.path.join(runs_dir, run_name)}")

    # === ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ===
    print("\n" + "="*50)
    print("BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP DỮ LIỆU TEST")
    print("="*50 + "\n")

    # Tải lại mô hình tốt nhất (best.pt) từ kết quả huấn luyện
    best_model_path = os.path.join(runs_dir, run_name, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        # Chạy đánh giá trên tập dữ liệu test được định nghĩa trong file yaml
        test_results = best_model.val(split='test', data=train_dataset_path)
    else:
        print(f"Không tìm thấy file best.pt tại {best_model_path} để đánh giá.")
    
    print("\n" + "="*50)
    print("ĐÁNH GIÁ TRÊN TẬP TEST HOÀN TẤT")
    print("="*50)
    print("Kết quả chi tiết đã được lưu trong thư mục của lần chạy.")

if __name__ == "__main__":
    train_yolov11()