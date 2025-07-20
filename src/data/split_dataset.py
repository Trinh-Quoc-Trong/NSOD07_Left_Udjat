import os
import json
import random
import shutil
from tqdm import tqdm

def split_coco_dataset(source_dir, output_dir, train_ratio=0.6, valid_ratio=0.2):
    """
    Chia một bộ dữ liệu COCO (ảnh và file JSON) thành các tập train, valid, và test.

    Args:
        source_dir (str): Đường dẫn đến thư mục chứa bộ dữ liệu nguồn (bao gồm thư mục 'images' và '_annotations.coco.json').
        output_dir (str): Đường dẫn đến thư mục để lưu các tập dữ liệu đã chia.
        train_ratio (float): Tỉ lệ cho tập huấn luyện.
        valid_ratio (float): Tỉ lệ cho tập xác thực.
    """
    # --- 1. Thiết lập các đường dẫn và tạo thư mục ---
    source_json_path = os.path.join(source_dir, '_annotations.coco.json')
    source_images_dir = os.path.join(source_dir, 'images')

    if not os.path.exists(source_json_path) or not os.path.exists(source_images_dir):
        print(f"Lỗi: Không tìm thấy file JSON hoặc thư mục 'images' trong: {source_dir}")
        return

    # Tạo các thư mục đầu ra
    splits = ['train', 'valid', 'test']
    output_paths = {}
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        output_paths[split] = {'dir': split_dir, 'images_dir': images_dir}
    
    print(">>> Đã tạo các thư mục đầu ra.")

    # --- 2. Đọc và xáo trộn dữ liệu nguồn ---
    print(f">>> Đang đọc dữ liệu từ: {source_json_path}")
    with open(source_json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    random.shuffle(images) # Xáo trộn ảnh để đảm bảo tính ngẫu nhiên
    print(f">>> Tổng số ảnh: {len(images)}. Đã xáo trộn thành công.")

    # --- 3. Chia danh sách ảnh theo tỉ lệ ---
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    valid_end = train_end + int(total_images * valid_ratio)

    split_images = {
        'train': images[:train_end],
        'valid': images[train_end:valid_end],
        'test': images[valid_end:]
    }

    # --- 4. Tạo một bản đồ tra cứu nhanh các chú thích theo image_id ---
    annotations_by_image_id = {}
    for ann in tqdm(coco_data['annotations'], desc="Đang lập chỉ mục chú thích"):
        image_id = ann['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)

    # --- 5. Tạo các file JSON và sao chép ảnh cho từng tập ---
    for split in splits:
        print(f"\n>>> Đang xử lý tập '{split}'...")
        
        split_data_images = split_images[split]
        split_data_annotations = []
        
        # Lấy ID của các ảnh trong tập hiện tại
        split_image_ids = {img['id'] for img in split_data_images}

        # Lấy tất cả các chú thích thuộc về các ảnh trong tập này
        for image_id in split_image_ids:
            if image_id in annotations_by_image_id:
                split_data_annotations.extend(annotations_by_image_id[image_id])
        
        # Tạo cấu trúc COCO mới cho tập này
        new_coco_split = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data.get('categories', []),
            'images': split_data_images,
            'annotations': split_data_annotations
        }

        # Lưu file JSON mới
        json_output_path = os.path.join(output_paths[split]['dir'], '_annotations.coco.json')
        print(f"  Đang lưu file JSON tại: {json_output_path}")
        with open(json_output_path, 'w') as f:
            json.dump(new_coco_split, f, indent=4)

        # Sao chép các file ảnh
        print(f"  Đang sao chép {len(split_data_images)} file ảnh...")
        for image_info in tqdm(split_data_images, desc=f"Sao chép ảnh {split}"):
            source_image_path = os.path.join(source_images_dir, image_info['file_name'])
            dest_image_path = os.path.join(output_paths[split]['images_dir'], image_info['file_name'])
            
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, dest_image_path)
            else:
                print(f"Cảnh báo: Không tìm thấy file ảnh nguồn, bỏ qua: {source_image_path}")

    print("\n>>> Hoàn tất! Đã chia bộ dữ liệu thành các tập train, valid, và test. <<<")


if __name__ == '__main__':
    # --- CẤU HÌNH CÁC THAM SỐ TẠI ĐÂY ---

    # 1. Thư mục chứa bộ dữ liệu đã gộp cần được chia.
    SOURCE_DATASET_DIR = "data/interim/merged_fighter_jets"

    # 2. Thư mục để lưu kết quả các tập train/valid/test đã chia.
    OUTPUT_PROCESSED_DIR = "data/processed"

    # 3. Tỉ lệ chia dữ liệu.
    # Tỉ lệ test sẽ được tự động tính toán (1 - TRAIN_RATIO - VALID_RATIO).
    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    
    # --- KẾT THÚC PHẦN CẤU HÌNH ---

    split_coco_dataset(
        source_dir=SOURCE_DATASET_DIR,
        output_dir=OUTPUT_PROCESSED_DIR,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO
    ) 