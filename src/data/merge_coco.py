import json
import os
import shutil
from tqdm import tqdm

def merge_and_filter_coco(input_json_paths, output_dir, class_names_to_keep):
    """
    Hàm này gộp nhiều bộ dữ liệu COCO, lọc theo một hoặc nhiều lớp đối tượng cụ thể,
    sao chép các file ảnh liên quan vào một thư mục mới và ánh xạ lại (remap) các ID để tránh xung đột.

    Args:
        input_json_paths (list): Một danh sách các đường dẫn đến file COCO JSON đầu vào.
        output_dir (str): Đường dẫn đến thư mục để lưu bộ dữ liệu mới (ảnh và file JSON).
        class_names_to_keep (list): Danh sách tên của các lớp đối tượng cần giữ lại.
    """
    
    # Chuẩn hóa tên lớp sang chữ thường để xử lý
    class_names_to_keep_lower = {name.lower() for name in class_names_to_keep}

    # Tạo các thư mục đầu ra nếu chúng chưa tồn tại.
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Khởi tạo cấu trúc cho file COCO tổng hợp cuối cùng.
    merged_coco = {
        'info': {'description': f'Bộ dữ liệu được gộp và lọc cho các lớp: {", ".join(class_names_to_keep)}'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [] # Sẽ được điền vào sau
    }

    # Ánh xạ từ tên lớp (chữ thường) sang thông tin category cuối cùng (bao gồm ID mới)
    final_categories_map = {}
    next_final_cat_id = 1

    current_max_image_id = 0
    current_max_annotation_id = 0

    print("Bắt đầu quá trình gộp, lọc và sao chép ảnh...")
    for json_path in tqdm(input_json_paths, desc="Đang xử lý các file"):
        print(f"\nĐang xử lý {json_path}")
        
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Cảnh báo: Không thể đọc file JSON từ {json_path}. Đang bỏ qua.")
                continue

        # --- Bước 1: Tìm và ánh xạ các category ID mục tiêu trong file hiện tại ---
        # Ánh xạ từ source_cat_id -> {final_id, name}
        source_cat_map = {}
        for category in data.get('categories', []):
            cat_name_lower = category['name'].lower()
            if cat_name_lower in class_names_to_keep_lower:
                # Nếu lớp này chưa có trong bộ dữ liệu cuối cùng, thêm vào
                if cat_name_lower not in final_categories_map:
                    final_categories_map[cat_name_lower] = {
                        'id': next_final_cat_id,
                        'name': category['name'], # Giữ nguyên định dạng gốc
                        'supercategory': category.get('supercategory', 'none')
                    }
                    next_final_cat_id += 1
                
                final_cat_id = final_categories_map[cat_name_lower]['id']
                source_cat_map[category['id']] = {'final_id': final_cat_id, 'name': cat_name_lower}
        
        if not source_cat_map:
            print(f"Thông tin: Không tìm thấy lớp nào trong danh sách cần giữ lại tại {json_path}. Đang bỏ qua.")
            continue

        # --- Bước 2: Lọc các chú thích thuộc về các lớp đối tượng mục tiêu ---
        source_ids_to_keep = set(source_cat_map.keys())
        relevant_annotations = [
            ann for ann in data.get('annotations', []) if ann['category_id'] in source_ids_to_keep
        ]
        
        if not relevant_annotations:
            print(f"Thông tin: Không có chú thích nào cho các lớp mục tiêu trong {json_path}. Đang bỏ qua.")
            continue
            
        relevant_image_ids = {ann['image_id'] for ann in relevant_annotations}
        relevant_images = [img for img in data.get('images', []) if img['id'] in relevant_image_ids]
        
        image_id_mapping = {}
        
        for image in tqdm(relevant_images, desc="  Sao chép ảnh"):
            source_image_path = os.path.join(os.path.dirname(json_path), image['file_name'])
            if not os.path.exists(source_image_path):
                print(f"\nCảnh báo: Không tìm thấy file ảnh nguồn, bỏ qua: {source_image_path}")
                continue
            old_image_id = image['id']
            new_image_id = old_image_id + current_max_image_id
            file_extension = os.path.splitext(image['file_name'])[-1]
            new_filename = f"{new_image_id}{file_extension}"
            dest_image_path = os.path.join(output_images_dir, new_filename)
            try:
                shutil.copy(source_image_path, dest_image_path)
            except Exception as e:
                print(f"\nCảnh báo: Không thể sao chép {source_image_path}. Lỗi: {e}. Đang bỏ qua.")
                continue
            image_id_mapping[old_image_id] = new_image_id
            image['id'] = new_image_id
            image['file_name'] = new_filename
            merged_coco['images'].append(image)

        # Xử lý các chú thích
        for annotation in relevant_annotations:
            old_image_id = annotation['image_id']
            if old_image_id in image_id_mapping:
                new_image_id = image_id_mapping[old_image_id]
                old_annotation_id = annotation['id']
                new_annotation_id = old_annotation_id + current_max_annotation_id
                
                # Lấy ID category mới từ bản đồ đã tạo
                source_cat_id = annotation['category_id']
                new_category_id = source_cat_map[source_cat_id]['final_id']

                annotation['id'] = new_annotation_id
                annotation['image_id'] = new_image_id
                annotation['category_id'] = new_category_id
                
                merged_coco['annotations'].append(annotation)

        if image_id_mapping:
            current_max_image_id = max(image_id_mapping.values()) + 1
        if merged_coco['annotations']:
            current_max_annotation_id = max(ann['id'] for ann in merged_coco['annotations']) + 1
    
    # Cập nhật danh sách category cuối cùng vào bộ dữ liệu gộp
    merged_coco['categories'] = sorted(list(final_categories_map.values()), key=lambda x: x['id'])

    print("\nQuá trình gộp và lọc đã hoàn tất.")
    print(f"Tổng số ảnh đã sao chép: {len(merged_coco['images'])}")
    print(f"Tổng số chú thích: {len(merged_coco['annotations'])}")

    output_json_path = os.path.join(output_dir, '_annotations.coco.json')
    print(f"Đang lưu file chú thích đã gộp vào {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(merged_coco, f, indent=4)

    print("Xong!")


# Đoạn code này sẽ chạy khi bạn thực thi `python merge_coco.py`
if __name__ == '__main__':
    # --- CẤU HÌNH CÁC THAM SỐ TRỰC TIẾP TẠI ĐÂY ---

    # 1. Danh sách các file COCO JSON đầu vào.
    # Các đường dẫn này là tương đối so với thư mục gốc của dự án.
    input_files_to_process = [
        "data/raw/333.v1i.coco/train/_annotations.coco.json",
        "data/raw/fight_10000_images/train/_annotations.coco.json",
        "data/raw/fight_10000_images/test/_annotations.coco.json",
        "data/raw/fight_10000_images/valid/_annotations.coco.json",
        "data/raw/yolov8veesles.v1i.coco/train/_annotations.coco.json",
        "data/raw/yolov8veesles.v1i.coco/test/_annotations.coco.json",
        "data/raw/yolov8veesles.v1i.coco/valid/_annotations.coco.json",
        r"data/raw/Cats.v1i.coco/test/_annotations.coco.json",
        r"data/raw/Cats.v1i.coco/train/_annotations.coco.json",
        r"data/raw/Cats.v1i.coco/valid/_annotations.coco.json",
    ]

    # 2. Thư mục để lưu bộ dữ liệu mới.
    output_dataset_directory = "data/interim/merged_fighter_jets"

    # 3. Danh sách tên của các lớp đối tượng cần giữ lại.
    classes_to_keep = ["F22", "F35", "J20"]
    
    # --- KẾT THÚC PHẦN CẤU HÌNH ---

    # Gọi hàm chính với các tham số đã cấu hình ở trên.
    print(">>> Bắt đầu kịch bản với các tham số được cấu hình sẵn trong code <<<")
    merge_and_filter_coco(
        input_json_paths=input_files_to_process, 
        output_dir=output_dataset_directory, 
        class_names_to_keep=classes_to_keep
    )
    print(">>> Kịch bản đã hoàn tất! <<<") 