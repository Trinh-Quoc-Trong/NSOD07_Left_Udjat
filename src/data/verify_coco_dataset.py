import json
import os
import random
import cv2
import argparse
from tqdm import tqdm # Import tqdm for progress bar

def perform_quantitative_check(input_json_paths, output_json_path, output_images_dir, class_names_to_keep):
    """
    Thực hiện kiểm tra định lượng bộ dữ liệu COCO đã gộp.
    So sánh số lượng ảnh và chú thích mong đợi với số lượng thực tế.
    """
    print(">>> Bắt đầu kiểm tra định lượng bộ dữ liệu... <<<")

    # Chuẩn hóa tên lớp sang chữ thường
    class_names_to_keep_lower = {name.lower() for name in class_names_to_keep}

    expected_images_count = 0
    expected_annotations_count = 0

    # Tính toán số lượng mong đợi từ các file JSON nguồn
    print("\nĐang tính toán số lượng ảnh và chú thích mong đợi từ các file nguồn...")
    for json_path in tqdm(input_json_paths, desc="Phân tích file nguồn"):
        if not os.path.exists(json_path):
            print(f"Cảnh báo: Không tìm thấy file JSON nguồn: {json_path}. Bỏ qua.")
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Cảnh báo: Không thể đọc JSON từ {json_path}. Bỏ qua.")
            continue

        source_category_ids_to_keep = set()
        # Lấy ID của các category cần giữ từ file nguồn hiện tại
        for category in data.get('categories', []):
            if category['name'].lower() in class_names_to_keep_lower:
                source_category_ids_to_keep.add(category['id'])
        
        if not source_category_ids_to_keep:
            # print(f"Thông tin: Không tìm thấy các lớp mục tiêu trong {json_path}.")
            continue

        # Đếm số chú thích phù hợp trong file nguồn này
        relevant_annotations_in_source = [
            ann for ann in data.get('annotations', []) if ann['category_id'] in source_category_ids_to_keep
        ]
        expected_annotations_count += len(relevant_annotations_in_source)

        # Đếm số ảnh phù hợp trong file nguồn này (chỉ đếm một lần cho mỗi ảnh)
        relevant_image_ids_in_source = {ann['image_id'] for ann in relevant_annotations_in_source}
        expected_images_count += len(relevant_image_ids_in_source)

    print(f"\nSố lượng mong đợi: {expected_images_count} ảnh và {expected_annotations_count} chú thích.")

    # --- Đọc và đếm số lượng thực tế trong file JSON đã gộp ---
    print(f"\nĐang đọc file JSON đã gộp tại: {output_json_path}")
    if not os.path.exists(output_json_path):
        print(f"Lỗi: Không tìm thấy file JSON đã gộp tại: {output_json_path}")
        return

    with open(output_json_path, 'r') as f:
        merged_coco_data = json.load(f)
    
    actual_images_in_json_count = len(merged_coco_data.get('images', []))
    actual_annotations_in_json_count = len(merged_coco_data.get('annotations', []))

    print(f"Số lượng thực tế trong JSON đã gộp: {actual_images_in_json_count} ảnh và {actual_annotations_in_json_count} chú thích.")

    # --- Đếm số lượng file ảnh thực tế đã sao chép ---
    actual_copied_image_files_count = 0
    if os.path.exists(output_images_dir):
        # Đếm các file ảnh (bỏ qua các thư mục con hoặc file không phải ảnh nếu có)
        for root, _, files in os.walk(output_images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    actual_copied_image_files_count += 1
    print(f"Số lượng file ảnh đã sao chép: {actual_copied_image_files_count}")

    # --- Báo cáo kết quả ---
    print("\n--- BÁO CÁO KIỂM TRA ĐỊNH LƯỢNG ---")
    images_match = (expected_images_count == actual_images_in_json_count and 
                    actual_images_in_json_count == actual_copied_image_files_count)
    annotations_match = (expected_annotations_count == actual_annotations_in_json_count)

    if images_match:
        print("✅ Số lượng ảnh: KHỚP hoàn toàn.")
    else:
        print("❌ Số lượng ảnh: KHÔNG KHỚP.")
        if expected_images_count != actual_images_in_json_count:
            print(f"   - JSON đã gộp có {actual_images_in_json_count} ảnh, nhưng dự kiến là {expected_images_count}.")
        if actual_images_in_json_count != actual_copied_image_files_count:
            print(f"   - JSON đã gộp có {actual_images_in_json_count} ảnh, nhưng thư mục ảnh có {actual_copied_image_files_count} file.")

    if annotations_match:
        print("✅ Số lượng chú thích (bbox): KHỚP hoàn toàn.")
    else:
        print("❌ Số lượng chú thích (bbox): KHÔNG KHỚP.")
        print(f"   - JSON đã gộp có {actual_annotations_in_json_count} chú thích, nhưng dự kiến là {expected_annotations_count}.")

    print(">>> Kiểm tra định lượng hoàn tất. <<<")

def verify_coco_dataset_visual(json_path, images_dir, num_samples=10):
    """
    Hàm này kiểm tra trực quan một bộ dữ liệu COCO bằng cách vẽ các bounding box
    lên một số lượng ảnh ngẫu nhiên và hiển thị chúng.
    """
    print(f">>> Đang tải bộ dữ liệu từ: {json_path}")
    if not os.path.exists(json_path):
        print(f"Lỗi: Không tìm thấy file JSON tại: {json_path}")
        return

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # --- Bước 1: Chuẩn bị dữ liệu để tra cứu nhanh ---
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Các lớp trong bộ dữ liệu: {list(category_map.values())}")
    
    annotations_by_image_id = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)

    # --- Bước 2: Chọn ngẫu nhiên các ảnh để kiểm tra ---
    all_images = coco_data['images']
    if len(all_images) == 0:
        print("Không có ảnh nào trong bộ dữ liệu để kiểm tra trực quan.")
        return

    if len(all_images) < num_samples:
        print(f"Cảnh báo: Số lượng ảnh trong bộ dữ liệu ({len(all_images)}) nhỏ hơn số lượng yêu cầu ({num_samples}). Sẽ kiểm tra tất cả ảnh.")
        num_samples = len(all_images)

    random_images = random.sample(all_images, num_samples)
    print(f"\n>>> Bắt đầu kiểm tra {num_samples} ảnh ngẫu nhiên... <<<")
    print("Một cửa sổ sẽ hiện lên, nhấn phím bất kỳ để xem ảnh tiếp theo. Nhấn 'q' để thoát.")

    # --- Bước 3: Lặp qua các ảnh ngẫu nhiên, vẽ và hiển thị ---
    for i, image_info in enumerate(random_images):
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = os.path.join(images_dir, file_name)

        print(f"\nĐang hiển thị ảnh {i+1}/{num_samples}: {file_name} (ID: {image_id})")

        if not os.path.exists(image_path):
            print(f"  Lỗi: Không tìm thấy file ảnh tại: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"  Lỗi: Không thể đọc file ảnh: {image_path}")
            continue

        annotations = annotations_by_image_id.get(image_id, [])
        if not annotations:
            print("  Thông tin: Không có chú thích (bounding box) nào cho ảnh này.")
        
        for ann in annotations:
            bbox = ann['bbox']  # Định dạng COCO: [x, y, width, height]
            category_id = ann['category_id']
            category_name = category_map.get(category_id, "Unknown")

            x, y, w, h = [int(coord) for coord in bbox]
            top_left = (x, y)
            bottom_right = (x + w, y + h)

            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            label = f"{category_name}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(y, label_size[1] + 10)
            cv2.rectangle(image, (x, y_label - label_size[1] - 10), (x + label_size[0], y_label), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (x, y_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        window_name = f'Kiểm tra: {file_name}'
        cv2.imshow(window_name, image)
        
        # Đợi người dùng nhấn một phím bất kỳ (27 là phím Esc, 'q' là 113) để tiếp tục
        # Nếu nhấn 'q' hoặc Esc, thoát vòng lặp
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        cv2.destroyWindow(window_name)

    print("\n>>> Đã hoàn tất kiểm tra trực quan! <<<")


if __name__ == '__main__':
    # --- Cấu hình bộ dữ liệu cần kiểm tra ở đây ---
    # Hãy đảm bảo các đường dẫn này khớp với cấu hình trong merge_coco.py

    # 1. Danh sách các file COCO JSON đầu vào gốc.
    # PHẢI KHỚP VỚI `input_files_to_process` TRONG `merge_coco.py`
    input_files_for_check = [
        r"data/raw/333.v1i.coco/train/_annotations.coco.json",
        r"data/raw/fight_10000_images/train/_annotations.coco.json",
        r"data/raw/fight_10000_images/test/_annotations.coco.json",
        r"data/raw/fight_10000_images/valid/_annotations.coco.json",
        r"data/raw/yolov8veesles.v1i.coco/train/_annotations.coco.json",
        r"data/raw/yolov8veesles.v1i.coco/test/_annotations.coco.json",
        r"data/raw/yolov8veesles.v1i.coco/valid/_annotations.coco.json",
        r"data\raw\Cats.v1i.coco\test\_annotations.coco.json",
        r"data\raw\Cats.v1i.coco\train\_annotations.coco.json",
        r"data\raw\Cats.v1i.coco\valid\_annotations.coco.json",
    ]

    # 2. Thư mục đầu ra của bộ dữ liệu đã gộp.
    # PHẢI KHỚP VỚI `output_dataset_directory` TRONG `merge_coco.py`
    merged_dataset_output_dir = "data/interim/merged_fighter_jets"

    # 3. Danh sách tên của các lớp đối tượng bạn đã lọc.
    # PHẢI KHỚP VỚI `classes_to_keep` TRONG `merge_coco.py`
    classes_to_verify = ["F22", "F35", "J20"]
    
    # --- Tùy chọn kiểm tra trực quan ---
    NUM_SAMPLES_TO_VISUALLY_CHECK = 10 # Số lượng ảnh ngẫu nhiên để kiểm tra trực quan (đặt 0 để bỏ qua).

    # --- Bắt đầu quá trình kiểm tra ---
    merged_json_path = os.path.join(merged_dataset_output_dir, '_annotations.coco.json')
    merged_images_dir = os.path.join(merged_dataset_output_dir, 'images')

    # 1. Thực hiện kiểm tra định lượng trước
    perform_quantitative_check(
        input_files_for_check,
        merged_json_path,
        merged_images_dir,
        classes_to_verify
    )

    # 2. Sau đó, nếu muốn, thực hiện kiểm tra trực quan
    if NUM_SAMPLES_TO_VISUALLY_CHECK > 0:
        verify_coco_dataset_visual(merged_json_path, merged_images_dir, NUM_SAMPLES_TO_VISUALLY_CHECK)

    print("\n>>> Tất cả các bước kiểm tra đã hoàn tất. <<<") 