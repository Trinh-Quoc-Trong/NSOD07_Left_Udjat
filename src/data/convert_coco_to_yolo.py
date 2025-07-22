import json
import os
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, save_dir):
    """
    Chuyển đổi tệp chú thích định dạng COCO (.json) sang định dạng YOLO (.txt).

    Args:
        coco_json_path (str): Đường dẫn đến tệp _annotations.coco.json.
        save_dir (str): Thư mục để lưu các tệp nhãn .txt của YOLO.
    """
    labels_dir = Path(save_dir) / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path) as f:
        data = json.load(f)

    # Tạo một từ điển để truy cập thông tin ảnh nhanh hơn
    images_info = {img['id']: img for img in data['images']}
    
    # Nhóm các chú thích theo image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    for img_id, image in images_info.items():
        img_name = Path(image['file_name']).stem
        img_width = image['width']
        img_height = image['height']

        yolo_labels = []
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                # COCO format: [x_min, y_min, width, height]
                x_min, y_min, bbox_width, bbox_height = ann['bbox']
                
                # YOLO format: [x_center, y_center, width, height] (normalized)
                x_center = (x_min + bbox_width / 2) / img_width
                y_center = (y_min + bbox_height / 2) / img_height
                norm_width = bbox_width / img_width
                norm_height = bbox_height / img_height
                
                # Chuyển đổi category_id từ 1-based (COCO) sang 0-based (YOLO)
                category_id = ann['category_id'] - 1
                
                yolo_labels.append(f"{category_id} {x_center} {y_center} {norm_width} {norm_height}")

        # Ghi vào tệp .txt
        if yolo_labels:
            with open(labels_dir / f"{img_name}.txt", 'w') as f:
                f.write("\n".join(yolo_labels))

    print(f"Hoàn tất chuyển đổi cho: {coco_json_path}")

if __name__ == '__main__':
    base_dir = Path('data/processed')
    
    # Chuyển đổi cho các tập train, valid, và test
    for split in ['train', 'valid', 'test']:
        coco_file = base_dir / split / '_annotations.coco.json'
        if coco_file.exists():
            convert_coco_to_yolo(coco_file, base_dir / split)
        else:
            print(f"Không tìm thấy tệp: {coco_file}") 