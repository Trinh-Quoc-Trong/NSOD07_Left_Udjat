import os
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import pandas as pd

# --- Các hàm trợ giúp từ kịch bản trước ---
def get_device():
    """Kiểm tra và trả về thiết bị có sẵn (GPU hoặc CPU)."""
    if torch.cuda.is_available():
        print(">>> Đang sử dụng GPU.")
        return torch.device("cuda")
    print(">>> Đang sử dụng CPU.")
    return torch.device("cpu")

def get_feature_extractor(device):
    """Tải mô hình ResNet50 đã được huấn luyện trước và loại bỏ lớp cuối cùng."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

def extract_features(image_path, model, device, transform):
    """Trích xuất vector đặc trưng từ một ảnh."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor).squeeze().cpu().numpy()
        return features
    except Exception as e:
        print(f"\nCảnh báo: Không thể trích xuất đặc trưng từ {os.path.basename(image_path)}. Lỗi: {e}")
        return None

# --- Hàm chính để loại bỏ ảnh trùng lặp ---
def deduplicate_dataset(target_dir, class_names_to_target, similarity_threshold=0.99):
    """
    Tìm và loại bỏ các ảnh trùng lặp có chứa các lớp cụ thể trong bộ dữ liệu COCO.

    Args:
        target_dir (str): Đường dẫn đến thư mục bộ dữ liệu.
        class_names_to_target (list): Danh sách các tên lớp cần được xử lý.
        similarity_threshold (float): Ngưỡng tương đồng Cosine để coi là trùng lặp.
    """
    images_dir = os.path.join(target_dir, 'images')
    json_path = os.path.join(target_dir, '_annotations.coco.json')

    if not os.path.exists(images_dir) or not os.path.exists(json_path):
        print(f"Lỗi: Không tìm thấy thư mục 'images' hoặc file '_annotations.coco.json' trong {target_dir}")
        return

    # --- Bước 0: Tải và chuẩn bị dữ liệu ---
    print(">>> Đang tải và chuẩn bị dữ liệu...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    df_categories = pd.DataFrame(coco_data['categories'])
    df_annotations = pd.DataFrame(coco_data['annotations'])
    df_images = pd.DataFrame(coco_data['images'])

    target_class_names_lower = {name.lower() for name in class_names_to_target}
    target_category_ids = set(df_categories[df_categories['name'].str.lower().isin(target_class_names_lower)]['id'])
    
    if not target_category_ids:
        print(f"Lỗi: Không tìm thấy bất kỳ lớp nào trong danh sách {class_names_to_target} trong file JSON.")
        return

    image_ids_to_process = set(df_annotations[df_annotations['category_id'].isin(target_category_ids)]['image_id'])
    image_files_to_process = list(df_images[df_images['id'].isin(image_ids_to_process)]['file_name'])
    
    print(f">>> Sẽ xử lý {len(image_files_to_process)} ảnh có chứa các lớp: {', '.join(class_names_to_target)}")

    # --- Bước 1 & 2: Trích xuất đặc trưng và nhóm các ảnh tương đồng ---
    device = get_device()
    feature_extractor = get_feature_extractor(device)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n>>> Bắt đầu trích xuất đặc trưng từ các ảnh...")
    features_dict = {}
    for filename in tqdm(image_files_to_process, desc="Trích xuất đặc trưng"):
        full_path = os.path.join(images_dir, filename)
        if not os.path.exists(full_path): continue
        features = extract_features(full_path, feature_extractor, device, transform)
        if features is not None:
            features_dict[filename] = features
    
    print("\n>>> Đang nhóm các ảnh tương đồng...")
    filenames = list(features_dict.keys())
    if len(filenames) < 2:
        print("Không đủ ảnh để so sánh, không có gì để làm.")
        return

    all_features = np.array(list(features_dict.values()))
    similarity_matrix = cosine_similarity(all_features)
    
    processed_indices = set()
    duplicate_groups = []
    for i in tqdm(range(len(filenames)), desc="Đang nhóm"):
        if i in processed_indices:
            continue
        
        current_group = {i}
        for j in range(i + 1, len(filenames)):
            if similarity_matrix[i, j] >= similarity_threshold:
                current_group.add(j)
        
        if len(current_group) > 1:
            duplicate_groups.append([filenames[idx] for idx in current_group])
        
        processed_indices.update(current_group)

    # --- Bước 3: Xác định các file cần xóa với ĐIỀU KIỆN AN TOÀN ---
    print("\n>>> Đang kiểm tra điều kiện an toàn trước khi xác định file cần xóa...")
    files_to_delete = set()
    
    # Tạo các bản đồ tra cứu cần thiết
    image_id_to_cats = df_annotations.groupby('image_id')['category_id'].apply(set).to_dict()
    filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
    # Tập hợp các ID của các lớp KHÔNG PHẢI là mục tiêu
    non_target_category_ids = set(df_categories[~df_categories['id'].isin(target_category_ids)]['id'])

    for group in tqdm(duplicate_groups, desc="Kiểm tra điều kiện xóa"):
        # Giữ lại file đầu tiên, kiểm tra các file còn lại trong nhóm
        for filename_to_check in group[1:]:
            image_id = filename_to_id.get(filename_to_check)
            if not image_id:
                continue
            
            # Lấy tất cả các category có trong ảnh này
            categories_in_image = image_id_to_cats.get(image_id, set())
            
            # Kiểm tra xem có category nào không phải là mục tiêu không.
            # isdisjoint() trả về True nếu hai tập hợp không có phần tử chung.
            contains_other_classes = not categories_in_image.isdisjoint(non_target_category_ids)
            
            if not contains_other_classes:
                # Nếu ảnh CHỈ chứa các lớp mục tiêu, thêm vào danh sách xóa
                files_to_delete.add(filename_to_check)
            else:
                # Nếu ảnh chứa các lớp khác, thông báo và KHÔNG xóa
                print(f"  CẢNH BÁO: Ảnh '{filename_to_check}' là trùng lặp nhưng sẽ không bị xóa vì chứa các lớp khác.")

    if not files_to_delete:
        print("\n>>> Không tìm thấy ảnh nào thỏa mãn điều kiện xóa. Không có gì để làm. <<<")
        return

    # --- Bước 4: Cập nhật file JSON ---
    print(f"\n>>> Tìm thấy {len(files_to_delete)} ảnh trùng lặp thỏa mãn điều kiện để xóa.")
    
    print(">>> Đang cập nhật file _annotations.coco.json...")
    ids_to_delete = {filename_to_id[fname] for fname in files_to_delete if fname in filename_to_id}

    original_image_count = len(coco_data['images'])
    original_annotation_count = len(coco_data['annotations'])

    coco_data['images'] = [img for img in coco_data['images'] if img['id'] not in ids_to_delete]
    coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['image_id'] not in ids_to_delete]

    updated_image_count = len(coco_data['images'])
    updated_annotation_count = len(coco_data['annotations'])

    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"  Đã xóa {original_image_count - updated_image_count} mục ảnh khỏi JSON.")
    print(f"  Đã xóa {original_annotation_count - updated_annotation_count} mục chú thích khỏi JSON.")

    # --- Bước 5: Xóa các file ảnh vật lý ---
    print("\n>>> Đang xóa các file ảnh trùng lặp...")
    deleted_count = 0
    for filename in tqdm(files_to_delete, desc="Đang xóa file"):
        try:
            os.remove(os.path.join(images_dir, filename))
            deleted_count += 1
        except OSError as e:
            print(f"Lỗi khi xóa file {filename}: {e}")

    print(f"\n>>> Hoàn tất! Đã xóa {deleted_count} file ảnh trùng lặp. <<<")


if __name__ == '__main__':
    print("!!! CẢNH BÁO: Kịch bản này sẽ XÓA VĨNH VIỄN các file ảnh và sửa đổi file JSON. !!!")
    print("!!! Hãy đảm bảo bạn đã sao lưu (backup) bộ dữ liệu trước khi tiếp tục. !!!")
    user_input = input("Nhập 'yes' để tiếp tục: ")

    if user_input.lower() == 'yes':
        TARGET_DATASET_DIR = "data/interim/merged_fighter_jets"
        SIMILARITY_THRESHOLD = 0.75
        # Chỉ định các lớp bạn muốn dọn dẹp sự trùng lặp
        CLASSES_TO_DEDUPLICATE = ["F35"]

        deduplicate_dataset(
            target_dir=TARGET_DATASET_DIR,
            class_names_to_target=CLASSES_TO_DEDUPLICATE,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
    else:
        print("Đã hủy bỏ thao tác.") 