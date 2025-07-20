import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt

def get_device():
    """Kiểm tra và trả về thiết bị có sẵn (GPU hoặc CPU)."""
    if torch.cuda.is_available():
        print(">>> Đang sử dụng GPU.")
        return torch.device("cuda")
    print(">>> Đang sử dụng CPU.")
    return torch.device("cpu")

def get_feature_extractor(device):
    """Tải mô hình ResNet50 đã được huấn luyện trước và loại bỏ lớp cuối cùng."""
    # Tải mô hình ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Loại bỏ lớp phân loại cuối cùng (fully connected layer)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()  # Chuyển mô hình sang chế độ đánh giá
    return feature_extractor

def extract_features(image_path, model, device, transform):
    """Trích xuất vector đặc trưng từ một ảnh."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor)
            # Chuyển vector đặc trưng về dạng phẳng (1D) và chuyển sang CPU
            features = features.squeeze().cpu().numpy()
        return features
    except Exception as e:
        print(f"\nCảnh báo: Không thể trích xuất đặc trưng từ {os.path.basename(image_path)}. Lỗi: {e}")
        return None

def find_similar_images_deep(images_dir, min_similarity=0.97, max_similarity=0.98, num_samples_to_show=5):
    """
    Tìm và hiển thị các cặp ảnh tương đồng trong một khoảng cụ thể bằng cách sử dụng Deep Learning.
    
    Args:
        images_dir (str): Đường dẫn đến thư mục chứa ảnh.
        min_similarity (float): Ngưỡng dưới của khoảng tương đồng cần xem.
        max_similarity (float): Ngưỡng trên của khoảng tương đồng cần xem.
        num_samples_to_show (int): Số cặp ngẫu nhiên cần hiển thị.
    """
    device = get_device()
    feature_extractor = get_feature_extractor(device)
    
    # Định nghĩa các bước tiền xử lý ảnh cho ResNet50
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n>>> Bắt đầu trích xuất đặc trưng từ các ảnh trong: {images_dir}")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    features_dict = {}

    for filename in tqdm(image_files, desc="Trích xuất đặc trưng"):
        image_path = os.path.join(images_dir, filename)
        features = extract_features(image_path, feature_extractor, device, transform)
        if features is not None:
            features_dict[filename] = features
    
    print(f"\n>>> Đã trích xuất xong đặc trưng cho {len(features_dict)} ảnh.")

    print(f">>> Đang tìm kiếm các cặp tương đồng trong khoảng [{min_similarity}, {max_similarity})...")
    similar_pairs = []
    filenames = list(features_dict.keys())
    all_features = np.array(list(features_dict.values()))

    # Tính toán ma trận tương đồng Cosine cho tất cả các cặp
    similarity_matrix = cosine_similarity(all_features)
    
    for i in tqdm(range(len(filenames)), desc="So sánh đặc trưng"):
        for j in range(i + 1, len(filenames)):
            similarity = similarity_matrix[i, j]
            if min_similarity <= similarity < max_similarity:
                similar_pairs.append((filenames[i], filenames[j], similarity))

    print(f"\n>>> Tìm thấy {len(similar_pairs)} cặp ảnh tương đồng trong khoảng đã cho.")
    
    if not similar_pairs:
        print("Không tìm thấy cặp ảnh nào tương đồng để hiển thị.")
        return

    # Hiển thị các mẫu ngẫu nhiên
    if len(similar_pairs) < num_samples_to_show:
        num_samples_to_show = len(similar_pairs)
    random_pairs_to_show = random.sample(similar_pairs, num_samples_to_show)

    print(f"\n>>> Đang hiển thị {num_samples_to_show} cặp ảnh tương đồng ngẫu nhiên... <<<")
    for file1, file2, similarity in random_pairs_to_show:
        path1 = os.path.join(images_dir, file1)
        path2 = os.path.join(images_dir, file2)
        
        try:
            img1 = Image.open(path1)
            img2 = Image.open(path2)
        except Exception as e:
            print(f"Cảnh báo: Không thể mở ảnh để hiển thị. Lỗi: {e}")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img1)
        ax1.set_title(f"Ảnh 1: {file1}")
        ax1.axis('off')
        
        ax2.imshow(img2)
        ax2.set_title(f"Ảnh 2: {file2}")
        ax2.axis('off')
        
        fig.suptitle(f'Cặp tương đồng - Độ tương đồng Cosine: {similarity:.4f}', fontsize=16)
        plt.show()

if __name__ == '__main__':
    # --- CẤU HÌNH CÁC THAM SỐ TẠI ĐÂY ---
    IMAGES_DIRECTORY_TO_CHECK = "data/interim/merged_fighter_jets/images"
    
    # Chỉ định khoảng tương đồng bạn muốn xem.
    # Ví dụ: xem các cặp có độ tương đồng từ 0.97 đến (nhưng không bao gồm) 0.98.
    MIN_SIMILARITY = 0.81
    MAX_SIMILARITY = 0.84
    
    NUM_SAMPLES_TO_SHOW = 10

    find_similar_images_deep(
        images_dir=IMAGES_DIRECTORY_TO_CHECK,
        min_similarity=MIN_SIMILARITY,
        max_similarity=MAX_SIMILARITY,
        num_samples_to_show=NUM_SAMPLES_TO_SHOW
    ) 