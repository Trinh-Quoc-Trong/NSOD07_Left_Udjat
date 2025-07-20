import os
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- Sao chép các hàm trợ giúp cần thiết ---
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def get_feature_extractor(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

def extract_features(image_path, model, device, transform):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor).squeeze().cpu().numpy()
        return features
    except Exception:
        return None

def visualize_similarity_distribution(images_dir, output_figure_path, similarity_threshold=0.99):
    """
    Tính toán và vẽ biểu đồ phân bố độ tương đồng của các ảnh trong một thư mục.

    Args:
        images_dir (str): Đường dẫn đến thư mục chứa ảnh.
        output_figure_path (str): Đường dẫn để lưu biểu đồ kết quả.
        similarity_threshold (float): Ngưỡng tương đồng để vẽ đường tham chiếu.
    """
    device = get_device()
    feature_extractor = get_feature_extractor(device)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n>>> Bắt đầu trích xuất đặc trưng từ các ảnh...")
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    features_dict = {}
    for filename in tqdm(image_files, desc="Trích xuất đặc trưng"):
        features = extract_features(os.path.join(images_dir, filename), feature_extractor, device, transform)
        if features is not None:
            features_dict[filename] = features
    
    if len(features_dict) < 2:
        print("Không đủ ảnh để so sánh.")
        return

    print(f"\n>>> Đã trích xuất xong. Bắt đầu tính toán ma trận tương đồng...")
    all_features = np.array(list(features_dict.values()))
    similarity_matrix = cosine_similarity(all_features)
    
    # Lấy các giá trị tương đồng từ tam giác trên của ma trận (không bao gồm đường chéo)
    # để tránh so sánh một ảnh với chính nó và lặp lại các cặp.
    upper_triangle_indices = np.triu_indices(len(all_features), k=1)
    similarity_values = similarity_matrix[upper_triangle_indices]

    print(">>> Đang vẽ biểu đồ phân bố...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Vẽ biểu đồ histogram
    ax.hist(similarity_values, bins=50, range=(0.8, 1.0), color='skyblue', edgecolor='black', alpha=0.7)
    
    # Vẽ đường thẳng đứng tại ngưỡng đã chọn
    ax.axvline(similarity_threshold, color='red', linestyle='--', linewidth=2, label=f'Ngưỡng xóa: {similarity_threshold}')
    
    # Đếm số cặp có độ tương đồng cao hơn ngưỡng
    high_similarity_pairs = np.sum(similarity_values >= similarity_threshold)
    
    # Đặt tiêu đề và nhãn
    ax.set_title(f'Phân bố Độ tương đồng Cosine giữa các Ảnh\n(Tìm thấy {high_similarity_pairs} cặp >= ngưỡng)', fontsize=16)
    ax.set_xlabel('Độ tương đồng Cosine', fontsize=12)
    ax.set_ylabel('Số lượng cặp ảnh', fontsize=12)
    ax.legend()
    
    # Lưu biểu đồ
    os.makedirs(os.path.dirname(output_figure_path), exist_ok=True)
    plt.savefig(output_figure_path, dpi=300, bbox_inches='tight')
    
    print(f"\n>>> Biểu đồ đã được lưu tại: {output_figure_path} <<<")
    plt.show()


if __name__ == '__main__':
    # --- CẤU HÌNH CÁC THAM SỐ TẠI ĐÂY ---
    IMAGES_DIRECTORY = "data/interim/merged_fighter_jets/images"
    OUTPUT_FIGURE_PATH = "reports/figures/similarity_histogram.png"
    # Ngưỡng tương đồng để vẽ đường tham chiếu
    SIMILARITY_THRESHOLD = 0.99

    visualize_similarity_distribution(
        images_dir=IMAGES_DIRECTORY,
        output_figure_path=OUTPUT_FIGURE_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD
    ) 