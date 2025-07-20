import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_coco_dataset(json_path, output_dir):
    """
    Phân tích và trực quan hóa các đặc điểm của một bộ dữ liệu COCO.

    Args:
        json_path (str): Đường dẫn đến file _annotations.coco.json.
        output_dir (str): Đường dẫn đến thư mục để lưu các biểu đồ.
    """
    print(f">>> Bắt đầu phân tích file: {json_path}")
    if not os.path.exists(json_path):
        print(f"Lỗi: Không tìm thấy file JSON tại: {json_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # --- 1. Chuyển đổi dữ liệu sang Pandas DataFrame để dễ dàng phân tích ---
    print(">>> Đang chuyển đổi dữ liệu sang DataFrame...")
    df_annotations = pd.DataFrame(data['annotations'])
    df_images = pd.DataFrame(data['images'])
    df_categories = pd.DataFrame(data['categories'])

    # Gộp các DataFrame để có thông tin đầy đủ
    df = pd.merge(df_annotations, df_categories, left_on='category_id', right_on='id', suffixes=('', '_cat'))
    df = pd.merge(df, df_images, left_on='image_id', right_on='id', suffixes=('', '_img'))
    
    # --- 2. Biểu đồ 1: Phân bố số lượng Chú thích mỗi Lớp (Cải tiến) ---
    print("Đang vẽ biểu đồ: Phân bố số lượng mỗi lớp...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Lấy số lượng của từng lớp để sắp xếp và thêm nhãn
    class_counts = df['name'].value_counts()
    
    # Vẽ biểu đồ
    sns.countplot(y='name', data=df, order=class_counts.index, palette='viridis', ax=ax)
    
    # Thêm số lượng cụ thể vào cuối mỗi cột
    for i, count in enumerate(class_counts):
        # Đặt vị trí văn bản ngay bên phải cột
        ax.text(count + (0.01 * class_counts.max()), i, str(count), ha='left', va='center', fontsize=10, color='black')

    # Thêm đường trung bình để dễ so sánh
    mean_count = class_counts.mean()
    ax.axvline(mean_count, color='r', linestyle='--', label=f'Trung bình: {mean_count:.1f}')

    ax.set_title('Phân bố số lượng Chú thích mỗi Lớp', fontsize=18)
    ax.set_xlabel('Số lượng Chú thích (Bounding Box)', fontsize=14)
    ax.set_ylabel('Tên Lớp', fontsize=14)
    # Bỏ thang đo log để trực quan hơn
    # plt.xscale('log') 
    
    # Tự động điều chỉnh giới hạn trục x để có không gian cho nhãn
    ax.set_xlim(right=class_counts.max() * 1.15)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_class_distribution.png'), dpi=300)
    plt.close()

    # Thêm các cột tính toán để phân tích bbox
    df['bbox_x'] = df['bbox'].apply(lambda x: x[0])
    df['bbox_y'] = df['bbox'].apply(lambda x: x[1])
    df['bbox_w'] = df['bbox'].apply(lambda x: x[2])
    df['bbox_h'] = df['bbox'].apply(lambda x: x[3])
    df['bbox_area'] = df['bbox_w'] * df['bbox_h']
    # Chuẩn hóa diện tích bbox theo diện tích ảnh
    df['normalized_area'] = df['bbox_area'] / (df['width'] * df['height'])
    
    # --- 3. Biểu đồ 2: Phân bố Diện tích Bounding Box (Đã chuẩn hóa) ---
    print("Đang vẽ biểu đồ: Phân bố diện tích Bounding Box...")
    plt.figure(figsize=(12, 7))
    sns.histplot(df['normalized_area'], bins=50, kde=True, color='skyblue')
    plt.title('Phân bố Diện tích Bounding Box (Đã chuẩn hóa)', fontsize=16)
    plt.xlabel('Diện tích tương đối so với ảnh (width*height)', fontsize=12)
    plt.ylabel('Số lượng', fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_bbox_area_distribution.png'), dpi=300)
    plt.close()

    # --- 4. Biểu đồ 3: Phân bố Tỉ lệ khung hình Bounding Box ---
    print("Đang vẽ biểu đồ: Phân bố tỉ lệ khung hình Bounding Box...")
    df['aspect_ratio'] = df['bbox_w'] / df['bbox_h']
    plt.figure(figsize=(12, 7))
    sns.histplot(df['aspect_ratio'], bins=50, kde=True, color='salmon')
    plt.title('Phân bố Tỉ lệ khung hình Bounding Box (width / height)', fontsize=16)
    plt.xlabel('Tỉ lệ khung hình', fontsize=12)
    plt.ylabel('Số lượng', fontsize=12)
    plt.xlim(0, 5) # Giới hạn trục x để dễ xem hơn
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_bbox_aspect_ratio_distribution.png'), dpi=300)
    plt.close()
    
    # --- 5. Biểu đồ 4: Phân bố số lượng Chú thích mỗi Ảnh ---
    print("Đang vẽ biểu đồ: Phân bố số lượng chú thích mỗi ảnh...")
    annotations_per_image = df.groupby('image_id').size()
    plt.figure(figsize=(12, 7))
    sns.histplot(annotations_per_image, bins=max(1, annotations_per_image.max()), discrete=True, color='gold')
    plt.title('Phân bố số lượng Chú thích trên mỗi Ảnh', fontsize=16)
    plt.xlabel('Số lượng Chú thích (Bounding Box)', fontsize=12)
    plt.ylabel('Số lượng Ảnh', fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_annotations_per_image_distribution.png'), dpi=300)
    plt.close()

    # --- 6. Biểu đồ 5: Biểu đồ tâm của Bounding Box (Heatmap) ---
    print("Đang vẽ biểu đồ: Heatmap vị trí tâm Bounding Box...")
    # Tính toán tọa độ tâm đã chuẩn hóa
    df['center_x_norm'] = (df['bbox_x'] + df['bbox_w'] / 2) / df['width']
    df['center_y_norm'] = (df['bbox_y'] + df['bbox_h'] / 2) / df['height']
    plt.figure(figsize=(10, 10))
    # Sử dụng jointplot để hiển thị cả heatmap và histogram
    sns.jointplot(x='center_x_norm', y='center_y_norm', data=df, kind='hex', cmap='viridis', height=10)
    plt.suptitle('Phân bố vị trí tâm Bounding Box trong Ảnh', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_bbox_center_heatmap.png'), dpi=300)
    plt.close()

    print("\n>>> Hoàn tất! Tất cả các biểu đồ đã được lưu vào thư mục 'reports/figures'. <<<")


if __name__ == '__main__':
    # --- CẤU HÌNH ---
    JSON_FILE_TO_ANALYZE = "data/interim/merged_fighter_jets/_annotations.coco.json"
    OUTPUT_FIGURES_DIR = "reports/figures"
    
    analyze_coco_dataset(
        json_path=JSON_FILE_TO_ANALYZE,
        output_dir=OUTPUT_FIGURES_DIR
    ) 