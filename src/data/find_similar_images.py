import os
import imagehash
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def find_similar_images(images_dir, hash_size=8, similarity_threshold=5, num_samples_to_show=5):
    """
    Tìm và hiển thị các cặp ảnh tương đồng trong một thư mục bằng cách sử dụng Perceptual Hashing.

    Args:
        images_dir (str): Đường dẫn đến thư mục chứa các file ảnh.
        hash_size (int): Kích thước của hash (số lớn hơn cho độ chính xác cao hơn nhưng chậm hơn).
        similarity_threshold (int): Ngưỡng khoảng cách Hamming. Các cặp ảnh có khoảng cách nhỏ hơn
                                    hoặc bằng ngưỡng này sẽ được coi là tương đồng.
        num_samples_to_show (int): Số lượng cặp tương đồng ngẫu nhiên cần hiển thị.
    """
    # --- Bước 1: Tính toán hash cho tất cả các ảnh ---
    print(f">>> Bắt đầu tính toán hash cho các ảnh trong: {images_dir}")
    if not os.path.exists(images_dir):
        print(f"Lỗi: Không tìm thấy thư mục ảnh: {images_dir}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    hashes = {}
    
    for filename in tqdm(image_files, desc="Đang tính toán hash"):
        try:
            image_path = os.path.join(images_dir, filename)
            image = Image.open(image_path)
            # Sử dụng average hash, một lựa chọn cân bằng giữa tốc độ và độ chính xác
            h = imagehash.average_hash(image, hash_size=hash_size)
            hashes[filename] = h
        except Exception as e:
            print(f"\nCảnh báo: Không thể xử lý file {filename}. Lỗi: {e}")
            continue
    
    print(f"\n>>> Đã tính toán xong hash cho {len(hashes)} ảnh.")

    # --- Bước 2: Tìm các cặp ảnh tương đồng ---
    print(f">>> Đang tìm kiếm các cặp tương đồng với ngưỡng khoảng cách <= {similarity_threshold}...")
    similar_pairs = []
    filenames = list(hashes.keys())
    
    for i in tqdm(range(len(filenames)), desc="Đang so sánh hash"):
        for j in range(i + 1, len(filenames)):
            file1 = filenames[i]
            file2 = filenames[j]
            
            # Tính khoảng cách Hamming
            distance = hashes[file1] - hashes[file2]
            
            if distance <= similarity_threshold:
                similar_pairs.append((file1, file2, distance))
    
    print(f"\n>>> Tìm thấy {len(similar_pairs)} cặp ảnh tương đồng.")

    # --- Bước 3: Hiển thị một số mẫu ngẫu nhiên ---
    if not similar_pairs:
        print("Không tìm thấy cặp ảnh nào tương đồng để hiển thị.")
        return

    if len(similar_pairs) < num_samples_to_show:
        num_samples_to_show = len(similar_pairs)
    
    random_pairs_to_show = random.sample(similar_pairs, num_samples_to_show)

    print(f"\n>>> Đang hiển thị {num_samples_to_show} cặp ảnh tương đồng ngẫu nhiên... <<<")
    
    for file1, file2, distance in random_pairs_to_show:
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
        
        fig.suptitle(f'Cặp tương đồng - Khoảng cách Hamming: {distance}', fontsize=16)
        plt.show()

if __name__ == '__main__':
    # --- CẤU HÌNH CÁC THAM SỐ TẠI ĐÂY ---

    # 1. Thư mục chứa các ảnh cần được kiểm tra.
    IMAGES_DIRECTORY_TO_CHECK = "data/interim/merged_fighter_jets/images"

    # 2. Ngưỡng tương đồng. Khoảng cách Hamming càng nhỏ, ảnh càng giống.
    #    - 0: Gần như trùng lặp hoàn toàn.
    #    - 1-5: Rất tương đồng.
    #    - > 10: Khác nhau.
    SIMILARITY_THRESHOLD = 0

    # 3. Số lượng cặp ảnh ngẫu nhiên muốn hiển thị kết quả.
    NUM_SAMPLES_TO_SHOW = 5

    # --- KẾT THÚC PHẦN CẤU HÌNH ---

    find_similar_images(
        images_dir=IMAGES_DIRECTORY_TO_CHECK,
        similarity_threshold=SIMILARITY_THRESHOLD,
        num_samples_to_show=NUM_SAMPLES_TO_SHOW
    ) 