import imageio
import os

def convert_video_to_gif(
    input_video_path: str,
    output_gif_path: str,
    fps: int = 10, # FPS cho GIF
    size: tuple = None # Kích thước (width, height) cho GIF, ví dụ: (320, 240). None để giữ nguyên kích thước gốc.
):
    """
    Chuyển đổi một file video thành file GIF bằng Imageio.

    Args:
        input_video_path (str): Đường dẫn đến file video đầu vào.
        output_gif_path (str): Đường dẫn để lưu file GIF đầu ra.
        fps (int): Số khung hình mỗi giây cho GIF. Mặc định là 10.
        size (tuple): Kích thước (width, height) của GIF. Mặc định là None (giữ nguyên kích thước gốc).
    """
    if not os.path.exists(input_video_path):
        print(f"Lỗi: Không tìm thấy video đầu vào tại: {input_video_path}")
        return

    # Đọc video
    try:
        reader = imageio.get_reader(input_video_path)
    except Exception as e:
        print(f"Lỗi khi đọc video: {e}")
        return

    # Lấy thông tin video gốc
    meta = reader.get_meta_data()
    original_fps = meta.get('fps', 30) # Mặc định 30 nếu không tìm thấy
    original_size = meta.get('size')

    # Thiết lập kích thước đầu ra
    output_size = original_size
    if size:
        output_size = size

    print(f"Đang chuyển đổi video '{input_video_path}' sang GIF...")
    print(f"Output GIF FPS: {fps}")
    if output_size:
        print(f"Output GIF Size: {output_size[0]}x{output_size[1]}")

    # Tạo writer cho GIF
    try:
        writer = imageio.get_writer(output_gif_path, fps=fps, mode='I') # mode='I' cho video to GIF
    except Exception as e:
        print(f"Lỗi khi tạo file GIF: {e}")
        reader.close()
        return

    # Lặp qua các frame và ghi vào GIF
    try:
        for i, frame in enumerate(reader):
            # Nếu cần resize frame
            if output_size and frame.shape[0:2] != output_size[::-1]: # OpenCV uses (height, width), imageio might use (width, height)
                # Resize chỉ khi kích thước thay đổi
                import cv2
                frame = cv2.resize(frame, output_size)

            writer.append_data(frame)
            # In tiến độ
            if i % (original_fps * 5) == 0: # In mỗi 5 giây video
                print(f"  Đã xử lý {i} khung hình...")
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý frame: {e}")
    finally:
        reader.close()
        writer.close()

    print(f"Chuyển đổi hoàn tất. GIF đã lưu tại: {output_gif_path}")

if __name__ == "__main__":
    input_video = r"runs\detect\f22_detection_result4\skills_f22_edited_2.avi"
    output_gif = "runs/detect/f22_detection_result4/skills_f22_edited_2.gif"
    
    # Điều chỉnh các tham số nếu cần
    gif_fps = 40  # FPS của GIF đầu ra
    gif_size = (960, 540) # Kích thước của GIF (width, height), hoặc None

    convert_video_to_gif(
        input_video_path=input_video,
        output_gif_path=output_gif,
        fps=gif_fps,
        size=gif_size
    ) 