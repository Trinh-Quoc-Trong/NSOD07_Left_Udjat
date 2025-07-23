import cv2
import argparse
import os

def process_video(
    input_path: str,
    output_path: str,
    start_time_sec: float = 0.0,
    end_time_sec: float = -1.0,
    target_fps: float = -1.0,
    speed_factor: float = 1.0
):
    """
    Cắt video, điều chỉnh FPS, và thay đổi tốc độ phát của video.

    Args:
        input_path (str): Đường dẫn đến video đầu vào.
        output_path (str): Đường dẫn để lưu video đầu ra.
        start_time_sec (float): Thời gian bắt đầu cắt (giây). Mặc định là 0.
        end_time_sec (float): Thời gian kết thúc cắt (giây). Mặc định là đến hết video (-1.0).
        target_fps (float): FPS mong muốn của video đầu ra. Mặc định giữ nguyên (-1.0).
        speed_factor (float): Hệ số thay đổi tốc độ phát (ví dụ: 0.5 để giảm một nửa, 2.0 để tăng gấp đôi). Mặc định là 1.0 (tốc độ gốc).
    """
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy video đầu vào tại: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video: {input_path}")
        return

    # Lấy thông tin video gốc
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tính toán thời gian bắt đầu và kết thúc theo frame
    start_frame = int(start_time_sec * original_fps)
    end_frame = int(end_time_sec * original_fps) if end_time_sec != -1.0 else total_frames

    if start_frame < 0:
        start_frame = 0
    if end_frame > total_frames or end_frame == -1:
        end_frame = total_frames
    if start_frame >= end_frame:
        print("Lỗi: Thời gian bắt đầu phải nhỏ hơn thời gian kết thúc hoặc video quá ngắn.")
        cap.release()
        return

    # Tính toán FPS đầu ra thực tế
    effective_base_fps = original_fps
    if target_fps != -1.0:
        effective_base_fps = target_fps

    output_fps = effective_base_fps * speed_factor

    # Đảm bảo FPS đầu ra hợp lệ
    if output_fps <= 0:
        print("Lỗi: FPS đầu ra tính toán không hợp lệ (nhỏ hơn hoặc bằng 0). Vui lòng kiểm tra lại tham số tốc độ hoặc FPS.")
        cap.release()
        return

    # Thiết lập VideoWriter
    # Sử dụng codec MP4V cho định dạng .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Lỗi: Không thể tạo file video đầu ra: {output_path}. Đảm bảo đường dẫn hợp lệ và codec được hỗ trợ.")
        cap.release()
        return

    print(f"Đang xử lý video '{input_path}'...")
    print(f"Original FPS: {original_fps:.2f}")
    if target_fps != -1.0:
        print(f"Target FPS (base for speed): {target_fps:.2f}")
    print(f"Playback Speed Factor: x{speed_factor:.2f}")
    print(f"Calculated Output FPS: {output_fps:.2f}")
    print(f"Cắt từ frame {start_frame} đến {end_frame} (thời gian: {start_time_sec:.2f}s đến {end_time_sec if end_time_sec != -1.0 else 'end'}s)")

    # Đặt con trỏ video về frame bắt đầu
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame_idx = start_frame
    while cap.isOpened() and current_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Viết frame vào video đầu ra
        out.write(frame)

        current_frame_idx += 1

    cap.release()
    out.release()
    print(f"Video đã xử lý được lưu tại: {output_path}")

if __name__ == "__main__":
    # Các tham số được định nghĩa sẵn trong code
    input_video_path = r"runs\detect\f22_detection_result4\skills_f22_edited.avi"
    output_video_path = r"runs\detect\f22_detection_result4\skills_f22_edited_2.avi"
    start_clip_time = 0
    end_clip_time = 20
    target_output_fps = 40.0
    playback_speed_factor = 1

    process_video(
        input_path=input_video_path,
        output_path=output_video_path,
        start_time_sec=start_clip_time,
        end_time_sec=end_clip_time,
        target_fps=target_output_fps,
        speed_factor=playback_speed_factor
    ) 