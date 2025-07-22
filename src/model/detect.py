import torch
from ultralytics import YOLO

def detect_on_test_data():
    """
    Runs YOLOv11 detection on the test dataset using a pretrained model.
    """
    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the pretrained YOLOv11 nano model
    # This will automatically download yolov11n.pt if not present
    model = YOLO('yolov11n.pt')
    model.to(device)

    # Path to the test images
    test_images_path = 'data/processed/test/images'

    # Run detection
    # show=True will display the results in a window
    # save=True will save the images with bounding boxes
    results = model.predict(source=test_images_path, show=True, save=True, project='runs/detect', name='yolov11n_pretrained_test')

    print("Detection complete.")
    # The save directory can be accessed from the results, but it's consistent for all images in the run.
    # The path is usually in runs/detect/experiment_name/
    print(f"Results saved in 'runs/detect/yolov11n_pretrained_test' directory.")

if __name__ == '__main__':
    detect_on_test_data() 