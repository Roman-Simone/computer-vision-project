import os
import cv2
import sys
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Add the parent directory to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

# Now you can import the utils module from the parent directory
from utils.utils import *
from utils.config import *

# Load calibration matrix
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)
SIZE = 800
END = 5100  

# Determine device to use for PyTorch
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

# Load YOLO model
pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
yolo_model = YOLO(pathWeight)  

def get_ball_center(bbox):
    x_min, y_min, x_max, y_max = bbox.xyxy[0]
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center

def detect_balls(yolo_model, frame, cameraInfo):
    frameUndistorted = undistorted(frame, cameraInfo)
    frameUndistorted = cv2.resize(frameUndistorted, (SIZE, SIZE))
    
    results = yolo_model(frameUndistorted, verbose=False, device=device)
    
    volleyball_detections = []
    
    for bbox in results[0].boxes:
        x_center, y_center = get_ball_center(bbox)
        volleyball_detections.append((int(x_center), int(y_center)))  # cast from tensor to int
    
    return volleyball_detections

def process_video(yolo_model, video_path, cam_id):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    detections = {}

    # Estimate the total number of frames if needed (cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if END is None else END

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened() and frame_idx < END:  
            ret, frame = cap.read()
            if not ret:
                break
            
            cameraInfo, _ = take_info_camera(cam_id, cameraInfos)
            volleyball_detections = detect_balls(yolo_model, frame, cameraInfo)
            detections[(cam_id, frame_idx)] = volleyball_detections
            
            frame_idx += 1
            pbar.update(1)

    cap.release()
    return detections

def process_all_cameras(yolo_model, video_paths):
    pathDetections = os.path.join(PATH_DETECTIONS, "detections.pkl")
    all_detections = load_pickle(pathDetections)
    
    for cam_id, video_path in video_paths.items():
        if cam_id not in VALID_CAMERA_NUMBERS:
            continue

        print(f"Processing camera {cam_id}...")
            
        countDetection = sum(1 for key in all_detections.keys() if key[0] == cam_id)
        
        if countDetection >= END - 1:
            print(f"Camera {cam_id} already processed. Skipping...")
            continue
            
        detections = process_video(yolo_model, video_path, cam_id)
        all_detections.update(detections)
        save_pickle(all_detections, pathDetections)

        print(f"Detections for camera {cam_id} saved to detections.pkl")
    
    return all_detections

def display_detections(detections, video_path, cam_id):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (SIZE, SIZE))
        
        if (cam_id, frame_idx) in detections:
            for (x, y) in detections[(cam_id, frame_idx)]:
                cv2.circle(frame_resized, (x, y), 10, (0, 255, 0), -1)  # Green circle
                cv2.putText(frame_resized, f'({x}, {y})', (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

        # Display the resized frame with detections
        cv2.namedWindow(f'Camera {cam_id} - Frame {frame_idx}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'Camera {cam_id} - Frame {frame_idx}', frame_resized)
        
        # Wait for key input
        key = cv2.waitKey(0)
        if key == ord('n'):  # 'n' to skip to next frame
            frame_idx += 1
        elif key == ord('q'):  # 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_paths = {num_cam: f'{PATH_VIDEOS}/out{num_cam}.mp4' for num_cam in VALID_CAMERA_NUMBERS}
    
    all_detections = process_all_cameras(yolo_model, video_paths)
    
    # Display detections for camera 6 by default
    selected_camera = 6
    if selected_camera in VALID_CAMERA_NUMBERS:
        display_detections(all_detections, video_paths[selected_camera], selected_camera)
    else:
        print(f"Camera {selected_camera} is not a valid camera number.")

    print("All detections processed and saved.")
