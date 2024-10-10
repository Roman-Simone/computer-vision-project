import pickle
import cv2
import torch
from config import *
from utils import *
from ultralytics import YOLO
import numpy as np

cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)
SIZE = 800
END = 5100  

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

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
    
    while cap.isOpened() and frame_idx < END:  
        ret, frame = cap.read()
        if not ret:
            break
        
        cameraInfo, _ = take_info_camera(cam_id, cameraInfos)
        volleyball_detections = detect_balls(yolo_model, frame, cameraInfo)
        detections[(cam_id, frame_idx)] = volleyball_detections
        
        print("(", cam_id, ", ", frame_idx, ") : ", detections[(cam_id, frame_idx)], "\n")
        frame_idx += 1
    
    cap.release()
    return detections

def process_all_cameras(yolo_model, video_paths):
    all_detections = {}
    
    for cam_id, video_path in video_paths.items():
        if cam_id > 2:
            print(f"Processing camera {cam_id}...")
            detections = process_video(yolo_model, video_path, cam_id)
            all_detections.update(detections)


            # il formato con cui vengono salvati i dati nel pkl è: 
            #           (camera_id, frame_idx) : [(x_center, y_center), ...]        
            
            with open(os.path.join(PATH_DETECTIONS, "detections.pkl"), "wb") as f:
                pickle.dump(all_detections, f)
            
            print(f"Detections for camera {cam_id} saved to detections.pkl")
    
    return all_detections

if __name__ == "__main__":
    video_paths = { num_cam : f'{PATH_VIDEOS}/out{num_cam}.mp4' for num_cam in VALID_CAMERA_NUMBERS}
    
    all_detections = process_all_cameras(yolo_model, video_paths)
    
    print("All detections processed and saved.")
