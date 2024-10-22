from config import *
from utils import *
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import tqdm
import json

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600),
    7: (5150, 5330)
}

# Determine device to use for PyTorch
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

SIZE = 800

cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)
pathDetections = os.path.join(PATH_DETECTIONS, 'detections.pkl')
camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
yolo_model = YOLO(os.path.join(PATH_WEIGHT, 'best_v11_800.pt'))

cam = [take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS]

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

def process_video(yolo_model, video_path, cam_id, END):
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


def main():
    try:
        action_id = int(input(f"Select an action from the available actions [1, 2, 3, 4, 5, 6, 7] : "))
        if action_id not in ACTIONS:
            print("Invalid action selected. Exiting.")
            exit()
    except ValueError:
        print("Invalid input. Please enter a number corresponding to the action.")
        exit()
    
    START, END = ACTIONS[action_id]
    
    video_paths = {num_cam: f'{PATH_VIDEOS}/out{num_cam}.mp4' for num_cam in VALID_CAMERA_NUMBERS}

    for video_path in video_paths.values():
        cap = cv2.VideoCapture(video_path)
    
        if not cap.isOpened():
            print("Errore nell'apertura del video")
            return
        
        # Set the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, START)
        
        frame_idx = START
        # paused = False  

        while cap.isOpened() and frame_idx <= END:
            # if not paused:
            ret, frame = cap.read()
            
            if not ret:
                break
                        
            frame_idx += 1

            all_detections = load_pickle(pathDetections)

            for cam_id, video_path in video_paths.items():
                if cam_id not in VALID_CAMERA_NUMBERS:
                    continue

                print(f"Processing camera {cam_id}...")
                    
                countDetection = sum(1 for key in all_detections.keys() if key[0] == cam_id)
                
                if countDetection >= END - 1:
                    print(f"Camera {cam_id} already processed. Skipping...")
                    continue
                    
                detections = process_video(yolo_model, video_path, cam_id, END)
                all_detections.update(detections)
                save_pickle(all_detections, pathDetections)

                print(f"Detections for camera {cam_id} saved to detections.pkl")
            
            return all_detections
                    
            key = cv2.waitKey(25) & 0xFF  # Wait for 25 ms for a key press
            
            if key == ord('q'):  # Quit on 'q' key
                break
            # elif key == ord(' '):  # Toggle pause/play on spacebar press
            #     paused = not paused  # Toggle the paused flag

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
