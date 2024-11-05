import re
import os
import cv2
import sys
import torch
import pickle
# from ultralytics import YOLO
from  yoloWindows import yoloWindow

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

CONFIDENCE = 0.3

# DICTIONARY (and pkl file) structure:
# {
#     '1' : {     # first camera
#         '1' : {frame1: (x1, y1), frame2 : (x2, y2), ...},  # first action, a point for each frame (or 0 points if no balls detected)
#         '2' : [...] 
#         ....
#     }, 
#     ...
# }

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)
model = yoloWindow(pathWeight)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

SIZE = 800

ACTIONS = {
    1: (48, 230),               # 182 frames
    2: (1050, 1230),            # 180 
    3: (1850, 2060),            # 210
    4: (2620, 2790),            # 170
    5: (3770, 3990),            # 220
    6: (4450, 4600)             # 150
}

output_file = os.path.join(PATH_DETECTIONS_WINDOW_05, 'all_detections.pkl')

def load_existing_detections(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

def select_regions(frame):
    """At the begining of the action, select regions to ignore for detection."""
    regions = []
    while True:
        region = cv2.selectROI("Select Region R", frame, showCrosshair=True)
        if region[2] == 0 or region[3] == 0:  # Check for zero-width or height
            # print("Region selection completed.")
            break
        regions.append(region)
        print(f"Selected region R: {region}")
    cv2.destroyWindow("Select Region R")
    return regions

def applyModel(frame, model, regions):
    
    return model.detect(frame, visualizeBBox=True, thresholdConfidence=CONFIDENCE, regions=regions)

def save_detections(detections, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(detections, f)

def process_single_action(camera_number, action_id):
    if action_id not in ACTIONS:
        print(f"Invalid action ID: {action_id}")
        return

    all_detections = load_existing_detections(output_file)
    
    if str(camera_number) not in all_detections:
        all_detections[str(camera_number)] = {}

    video_filename = f"out{camera_number}.mp4"
    video_path = os.path.join(PATH_VIDEOS, video_filename)
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    cameraInfo, _ = take_info_camera(camera_number, cameraInfos)
    videoCapture = cv2.VideoCapture(video_path)
    
    start_frame, end_frame = ACTIONS[action_id]
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = videoCapture.read()
    
    if not ret:
        print("Failed to read video")
        return
    
    first_frame = undistorted(first_frame, cameraInfo)
    regions = select_regions(first_frame)
    
    action_detections = {}

    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while videoCapture.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        frame_idx = int(videoCapture.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = videoCapture.read()
        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)
        detection_point, frame = applyModel(frameUndistorted, model, regions)
        
        action_detections[frame_idx] = detection_point

        # if detection_point:
        #     cv2.rectangle(frameUndistorted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #     cv2.circle(frameUndistorted, detection_point, 5, (0, 255, 0), -1)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break
    
    videoCapture.release()
    cv2.destroyAllWindows()

    # Ask for confirmation before saving detections
    print(f"Process finished for camera {camera_number}, action {action_id}")
    confirm = input("Save detections to file? (y/n): ").strip().lower()
    if confirm == 'y':
        all_detections[str(camera_number)][str(action_id)] = action_detections
        save_detections(all_detections, output_file)
        print(f"Detections saved for camera {camera_number}, action {action_id}")
    else:
        print("Detections discarded.")

if __name__ == '__main__':
    camera_number = int(input("Enter camera number (1-8, 12-13): "))
    action_id = int(input("Enter action ID (1-6): "))
    process_single_action(camera_number, action_id)
