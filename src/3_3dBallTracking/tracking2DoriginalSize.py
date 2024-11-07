import os
import cv2
import sys
import torch
import pickle
import random
import numpy as np
from ultralytics import YOLO

# Add the parent directory to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

# Now you can import the utils module from the parent directory
from utils.utils import *
from utils.config import *
from utils.particleFilter import *

# Action frame ranges
ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),            # bad 
    5: (3770, 3990),            
    6: (4450, 4600)             # bad
}

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)
model = YOLO(pathWeight)

# Select the device to use (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

YOLO_INPUT_SIZE = 800  # Size for YOLO model input
DISTANCE_THRESHOLD = 800  # Threshold distance to detect a new ball

def applyModel(frame, model):
    height, width = frame.shape[:2]
    
    # Resize for YOLO model
    frameResized = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    
    results = model.track(frameResized, verbose=False, device=device)
    
    center_ret = (-1, -1)
    confidence = -1
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        # Scale back to original size
        x1 = x1 * width / YOLO_INPUT_SIZE
        y1 = y1 * height / YOLO_INPUT_SIZE
        x2 = x2 * width / YOLO_INPUT_SIZE
        y2 = y2 * height / YOLO_INPUT_SIZE
        confidence = box.conf[0]
        class_id = box.cls[0]

        if class_id == 0 and confidence > 0.35:
            x_center = (x1 + x2) / 2    
            y_center = (y1 + y2) / 2
            center_ret = (int(x_center), int(y_center))
            detections.append(center_ret)
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    return detections, center_ret, confidence

def testModel(num_cam, action):
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameraInfos)
    videoCapture = cv2.VideoCapture(pathVideo)

    # Get video dimensions
    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    START, END = ACTIONS[action]
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, START)

    trackers = []
    trajectory_points = []

    while True:
        current_frame = int(videoCapture.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > END:
            break

        ret, frame = videoCapture.read()
        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)
        detections, center_ret, confidence = applyModel(frameUndistorted, model)

        new_trackers = []
        for detection in detections:
            # Convert detection to numpy array
            detection_np = np.array(detection)

            matched = False
            for tracker in trackers:
                if tracker.last_position is not None:  # Check if last_position is not None
                    distance = np.linalg.norm(np.array(tracker.last_position) - detection_np)
                    if distance < DISTANCE_THRESHOLD:
                        tracker.update(detection_np)  # Pass the numpy array
                        matched = True
                        break

            if not matched:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_tracker = ParticleFilterBallTracker(len(trackers), color, frame_size)
                new_tracker.update(detection_np)  # Pass the numpy array
                new_trackers.append(new_tracker)

        trackers.extend(new_trackers)

        for tracker in trackers:
            if tracker.last_position is not None:  # Check if last_position is not None
                trajectory_points.append(tracker.last_position)
            tracker.update(tracker.last_position)
            tracker.predict()
            tracker.draw_particles(frameUndistorted)
            tracker.draw_estimated_position(frameUndistorted)
            tracker.draw_trajectory(frameUndistorted)


        cv2.imshow('Frame', frameUndistorted)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()
    return trajectory_points


def load_existing_results(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

if __name__ == '__main__':

    cam = int(input("Enter camera number: "))
    while cam not in VALID_CAMERA_NUMBERS:
        print("Invalid camera number.")
        cam = int(input("Enter camera number: "))
        
    action = int(input("Enter action number: "))
    while action not in ACTIONS:
        print("Invalid action number.")
        action = int(input("Enter action number: "))
    
    print(f"Processing Camera {cam}, Action {action}...")
    trajectory = testModel(cam, action)
    
