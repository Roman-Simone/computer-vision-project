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

# Action frame ranges
ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600),
    7: (5150, 5330)
}

# {
#     '1' : {     # first camera
#         '1' : {frame1: (x1, y1), frame2 : (x2, y2), ...},  # first action, a point for each frame (or 0 points if no balls detected)
#         '2' : [...] 
#         ....
#     }, 
#     ...
# }


# Ask user to select an action (1-8)
# action = int(input("Select the action to process (1-7): "))
# if action not in ACTIONS:
#     print("Invalid action selected. Please choose between 1 and 7.")
#     exit()

# Set START and END based on the action chosen
# START, END = ACTIONS[action]

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

size = 800
DISTANCE_THRESHOLD = 200  # Define a threshold distance to detect a new ball


def applyModel(frame, model):
    results = model.track(frame, save=False, verbose=False, device=device)
    
    center_ret = (-1, -1)
    confidence = -1
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]

        if class_id == 0 and confidence > 0.5:
            x_center = (x1 + x2) / 2    
            y_center = (y1 + y2) / 2
            center_ret = (int(x_center), int(y_center))
            detections.append(center_ret)
            
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    return detections, center_ret, confidence


def testModel(num_cam, action):
    """Process the video for the given camera and action, return trajectory points"""
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameraInfos)
    videoCapture = cv2.VideoCapture(pathVideo)

    START, END = ACTIONS[action]  # Set frame range based on the action
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
        frameUndistorted = cv2.resize(frameUndistorted, (size, size))
        detections, center, confidence = applyModel(frameUndistorted, model)

        new_trackers = []
        for detection in detections:
            matched = False
            for tracker in trackers:
                distance = np.linalg.norm(np.array(tracker.last_position) - np.array(detection)) if tracker.last_position else float('inf')
                if distance < DISTANCE_THRESHOLD:
                    tracker.update(detection)
                    matched = True
                    break

            if not matched:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_tracker = ParticleFilterBallTracker(len(trackers), color)
                new_tracker.update(detection)
                new_trackers.append(new_tracker)

        trackers.extend(new_trackers)

        for tracker in trackers:
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
    """Helper function to load existing results from a pickle file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

if __name__ == '__main__':
    pickle_file = 'ball_trajectories.pkl'
    results = load_existing_results(pickle_file)  # Load existing data if available

    
    while cam not in VALID_CAMERA_NUMBERS: 
        cam = input("Enter the camera number to process (1, 2, 3, 4, 5, 6, 7, 8, 12, 13):")

    while action not in ACTIONS:
        action = input("Enter the action number to process (1, 2, 3, 4, 5, 6, 7):")
    
   

    print(f"Processing complete. Results saved in {pickle_file}")
