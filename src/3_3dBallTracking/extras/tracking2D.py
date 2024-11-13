import os
import cv2
import sys
import torch
import random
import numpy as np
from ultralytics import YOLO
from particleFilter2D import ParticleFilter

current_path = os.path.dirname(os.path.abspath(__file__))
grandparent_path = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
sys.path.append(grandparent_path)

from utils.utils import *
from utils.config import *

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)
model = YOLO(pathWeight)

CONFIDENCE = 0.4  # confidence threshold for YOLO detection

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

DISTANCE_THRESHOLD = 800  # threshold distance to associate detections with trackers

def apply_model(frame, model):
    """
    Applies the YOLO model to detect objects in a given frame.

    Parameters:
        frame (numpy.ndarray): the input video frame on which detection is performed.
        model (YOLO): the pre-trained YOLO model used for object detection.

    Returns:
        tuple:
            - detections (list): list of tuples with detected object center coordinates.
            - center_ret (tuple): coordinates of the detected object's center, or (-1, -1) if none detected.
            - confidence (float): confidence score of the detection.
    """
    
    height, width = frame.shape[:2]
    frameResized = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        
    results = model.track(frameResized, verbose=False, device=device)
    
    center_ret = (-1, -1)
    confidence = -1
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        # Scale back to original frame size
        x1, y1, x2, y2 = (
            x1 * width / YOLO_INPUT_SIZE, y1 * height / YOLO_INPUT_SIZE, 
            x2 * width / YOLO_INPUT_SIZE, y2 * height / YOLO_INPUT_SIZE
        )

        confidence = box.conf[0]
        class_id = box.cls[0]

        if class_id == 0 and confidence > CONFIDENCE:  
            x_center = (x1 + x2) / 2    
            y_center = (y1 + y2) / 2
            center_ret = (int(x_center), int(y_center))
            detections.append(center_ret)
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)  

    return detections, center_ret, confidence

    
def test_model(num_cam, action):
    """
    Processes a video from a specific camera and action, using YOLO and particle filters to track objects.

    Parameters:
        num_cam (int): camera number to select video.
        action (int): action identifier to specify start and end frames.

    Returns:
        list: trajectory points recorded during the tracking.
    """
    
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameraInfos)
    videoCapture = cv2.VideoCapture(pathVideo)

    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    START, END = ACTIONS[action]
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, START)

    trackers = []  # list to hold active particle trackers
    trajectory_points = []  

    while True:
        current_frame = int(videoCapture.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > END:
            break

        ret, frame = videoCapture.read()
        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)

        detections, _, _ = apply_model(frameUndistorted, model)

        new_trackers = []
        for detection in detections:
            detection_np = np.array(detection)

            matched = False
            for tracker in trackers:
                if tracker.last_position is not None:
                    distance = np.linalg.norm(np.array(tracker.last_position) - detection_np)
                    if distance < DISTANCE_THRESHOLD:
                        tracker.update(detection_np)  # update tracker if within distance threshold
                        matched = True
                        break

            # If no match, initialize a new tracker for this detection
            if not matched:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  
                new_tracker = ParticleFilter(len(trackers), color, frame_size)
                new_tracker.update(detection_np)
                new_trackers.append(new_tracker)

        trackers.extend(new_trackers)  # add new trackers to the main tracker list

        for tracker in trackers:
            if tracker.last_position is not None:
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
    test_model(cam, action)
