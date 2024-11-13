import re
import os
import cv2
import sys
import torch
import pickle
from ultralytics import YOLO
from utils3DBallTracking.yoloWindows import yoloWindows

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

CONFIDENCE = 0.4  # confidence threshold for YOLO detection

output_file = os.path.join(PATH_DETECTIONS_04, 'all_detections.pkl')  
pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

def select_regions(frame):
    """
    Allows the user to manually select regions within a video frame to ignore during object detection,
    improving detection accuracy by filtering out irrelevant areas.

    parameters:
        frame (numpy.ndarray): the video frame where regions are selected by the user.

    returns:
        list: a list of tuples, where each tuple represents a selected region defined by
              (x, y, width, height) coordinates.
    """
    
    regions = []
    while True:
        region = cv2.selectROI("Select Region R", frame, showCrosshair=True)
        if region[2] == 0 or region[3] == 0:  
            break
        regions.append(region)
        print(f"Selected region R: {region}")
    cv2.destroyWindow("Select Region R")
    return regions


def inside_ignore_region(x_center, y_center, regions):
    """
    Determines if a given point (x_center, y_center) lies within any of the specified regions,
    which are set to be ignored during detection.

    parameters:
        x_center (float): x-coordinate of the point to be checked.
        y_center (float): y-coordinate of the point to be checked.
        regions (list): list of ignored regions, where each region is defined by (x, y, width, height).

    returns:
        bool: returns true if the point lies within any of the ignored regions; otherwise, returns false.
    """
    
    for region in regions:
        x, y, w, h = region
        if x <= x_center <= x + w and y <= y_center <= y + h:
            return True
    return False


def apply_model(frame, model, regions):
    """
    Applies a YOLO object detection model to a video frame to detect objects,
    filtering out any detections within specified ignored regions.

    parameters:
        frame (numpy.ndarray): the input video frame on which detection is performed.
        model (YOLO): the pre-trained YOLO model used for object detection.
        regions (list): list of ignored regions specified by (x, y, width, height).

    returns:
        tuple:
            - detection_point (tuple or None): coordinates of the detected object's center (x_center, y_center),
              or None if no object is detected.
            - x1 (float), y1 (float), x2 (float), y2 (float): coordinates of the bounding box
              around the detected object. if no object is detected, all coordinates are set to -1.
    """
    
    originalSizeHeight, originalSizeWidth, _ = frame.shape
    frameResized = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    results = model.track(frameResized, verbose=False, device=device)

    detection_point = None
    x1, y1, x2, y2 = -1, -1, -1, -1

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1 = x1 * originalSizeWidth / YOLO_INPUT_SIZE
        y1 = y1 * originalSizeHeight / YOLO_INPUT_SIZE
        x2 = x2 * originalSizeWidth / YOLO_INPUT_SIZE
        y2 = y2 * originalSizeHeight / YOLO_INPUT_SIZE

        confidence = box.conf[0]
        if confidence < CONFIDENCE:
            continue

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        if inside_ignore_region(x_center, y_center, regions):
            continue

        detection_point = (int(x_center), int(y_center))
        break

    return detection_point, x1, x2, y1, y2


def save_detections(detections, file_path):
    """
    Saves detection data to a specified file in pickle (.pkl) format, allowing for persistence of detections.

    parameters:
        detections (dict): the dictionary containing detection data to save.
        file_path (str): the path to the file where detection data will be stored.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(detections, f)


def process_single_action(camera_number, action_id):
    """
    Processes video data for a specific camera and action, applying object detection to frames
    within the action's frame range and saving detected objects. allows user interaction
    to select ignored regions and confirm saving results.

    parameters:
        camera_number (int): the identifier for the camera source of the video.
        action_id (int): the action identifier, determining the frame range to process.

    returns:
        none
    """
    
    if action_id not in ACTIONS:
        print(f"Invalid action ID: {action_id}")
        return

    all_detections = load_existing_detections(output_file)
    
    # dictionary (and pkl file) structure:
    # {
    #     '1' : {     # first camera
    #         '1' : {frame1: (x1, y1), frame2 : (x2, y2), ...},  # first action, a point for each frame (or 0 points if no balls detected)
    #         '2' : [...] 
    #         ....
    #     }, 
    #     ...
    # }
    
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
        if camera_number != 2:
            detection_point, x1, x2, y1, y2 = apply_model(frameUndistorted, model, regions)
            action_detections[frame_idx] = detection_point

            if detection_point:
                # draw bounding box and detection point
                cv2.rectangle(frameUndistorted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frameUndistorted, detection_point, 5, (0, 255, 0), -1)
            cv2.imshow('Frame', frameUndistorted)
            
        else:
            # if camera 2, use yoloWindows because image shape is wider than the other cameras
            detection_point, frame = model.detect(frame, visualizeBBox=True, thresholdConfidence=CONFIDENCE, regions=regions)
            action_detections[frame_idx] = detection_point
            cv2.imshow('Frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

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
    while camera_number not in VALID_CAMERA_NUMBERS:
        camera_number = int(input("[INVALID INPUT] Enter camera number (1-8, 12-13): "))

    if camera_number == 2:
        model = yoloWindows(pathWeight)
    else:
        model = YOLO(pathWeight)

    action_id = int(input("Enter action ID (1-6): "))
    while action_id not in ACTIONS:
        action_id = int(input("[INVALID INPUT] Enter action ID (1-6): "))
    process_single_action(camera_number, action_id)
