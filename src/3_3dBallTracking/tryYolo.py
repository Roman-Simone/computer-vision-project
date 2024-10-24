import re
import os
import cv2
import sys
import torch
from ultralytics import YOLO

# Add the parent directory to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

# Now you can import the utils module from the parent directory
from utils.utils import *
from utils.config import *


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

size = 800

def applyModel(frame, model):

    originalSizeHeight, originalSizeWidth, _ = frame.shape

    frameResized = cv2.resize(frame, (size, size))
    
    results = model.track(frameResized, verbose=False, device=device)
    
    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = x1 * originalSizeWidth / size
        y1 = y1 * originalSizeHeight / size
        x2 = x2 * originalSizeWidth / size
        y2 = y2 * originalSizeHeight / size

        confidence = box.conf[0]
        class_id = box.cls[0]

        if confidence < 0.5:
            continue

        # Draw the bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Prepare the confidence label
        label = f'{confidence:.2f}'

        # Determine position for the label (slightly above the top-left corner of the bbox)
        label_position = (int(x1), int(y1) - 10)

        # Add the confidence score text
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #center of the bounding box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        #draw the center of the bounding box
        cv2.circle(frame, (int(x_center), int(y_center)), 3, (0, 255, 0), -1)

    return frame 

def testModel():
    videos = find_files(PATH_VIDEOS)
    videos.sort()
    cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

    for video in videos:
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])
        
        if numero_camera not in VALID_CAMERA_NUMBERS:
            continue

        cameraInfo, _ = take_info_camera(numero_camera, cameraInfos)

        pathVideo = os.path.join(PATH_VIDEOS, video)

        videoCapture = cv2.VideoCapture(pathVideo)

        while True:
            ret, frame = videoCapture.read()

            if not ret:
                break

            frameUndistorted = undistorted(frame, cameraInfo)

            frameWithBbox = applyModel(frameUndistorted, model)

            cv2.imshow('Frame', frameWithBbox)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                break
        videoCapture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    testModel()