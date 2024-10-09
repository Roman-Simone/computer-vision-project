import re
import cv2
import torch
from utils import *
from config import *
from ultralytics import YOLO


pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


size = 800
model = YOLO(pathWeight)  

def applyModel(frame, model):
    results = model.track(frame, verbose = False, device=device)
    
    flagResults = False

    for box in results[0].boxes:
        flagResults = True
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]

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

        center_ret = (int(x_center), int(y_center))
    
    if flagResults == False:
        center_ret = (-1, -1)
        confidence = -1


    return frame, center_ret, confidence




def testModel():
    videosCalibration = find_files(PATH_VIDEOS)
    videosCalibration.sort()
    cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

    for video in videosCalibration:
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

            frameUndistorted = cv2.resize(frameUndistorted, (size, 480))

            frameWithBbox, center, confidence = applyModel(frameUndistorted, model)

            print(center, confidence)

            cv2.imshow('Frame', frameWithBbox)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                break
        videoCapture.release()
        cv2.destroyAllWindows()





if __name__ == '__main__':
    testModel()