import re
import cv2
from utils import *
from config import *
from ultralytics import YOLO

model = YOLO(PATH_WEIGHT)  

def applyModel(frame, model):
    results = model(frame)

    for box in results[0].boxes:
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

    return frame


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

            frameWithBbox = applyModel(frameUndistorted, model)

            cv2.imshow('Frame', frameWithBbox)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                break
        videoCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    testModel()