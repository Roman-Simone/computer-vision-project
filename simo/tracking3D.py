import re
import cv2
from utils import *
from config import *
from ultralytics import YOLO
from tqdm import tqdm  # Import tqdm

pathWeight = os.path.join(PATH_WEIGHT, 'best_v8_800.pt')
model = YOLO(pathWeight, verbose=False)

SIZE = 800
NUMBER_FRAMES = 50

def applyModel(frame, model):
    results = model(frame, verbose=False)

    flagResults = False
    center_ret = []
    confidence_ret = []

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

        # Center of the bounding box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Draw the center of the bounding box
        cv2.circle(frame, (int(x_center), int(y_center)), 3, (0, 255, 0), -1)

        center_ret.append((int(x_center), int(y_center)))
        confidence_ret.append(confidence)
        

    if len(center_ret) > 1:
        max_confidence = max(confidence_ret)
        for pos, elem in enumerate(confidence_ret):
            if elem == max_confidence:
                center_rett = center_ret[pos]
                confidence_rett = max_confidence
                break
    elif len(center_ret) == 1:
        center_rett = center_ret[0]
        confidence_rett = confidence_ret[0]
    else:
        center_rett = (-1, -1)
        confidence_rett = -1

    return frame, center_rett, confidence_rett


def takeBallCoordinates(pathVideo, cameraInfo, model):
    videoCapture = cv2.VideoCapture(pathVideo)

    countFrame = 0
    retPoints = []

    while True:
        ret, frame = videoCapture.read()

        if not ret or countFrame > NUMBER_FRAMES:
            break

        countFrame += 1

        frameUndistorted = undistorted(frame, cameraInfo)
        frameUndistorted = cv2.resize(frameUndistorted, (SIZE, SIZE))

        frameWithBbox, center, confidence = applyModel(frameUndistorted, model)

        element = [center[0], center[1], confidence]
        retPoints.append(element)
        

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

    return retPoints


def takePoints():
    videosCalibration = find_files(PATH_VIDEOS)
    videosCalibration.sort()
    cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

    points = {}

    # Wrap the video processing loop with tqdm to show progress
    for video in tqdm(videosCalibration, desc="Processing videos"):
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])

        if numero_camera not in VALID_CAMERA_NUMBERS:
            continue

        cameraInfo, _ = take_info_camera(numero_camera, cameraInfos)

        pathVideo = os.path.join(PATH_VIDEOS, video)

        points[numero_camera] = takeBallCoordinates(pathVideo, cameraInfo, model)


    print(points)
    save_pickle(points, 'points.pkl')


def triangulate_points(points, camera_matrices):
    # Convert 2D points and camera matrices into suitable format
    points_2d = np.array(points).T  # Transpose to 2xN format
    proj_mats = [np.array(cam_matrix) for cam_matrix in camera_matrices]

    # Perform triangulation
    points_4d = cv2.triangulatePoints(proj_mats[0], proj_mats[1], points_2d[:, 0], points_2d[:, 1])
    
    # Convert homogeneous coordinates (4D) to 3D by dividing by the last component
    points_3d = points_4d[:3] / points_4d[3]
    
    return points_3d.T  # Return as 3xN format


def tracking3D():
    points = load_pickle('points.pkl')
    cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

    pointsForFrame = {}

    for frame in range(NUMBER_FRAMES):
        pointsForFrame[frame] = []

        camera_matrices = []

        for camera, point in points.items():
            if point[frame][0] != -1 and point[frame][1] != -1:
                pointsForFrame[frame].append(point[frame][:2])  # Only take x, y coordinates

                cameraInfo, _ = take_info_camera(camera, cameraInfos)
                extrinsic_matrix_3x4 = cameraInfo.extrinsic_matrix[:3, :]
                camera_matrices.append(extrinsic_matrix_3x4)  # Projection matrix of each camera


        if len(pointsForFrame[frame]) >= 2:  # We need at least two points to triangulate
            points_3d = triangulate_points(pointsForFrame[frame], camera_matrices)
            print(f"3D point at frame {frame}: {points_3d}")



if __name__ == '__main__':
    # takePoints()
    tracking3D()
