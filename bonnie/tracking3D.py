import re
import cv2
from utils import *
from config import *
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  

pathWeight = os.path.join(PATH_WEIGHT, 'best_v8_800.pt')
model = YOLO(pathWeight, verbose=False)

SIZE = 800
NUMBER_FRAMES = 50

def plot_3D_trajectory(points_3D):
    """
    Plotta la traiettoria 3D del pallone.
    
    Args:
        points_3D (np.array): Lista di punti 3D triangolati.
    """
    x_vals = points_3D[:, 0]
    y_vals = points_3D[:, 1]
    z_vals = points_3D[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Punti in 3D
    ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o', label='Points')

    # Traiettoria (linea che unisce i punti)
    ax.plot(x_vals, y_vals, z_vals, color='b', linewidth=2, label='Trajectory')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Ball trajectory')

    ax.legend()
    plt.show()

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

def get_projection_matrix(camera_number, cameraInfos):
    """
    Gets the projection matrix for a camera.

    Args:
        camera_number (int): Camera number.
        cameraInfos (dict): Dictionary containing the camera information.
    
    Returns:
        np.array: Projection matrix for the camera.
    """

    cameraInfo, _ = take_info_camera(camera_number, cameraInfos)
    K = cameraInfo.newcameramtx  # 3x3 
    
    # (4x4)
    extrinsic_matrix = cameraInfo.extrinsic_matrix  
    
    # get the top 3x4 part (first 3 rows and 4 columns)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]  
    
    print("K: ", K)
    print("Extrinsic matrix (3x4): ", extrinsic_matrix_3x4)
    
    # return projection matrix P = K * [R | t]    
    return np.dot(K, extrinsic_matrix_3x4)

def tracking3D():
    videosCalibration = find_files(PATH_VIDEOS)
    videosCalibration.sort()
    cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

    points_2D = {}
    
    for video in tqdm(videosCalibration, desc="Processing videos"):
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])

        if numero_camera not in VALID_CAMERA_NUMBERS:
            continue

        cameraInfo, _ = take_info_camera(numero_camera, cameraInfos)
        pathVideo = os.path.join(PATH_VIDEOS, video)
        
        points_2D[numero_camera] = takeBallCoordinates(pathVideo, cameraInfo, model)

    # Assumendo che abbiamo punti da almeno due videocamere per triangolare
    if 1 in points_2D and 2 in points_2D:  
        camera1_points = [p[:2] for p in points_2D[1] if p[0] != -1 and p[1] != -1]  # Filter valid points
        camera2_points = [p[:2] for p in points_2D[2] if p[0] != -1 and p[1] != -1]  # Filter valid points

        # Print shapes for debugging
        print("Camera 1 Points Shape:", np.array(camera1_points).shape)
        print("Camera 2 Points Shape:", np.array(camera2_points).shape)

        # Ensure the number of points is the same for triangulation
        min_length = min(len(camera1_points), len(camera2_points))

        # Truncate both lists to the minimum length
        camera1_points = camera1_points[:min_length]
        camera2_points = camera2_points[:min_length]

        # Check if there are valid points to triangulate
        if len(camera1_points) == 0 or len(camera2_points) == 0:
            print("No valid points to triangulate. Skipping.")
            return

        # Matrici di proiezione delle due videocamere
        P1 = get_projection_matrix(6, cameraInfos)
        P2 = get_projection_matrix(7, cameraInfos)

        if P1.shape != (3, 4) or P2.shape != (3, 4):
            print("Projection matrix shapes are incorrect")
            return

        # Triangolazione
        # points_3D = triangulate_3D_points(camera1_points, camera2_points, P1, P2)

        # Plot dei punti triangolati
        plot_3D_trajectory(points_3D)

    save_pickle(points_2D, 'points_2D.pkl')

# def triangulate_3D_points(camera1_points, camera2_points, P1, P2):
#     """
#     Triangola i punti 3D usando i punti 2D di due videocamere.
#     """
#     # Ensure that points are numpy arrays and transpose them correctly
#     camera1_points = np.array(camera1_points).T  # (2, N)
#     camera2_points = np.array(camera2_points).T  # (2, N)

#     print("Camera 1 Points Shape:", camera1_points.shape)
#     print("Camera 2 Points Shape:", camera2_points.shape)

#     # Check if the points array have matching dimensions
#     if camera1_points.shape != camera2_points.shape:
#         raise ValueError("Camera 1 and Camera 2 points do not have the same dimensions")

#     # Perform triangulation
#     points_3D_homogeneous = cv2.triangulatePoints(P1, P2, camera1_points, camera2_points)
    
#     # Convert from homogeneous to Cartesian coordinates
#     points_3D = cv2.convertPointsFromHomogeneous(points_3D_homogeneous.T)
    
#     return points_3D.squeeze()


def testPoints():
    points = load_pickle('points.pkl')

    print("\n\n\n\n")

    for camera, coords in points.items():
        print(f"Camera {camera}: {coords}")


if __name__ == '__main__':
    tracking3D()
    testPoints()
