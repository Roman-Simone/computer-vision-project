import pickle
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import json

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

pathPickle = os.path.join(PATH_DETECTIONS_04, 'all_detections.pkl')
pathPickle_cam2 = os.path.join(PATH_DETECTIONS_WINDOW_04, 'all_detections.pkl')

detections = load_pickle(pathPickle)
detections_cam2 = load_pickle(pathPickle_cam2)

# DICTIONARY (and pkl file) structure:
# {
#     '1' : {     # first camera
#         '1' : {frame1: (x1, y1), frame2 : (x2, y2), ...},  # first action, a point for each frame (or 0 points if no balls detected)
#         '2' : [...] 
#         ....
#     }, 
#     ...
# }

camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
cam = {n: take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS}
projection_matrices = {n: get_projection_matrix(cam[n]) for n in VALID_CAMERA_NUMBERS}

def triangulate(cam1, cam2, point2d1, point2d2):
    """
    Triangulate a 3D point from two 2D points in different camera views.

    Parameters:
        cam1 (Camera): first camera object with calibration data.
        cam2 (Camera): second camera object with calibration data.
        point2d1 (tuple): 2D coordinates of the point in the first camera view.
        point2d2 (tuple): 2D coordinates of the point in the second camera view.

    Returns:
        numpy.ndarray: 3D coordinates of the triangulated point.
    """
    proj1 = get_projection_matrix(cam1)
    proj2 = get_projection_matrix(cam2)

    # Format points for OpenCV triangulatePoints function
    point2d1 = np.array([point2d1], dtype=np.float32)  # Shape (1, 2)
    point2d2 = np.array([point2d2], dtype=np.float32)  # Shape (1, 2)

    # Triangulation
    point4d = cv2.triangulatePoints(proj1, proj2, point2d1.T, point2d2.T)

    # Convert to 3D coordinates by normalizing the homogeneous coordinates
    point3d = cv2.convertPointsFromHomogeneous(point4d.T)[0][0]
    
    return point3d

def main():
    """
    Main function to triangulate 3D points from 2D detections across multiple cameras
    and save the results for each action.
    """
    for action_number in ACTIONS:
        frame_start, frame_end = ACTIONS[action_number]

        points_3d = {frame: [] for frame in range(frame_start, frame_end + 1)}

        for frame in range(frame_start, frame_end + 1):
            points_2d = []

            for camera in VALID_CAMERA_NUMBERS:
                if camera != 2:  
                    if frame in detections[str(camera)][str(action_number)]:
                        point2d = detections[str(camera)][str(action_number)][frame]
                        points_2d.append((camera, point2d))
                else:  
                    if frame in detections_cam2[str(camera)][str(action_number)]:
                        point2d = detections_cam2[str(camera)][str(action_number)][frame]
                        if point2d is not None:
                            point2d = list(point2d)
                            x = int(point2d[0][0])
                            y = int(point2d[0][1])                        
                            points_2d.append((camera, (x, y)))
                        else:
                            points_2d.append((camera, None))

            # Triangulate 3D points if we have at least two 2D points from different cameras
            if len(points_2d) >= 2:
                for i in range(len(points_2d)):
                    for j in range(i + 1, len(points_2d)):
                        cam1, point2d1 = points_2d[i]
                        cam2, point2d2 = points_2d[j]
                        if point2d1 is not None and point2d2 is not None:
                            print(f'Triangulating between cameras {cam1} and {cam2} with points {point2d1} and {point2d2}')
                            point3d = triangulate(cam[cam1], cam[cam2], point2d1, point2d2)
                            points_3d[frame].append(point3d)
                            print(f"3D point: {point3d}")

        output_path = os.path.join(PATH_3D_DETECTIONS_04, f'points_3d_action{action_number}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(points_3d, f)
        print(f"3D points saved successfully at {output_path}")

if __name__ == "__main__":
    main()
