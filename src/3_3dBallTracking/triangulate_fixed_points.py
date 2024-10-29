import pickle
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import json

# Paths and configuration
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

# Load pickle files
pathPickle = os.path.join(PATH_DETECTIONS, 'all_detections.pkl')
detections = load_pickle(pathPickle)

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

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600)
}

# User input for selecting action
# action_number = -1
# while action_number not in ACTIONS:
#     action_number = int(input('Enter the action to analyze: '))

cam = {n: take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS}

def get_projection_matrix(cam):
    """Create projection matrix from intrinsic and extrinsic matrices."""
    K = cam.newcameramtx
    extrinsic_matrix = np.linalg.inv(cam.extrinsic_matrix)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]
    return np.dot(K, extrinsic_matrix_3x4)

projection_matrices = {n: get_projection_matrix(cam[n]) for n in VALID_CAMERA_NUMBERS}

def triangulate(cam1, cam2, point2d1, point2d2):
    """
    Triangulate 3D point from 2D points in two different camera views.
    
    Args:
        cam1: The camera object for the first camera.
        cam2: The camera object for the second camera.
        point2d1: 2D point from the first camera (tuple).
        point2d2: 2D point from the second camera (tuple).
        
    Returns:
        point3d: The 3D point resulting from the triangulation (numpy array).
    """
    
    proj1 = get_projection_matrix(cam1)
    proj2 = get_projection_matrix(cam2)

    point2d1 = np.array([point2d1], dtype=np.float32)  # Shape (1, 2)
    point2d2 = np.array([point2d2], dtype=np.float32)  # Shape (1, 2)

    point4d = cv2.triangulatePoints(proj1, proj2, point2d1.T, point2d2.T)

    point3d = cv2.convertPointsFromHomogeneous(point4d.T)[0][0]
    
    return point3d

def get_positions():
    """Retrieve field corner positions for visualization."""
    with open(PATH_CAMERA_POS, "r") as file:
        data = json.load(file)
        return np.array(data["field_corners"])

frame = 49

# DICTIONARY (and pkl file) structure:
# {
#     '1' : {     # first camera
#         '1' : {frame1: (x1, y1), frame2 : (x2, y2), ...},  # first action, a point for each frame (or 0 points if no balls detected)
#         '2' : [...] 
#         ....
#     }, 
#     ...
# }

points_2d = []

# 12: "world_coordinate": [
#                     -9.0,
#                     4.5,
#                     0.0
#                 ],
#                 "image_coordinate": [
#                     1282,
#                     949
#                 ]

# 13: "world_coordinate": [
#                     -9.0,
#                     4.5,
#                     0.0
#                 ],
#                 "image_coordinate": [
#                     2127,
#                     540
#                 ]
#             },

# 7: "world_coordinate": [
#                     -9.0,
#                     4.5,
#                     0.0
#                 ],
#                 "image_coordinate": [
#                     838,
#                     1852
#                 ]
#             },

# 8: "world_coordinate": [
#                     -9.0,
#                     4.5,
#                     0.0
#                 ],
#                 "image_coordinate": [
#                     1440,
#                     899
#                 ]

cam1 = 7
point2d1 = (838, 1852)
cam2 = 8
point2d2 = (1440, 899)

if point2d1 is not None and point2d2 is not None:
    print(f'Triangle between cameras {cam1} and {cam2} with points {point2d1} and {point2d2}')

    point3d = triangulate(cam[cam1], cam[cam2], point2d1, point2d2)
    print(f"3D point: {point3d} [-9, 4.5, 0]")