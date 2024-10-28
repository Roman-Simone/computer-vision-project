import pickle
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

pathPickle = os.path.join(PATH_DETECTIONS, 'all_detections.pkl')
detections = load_pickle(pathPickle)
camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
ACTIONS = [1, 2, 3, 4, 5, 6]

action_number = -1
while action_number not in ACTIONS:
    action_number = int(input('Enter the action to analyze: '))

cam = {n: take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS}

def get_positions():
    with open(PATH_CAMERA_POS, "r") as file:
        data = json.load(file)
        return np.array(data["field_corners"]) 

def get_projection_matrix(cam):
    K = cam.newcameramtx
    extrinsic_matrix = np.linalg.inv(cam.extrinsic_matrix)  
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]  
    return np.dot(K, extrinsic_matrix_3x4)

def triangulate_points(detections, action_number):
    points_3D = []
    
    for frame in detections['1'][str(action_number)]:  
        points_2D = []
        projection_matrices = []
        
        for camera_number in cam:
            if frame in detections[str(camera_number)][str(action_number)]:
                point = detections[str(camera_number)][str(action_number)][frame]
                if point is not None:
                    points_2D.append(point)
                    projection_matrices.append(get_projection_matrix(cam[camera_number]))

        if len(points_2D) >= 2:
            points_2D_cam1 = np.array(points_2D[0], dtype=np.float32).reshape(2, 1)
            points_2D_cam2 = np.array(points_2D[1], dtype=np.float32).reshape(2, 1)
            
            points_3D_homogeneous = cv2.triangulatePoints(
                projection_matrices[0], projection_matrices[1], points_2D_cam1, points_2D_cam2
            )
            points_3D_cartesian = cv2.convertPointsFromHomogeneous(points_3D_homogeneous.T)[0][0]
            
            points_3D.append(points_3D_cartesian)

    return np.array(points_3D)

points_3D = triangulate_points(detections, action_number)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])  

field_corners = get_positions()
ax.scatter(field_corners[:, 0], field_corners[:, 1], field_corners[:, 2], c="red", label="Court Corners")
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title(f'Triangulated 3D Points for Action {action_number}')
plt.show()
