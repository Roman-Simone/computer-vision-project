from config import *
from utils import *
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
import json

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600),
    7: (5150, 5330)
}

pathPickle = os.path.join(PATH_DETECTIONS, 'detections.pkl')
detections = load_pickle(pathPickle)
camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

cam = [take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS]

def get_projection_matrix(cam):
    K = cam.newcameramtx  
    extrinsic_matrix = cam.extrinsic_matrix  
    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]  
    return np.dot(K, extrinsic_matrix_3x4)

def get_image_resolution_for_frame(cam_num):
    image_path = os.path.join(PATH_FRAME_DISTORTED, f"cam_{cam_num}.png")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image.shape[1], image.shape[0]  # Return width and height directly

def scale_detection_to_original(point2d, img_width, img_height):
    resized_size = 800
    scale_x = img_width / resized_size
    scale_y = img_height / resized_size
    return np.array([point2d[0] * scale_x, point2d[1] * scale_y])

def triangulate(cam1, cam2, point2d1, point2d2):
    img_width1, img_height1 = get_image_resolution_for_frame(cam1.camera_number)
    img_width2, img_height2 = get_image_resolution_for_frame(cam2.camera_number)

    point2d1_scaled = scale_detection_to_original(point2d1, img_width1, img_height1)
    point2d2_scaled = scale_detection_to_original(point2d2, img_width2, img_height2)

    proj1 = get_projection_matrix(cam1)
    proj2 = get_projection_matrix(cam2)

    point2d1_scaled = np.array([point2d1_scaled], dtype=np.float32)
    point2d2_scaled = np.array([point2d2_scaled], dtype=np.float32)

    point4d = cv2.triangulatePoints(proj1, proj2, point2d1_scaled.T, point2d2_scaled.T)

    # Convert to 3D homogeneous coordinates (x, y, z) by dividing by the last coordinate
    point3d = cv2.convertPointsFromHomogeneous(point4d.T)
    return point3d[0][0]

def is_valid_point(point3d):
    x, y, z = point3d
    return not (z < 0 or x < -14 or y < -7.5 or x > 14 or y > 7.5)

def get_positions():
    with open(PATH_CAMERA_POS, "r") as file:
        data = json.load(file)
        return np.array(data["field_corners"]) 

def draw_detection_on_image(image, point2d):
    if image is None:
        raise FileNotFoundError(f"Image not found")
    
    orig_height, orig_width = image.shape[:2]  # Shape gives (height, width, channels)
    
    scale_x = orig_width / 800.0
    scale_y = orig_height / 800.0
    
    scaled_point2d = (point2d[0] * scale_x, point2d[1] * scale_y)
    scaled_point2d = tuple(map(int, scaled_point2d))  # Convert to integer pixel coordinates
    
    cv2.circle(image, scaled_point2d, 20, (255, 0, 0), -1)
    
    return image

def main():
    det_3D = {}
    
    try:
        action_id = int(input(f"Select an action from the available actions [1, 2, 3, 4, 5, 6, 7] : "))
        if action_id not in ACTIONS:
            print("Invalid action selected. Exiting.")
            exit()
    except ValueError:
        print("Invalid input. Please enter a number corresponding to the action.")
        exit()
        
    START, END = ACTIONS[action_id]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])  # Set aspect ratio for the 3D plot

    field_corners = get_positions()
    ax.scatter(field_corners[:, 0], field_corners[:, 1], field_corners[:, 2], c="red", label="Court Corners")

    previous_points = []  # Store points from all previous frames
    scatter_current = ax.scatter([], [], [], c='darkgreen', s=30, marker='o', label="Current frame detections")
    scatter_previous = ax.scatter([], [], [], c='lightgreen', s=20, marker='o', label="Previous frame detections")

    for i in range(START, END):
        print("Frame: ", i)
        detections_frame = {cam: detections.get((cam, i), []) for cam in VALID_CAMERA_NUMBERS}
        
        det_frame_3D = {}

        # Create a list to hold 3D points for triangulation
        triangulated_points = []

        for idx1, cam1 in enumerate(VALID_CAMERA_NUMBERS):
            for idx2 in range(idx1 + 1, len(VALID_CAMERA_NUMBERS)):
                cam2 = VALID_CAMERA_NUMBERS[idx2]

                detections_cam1 = detections_frame[cam1]
                detections_cam2 = detections_frame[cam2]

                if not detections_cam1 or not detections_cam2:
                    continue

                # Triangulate all pairs of detections from both cameras
                for point2d1 in detections_cam1:
                    for point2d2 in detections_cam2:
                        point3d = triangulate(cam[idx1], cam[idx2], point2d1, point2d2)
                        if is_valid_point(point3d):
                            triangulated_points.append(point3d)

        # If triangulated points exist, select the closest based on previous points
        # Inside the main loop
        if triangulated_points:
            triangulated_points = np.array(triangulated_points)
            
            # Ensure that they are in the right shape (N, 3) for plotting
            if triangulated_points.shape[1] == 3:
                # If previous points exist, calculate distances and select closest points
                if previous_points:
                    previous_points_np = np.array(previous_points)
                    distances = np.linalg.norm(triangulated_points[:, np.newaxis] - previous_points_np, axis=2)
                    closest_indices = np.argmin(distances, axis=1)
                    det_frame_3D[i] = triangulated_points[closest_indices].tolist()
                else:
                    det_frame_3D[i] = triangulated_points.tolist()

        if det_frame_3D:
            det_3D[i] = det_frame_3D

        # Update scatter plots for current and previous points
        current_points = list(det_frame_3D.values())
        if current_points:
            current_points = np.array(current_points)
            scatter_current._offsets3d = (current_points[:, 0], current_points[:, 1], current_points[:, 2])

            # Draw lines connecting previous points to current points
            if previous_points:
                previous_points_np = np.array(previous_points)
                for current_point in current_points:
                    ax.plot([previous_points_np[0][0], current_point[0]], 
                            [previous_points_np[0][1], current_point[1]], 
                            [previous_points_np[0][2], current_point[2]], 
                            color='blue', linewidth=1)

        if previous_points:
            temp_previous = np.array(previous_points)
            scatter_previous._offsets3d = (temp_previous[:, 0], temp_previous[:, 1], temp_previous[:, 2])

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim([-15, 15])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-0.5, 30])
        ax.set_title(f"3D Points for Frame {i}")
        plt.draw()
        plt.pause(0.1)

        # Add the current points to the previous points list
        if current_points.size > 0:
            previous_points.append(current_points[0])

    save_pickle(det_3D, os.path.join(PATH_DETECTIONS, 'detections_3D.pkl'))

if __name__ == "__main__":
    main()
