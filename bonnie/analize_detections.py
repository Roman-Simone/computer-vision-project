from config import *
from utils import *
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
import json

MAX_FRAME = 5100

pathPickle = os.path.join(PATH_DETECTIONS, 'detections.pkl')
detections = load_pickle(pathPickle)
camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

cam = [take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS]

TOLERANCE = 0.5

def get_projection_matrix(cam):
    K = cam.newcameramtx  
    extrinsic_matrix = cam.extrinsic_matrix  
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
    return cv2.convertPointsFromHomogeneous(point4d.T)[0][0]

def is_valid_point(point3d):
    x, y, z = point3d
    return not (z < 0 or x < -14 or y < -7.5 or x > 14 or y > 7.5)

def get_positions():
    with open(PATH_CAMERA_POS, "r") as file:
        data = json.load(file)
        return data["positions"], np.array(data["field_corners"]) 

def are_points_close(point1, point2):
    return np.linalg.norm(point1 - point2) < TOLERANCE

def main():
    det_3D = {}
    plt.ion()  # Turn on interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])  # Set aspect ratio for the 3D plot

    positions, field_corners = get_positions()

    # Pre-plot the static field corners once
    ax.scatter(field_corners[:, 0], field_corners[:, 1], field_corners[:, 2], c="red", label="Court Corners")

    previous_points = set()  # Store points from all previous frames
    scatter_current = ax.scatter([], [], [], c='darkgreen', s=30, marker='o', label="Current frame detections")
    scatter_previous = ax.scatter([], [], [], c='yellow', s=20, marker='o', label="Previous frame detections")

    for i in range(1, MAX_FRAME):
        det_frame_3D = {}
        detections_frame = {cam: detections.get((cam, i), []) for cam in VALID_CAMERA_NUMBERS}

        # Iterate over pairs of cameras
        for idx1, cam1 in enumerate(VALID_CAMERA_NUMBERS):
            for idx2 in range(idx1 + 1, len(VALID_CAMERA_NUMBERS)):
                cam2 = VALID_CAMERA_NUMBERS[idx2]

                detections_cam1 = detections_frame[cam1]
                detections_cam2 = detections_frame[cam2]

                if not detections_cam1 or not detections_cam2:
                    continue

                for point2d1 in detections_cam1:
                    for point2d2 in detections_cam2:
                        point3d = triangulate(cam[idx1], cam[idx2], point2d1, point2d2)

                        if is_valid_point(point3d):
                            found_close = False
                            for existing_point in det_frame_3D.keys():
                                if are_points_close(existing_point, point3d):
                                    det_frame_3D[existing_point] += 1
                                    found_close = True
                                    break

                            if not found_close:
                                det_frame_3D[tuple(point3d)] = 1  

        valid_points = {point: count for point, count in det_frame_3D.items() if count >= 2}
        det_3D[i] = list(valid_points.keys())

        current_points = list(valid_points.keys())
        if current_points:
            previous_points.update(current_points)

        # Update scatter plots for current and previous points
        if current_points:
            scatter_current._offsets3d = (np.array(current_points)[:, 0], np.array(current_points)[:, 1], np.array(current_points)[:, 2])
        
        if previous_points:
            scatter_previous._offsets3d = (np.array(list(previous_points))[:, 0], np.array(list(previous_points))[:, 1], np.array(list(previous_points))[:, 2])

        print(f"Frame {i} - valid points: {len(valid_points)}")

        # Update plot limits and labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim([-15, 15])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-0.5, 30])
        ax.set_title('3D Tracked Points and Real Corners (with Path)')
        ax.legend()

        # Pause briefly to allow the plot to update
        plt.pause(0.001)

    # Finalize the plot window
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the window open for interaction

    # Save the final detections
    save_pickle(det_3D, os.path.join(PATH_DETECTIONS, 'detections_3D.pkl'))

if __name__ == "__main__":
    main()
