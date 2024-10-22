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

TOLERANCE = 0.45

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
        return np.array(data["field_corners"]) 

def are_points_close(point1, point2):
    
    if np.linalg.norm(point1 - point2) < TOLERANCE:
        print("Distance: ", np.linalg.norm(point1 - point2), " - ACCEPTED")
        return True
    elif np.linalg.norm(point1 - point2) >= TOLERANCE:
        print("Distance: ", np.linalg.norm(point1 - point2), " - DISCARDED")
        return False

def draw_detection_on_image(image, point2d):
    if image is None:
        raise FileNotFoundError(f"Image not found")
    
    # Get original image dimensions
    orig_height, orig_width = image.shape[:2]  # Shape gives (height, width, channels)
    
    # Calculate scaling factors (resized image is 800x800)
    scale_x = orig_width / 800.0
    scale_y = orig_height / 800.0
    
    print("Original image size ----> ", (orig_width, orig_height))
    print("Original point on resized image ----> ", point2d)
    
    # Scale the detected point2d back to the original image size
    scaled_point2d = (point2d[0] * scale_x, point2d[1] * scale_y)
    scaled_point2d = tuple(map(int, scaled_point2d))  # Convert to integer pixel coordinates
    
    print("Scaled point to original size ----> ", scaled_point2d)
    
    # Draw a blue circle at the detected point on the original image
    cv2.circle(image, scaled_point2d, 20, (255, 0, 0), -1)
    
    return image

def main():
    det_3D = {}
    plt.ion()  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])  

    field_corners = get_positions()

    ax.scatter(field_corners[:, 0], field_corners[:, 1], field_corners[:, 2], c="red", label="Court Corners")

    action_id = 1
        
    START, END = ACTIONS[int(action_id)]

    while True:
            previous_points = set()  
            scatter_current = ax.scatter([], [], [], c='darkgreen', s=30, marker='o', label="Current frame detections")
            scatter_previous = ax.scatter([], [], [], c='lightgreen', s=20, marker='o', label="Previous frame detections")

            for i in range(START, END):
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

                if previous_points:
                    temp_previous = np.array(list(set(previous_points) - set(current_points)))
                    
                    if temp_previous.size > 0:
                        scatter_previous._offsets3d = (temp_previous[:, 0], temp_previous[:, 1], temp_previous[:, 2])
                    else:
                        scatter_previous._offsets3d = ([], [], [])
                        
                if current_points:
                    scatter_current._offsets3d = (np.array(current_points)[:, 0], np.array(current_points)[:, 1], np.array(current_points)[:, 2])
                

                print(f"Frame {i} - valid points: {len(valid_points)}")

                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                ax.set_xlim([-15, 15])
                ax.set_ylim([-10, 10])
                ax.set_zlim([-0.5, 30])
                ax.set_title('3D Tracked Points and Real Corners (with Path)')
                ax.legend()

                plt.savefig("3D_tracking.png")
                plt.show()

                save_pickle(det_3D, os.path.join(PATH_DETECTIONS, 'detections_3D.pkl'))

if __name__ == "__main__":
    main()