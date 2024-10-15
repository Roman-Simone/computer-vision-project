import torch
import numpy as np
import cv2
from config import *
from utils import *
import os
import json
from matplotlib import pyplot as plt

torch.set_default_dtype(torch.float32)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Metal Performance Shaders) is available. Using GPU acceleration.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU.")

MAX_FRAME = 5100
TOLERANCE = 0.5

def get_projection_matrix(cam):
    K = torch.tensor(cam.newcameramtx, dtype=torch.float32, device=device)
    extrinsic_matrix = torch.tensor(cam.extrinsic_matrix, dtype=torch.float32, device=device)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]
    return torch.matmul(K, extrinsic_matrix_3x4)

def scale_detection_to_original(point2d, img_width, img_height):
    resized_size = 800
    scale_x = img_width / resized_size
    scale_y = img_height / resized_size
    return torch.tensor([point2d[0] * scale_x, point2d[1] * scale_y], dtype=torch.float32, device=device)

def get_image_resolution_for_frame(cam_num):
    image_path = os.path.join(PATH_FRAME_DISTORTED, f"cam_{cam_num}.png")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image.shape[1], image.shape[0]  # Return width and height directly

def triangulate_torch(proj1, proj2, points2d1, points2d2):
    A = torch.zeros((4, 4), dtype=torch.float32, device=device)
    A[0] = points2d1[0] * proj1[2] - proj1[0]
    A[1] = points2d1[1] * proj1[2] - proj1[1]
    A[2] = points2d2[0] * proj2[2] - proj2[0]
    A[3] = points2d2[1] * proj2[2] - proj2[1]
    
    _, _, V = torch.linalg.svd(A)
    point4d = V[:, -1]
    point3d = point4d[:3] / point4d[3]
    return point3d

def triangulate(cam1, cam2, point2d1, point2d2):
    img_width1, img_height1 = get_image_resolution_for_frame(cam1.camera_number)
    img_width2, img_height2 = get_image_resolution_for_frame(cam2.camera_number)

    point2d1_scaled = scale_detection_to_original(point2d1, img_width1, img_height1)
    point2d2_scaled = scale_detection_to_original(point2d2, img_width2, img_height2)

    proj1 = get_projection_matrix(cam1)
    proj2 = get_projection_matrix(cam2)

    return triangulate_torch(proj1, proj2, point2d1_scaled, point2d2_scaled)

def is_valid_point(point3d):
    x, y, z = point3d
    return not (z < 0 or x < -14 or y < -7.5 or x > 14 or y > 7.5)

def main():
    pathPickle = os.path.join(PATH_DETECTIONS, 'detections.pkl')
    detections = load_pickle(pathPickle)
    camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

    cam = [take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS]

    det_3D = {}
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    with open(PATH_CAMERA_POS, "r") as file:
        data = json.load(file)
        field_corners = torch.tensor(data["field_corners"], dtype=torch.float32, device=device)

    ax.scatter(field_corners[:, 0].cpu().numpy(), field_corners[:, 1].cpu().numpy(), field_corners[:, 2].cpu().numpy(), c="red", label="Court Corners")

    previous_points = set()
    scatter_current = ax.scatter([], [], [], c='darkgreen', s=30, marker='o', label="Current frame detections")
    scatter_previous = ax.scatter([], [], [], c='yellow', s=20, marker='o', label="Previous frame detections")

    for i in range(1, MAX_FRAME):
        det_frame_3D = {}
        detections_frame = {cam: detections.get((cam, i), []) for cam in VALID_CAMERA_NUMBERS}

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
                            point_tuple = tuple(point3d.cpu().numpy())
                            if point_tuple in det_frame_3D:
                                det_frame_3D[point_tuple] += 1
                            else:
                                det_frame_3D[point_tuple] = 1

        valid_points = {point: count for point, count in det_frame_3D.items() if count >= 2}
        det_3D[i] = list(valid_points.keys())

        current_points = list(valid_points.keys())
        if current_points:
            previous_points.update(current_points)

        if current_points:
            scatter_current._offsets3d = (np.array(current_points)[:, 0], np.array(current_points)[:, 1], np.array(current_points)[:, 2])
        
        if previous_points:
            scatter_previous._offsets3d = (np.array(list(previous_points))[:, 0], np.array(list(previous_points))[:, 1], np.array(list(previous_points))[:, 2])

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

    save_pickle(det_3D, os.path.join(PATH_DETECTIONS, 'detections_3D.pkl'))

if __name__ == "__main__":
    main()