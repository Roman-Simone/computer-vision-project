import os
import cv2
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

def calculate_extrinsics(camera_number, undistortedFlag=False):
    """
    Calculates the extrinsic matrix for a specified camera, including rotation matrix and translation vector.

    Parameters:
        camera_number (int): the number of the camera to calculate extrinsics for.
        undistortedFlag (bool): if True, undistorts the image points and later compute extrinsic parameter with new_mtx.

    Returns:
        numpy.ndarray: the 4x4 extrinsic matrix, including rotation matrix and translation vector if calculation is successful;
                      otherwise, None.
    """
    
    pathToRead = PATH_JSON_DISTORTED
    coordinates_by_camera = read_json_file_and_structure_data(pathToRead)
    all_camera_coordinates = {}

    if str(camera_number) not in coordinates_by_camera:
        print(f"Camera {camera_number} not found in the dataset.")
        return None

    for camera_id, coords in coordinates_by_camera.items():
        if int(camera_id) == camera_number:
            world_points = np.array(coords["world_coordinates"], dtype=np.float32)
            image_points = np.array(coords["image_coordinates"], dtype=np.float32)

        if "camera_coordinates" in coords:
            cam_coords = np.array(coords["camera_coordinates"], dtype=np.float32)
            all_camera_coordinates[camera_id] = cam_coords

    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)
    camera_info, _ = take_info_camera(camera_number, camera_infos)

    if camera_info is None:
        print(f"Camera info for camera {camera_number} not found.")
        return None

    if undistortedFlag:
        image_points = np.expand_dims(image_points, axis=1)
        image_points = cv2.undistortPoints(image_points, camera_info.mtx, camera_info.dist, None, camera_info.newcameramtx)
        image_points = image_points.reshape(-1, 2)
        camera_matrix = np.array(camera_info.newcameramtx, dtype=np.float32)
    else:
        camera_matrix = np.array(camera_info.mtx, dtype=np.float32)

    distortion_coefficients = np.array(camera_info.dist, dtype=np.float32)

    _, rotation_vector, translation_vector = cv2.solvePnP(
        world_points, image_points, camera_matrix, distortion_coefficients
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)
    extrinsic_matrix = np.hstack((inverse_rotation_matrix, inverse_translation_vector))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

    return extrinsic_matrix


def compute_all_extrinsics(undistortedFlag=False):
    """
    Compute the extrinsic parameters for all valid cameras and updates them in the calibration file.

    Parameters:
        undistortedFlag (bool): if True, calculates undistorted extrinsics for each camera.
    """
    
    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    for camera_number in VALID_CAMERA_NUMBERS:
        _, pos = take_info_camera(camera_number, camera_infos)
        extrinsic_matrix = calculate_extrinsics(camera_number, undistortedFlag)
        camera_infos[pos].extrinsic_matrix = extrinsic_matrix

    save_pickle(camera_infos, PATH_CALIBRATION_MATRIX)


def plot_3d_data(extrinsic_matrices, camera_numbers=None):
    """
    Plots 3D data of camera positions, world points, and their orientations based on extrinsic matrices.

    Parameters:
        extrinsic_matrices (list or numpy.ndarray): extrinsic matrices of the cameras.
        camera_numbers (list, optional): list of camera numbers corresponding to extrinsic matrices.
                                         If None, uses sequential numbering.
    """
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    with open(PATH_JSON_UNDISTORTED, 'r') as file:
        data = json.load(file)

    calculated_camera_color = 'red'
    real_camera_color = 'green'
    points_color = 'blue'

    all_points = []

    if not isinstance(extrinsic_matrices, list):
        extrinsic_matrices = [extrinsic_matrices]
    
    if camera_numbers is None:
        camera_numbers = list(range(1, len(extrinsic_matrices) + 1))
    elif not isinstance(camera_numbers, list):
        camera_numbers = [camera_numbers]

    for extrinsic_matrix, camera_number in zip(extrinsic_matrices, camera_numbers):
        camera_position = extrinsic_matrix[:3, 3]
        ax.scatter(camera_position[0], camera_position[1], camera_position[2], 
                   c=calculated_camera_color, marker="o", s=100, 
                   label='Calculated Camera Positions' if camera_number == camera_numbers[0] else '')

        direction_vector_size = 5
        camera_direction = extrinsic_matrix[:3, :3] @ np.array([0, 0, direction_vector_size]) + camera_position
        ax.plot([camera_position[0], camera_direction[0]],
                [camera_position[1], camera_direction[1]],
                [camera_position[2], camera_direction[2]],
                c="red", label="Camera Direction" if camera_number == camera_numbers[0] else '')

        ax.text(camera_position[0], camera_position[1], camera_position[2], 
                f'Calc Cam {camera_number}', fontsize=8)

        if str(camera_number) in data:
            real_camera_coords = data[str(camera_number)]['camera_coordinates']
            ax.scatter(*real_camera_coords, color=real_camera_color, s=100, 
                       label='Real Camera Positions' if camera_number == camera_numbers[0] else '')
            ax.text(real_camera_coords[0], real_camera_coords[1], real_camera_coords[2], 
                    f'Real Cam {camera_number}', fontsize=8)

    first_point = True
    for camera_data in data.values():
        for point in camera_data['points']:
            world_coord = point['world_coordinate']
            all_points.append(world_coord)
            ax.scatter(*world_coord, color=points_color, s=50, 
                       label='World Points' if first_point else '')
            first_point = False

    all_points = np.array(all_points)
    center = all_points.mean(axis=0)
    radius = np.max(np.linalg.norm(all_points - center, axis=1))

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2], center[2] + radius)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Cameras and World Points')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_camera(camera_number):
    """
    Plots the extrinsic matrix and 3D data for a specified camera.

    Parameters:
        camera_number (int): camera number to plot.
    """
    
    if camera_number not in VALID_CAMERA_NUMBERS:
        print(f"Camera {camera_number} is not a valid camera number.")
        return

    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)
    _, pos = take_info_camera(camera_number, camera_infos)
    extrinsic_matrix = camera_infos[pos].extrinsic_matrix

    display_extrinsic_matrix(extrinsic_matrix, camera_number)
    plot_3d_data(extrinsic_matrix, camera_number)


def plot_all_cameras():
    """
    Plots the extrinsic matrices and 3D data for all available cameras.
    """
    
    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)
    extrinsic_matrices = []
    camera_numbers = []

    for camera_info in camera_infos:
        extrinsic_matrices.append(camera_info.extrinsic_matrix)
        camera_numbers.append(camera_info.camera_number)
        display_extrinsic_matrix(camera_info.extrinsic_matrix, camera_info.camera_number)

    plot_3d_data(extrinsic_matrices, camera_numbers)


def display_extrinsic_matrix(extrinsic_matrix, camera_number=None):
    """
    Displays the extrinsic matrix, rotation matrix, translation vector, and Euler angles for a camera.

    Parameters:
        extrinsic_matrix (numpy.ndarray): extrinsic matrix to display.
        camera_number (int, optional): camera number, if applicable, to identify the matrix.
    """
    
    if extrinsic_matrix is not None:
        print(f"\nExtrinsic Matrix for Camera {camera_number}:")
        print(extrinsic_matrix)

        rotation = extrinsic_matrix[:3, :3]
        translation = extrinsic_matrix[:3, 3]

        print("\nRotation Matrix:")
        print(rotation)
        print("\nTranslation Vector:")
        print(translation)

        euler_angles = rotation_matrix_to_euler_angles(rotation)
        print("\nEuler Angles (in degrees):")
        print(f"Roll: {np.degrees(euler_angles[0]):.2f}")
        print(f"Pitch: {np.degrees(euler_angles[1]):.2f}")
        print(f"Yaw: {np.degrees(euler_angles[2]):.2f}")
    else:
        print(f"Unable to calculate extrinsic matrix for Camera {camera_number}")


def rotation_matrix_to_euler_angles(R):
    """
    Converts a rotation matrix to Euler angles.

    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        numpy.ndarray: euler angles (roll, pitch, yaw) in radians.
    """
    
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

if __name__ == "__main__":

    #find extrinsic parameter for all cameras
    undistortedFlag = False
    compute_all_extrinsics(undistortedFlag)

    # camera_number = 1
    # plot_camera(camera_number)
    plot_all_cameras()
    