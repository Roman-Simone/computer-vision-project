import os
import cv2
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

# Now you can import the utils module from the parent directory
from utils.utils import *
from utils.config import *



def calculate_extrinsics(camera_number, undistortedFlag = False):
    # Read the data
    pathToRead = PATH_JSON_DISTORTED
    if undistortedFlag:
        pathToRead = PATH_JSON_UNDISTORTED
    coordinates_by_camera = read_json_file_and_structure_data(pathToRead)

    all_camera_coordinates = {}

    if str(camera_number) not in coordinates_by_camera:
        print(f"Camera {camera_number} not found in the dataset.")
        return None

    for camera_id, coords in coordinates_by_camera.items():
        if int(camera_id) == camera_number:
            world_points = np.array(coords["world_coordinates"], dtype=np.float32)
            image_points = np.array(coords["image_coordinates"], dtype=np.float32)

        # Collect all camera coordinates (assuming they are provided)
        if "camera_coordinates" in coords:
            cam_coords = np.array(coords["camera_coordinates"], dtype=np.float32)
            all_camera_coordinates[camera_id] = cam_coords

    # Load camera calibration data
    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    camera_info, _ = take_info_camera(camera_number, camera_infos)

    if camera_info is None:
        print(f"Camera info for camera {camera_number} not found.")
        return None

    if undistortedFlag:
        camera_matrix = np.array(camera_info.newcameramtx, dtype=np.float32)
    else:
        camera_matrix = np.array(camera_info.mtx, dtype=np.float32)

    distortion_coefficients = np.array(camera_info.dist, dtype=np.float32)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        world_points, image_points, camera_matrix, distortion_coefficients
    )

    # Convert the rotation vector to a rotation matrix using Rodrigues
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)

    extrinsic_matrix = np.hstack((inverse_rotation_matrix, inverse_translation_vector))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

    return extrinsic_matrix


def findAllExtrinsics(undistortedFlag = False):

    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    for camera_number in VALID_CAMERA_NUMBERS:
        _, pos = take_info_camera(camera_number, camera_infos)
        extrinsic_matrix = calculate_extrinsics(camera_number, undistortedFlag)
        display_extrinsic_matrix(extrinsic_matrix, camera_number)
        camera_infos[pos].extrinsic_matrix = extrinsic_matrix
    
    save_pickle(camera_infos, PATH_CALIBRATION_MATRIX)


def find_cam_extrinsic(camera_number):
    if camera_number not in VALID_CAMERA_NUMBERS:
        print(f"Camera {camera_number} is not a valid camera number.")
        return

    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)
    _, pos = take_info_camera(camera_number, camera_infos)
    extrinsic_matrix = calculate_extrinsics(camera_number)
    display_extrinsic_matrix(extrinsic_matrix)
    camera_infos[pos].extrinsic_matrix = extrinsic_matrix
    save_pickle(camera_infos, PATH_CALIBRATION_MATRIX)

def plot_3d_data(extrinsic_matrices, camera_numbers=None):
    # Create a 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Load data from JSON file
    with open(PATH_JSON_UNDISTORTED, 'r') as file:
        data = json.load(file)

    # Colors for cameras and points
    calculated_camera_color = 'red'
    real_camera_color = 'green'
    point_color = 'blue'

    all_points = []
    
    # Ensure extrinsic_matrices is a list
    if not isinstance(extrinsic_matrices, list):
        extrinsic_matrices = [extrinsic_matrices]
    
    # If camera_numbers is not provided, generate them
    if camera_numbers is None:
        camera_numbers = list(range(1, len(extrinsic_matrices) + 1))
    elif not isinstance(camera_numbers, list):
        camera_numbers = [camera_numbers]

    # Plot each camera
    for extrinsic_matrix, camera_number in zip(extrinsic_matrices, camera_numbers):
        camera_position = extrinsic_matrix[:3, 3]

        # Plot calculated camera location
        ax.scatter(camera_position[0], camera_position[1], camera_position[2], 
                   c=calculated_camera_color, marker="o", s=100, 
                   label='Calculated Camera Positions' if camera_number == camera_numbers[0] else '')

        # Plot camera direction
        direction_vector_size = 5
        camera_direction = extrinsic_matrix[:3, :3] @ np.array([0, 0, direction_vector_size]) + camera_position
        ax.plot([camera_position[0], camera_direction[0]],
                [camera_position[1], camera_direction[1]],
                [camera_position[2], camera_direction[2]],
                c="red", label="Camera Direction" if camera_number == camera_numbers[0] else '')

        # Add camera label for calculated position
        ax.text(camera_position[0], camera_position[1], camera_position[2], 
                f'Calc Cam {camera_number}', fontsize=8)

        # Plot real camera position if available in JSON data
        if str(camera_number) in data:
            real_camera_coords = data[str(camera_number)]['camera_coordinates']
            ax.scatter(*real_camera_coords, color=real_camera_color, s=100, 
                       label='Real Camera Positions' if camera_number == camera_numbers[0] else '')
            
            # Add camera label for real position
            ax.text(real_camera_coords[0], real_camera_coords[1], real_camera_coords[2], 
                    f'Real Cam {camera_number}', fontsize=8)

    # Plot world points
    first_point = True
    for camera_data in data.values():
        for point in camera_data['points']:
            world_coord = point['world_coordinate']
            all_points.append(world_coord)
            ax.scatter(*world_coord, color=point_color, s=50, 
                       label='World Points' if first_point else '')
            first_point = False

    # Calculate scene center and radius
    all_points = np.array(all_points)
    center = all_points.mean(axis=0)
    radius = np.max(np.linalg.norm(all_points - center, axis=1))

    # Set axis limits
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2], center[2] + radius)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Cameras and World Points')

    # Add legend
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_camera(camera_number):
    if camera_number not in VALID_CAMERA_NUMBERS:
        print(f"Camera {camera_number} is not a valid camera number.")
        return

    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)
    _, pos = take_info_camera(camera_number, camera_infos)

    extrinsic_matrix = camera_infos[pos].extrinsic_matrix

    display_extrinsic_matrix(extrinsic_matrix)
    plot_3d_data(extrinsic_matrix, camera_number)

def plotAllCameras():
    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    extrinsic_matrices = []
    camera_numbers = []

    for camera_info in camera_infos:
        extrinsic_matrices.append(camera_info.extrinsic_matrix)
        camera_numbers.append(camera_info.camera_number)
        display_extrinsic_matrix(camera_info.extrinsic_matrix, camera_info.camera_number)

    plot_3d_data(extrinsic_matrices, camera_numbers)


def display_extrinsic_matrix(extrinsic_matrix, camera_number=None):
    
    if extrinsic_matrix is not None:
        print(f"\nExtrinsic Matrix for Camera {camera_number}:")
        print(extrinsic_matrix)
        
        # Estrai rotazione e traslazione dalla matrice degli estrinseci
        rotation = extrinsic_matrix[:3, :3]
        translation = extrinsic_matrix[:3, 3]
        
        print("\nRotation Matrix:")
        print(rotation)
        print("\nTranslation Vector:")
        print(translation)
        
        # Calcola gli angoli di Eulero dalla matrice di rotazione
        euler_angles = rotationMatrixToEulerAngles(rotation)
        print("\nEuler Angles (in degrees):")
        print(f"Roll: {np.degrees(euler_angles[0]):.2f}")
        print(f"Pitch: {np.degrees(euler_angles[1]):.2f}")
        print(f"Yaw: {np.degrees(euler_angles[2]):.2f}")
    else:
        print(f"Unable to calculate extrinsic matrix for Camera {camera_number}")

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])




if __name__ == "__main__":

    #find extrinsic parameter for all cameras
    undistortedFlag = False
    findAllExtrinsics(undistortedFlag)

    #find the extrinsic matrix for specific camera
    # camera_number = 2 
    # find_cam_extrinsic(camera_number)


    #plot all camera extrinsic matrix
    plotAllCameras()
    
    # plot specific camera extrinsic matrix
    # camera_number = 12
    # plot_camera(camera_number)
    
