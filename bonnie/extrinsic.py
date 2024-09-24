import os
import cv2
import numpy as np
from utils import *
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)

print("Parent_path: ", parent_path)

path_json = os.path.join(parent_path, 'data/world_points_all_cameras.json')
path_calibrationMTX = os.path.join(parent_path, 'data/calibrationMatrix/calibration.pkl')
json_file_path = parent_path + '/bonnie/camera_data.json'

def update_json(camera_number, inverse_rotation_matrix, inverse_translation_vector):

    with open(json_file_path, 'r') as json_file:
        camera_data = json.load(json_file)
       
    camera_data[str(camera_number)]["inverse_rotation_matrix"] = inverse_rotation_matrix.tolist()
    camera_data[str(camera_number)]["inverse_translation_vector"] = inverse_translation_vector.tolist()

    with open(json_file_path, 'w') as json_file:
        json.dump(camera_data, json_file, indent=4)

def calculate_extrinsics(camera_number):
    # Read the data
    coordinates_by_camera = read_json_file_and_structure_data(path_json)

    all_camera_coordinates = {}

    if str(camera_number) not in coordinates_by_camera:
        print(f"Camera {camera_number} not found in the dataset.")
        return None, None, None

    for camera_id, coords in coordinates_by_camera.items():
        if int(camera_id) == camera_number:
            world_points = np.array(coords["world_coordinates"], dtype=np.float32)
            image_points = np.array(coords["image_coordinates"], dtype=np.float32)

        # Collect all camera coordinates (assuming they are provided)
        if "camera_coordinates" in coords:
            cam_coords = np.array(coords["camera_coordinates"], dtype=np.float32)
            all_camera_coordinates[camera_id] = cam_coords

    print(f"World points for Camera {camera_number}:")
    print(world_points)
    print(f"Image points for Camera {camera_number}:")
    print(image_points)

    # Load camera calibration data
    camera_infos = load_pickle(path_calibrationMTX)

    camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)

    if camera_info is None:
        print(f"Camera info for camera {camera_number} not found.")
        return None, None, None

    camera_matrix = camera_info.mtx
    distortion_coefficients = np.zeros((1, 5), dtype=np.float32)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        world_points, image_points, camera_matrix, distortion_coefficients
    )

    # Convert the rotation vector to a rotation matrix using Rodrigues
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)

    extrinsic_matrix = np.hstack((inverse_rotation_matrix, inverse_translation_vector))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
    
    update_json(camera_number, inverse_rotation_matrix, inverse_translation_vector)

    return extrinsic_matrix, all_camera_coordinates, camera_number

def pretty_print_matrix(matrix):
    print("Extrinsic Matrix:")
    for row in matrix:
        print(" ".join(f"{val:8.4f}" for val in row))

# The size parameter is used to set the limits of the plot, so that the camera and points are visible
def plot_camera(extrinsic_matrix, all_camera_coordinates, size, camera_number, ax):
    ax.clear()  # Clear previous plot data

    camera_position = extrinsic_matrix[:3, 3]

    # Plot camera location obtained from extrinsic matrix
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c="r", marker="o", label=f"Camera {camera_number}")

    # Plot camera direction obtained from extrinsic matrix
    direction_vector_size = 10
    camera_direction = extrinsic_matrix[:3, :3] @ np.array([0, 0, direction_vector_size]) + camera_position
    ax.plot(
        [camera_position[0], camera_direction[0]],
        [camera_position[1], camera_direction[1]],
        [camera_position[2], camera_direction[2]],
        c="g",
        label="Camera Direction",
    )

    # Plot other camera positions
    if all_camera_coordinates:
        ax.scatter(
            [coordinates[0] for coordinates in all_camera_coordinates.values()],
            [coordinates[1] for coordinates in all_camera_coordinates.values()],
            [coordinates[2] for coordinates in all_camera_coordinates.values()],
            c="y",
            marker="o",
            label="Other Cameras",
        )

        for cam_id, coordinates in all_camera_coordinates.items():
            ax.text(coordinates[0], coordinates[1], coordinates[2], f"{cam_id}")

    # Plot volleyball court points
    volleyball_points = np.array(
        [
            [9.0, 4.5, 0.0],
            [3.0, 4.5, 0.0],
            [-3.0, 4.5, 0.0],
            [-9.0, 4.5, 0.0],
            [9.0, -4.5, 0.0],
            [3.0, -4.5, 0.0],
            [-3.0, -4.5, 0.0],
            [-9.0, -4.5, 0.0],
        ],
        dtype=np.float32,
    )
    ax.scatter(
        volleyball_points[:, 0], volleyball_points[:, 1], volleyball_points[:, 2], c="b", marker="o", label="Court Points"
    )

    ax.set_xlim([camera_position[0] - size, camera_position[0] + size])
    ax.set_ylim([camera_position[1] - size, camera_position[1] + size])
    ax.set_zlim([camera_position[2] - size, camera_position[2] + size])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Camera {camera_number} Position and Points")
    ax.legend()

    plt.draw()  # Update the plot

if __name__ == "__main__":
    # Prepare the plot figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.ion()  # Enable interactive mode

    camera_number = 13  # Initial camera
    extrinsic_matrix, all_camera_coordinates, camera_number = calculate_extrinsics(camera_number)

    if extrinsic_matrix is not None:
        plot_camera(extrinsic_matrix, all_camera_coordinates, size=35, camera_number=camera_number, ax=ax)
        plt.show()

    while True:
        try:
            # Ask the user for the camera number
            camera_number = input("Enter the camera number to use (or type 'exit' to quit): ")
            if camera_number.lower() == 'exit':
                print("Exiting the program.")
                break

            camera_number = int(camera_number)

            extrinsic_matrix, all_camera_coordinates, camera_number = calculate_extrinsics(camera_number)

            if extrinsic_matrix is not None:
                plot_camera(extrinsic_matrix, all_camera_coordinates, size=35, camera_number=camera_number, ax=ax)

        except ValueError:
            print("Invalid input. Please enter a valid camera number.")
