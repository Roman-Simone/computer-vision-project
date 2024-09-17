import yaml
import numpy as np
from utils import *
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)
path_yaml = os.path.join(parent_path, 'data/points.yaml')
path_calibrationMTX = os.path.join(parent_path, 'data/calibrationMatrix/calibration.pkl')

def calculate_extrinsics():
    camera_number = 3

    with open("/Users/simoneroman/Desktop/CV/Computer_Vision_project/data/points.yaml", "r") as file:
        data = yaml.safe_load(file)

        camera_points = data[camera_number]["undistorted_with_crop"]

        world_points = camera_points["world_points"]
        image_points = camera_points["image_points"]

        world_points = np.array(world_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        all_camera_coordinates = {}
        for key in data.keys():
            all_camera_coordinates[key] = np.array(data[key]["coordinates"], dtype=np.float32)

    camera_infos = load_pickle(path_calibrationMTX)

    camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)


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

    pretty_print_matrix(extrinsic_matrix)

    size = 35
    plot_camera(extrinsic_matrix, all_camera_coordinates, size)

def pretty_print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:8.4f}" for val in row))


# The size parameter is used to set the limits of the plot, so that the camera and points are visible
# 10 should be a good value for the size in most cases, but you can change it if points overflow the plot (some cameras sometimes are placed far from the volleyball court, so the points may not be visible in the plot)
def plot_camera(extrinsic_matrix, all_camera_coordinates, size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # The camera positions are in order of the camera numbers

    camera_position = extrinsic_matrix[:3, 3]

    # Plot camera location obtained from extrinsic matrix
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c="r", marker="o", label="Camera")

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
    ax.scatter(
        [coordinates[0] for coordinates in all_camera_coordinates.values()],
        [coordinates[1] for coordinates in all_camera_coordinates.values()],
        [coordinates[2] for coordinates in all_camera_coordinates.values()],
        c="y",
        marker="o",
        label="Other Cameras",
    )

    for camera_number, coordinates in all_camera_coordinates.items():
        ax.text(coordinates[0], coordinates[1], coordinates[2], str(camera_number))

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
        volleyball_points[:, 0], volleyball_points[:, 1], volleyball_points[:, 2], c="b", marker="o", label="Points"
    )

    ax.set_xlim([camera_position[0] - size, camera_position[0] + size])
    ax.set_ylim([camera_position[1] - size, camera_position[1] + size])
    ax.set_zlim([camera_position[2] - size, camera_position[2] + size])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Position and Points")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    calculate_extrinsics()
