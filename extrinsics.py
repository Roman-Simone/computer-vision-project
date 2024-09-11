# This script is used to show the extrinsic matrix of a camera.

import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import yaml


def main(args):
    camera_number = args.camera_number

    if args.with_crop:
        with_crop = True
    elif args.without_crop:
        with_crop = False
    else:
        print("Invalid arguments", file=sys.stderr)
        sys.exit(1)

    with open("points.yaml", "r") as file:
        data = yaml.safe_load(file)
        camera_points = data[camera_number]["undistorted_with_crop" if with_crop else "undistorted_without_crop"]

        world_points = camera_points["world_points"]
        image_points = camera_points["image_points"]

        world_points = np.array(world_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        all_camera_coordinates = {}
        for key in data.keys():
            all_camera_coordinates[key] = np.array(data[key]["coordinates"], dtype=np.float32)

    camera_parameters_path = os.path.join(
        "camera_parameters", "with_crop" if with_crop else "without_crop", f"out{camera_number}F.p"
    )
    camera_parameters = pickle.load(open(camera_parameters_path, "rb"))

    if not os.path.exists(camera_parameters_path):
        print("Pickle file does not exist")
        sys.exit(1)

    

    camera_matrix = camera_parameters["mtx"]
    # new_mtx does not exist in the pickle file if the image was undistorted with crop
    new_camera_matrix = None if with_crop else camera_parameters["new_mtx"]
    distortion_coefficients = np.zeros((1, 5), dtype=np.float32)

    if with_crop:
        success, rotation_vector, translation_vector = cv.solvePnP(
            world_points, image_points, camera_matrix, distortion_coefficients
        )
    else:
        success, rotation_vector, translation_vector = cv.solvePnP(
            world_points, image_points, new_camera_matrix, distortion_coefficients
        )

    if not success:
        print("Failed to solve PnP", file=sys.stderr)
        sys.exit(1)

    # Convert the rotation vector to a rotation matrix using Rodrigues
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # THIS IS IMPORTANT
    # The output from solvePnP is the rotation vector and the translation vector
    # of the world coordinate system with respect to the camera coordinate system
    # We want the inverse of this transformation, so we invert the rotation matrix
    # to get the position and rotation of the camera with respect to the world
    # Note: the rotation_matrix can be inverted by transposing it, since it is orthonormal
    # I don't know if it gives some performance improvements, but for clarity I chose to use
    # the np.linalg.inv function
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)

    extrinsic_matrix = np.hstack((inverse_rotation_matrix, inverse_translation_vector))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

    print(f"Camera {camera_number} extrinsic matrix:")

    pretty_print_matrix(extrinsic_matrix)

    size = args.size if args.size else 10
    plot_camera(extrinsic_matrix, all_camera_coordinates, size)

    # Here you can save the extrinsic matrix to a file if you want
    # with open("extrinsic_matrix.yaml", "w") as file:
    #     yaml.dump(extrinsic_matrix.tolist(), file)


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
    parser = argparse.ArgumentParser(description="Show the extrinsic matrix of a camera")

    parser.add_argument("camera_number", type=int, help="The camera number")

    undistortion_mode_group = parser.add_mutually_exclusive_group(required=True)
    undistortion_mode_group.add_argument(
        "-w",
        "--with-crop",
        action="store_true",
        help="Extract the extrinsic parameters from images undistorted with crop",
    )
    undistortion_mode_group.add_argument(
        "-wo",
        "--without-crop",
        action="store_true",
        help="Extract the extrinsic parameters from images undistorted without crop",
    )

    parser.add_argument("-s", "--size", type=int, help="The size of the plot")

    args = parser.parse_args()

    main(args)
