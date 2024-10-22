import os
import cv2
import json
import pickle
import numpy as np
from config import *

def undistorted(frame1, camera_info):   

    undistorted_frame = cv2.undistort(frame1, camera_info.mtx, camera_info.dist, None, camera_info.newcameramtx)
    x1, y1, w1, h1 = camera_info.roi
    undistorted_frame = undistorted_frame[y1:y1+h1, x1:x1+w1]

    return undistorted_frame

def save_pickle(camerasInfo, filename):
    with open(filename, 'wb') as file:
        pickle.dump(camerasInfo, file)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        camerasInfo = pickle.load(file)
    return camerasInfo


def find_files(directory):
    file_mp4 = []
    for file in os.listdir(directory):
        if file.endswith(".mp4") or file.endswith(".png"):
            file_mp4.append(file)
    return file_mp4


def read_json_file_and_structure_data(file_name):
    # Dictionary to store coordinates organized by camera
    coordinates_by_camera = {}

    # Read the JSON file
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding the JSON file {file_name}.")
        return {}

    # Iterate over each camera in the JSON
    for camera_id, camera_info in data.items():
        # Create a structure for this camera
        coordinates_by_camera[camera_id] = {
            "world_coordinates": [],
            "image_coordinates": []
        }

        # Add camera coordinates if available
        if 'camera_coordinates' in camera_info:
            coordinates_by_camera[camera_id]['camera_coordinates'] = camera_info['camera_coordinates']
        else:
            print(f"Camera coordinates not found for camera {camera_id}.")

        # Iterate over each point and save the coordinates in the respective lists
        points = camera_info.get('points', [])
        for point in points:
            world_coord = point.get('world_coordinate', [])
            image_coord = point.get('image_coordinate', [])
            
            if world_coord and image_coord:
                coordinates_by_camera[camera_id]["world_coordinates"].append(world_coord)
                coordinates_by_camera[camera_id]["image_coordinates"].append(image_coord)

    return coordinates_by_camera


def take_info_camera(camera_number, camera_infos):

    for pos, camera_info in enumerate(camera_infos):
        if camera_info.camera_number == camera_number:
            return camera_info, pos

    return None, None


def get_positions():
    with open(PATH_CAMERA_POS, "r") as file:
        data = json.load(file)
        positions = data["positions"]
        field_corners = np.array(data["field_corners"]) * 1000
    return positions, field_corners


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
