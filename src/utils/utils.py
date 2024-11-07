import os
import cv2
import json
import pickle
import numpy as np
from utils.config import *


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

def moving_average(data, window_size):
    if window_size < 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def get_positions():
    """Get the field corners from the pkl file."""
    with open(PATH_CAMERA_POS, 'r') as file:  
        data = json.load(file)
        return np.array(data["field_corners"]) 

def set_axes_equal_scaling(ax):
    """Set equal scaling for 3D plot axes (preservate le proporzioni)."""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    mean_vals = np.mean(limits, axis=1)
    range_vals = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim([mean_vals[0] - range_vals, mean_vals[0] + range_vals])
    ax.set_ylim([mean_vals[1] - range_vals, mean_vals[1] + range_vals])
    ax.set_zlim([mean_vals[2] - range_vals, mean_vals[2] + range_vals])

def load_existing_detections(file_path):
    """
    loads previously saved detections from a specified pickle (.pkl) file.

    parameters:
        file_path (str): path to the .pkl file containing previously saved detection data.

    returns:
        dict: a dictionary containing the loaded detections data. if the file does not exist, an empty dictionary is returned.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}


def get_projection_matrix(cam):
    """
    Create the projection matrix for a given camera using its intrinsic and extrinsic matrices.

    Parameters:
        cam (Camera): camera object containing calibration data.

    Returns:
        numpy.ndarray: 3x4 projection matrix for the camera.
    """
    K = cam.newcameramtx
    extrinsic_matrix = np.linalg.inv(cam.extrinsic_matrix)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]
    return np.dot(K, extrinsic_matrix_3x4)
