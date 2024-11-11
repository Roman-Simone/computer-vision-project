import os
import cv2
import json
import pickle
import numpy as np
from utils.config import *


def undistorted(frame1, camera_info):
    """
    Undistorts a given frame using the provided camera calibration data.

    Parameters:
        frame1 (numpy.ndarray): the original image frame to be undistorted.
        camera_info (CameraInfo): contains the camera calibration parameters including intrinsic matrix, 
                                  distortion coefficients, new camera matrix, and region of interest (ROI).

    Returns:
        numpy.ndarray: the undistorted frame cropped to the region of interest.
    """
    undistorted_frame = cv2.undistort(frame1, camera_info.mtx, camera_info.dist, None, camera_info.newcameramtx)
    x1, y1, w1, h1 = camera_info.roi
    undistorted_frame = undistorted_frame[y1:y1+h1, x1:x1+w1]
    return undistorted_frame

def save_pickle(camerasInfo, filename):
    """
    Saves the camera information to a pickle file.

    Parameters:
        camerasInfo (object): The camera information object to be saved.
        filename (str): The path and name of the file to save the data to.
    """
    with open(filename, 'wb') as file:
        pickle.dump(camerasInfo, file)

def load_pickle(filename):
    """
    Loads camera information from a pickle file.

    Parameters:
        filename (str): The path and name of the pickle file to load data from.

    Returns:
        object: The loaded camera information.
    """
    with open(filename, 'rb') as file:
        camerasInfo = pickle.load(file)
    return camerasInfo

def find_files(directory):
    """
    Searches for all .mp4 and .png files in a specified directory.

    Parameters:
        directory (str): The directory path to search.

    Returns:
        list: A list of file names with .mp4 or .png extensions.
    """
    file_mp4 = []
    for file in os.listdir(directory):
        if file.endswith(".mp4") or file.endswith(".png"):
            file_mp4.append(file)
    return file_mp4

def read_json_file_and_structure_data(file_name):
    """
    Reads a JSON file and structures data by organizing coordinates by camera.

    Parameters:
        file_name (str): The path to the JSON file to be read.

    Returns:
        dict: A dictionary with organized camera data, including world and image coordinates.
    """
    coordinates_by_camera = {}
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding the JSON file {file_name}.")
        return {}

    for camera_id, camera_info in data.items():
        coordinates_by_camera[camera_id] = {
            "world_coordinates": [],
            "image_coordinates": []
        }

        if 'camera_coordinates' in camera_info:
            coordinates_by_camera[camera_id]['camera_coordinates'] = camera_info['camera_coordinates']
        else:
            print(f"Camera coordinates not found for camera {camera_id}.")

        points = camera_info.get('points', [])
        for point in points:
            world_coord = point.get('world_coordinate', [])
            image_coord = point.get('image_coordinate', [])
            
            if world_coord and image_coord:
                coordinates_by_camera[camera_id]["world_coordinates"].append(world_coord)
                coordinates_by_camera[camera_id]["image_coordinates"].append(image_coord)

    return coordinates_by_camera

def take_info_camera(camera_number, camera_infos):
    """
    Retrieves information about a specific camera by camera number.

    Parameters:
        camera_number (int): The number of the camera to retrieve information for.
        camera_infos (list): A list of camera information objects.

    Returns:
        tuple: A tuple containing the camera info object and its position in the list, or (None, None) if not found.
    """
    for pos, camera_info in enumerate(camera_infos):
        if camera_info.camera_number == camera_number:
            return camera_info, pos 
    return None, None

def moving_average(data, window_size):
    """
    Calculates the moving average of a data sequence over a specified window size.

    Parameters:
        data (list or numpy.ndarray): The data sequence for which to calculate the moving average.
        window_size (int): The window size for the moving average calculation.

    Returns:
        numpy.ndarray: The calculated moving average values.
    """
    if window_size < 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def get_positions():
    """
    Retrieves field corner positions from a predefined pickle file.

    Returns:
        numpy.ndarray: An array of field corner positions as specified in the pickle file.
    """
    with open(PATH_CAMERA_POS, 'r') as file:  
        data = json.load(file)
        return np.array(data["field_corners"])

def set_axes_equal_scaling(ax):
    """
    Sets equal scaling on the 3D plot axes to preserve proportions.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes of a Matplotlib plot.
    """
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    mean_vals = np.mean(limits, axis=1)
    range_vals = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim([mean_vals[0] - range_vals, mean_vals[0] + range_vals])
    ax.set_ylim([mean_vals[1] - range_vals, mean_vals[1] + range_vals])
    ax.set_zlim([mean_vals[2] - range_vals, mean_vals[2] + range_vals])

def load_existing_detections(file_path):
    """
    Loads previously saved detections from a pickle file if it exists.

    Parameters:
        file_path (str): The path to the pickle file containing detection data.

    Returns:
        dict: A dictionary with detection data, or an empty dictionary if the file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

def get_projection_matrix(cam):
    """
    Creates the projection matrix for a camera using its intrinsic and extrinsic matrices.

    Parameters:
        cam (Camera): Camera object containing calibration data such as the intrinsic matrix and extrinsic matrix.

    Returns:
        numpy.ndarray: A 3x4 projection matrix for the camera.
    """
    K = cam.newcameramtx
    extrinsic_matrix = np.linalg.inv(cam.extrinsic_matrix)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]
    return np.dot(K, extrinsic_matrix_3x4)
