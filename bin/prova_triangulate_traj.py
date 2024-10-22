import numpy as np
import cv2
from config import *
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
cam = [take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS]

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600),
    7: (5150, 5330)
}


def load_existing_results(filename):
    """Helper function to load existing results from a pickle file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def triangulate_points_multicam(points_2d_list, projection_matrices):
    """
    Triangulate the 3D point using multiple cameras by solving a least-squares problem.

    :param points_2d_list: List of 2D points from each camera [Nx2]
    :param projection_matrices: List of projection matrices for each camera [Nx3x4]
    :return: The triangulated 3D point (1x3)
    """
    A = []
    
    for i in range(len(points_2d_list)):
        P = projection_matrices[i]
        x, y = points_2d_list[i]

        # Each 2D point gives two equations from the projection matrix
        A.append(x * P[2] - P[0])  # x_row
        A.append(y * P[2] - P[1])  # y_row

    # Solve the system of equations using SVD (Singular Value Decomposition)
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    
    X = Vt[-1]  # Last row of V gives the solution

    # Convert homogeneous coordinates to 3D
    X_3d = X[:3] / X[3]
    
    return X_3d


def get_projection_matrix(cam):
    K = cam.newcameramtx  
    
    extrinsic_matrix = cam.extrinsic_matrix  
    
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]
    
    return np.dot(K, extrinsic_matrix_3x4)
    

def remove_consecutive_duplicates(points):
    """Helper function to remove consecutive identical points from a list."""
    cleaned_points = []
    for i, point in enumerate(points):
        if i == 0 or point != points[i - 1]:  # Keep if it's not a consecutive duplicate
            cleaned_points.append(point)
    return cleaned_points

def match_point_counts(points1, points2):
    """
    Adjust the number of points in the longer list by removing consecutive duplicates
    until both lists have the same length.
    """
    # Identify the longer and shorter list
    if len(points1) > len(points2):
        longer_list = points1
        shorter_list = points2
        print("Longer list is points1, with delta: ", len(points1) - len(points2))
    else:
        longer_list = points2
        shorter_list = points1
        print("Longer list is points2, with delta: ", len(points2) - len(points1))
        

    # Remove consecutive duplicates from the longer list until the lengths match
    cleaned_longer_list = remove_consecutive_duplicates(longer_list)

    while len(cleaned_longer_list) > len(shorter_list):
        # Check if there are still consecutive duplicates to remove
        cleaned_longer_list = remove_consecutive_duplicates(cleaned_longer_list)

        # If the lengths match, stop
        if len(cleaned_longer_list) == len(shorter_list):
            break

        # If no more consecutive duplicates, we might randomly remove an extra point
        if len(cleaned_longer_list) > len(shorter_list):
            cleaned_longer_list.pop()  # Simply remove the last point

    # Return the updated lists with matched lengths
    if len(points1) > len(points2):
        return cleaned_longer_list, shorter_list
    else:
        return shorter_list, cleaned_longer_list

def process_trajectories_for_action_multicam(results, action_id, cam_numbers):
    """
    Process the trajectories for a given action using multiple cameras and triangulate the 3D points.

    :param results: Dictionary containing the trajectories for each camera-action pair
    :param action_id: The action ID to process (1, 2, 3, ...)
    :param cam_numbers: A list of camera numbers to use for triangulation
    :return: A list of 3D points representing the triangulated trajectory
    """
    points_2d_list = []  # List of 2D points for each camera
    projection_matrices = []  # List of projection matrices for each camera

    for cam_number in cam_numbers:
        points_2d = results[str(cam_number)].get(str(action_id), [])
        points_2d = remove_consecutive_duplicates(points_2d)

        if len(points_2d) == 0:
            print(f"No valid points found for Camera {cam_number} for Action {action_id}")
            continue
        
        # Get projection matrix for this camera
        cam = take_info_camera(cam_number, camerasInfo)[0]
        P = get_projection_matrix(cam)

        points_2d_list.append(points_2d)
        projection_matrices.append(P)

    # Ensure all camera lists have the same number of points by adjusting the longest list
    min_len = min([len(pts) for pts in points_2d_list])
    for i in range(len(points_2d_list)):
        points_2d_list[i] = points_2d_list[i][:min_len]

    if len(points_2d_list) < 2:
        print(f"Not enough cameras with valid points to triangulate for Action {action_id}")
        return []

    # Now triangulate using multiple cameras
    points_3d = []

    for i in range(min_len):
        # Collect the 2D points for this frame from all cameras
        frame_points_2d = [points_2d_list[cam_idx][i] for cam_idx in range(len(cam_numbers))]

        # Triangulate the 3D point using all cameras
        point_3d = triangulate_points_multicam(frame_points_2d, projection_matrices)
        points_3d.append(point_3d)

    return np.array(points_3d)

def get_positions():
    with open(PATH_CAMERA_POS, "r") as file:
        data = json.load(file)
        return np.array(data["field_corners"]) 


def plot_3d_trajectory(points_3d, action_id):
    """
    Plot the 3D trajectory using matplotlib.

    :param points_3d: List of 3D points (Nx3 array)
    :param action_id: The action ID for which the trajectory is being plotted
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    field_corners = get_positions()

    # Pre-plot the static field corners once
    # ax.scatter(field_corners[:, 0], field_corners[:, 1], field_corners[:, 2], c="red", label="Court Corners")
    
    points_3d = np.array(points_3d)
    
    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], marker='o', label=f"Action {action_id}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Trajectory for Action {action_id}')
    ax.legend()
    
    plt.show()

if __name__ == '__main__':
    # Load your pickled trajectory results
    pickle_file = os.path.join(PATH_DETECTIONS, 'traj_detections.pkl')
    results = load_existing_results(pickle_file)  # Load 2D trajectories

    # Prompt user for input
    available_actions = list(ACTIONS.keys())
    print(f"Available actions: {available_actions}")
    
    try:
        action_id = int(input(f"Select an action from the available actions {available_actions}: "))
        if action_id not in ACTIONS:
            print("Invalid action selected. Exiting.")
            exit()
    except ValueError:
        print("Invalid input. Please enter a number corresponding to the action.")
        exit()

    # Use all valid cameras for triangulation
    cam_numbers = VALID_CAMERA_NUMBERS  # This contains all camera IDs

    # Process the selected action using multiple cameras
    print(f"Processing Action {action_id} with Cameras {cam_numbers}...")

    points_3d = process_trajectories_for_action_multicam(results, action_id, cam_numbers)

    if len(points_3d) > 0:
        # Plot the 3D trajectory for the selected action
        plot_3d_trajectory(points_3d, action_id)
    else:
        print(f"No 3D points found for Action {action_id} using Cameras {cam_numbers}.")
