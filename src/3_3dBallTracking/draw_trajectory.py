import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.config import *
from utils.utils import *

CONFIDENCE = 0.4

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def get_positions():
    with open(PATH_CAMERA_POS, 'r') as file:  
        data = json.load(file)
        return np.array(data["field_corners"]) 

def set_axes_equal_scaling(ax):
    """Set equal scaling for 3D plot axes."""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    mean_vals = np.mean(limits, axis=1)
    range_vals = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim([mean_vals[0] - range_vals, mean_vals[0] + range_vals])
    ax.set_ylim([mean_vals[1] - range_vals, mean_vals[1] + range_vals])
    ax.set_zlim([mean_vals[2] - range_vals, mean_vals[2] + range_vals])


def main():

    action_number = int(input("Enter the action number: "))
    if CONFIDENCE == 0.4:
        trajectory_data = load_pickle(os.path.join(PATH_3D_DETECTIONS_04, f'points_3D_action{action_number}.pkl'))  
    elif CONFIDENCE == 0.5:
        trajectory_data = load_pickle(os.path.join(PATH_3D_DETECTIONS_05, f'points_3D_action{action_number}.pkl'))
        
    selected_frames = []
    selected_points = []
    threshold_distance = 5  

    for frame in sorted(trajectory_data.keys()):
        points = trajectory_data[frame]
        
        if not points:
            continue  # skip frames with no points

        points = [point for point in points if (-15 < point[0] < 15 and -8 < point[1] < 8 and 0 < point[2] < 10)]  # out of field 
        
        if not points:
            continue

        if not selected_points:
            best_point = points[0]  
        else:
            # point closest to last selected point
            last_point = selected_points[-1]
            distances = [np.linalg.norm(point - last_point) for point in points]
            best_point = points[np.argmin(distances)]
            
            if np.linalg.norm(best_point - last_point) > threshold_distance:
                continue  

        selected_frames.append(frame)
        selected_points.append(best_point)

    selected_points = np.array(selected_points)

    # if selected_points.ndim != 2 or selected_points.shape[1] != 3:
    #     print(f"Warning: Selected points shape is {selected_points.shape}. Expected shape is (N, 3).")
    #     return  

    window_size = 5  
    smoothed_points = np.convolve(selected_points[:, 0], np.ones(window_size) / window_size, mode='valid')
    smoothed_points = np.vstack([smoothed_points, 
                                 np.convolve(selected_points[:, 1], np.ones(window_size) / window_size, mode='valid'),
                                 np.convolve(selected_points[:, 2], np.ones(window_size) / window_size, mode='valid')]).T

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')
    set_axes_equal_scaling(ax)
    
    ax.set_xlim([-15, 15])
    ax.set_ylim([-8.5, 8.5])
    ax.set_zlim([-0.1, 10])

    field_points = get_positions()
    ax.scatter(field_points[:, 0], field_points[:, 1], field_points[:, 2], color='red', label='Field Corners')

    ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], color='lightblue', marker='o', label='Detections')

    ax.plot(smoothed_points[:, 0], smoothed_points[:, 1], smoothed_points[:, 2], color='blue', linewidth=2, label='Smoothed Trajectory')

    ax.set_title(f'3D Ball Trajectory for action {action_number}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
