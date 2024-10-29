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

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def get_positions():
    with open(PATH_CAMERA_POS, 'r') as file:  
        data = json.load(file)
        return np.array(data["field_corners"]) 

def main():

    action_number = int(input("Enter the action number: "))
    trajectory_data = load_pickle(os.path.join(PATH_3D_DETECTIONS, f'points_3D_action{action_number}.pkl'))  

    selected_frames = []
    selected_points = []
    threshold_distance = 5  

    for frame in sorted(trajectory_data.keys()):
        points = trajectory_data[frame]
        
        if not points:
            continue  # skip frames with no points

        points = [point for point in points if (-15 < point[0] < 15 and -10 < point[1] < 10 and 0 < point[2] < 10)]  # out of field 
        
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
