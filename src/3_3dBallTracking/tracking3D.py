import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from utils3DBallTracking.particleFilter3D import ParticleFilter

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

outlier_threshold = 4.0
SMOOTHING_FACTOR = 2
WINDOW_SIZE = 8

SEED_VAL = 61
np.random.seed(SEED_VAL)

def smooth_trajectory(trajectory):
    """
    Smooths the 3D trajectory data using spline interpolation and a moving average filter.

    Parameters:
        trajectory (list): list of tuples containing frame number and (x, y, z) coordinates.

    Returns:
        tuple: containing arrays of interpolated and smoothed x, y, z coordinates along with frames.
    """
    
    frames, xs, ys, zs = zip(*trajectory)

    spline_x = UnivariateSpline(frames, xs, s=SMOOTHING_FACTOR)
    spline_y = UnivariateSpline(frames, ys, s=SMOOTHING_FACTOR)
    spline_z = UnivariateSpline(frames, zs, s=SMOOTHING_FACTOR)

    interp_frames = np.arange(frames[0], frames[-1] + 1, 1)
    interp_xs = spline_x(interp_frames)
    interp_ys = spline_y(interp_frames)
    interp_zs = spline_z(interp_frames)

    interp_xs = moving_average(interp_xs, WINDOW_SIZE)
    interp_ys = moving_average(interp_ys, WINDOW_SIZE)
    interp_zs = moving_average(interp_zs, WINDOW_SIZE)
    interp_frames = interp_frames[WINDOW_SIZE - 1:]

    return interp_frames, interp_xs, interp_ys, interp_zs


def plot_trajectory(interp_xs, interp_ys, interp_zs, trajectory, action_number):
    """
    Plots the original and smoothed 3D trajectory along with field points.

    Parameters:
        interp_frames (array-like): list of interpolated frame numbers.
        interp_xs (array-like): interpolated x-coordinates.
        interp_ys (array-like): interpolated y-coordinates.
        interp_zs (array-like): interpolated z-coordinates.
        trajectory (list): list of tuples containing original frame number and (x, y, z) coordinates.
        action_number (int): action number for labeling the plot.

    Returns:
        None
    """

    _, xs, ys, zs = zip(*trajectory)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    set_axes_equal_scaling(ax)

    ax.set_xlim([-15, 15])
    ax.set_ylim([-8.5, 8.5])
    ax.set_zlim([-0.1, 10])

    field_points = get_positions()
    ax.scatter(field_points[:, 0], field_points[:, 1], field_points[:, 2], color='red', label='Field Corners')

    ax.plot(interp_xs, interp_ys, interp_zs, label="Smoothed Trajectory", color="blue", lw=2)
    ax.scatter(xs, ys, zs, label="Detections", color="lightblue", s=15)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.legend()
    plt.title(f"3D Ball Tracking of action {action_number}")
    plt.show()


def track_ball_pf_3D():
    
    """
    Tracks a 3D ball using a particle filter based on detections from a specified action.
    The function initializes a particle filter with the first valid detection, and iteratively updates the filter
    based on subsequent detections to estimate the ball's trajectory. At the end it applies smoothing 
    operations to the trajectory and plotting the results.    
    """
    
    action_number = int(input("Enter the action number: "))
    while action_number not in ACTIONS:
        action_number = int(input("Enter the action number: "))

    detections_path = os.path.join(PATH_3D_DETECTIONS_04, f'points_3d_action{action_number}.pkl')
    with open(detections_path, "rb") as f:
        detections = pickle.load(f)

    first_detection_frame = next((k for k, v in detections.items() if len(v) > 0), None)
    if first_detection_frame is not None:
        first_detection_frame_cut = [
            d for d in detections[first_detection_frame] if -15 < d[0] < 15 and -8 < d[1] < 8 and 0 < d[2] < 10
        ]
        first_detection = np.mean(first_detection_frame_cut, axis=0)
    else:
        raise ValueError("No detections available to initialize")

    particle_filter = ParticleFilter(
        initial_state=first_detection,
        num_particles=1000,
        process_noise_std=1.0,
        measurement_noise_std=2.0,
        initial_state_std=1.0
    )
    
    trajectory = []
    frames_with_detections = sorted(detections.keys())
    for frame in range(frames_with_detections[0], frames_with_detections[-1] + 1):
        detection_list = detections.get(frame, [])

        if detection_list:
            particle_filter.predict()
            particle_mean = particle_filter.estimate()
            closest_detection = min(detection_list, key=lambda d: np.linalg.norm(d - particle_mean))

            if -15 < closest_detection[0] < 15 and -8 < closest_detection[1] < 8 and 0 < closest_detection[2] < 10:
                if trajectory:
                    last_position = np.array(trajectory[-1][1:])
                    distance_to_last = np.linalg.norm(last_position - closest_detection)
                    if distance_to_last > outlier_threshold:
                        continue

                weights = particle_filter.update_weights(closest_detection)
                particle_filter.resample(weights)
                estimated_state = particle_filter.estimate()
                trajectory.append((frame, *estimated_state))
        else:
            particle_filter.predict()
            estimated_state = particle_filter.estimate()
            if -15 < estimated_state[0] < 15 and -8 < estimated_state[1] < 8 and 0 < estimated_state[2] < 10:
                trajectory.append((frame, *estimated_state))

    _, interp_xs, interp_ys, interp_zs = smooth_trajectory(trajectory)

    plot_trajectory(interp_xs, interp_ys, interp_zs, trajectory, action_number)


if __name__ == "__main__":
    
    track_ball_pf_3D()
