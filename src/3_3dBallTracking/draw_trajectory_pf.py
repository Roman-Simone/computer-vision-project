import numpy as np
import pickle
import os
import sys
import json
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

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

action_number = input("Enter the action number: ")
while not action_number.isdigit() and int(action_number) not in ACTIONS:
    action_number = input("Enter a valid action number: ")

detections_path = os.path.join(PATH_3D_DETECTIONS_04, f'points_3d_action{action_number}.pkl')
with open(detections_path, "rb") as f:
    detections = pickle.load(f)

num_particles = 1000
process_noise_std = 1.0  # process noise 
measurement_noise_std = 2.0  # measurement noise
initial_state_std = 1.0
outlier_threshold = 4.0  # thresh for outlier rejection

# particle states for first detection
first_detection_frame = next((k for k, v in detections.items() if len(v) > 0), None)
if first_detection_frame is not None:
    first_detection = np.mean(detections[first_detection_frame], axis=0)
else:
    raise ValueError("No detections available to initialize")

particles = np.random.normal(
    np.concatenate([first_detection, [0, 0, 0]]),
    initial_state_std,
    size=(num_particles, 6)
)

def predict(particles, dt=1.0):
    noise = np.random.normal(0, process_noise_std, size=particles.shape)    # process noise for each particle
    particles[:, :3] += particles[:, 3:] * dt + noise[:, :3]                # update position based on velocity
    particles[:, 3:] += noise[:, 3:]                                        # update velocity with noise
    return particles

def update_weights(particles, detection):
    ''''We decide to update weights based on the closest detection.'''
    if detection is not None:
        distances = np.linalg.norm(particles[:, :3] - detection, axis=1)
        weights = np.exp(-0.5 * (distances / measurement_noise_std) ** 2)
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights /= weights_sum  
        else:
            weights = np.ones(len(particles)) / len(particles)  
    else:
        weights = np.ones(len(particles)) / len(particles)  # if no valid detection set uniform weights
    return weights

def resample(particles, weights):
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

def moving_average(data, window_size):
    ''''This moving average function is used to smooth the trajectory.'''
    if window_size < 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

trajectory = []
frames_with_detections = sorted(detections.keys())
for frame in range(frames_with_detections[0], frames_with_detections[-1] + 1):
    detection_list = detections.get(frame, [])
    
    if detection_list:
        particles = predict(particles)          # predict the next state

        particle_mean = particles[:, :3].mean(axis=0)
        closest_detection = min(detection_list, key=lambda d: np.linalg.norm(d - particle_mean))    # closest detection

        # bounds and outlier check
        if -15 < closest_detection[0] < 15 and -8 < closest_detection[1] < 8 and 0 < closest_detection[2] < 10:
            if trajectory:
                last_position = np.array(trajectory[-1][1:])
                distance_to_last = np.linalg.norm(last_position - closest_detection)
                # skip detection if it's an outlier
                if distance_to_last > outlier_threshold:
                    continue

            # update weights and resample
            weights = update_weights(particles, closest_detection)
            particles = resample(particles, weights)

            # estimate current position
            estimated_state = particles[:, :3].mean(axis=0)
            trajectory.append((frame, *estimated_state))
    else:
        # case in which we have only prediction (no detection available)
        particles = predict(particles)
        estimated_state = particles[:, :3].mean(axis=0)

        if -15 < estimated_state[0] < 15 and -8 < estimated_state[1] < 8 and 0 < estimated_state[2] < 10:
            trajectory.append((frame, *estimated_state))

frames, xs, ys, zs = zip(*trajectory)

# Spline smoothing for x, y, z coordinates
smoothing_factor = 10     # Adjust this factor for more smoothing
spline_x = UnivariateSpline(frames, xs, s=smoothing_factor)
spline_y = UnivariateSpline(frames, ys, s=smoothing_factor)
spline_z = UnivariateSpline(frames, zs, s=smoothing_factor)

interp_frames = np.arange(frames[0], frames[-1] + 1, 1)
interp_xs = spline_x(interp_frames)
interp_ys = spline_y(interp_frames)
interp_zs = spline_z(interp_frames)

# Apply moving average for smoothing
window_size = 9  # Adjust window size based on preference
interp_xs = moving_average(interp_xs, window_size)
interp_ys = moving_average(interp_ys, window_size)
interp_zs = moving_average(interp_zs, window_size)

interp_frames = interp_frames[window_size - 1:]  

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