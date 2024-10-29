import numpy as np
import pickle
import os
import sys
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# File path and loading configuration
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

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

action_number = input("Enter the action number: ")

# Load detections
detections_path = os.path.join(PATH_3D_DETECTIONS_05, f'points_3d_action{action_number}.pkl')
with open(detections_path, "rb") as f:
    detections = pickle.load(f)

num_particles = 1000
process_noise_std = 0.5
measurement_noise_std = 1.0
initial_state_std = 1.0
outlier_threshold = 4.0  # Define threshold for outlier rejection

# Initial particle states around the first detection
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

# Prediction function
def predict(particles, dt=1.0):
    noise = np.random.normal(0, process_noise_std, size=particles.shape)
    particles[:, :3] += particles[:, 3:] * dt + noise[:, :3]
    particles[:, 3:] += noise[:, 3:]
    return particles

# Update weights based on the closest detection
def update_weights(particles, detection):
    if detection is not None:
        distances = np.linalg.norm(particles[:, :3] - detection, axis=1)
        weights = np.exp(-0.5 * (distances / measurement_noise_std) ** 2)
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights /= weights_sum  # Normalize weights
        else:
            # If sum is zero, reset weights uniformly to avoid NaNs
            weights = np.ones(len(particles)) / len(particles)
    else:
        # No valid detection; set uniform weights
        weights = np.ones(len(particles)) / len(particles)
    return weights

# Resample particles
def resample(particles, weights):
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

# Particle filter with trajectory interpolation
trajectory = []
frames_with_detections = sorted(detections.keys())
for frame in range(frames_with_detections[0], frames_with_detections[-1] + 1):
    detection_list = detections.get(frame, [])
    
    if detection_list:
        # Predict step
        particles = predict(particles)

        # Select closest detection
        particle_mean = particles[:, :3].mean(axis=0)
        closest_detection = min(detection_list, key=lambda d: np.linalg.norm(d - particle_mean))

        # Apply bounds and outlier check
        if -15 < closest_detection[0] < 15 and -8 < closest_detection[1] < 8 and 0 < closest_detection[2] < 10:
            # Check if the closest detection is too far from the last point in the trajectory
            if trajectory:
                last_position = np.array(trajectory[-1][1:])
                distance_to_last = np.linalg.norm(last_position - closest_detection)
                # if distance_to_last > outlier_threshold:
                #     continue  # Skip this detection if it's an outlier

            # Update and resample
            weights = update_weights(particles, closest_detection)
            particles = resample(particles, weights)

            # Estimate current position
            estimated_state = particles[:, :3].mean(axis=0)
            trajectory.append((frame, *estimated_state))
    else:
        # Only prediction (no detection available)
        particles = predict(particles)
        estimated_state = particles[:, :3].mean(axis=0)

        # Apply bounds check for predicted state
        if -15 < estimated_state[0] < 15 and -8 < estimated_state[1] < 8 and 0 < estimated_state[2] < 10:
            trajectory.append((frame, *estimated_state))

# Prepare data for interpolation and plotting
frames, xs, ys, zs = zip(*trajectory)

# Spline smoothing for x, y, z coordinates
smoothing_factor = 5  # Adjust this factor for more or less smoothing
spline_x = UnivariateSpline(frames, xs, s=smoothing_factor)
spline_y = UnivariateSpline(frames, ys, s=smoothing_factor)
spline_z = UnivariateSpline(frames, zs, s=smoothing_factor)

# Define frames for complete interpolation
interp_frames = np.arange(frames[0], frames[-1] + 1, 1)
interp_xs = spline_x(interp_frames)
interp_ys = spline_y(interp_frames)
interp_zs = spline_z(interp_frames)

# Plotting the smoothed 3D trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
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
