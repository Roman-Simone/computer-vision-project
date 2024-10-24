import os
import cv2
import sys
import torch
import pickle
import random
import numpy as np
from ultralytics import YOLO

# Add the parent directory to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

# Now you can import the utils module from the parent directory
from utils.utils import *
from utils.config import *

# Action frame ranges
ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600),
    7: (5150, 5330)
}

# {
#     '1' : {     # first camera
#         '1' : [...],  # list of points of the trajectory of the first action from the first camera
#         '2' : [...] # list of points of the trajectory of the second action from the first camera
#         ....
#     }, 
#     '2' : {     # second camera
#         '1' : [...],  # list of points of the trajectory of the first action from the second camera
#         '2' : [...] # list of points of the etrajectory of the second action from the second camera
#         ....
#     }
#     ...
# }


# Ask user to select an action (1-8)
# action = int(input("Select the action to process (1-7): "))
# if action not in ACTIONS:
#     print("Invalid action selected. Please choose between 1 and 7.")
#     exit()

# Set START and END based on the action chosen
# START, END = ACTIONS[action]

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)
model = YOLO(pathWeight)

# Select the device to use (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

size = 800
DISTANCE_THRESHOLD = 200  # Define a threshold distance to detect a new ball

# Define particle filter ball tracker class
class ParticleFilterBallTracker:
    def __init__(self, tracker_id, color, num_particles=1000, process_noise=5.0, measurement_noise=2.0):
        self.tracker_id = tracker_id
        self.color = color
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.rand(self.num_particles, 2) * size
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.ball_positions = []
        self.last_position = None
        self.active = True
        self.frames_without_detection = 0
        
    def predict(self):
        if len(self.ball_positions) >= 2:
            # Calculate velocity from last two positions
            velocity = np.array(self.ball_positions[-1]) - np.array(self.ball_positions[-2])
            self.particles += velocity
        
        # Add noise to particles
        self.particles += np.random.randn(self.num_particles, 2) * self.process_noise
        self.particles = np.clip(self.particles, 0, size-1)

    def update(self, measurement):
        if measurement == (-1, -1):
            self.frames_without_detection += 1
            if self.frames_without_detection > 10:  # Deactivate tracker if no detections for too long
                self.active = False
            return

        self.frames_without_detection = 0
        
        if self.last_position is not None:
            distance = np.linalg.norm(np.array(measurement) - np.array(self.last_position))
            if distance > DISTANCE_THRESHOLD:
                self.reset_tracker(measurement)
                return
                
        self.last_position = measurement
        self.ball_positions.append(measurement)
        
        # Calculate weights based on distance to measurement
        distances = np.linalg.norm(self.particles - np.array(measurement), axis=1)
        self.weights = np.exp(-distances**2 / (2 * self.measurement_noise**2))
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        
        # Resample particles
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        
        # Add some particles near the measurement
        num_measurement_particles = self.num_particles // 4  # 25% of particles
        measurement_particles = np.random.normal(measurement, self.measurement_noise/2, 
                                               size=(num_measurement_particles, 2))
        self.particles[-num_measurement_particles:] = measurement_particles
        
        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles

    def reset_tracker(self, measurement):
        print(f"Resetting tracker {self.tracker_id}")
        self.ball_positions = [measurement]
        self.last_position = measurement
        spread = 20  # Initial spread around the measurement
        self.particles = np.random.normal(measurement, spread, size=(self.num_particles, 2))
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def estimate(self):
        if len(self.ball_positions) > 0:
            return np.array(self.ball_positions[-1]).astype(int)
        return np.average(self.particles, weights=self.weights, axis=0).astype(int)

    def draw_particles(self, frame):
        # Draw only a subset of particles for better visualization
        subset_size = min(100, self.num_particles)
        particle_indices = np.random.choice(self.num_particles, subset_size, replace=False)
        for particle in self.particles[particle_indices]:
            pos = tuple(np.clip(particle, 0, size-1).astype(int))
            cv2.circle(frame, pos, 1, self.color, -1)

    def draw_estimated_position(self, frame):
        if len(self.ball_positions) > 0:
            pos = tuple(np.clip(self.estimate(), 0, size-1).astype(int))
            cv2.circle(frame, pos, 5, self.color, -1)

    def draw_trajectory(self, frame):
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                pt1 = tuple(np.clip(self.ball_positions[i-1], 0, size-1).astype(int))
                pt2 = tuple(np.clip(self.ball_positions[i], 0, size-1).astype(int))
                cv2.line(frame, pt1, pt2, self.color, 2)

def applyModel(frame, model):
    results = model.track(frame, save=False, verbose=False, device=device)
    
    center_ret = (-1, -1)
    confidence = -1
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]

        if class_id == 0 and confidence > 0.5:
            x_center = (x1 + x2) / 2    
            y_center = (y1 + y2) / 2
            center_ret = (int(x_center), int(y_center))
            detections.append(center_ret)
            
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    return detections, center_ret, confidence


def testModel(num_cam, action):
    """Process the video for the given camera and action, return trajectory points"""
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameraInfos)
    videoCapture = cv2.VideoCapture(pathVideo)

    START, END = ACTIONS[action]  # Set frame range based on the action
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, START)

    trackers = []
    trajectory_points = []

    while True:
        current_frame = int(videoCapture.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > END:
            break

        ret, frame = videoCapture.read()
        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)
        frameUndistorted = cv2.resize(frameUndistorted, (size, size))
        detections, center, confidence = applyModel(frameUndistorted, model)

        new_trackers = []
        for detection in detections:
            matched = False
            for tracker in trackers:
                distance = np.linalg.norm(np.array(tracker.last_position) - np.array(detection)) if tracker.last_position else float('inf')
                if distance < DISTANCE_THRESHOLD:
                    tracker.update(detection)
                    matched = True
                    break

            if not matched:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_tracker = ParticleFilterBallTracker(len(trackers), color)
                new_tracker.update(detection)
                new_trackers.append(new_tracker)

        trackers.extend(new_trackers)

        for tracker in trackers:
            trajectory_points.append(tracker.last_position)
            tracker.update(tracker.last_position)
            tracker.predict()
            tracker.draw_particles(frameUndistorted)
            tracker.draw_estimated_position(frameUndistorted)
            tracker.draw_trajectory(frameUndistorted)

        cv2.imshow('Frame', frameUndistorted)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

    return trajectory_points

def load_existing_results(filename):
    """Helper function to load existing results from a pickle file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

if __name__ == '__main__':
    pickle_file = 'ball_trajectories.pkl'
    results = load_existing_results(pickle_file)  # Load existing data if available

    for cam in VALID_CAMERA_NUMBERS:
        if str(cam) not in results:
            results[str(cam)] = {}  # Initialize a dictionary for each camera

        for action in ACTIONS:
            # If this action for this camera is already processed, skip it
            # if str(action) in results[str(cam)]:
            #     print(f"Skipping Camera {cam}, Action {action} (already processed).")
            #     continue

            print(f"Processing Camera {cam}, Action {action}...")
            trajectory = testModel(cam, action)
            results[str(cam)][str(action)] = trajectory  # Store the trajectory points for this camera-action pair

            # Save the updated results after processing each action
            with open(pickle_file, 'wb') as f:
                pickle.dump(results, f)

            print(f"Camera {cam}, Action {action} saved to {pickle_file}")

    print(f"Processing complete. Results saved in {pickle_file}")
