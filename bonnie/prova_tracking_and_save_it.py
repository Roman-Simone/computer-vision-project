import numpy as np
import cv2
import torch
from ultralytics import YOLO
from config import *
from utils import *
import os
import pickle  # Import pickle for saving the data
import random  # Import random for color generation

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)
END = 30 * 120  # 120 seconds at 30 fps
model = YOLO(pathWeight)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

size = 800
DISTANCE_THRESHOLD = 200  # Define a threshold distance to detect a new ball

# Define particle filter ball tracker
class ParticleFilterBallTracker:
    def __init__(self, tracker_id, color, num_particles=1000, process_noise=1.0, measurement_noise=2.0):
        self.tracker_id = tracker_id  # Unique ID for the tracker
        self.color = color  # Color for this tracker
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.rand(self.num_particles, 2) * size  # Initialize particles randomly within frame size
        self.weights = np.ones(self.num_particles) / self.num_particles  # Equal weights initially
        self.ball_positions = []  # Store the ball's previous positions
        self.last_position = None  # Store the last detected ball position

    def predict(self):
        # Add random motion to particles (process noise)
        self.particles += np.random.randn(self.num_particles, 2) * self.process_noise
        self.particles = np.clip(self.particles, 0, size)  # Ensure particles stay within frame bounds

    def update(self, measurement):
        if measurement != (-1, -1):
            # Check if this is a new ball by comparing distances
            if self.last_position is not None:
                distance = np.linalg.norm(np.array(measurement) - np.array(self.last_position))
                if distance > DISTANCE_THRESHOLD:  # New ball detected
                    self.ball_positions.clear()  # Reset trajectory
                    print(f"New ball detected, resetting trajectory for Tracker ID {self.tracker_id}. Distance: {distance}")
            
            self.last_position = measurement

            # Compute weights based on distance from measurement (ball center)
            distances = np.linalg.norm(self.particles - np.array(measurement), axis=1)
            self.weights = np.exp(-distances / (2 * self.measurement_noise**2))
            self.weights += 1e-300  # Avoid division by zero
            self.weights /= np.sum(self.weights)  # Normalize weights

            # Resample particles based on their weights
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = self.weights[indices]

            # Add the detected ball position to the list
            self.ball_positions.append(measurement)

    def estimate(self):
        # Return the weighted average of the particles as the estimated position
        return np.average(self.particles, weights=self.weights, axis=0).astype(int)

    def draw_particles(self, frame):
        for particle in self.particles:
            cv2.circle(frame, tuple(particle.astype(int)), 1, self.color, -1)  # Draw each particle with the tracker's color

    def draw_estimated_position(self, frame):
        estimated_position = self.estimate()
        cv2.circle(frame, tuple(estimated_position), 5, self.color, -1)  # Draw estimated ball position with the tracker's color

    def draw_trajectory(self, frame):
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                # Draw line connecting the previous position to the current one
                cv2.line(frame, self.ball_positions[i-1], self.ball_positions[i], self.color, 2)  # Trajectory in tracker's color

def applyModel(frame, model):
    results = model.track(frame, save=True, verbose=False, device=device)
    
    center_ret = (-1, -1)
    confidence = -1
    detections = []  # To store detection coordinates

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]

        if class_id == 0 and confidence > 0.5:  # Assuming class_id 0 is for the ball
            x_center = (x1 + x2) / 2    
            y_center = (y1 + y2) / 2
            center_ret = (int(x_center), int(y_center))
            detections.append(center_ret)  # Add detected center coordinates to the list
            
            # Optionally draw bounding box and center circle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    return detections, center_ret, confidence

def testModel(num_cam):
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameraInfos)

    videoCapture = cv2.VideoCapture(pathVideo)
    trackers = []  # List to hold multiple tracker instances

    frame_count = 0
    trajectory_data = []  # List to store the trajectory data for each frame

    while frame_count < END:
        ret, frame = videoCapture.read()
        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)
        frameUndistorted = cv2.resize(frameUndistorted, (size, size))
        detections, center, confidence = applyModel(frameUndistorted, model)

        # Update trackers
        new_trackers = []
        for detection in detections:
            # Check if an existing tracker can be updated
            matched = False
            for tracker in trackers:
                distance = np.linalg.norm(np.array(tracker.last_position) - np.array(detection)) if tracker.last_position else float('inf')
                if distance < DISTANCE_THRESHOLD:
                    tracker.update(detection)
                    matched = True
                    break
            
            # If no tracker matched the detection, create a new tracker
            if not matched:
                # Generate a random color for the new tracker
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_tracker = ParticleFilterBallTracker(len(trackers), color)  # Pass the random color to the tracker
                new_tracker.update(detection)
                new_trackers.append(new_tracker)

        # Add new trackers to the list
        trackers.extend(new_trackers)

        # Draw all tracker trajectories
        for tracker in trackers:
            tracker.draw_particles(frameUndistorted)  # Draw particles
            tracker.draw_estimated_position(frameUndistorted)  # Draw the estimated position of the ball
            tracker.draw_trajectory(frameUndistorted)  # Draw the trajectory of the ball

        # Save the trajectory points
        trajectory_data.append([tracker.ball_positions for tracker in trackers])  # Store trajectories for each tracker

        # Display the frame
        cv2.imshow('Frame', frameUndistorted)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break
        
        frame_count += 1

    # Save trajectory data to a pickle file
    with open('detections_traj.pkl', 'wb') as f:
        pickle.dump(trajectory_data, f)

    videoCapture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    testModel(6)
