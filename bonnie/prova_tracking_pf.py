import numpy as np
import cv2
import torch
from ultralytics import YOLO
from config import *
from utils import *
import os

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

model = YOLO(pathWeight)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

size = 800

# Define particle filter ball tracker
class ParticleFilterBallTracker:
    def __init__(self, num_particles=1000, process_noise=1.0, measurement_noise=2.0):
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.rand(self.num_particles, 2) * size  # Initialize particles randomly within frame size
        self.weights = np.ones(self.num_particles) / self.num_particles  # Equal weights initially
        self.ball_positions = []  # Store the ball's previous positions

    def predict(self):
        # Add random motion to particles (process noise)
        self.particles += np.random.randn(self.num_particles, 2) * self.process_noise
        self.particles = np.clip(self.particles, 0, size)  # Ensure particles stay within frame bounds

    def update(self, measurement):
        if measurement != (-1, -1):
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
            cv2.circle(frame, tuple(particle.astype(int)), 1, (255, 0, 0), -1)  # Draw each particle

    def draw_estimated_position(self, frame):
        estimated_position = self.estimate()
        cv2.circle(frame, tuple(estimated_position), 5, (0, 0, 255), -1)  # Draw estimated ball position

    def draw_trajectory(self, frame):
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                # Draw line connecting the previous position to the current one
                cv2.line(frame, self.ball_positions[i-1], self.ball_positions[i], (0, 255, 0), 2)  # Trajectory in green

def applyModel(frame, model, tracker):
    
    results = model(frame, verbose=False, device=device)
    
    center_ret = (-1, -1)
    confidence = -1

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]

        if class_id == 0 and confidence > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            center_ret = (int(x_center), int(y_center))
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    tracker.update(center_ret)
    tracker.draw_particles(frame)  # Draw particles
    tracker.draw_estimated_position(frame)  # Draw the estimated position of the ball
    tracker.draw_trajectory(frame)  # Draw the trajectory of the ball

    return frame, center_ret, confidence

def testModel(num_cam):
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameraInfos)

    videoCapture = cv2.VideoCapture(pathVideo)
    tracker = ParticleFilterBallTracker()

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)
        frameUndistorted = cv2.resize(frameUndistorted, (size, size))
        frameWithBbox, center, confidence = applyModel(frameUndistorted, model, tracker)
        cv2.imshow('Frame', frameWithBbox)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    testModel(1)
