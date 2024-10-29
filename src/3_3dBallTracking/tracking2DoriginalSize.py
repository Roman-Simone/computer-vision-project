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

YOLO_INPUT_SIZE = 800  # Size for YOLO model input
DISTANCE_THRESHOLD = 800  # Threshold distance to detect a new ball

class ParticleFilterBallTracker:
    def __init__(self, tracker_id, color, frame_size, num_particles=1000, process_noise=5.0, measurement_noise=2.0):
        self.tracker_id = tracker_id
        self.color = color
        self.frame_width, self.frame_height = frame_size
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # Initialize particles with respect to frame size
        self.particles = np.random.rand(self.num_particles, 2) * [self.frame_width, self.frame_height]
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.ball_positions = []
        self.last_position = None
        self.active = True
        self.frames_without_detection = 0
        
    def predict(self):
        if len(self.ball_positions) >= 2:
            velocity = np.array(self.ball_positions[-1]) - np.array(self.ball_positions[-2])
            self.particles += velocity
        
        self.particles += np.random.randn(self.num_particles, 2) * self.process_noise
        self.particles = np.clip(self.particles, 
                               [0, 0], 
                               [self.frame_width-1, self.frame_height-1])

    def update(self, measurement):
        if measurement == (-1, -1):
            self.frames_without_detection += 1
            if self.frames_without_detection > 10:
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
        
        distances = np.linalg.norm(self.particles - np.array(measurement), axis=1)
        self.weights = np.exp(-distances**2 / (2 * self.measurement_noise**2))
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        
        num_measurement_particles = self.num_particles // 4
        measurement_particles = np.random.normal(measurement, self.measurement_noise/2, 
                                               size=(num_measurement_particles, 2))
        self.particles[-num_measurement_particles:] = measurement_particles
        
        self.weights = np.ones(self.num_particles) / self.num_particles

    def reset_tracker(self, measurement):
        print(f"Resetting tracker {self.tracker_id}")
        self.ball_positions = [measurement]
        self.last_position = measurement
        spread = 20
        self.particles = np.random.normal(measurement, spread, size=(self.num_particles, 2))
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def estimate(self):
        if len(self.ball_positions) > 0:
            return np.array(self.ball_positions[-1]).astype(int)
        return np.average(self.particles, weights=self.weights, axis=0).astype(int)

    def draw_particles(self, frame):
        subset_size = min(100, self.num_particles)
        particle_indices = np.random.choice(self.num_particles, subset_size, replace=False)
        for particle in self.particles[particle_indices]:
            pos = tuple(np.clip(particle, [0, 0], 
                              [self.frame_width-1, self.frame_height-1]).astype(int))
            cv2.circle(frame, pos, 1, self.color, -1)

    def draw_estimated_position(self, frame):
        if len(self.ball_positions) > 0:
            pos = tuple(np.clip(self.estimate(), [0, 0], 
                              [self.frame_width-1, self.frame_height-1]).astype(int))
            cv2.circle(frame, pos, 5, self.color, -1)

    def draw_trajectory(self, frame):
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                pt1 = tuple(np.clip(self.ball_positions[i-1], [0, 0], 
                                  [self.frame_width-1, self.frame_height-1]).astype(int))
                pt2 = tuple(np.clip(self.ball_positions[i], [0, 0], 
                                  [self.frame_width-1, self.frame_height-1]).astype(int))
                cv2.line(frame, pt1, pt2, self.color, 2)

def applyModel(frame, model):
    height, width = frame.shape[:2]
    
    # Resize for YOLO model
    frameResized = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    
    results = model.track(frameResized, verbose=False, device=device)
    
    center_ret = (-1, -1)
    confidence = -1
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        # Scale back to original size
        x1 = x1 * width / YOLO_INPUT_SIZE
        y1 = y1 * height / YOLO_INPUT_SIZE
        x2 = x2 * width / YOLO_INPUT_SIZE
        y2 = y2 * height / YOLO_INPUT_SIZE
        confidence = box.conf[0]
        class_id = box.cls[0]

        if class_id == 0 and confidence > 0.35:
            x_center = (x1 + x2) / 2    
            y_center = (y1 + y2) / 2
            center_ret = (int(x_center), int(y_center))
            detections.append(center_ret)
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    return detections, center_ret, confidence

def testModel(num_cam, action):
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo = take_info_camera(num_cam, cameraInfos)
    videoCapture = cv2.VideoCapture(pathVideo)

    # Get video dimensions
    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    START, END = ACTIONS[action]
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
        detections, center, confidence = applyModel(frameUndistorted, model)

        new_trackers = []
        for detection in detections:
            matched = False
            for tracker in trackers:
                if tracker.last_position:
                    distance = np.linalg.norm(np.array(tracker.last_position) - np.array(detection))
                    if distance < DISTANCE_THRESHOLD:
                        tracker.update(detection)
                        matched = True
                        break

            if not matched:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_tracker = ParticleFilterBallTracker(len(trackers), color, frame_size)
                new_tracker.update(detection)
                new_trackers.append(new_tracker)

        trackers.extend(new_trackers)

        for tracker in trackers:
            if tracker.last_position:
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
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

if __name__ == '__main__':
    # pickle_file = 'ball_trajectories.pkl'
    # results = load_existing_results(pickle_file)

    cam = int(input("Enter camera number: "))
    
    # if str(cam) not in results:
    #         results[str(cam)] = {}
            
    action = int(input("Enter action number: "))

    print(f"Processing Camera {cam}, Action {action}...")
    trajectory = testModel(cam, action)
    # results[str(cam)][str(action)] = trajectory

    # with open(pickle_file, 'wb') as f:
    #     pickle.dump(results, f)

    # print(f"Camera {cam}, Action {action} saved to {pickle_file}")
