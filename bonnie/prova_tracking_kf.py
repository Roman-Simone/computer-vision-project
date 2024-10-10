import re
import cv2
import torch
import numpy as np
from utils import *
from config import *
from ultralytics import YOLO

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

size = 800
model = YOLO(pathWeight)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Structure to hold ball trajectories and IDs
class BallTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.max_distance = 50  # Adjusted maximum distance to consider it the same ball
        self.smoothing_factor = 0.5  # Smoothing factor for fast balls
        self.persistent_threshold = 20  # Distance to keep the same ID even if detected again

    def update(self, center):
        if center != (-1, -1):
            assigned_id = None
            min_distance = float('inf')

            # Check for closest existing ball to determine if it's the same ball
            for ball_id, (last_position, trajectory) in self.trackers.items():
                distance = np.linalg.norm(np.array(center) - np.array(last_position))
                velocity = np.array(center) - np.array(last_position)
                speed = np.linalg.norm(velocity)

                # Check if the detected center is close enough to an existing ID
                if distance < self.max_distance:
                    if speed < 30:  # Speed threshold to determine a likely match
                        if distance < min_distance:
                            min_distance = distance
                            assigned_id = ball_id

            if assigned_id is not None:
                # Smooth the transition of the ball's position
                last_position, trajectory = self.trackers[assigned_id]
                smoothed_position = (
                    int(self.smoothing_factor * center[0] + (1 - self.smoothing_factor) * last_position[0]),
                    int(self.smoothing_factor * center[1] + (1 - self.smoothing_factor) * last_position[1])
                )
                trajectory.append(smoothed_position)
                self.trackers[assigned_id] = (smoothed_position, trajectory)
            else:
                # Assign new ID for a new ball if not persistent
                for ball_id, (last_position, _) in self.trackers.items():
                    distance = np.linalg.norm(np.array(center) - np.array(last_position))
                    if distance < self.persistent_threshold:
                        assigned_id = ball_id
                        break

                if assigned_id is not None:
                    last_position, trajectory = self.trackers[assigned_id]
                    smoothed_position = (
                        int(self.smoothing_factor * center[0] + (1 - self.smoothing_factor) * last_position[0]),
                        int(self.smoothing_factor * center[1] + (1 - self.smoothing_factor) * last_position[1])
                    )
                    trajectory.append(smoothed_position)
                    self.trackers[assigned_id] = (smoothed_position, trajectory)
                else:
                    # Assign new ID for a new ball
                    self.trackers[self.next_id] = (center, [center])
                    self.next_id += 1

        # Predict the positions of balls that were not detected
        for ball_id in list(self.trackers.keys()):
            last_position, trajectory = self.trackers[ball_id]
            if center == (-1, -1):  # Ball not detected
                if len(trajectory) > 1:
                    dx = trajectory[-1][0] - trajectory[-2][0]
                    dy = trajectory[-1][1] - trajectory[-2][1]
                    predicted_position = (
                        int(last_position[0] + dx),
                        int(last_position[1] + dy)
                    )

                    smoothed_predicted_position = (
                        int(self.smoothing_factor * predicted_position[0] + (1 - self.smoothing_factor) * last_position[0]),
                        int(self.smoothing_factor * predicted_position[1] + (1 - self.smoothing_factor) * last_position[1])
                    )
                    self.trackers[ball_id] = (smoothed_predicted_position, trajectory)

    def draw_trajectories(self, frame):
        for ball_id, (last_position, trajectory) in self.trackers.items():
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)  # Draw trajectory
            cv2.putText(frame, f'ID: {ball_id}', (last_position[0] + 10, last_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def applyModel(frame, model, ball_tracker):
    
    results = model.track(frame, verbose=False, device=device)
    
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

    ball_tracker.update(center_ret)
    ball_tracker.draw_trajectories(frame)

    return frame, center_ret, confidence

def testModel(num_cam):

    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')  
    cameraInfo, _ = take_info_camera(num_cam, cameraInfos)

    videoCapture = cv2.VideoCapture(pathVideo)

    ball_tracker = BallTracker()

    while True:
        ret, frame = videoCapture.read()

        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)
        frameUndistorted = cv2.resize(frameUndistorted, (size, size))

        frameWithBbox, center, confidence = applyModel(frameUndistorted, model, ball_tracker)

        cv2.imshow('Frame', frameWithBbox)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    testModel(1)
