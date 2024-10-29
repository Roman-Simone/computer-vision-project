import os
import cv2
import sys
import torch
import random
import numpy as np
from cameraInfo import *
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, request

# Add the parent directory to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

# Now you can import the utils module from the parent directory
from utils.utils import *
from utils.config import *
from utils.particleFilter import *

app = Flask(__name__, static_folder=PATH_STATIC)

# Configuration
available_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
interInfo = load_pickle(PATH_HOMOGRAPHY_MATRIX)
cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)
pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
model = YOLO(pathWeight, verbose=False)
DISTANCE_THRESHOLD = 200

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600),
    7: (5150, 5330)
}

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

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
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    return detections, center_ret, confidence

def testModel(num_cam, action):
    """Process the video for the given camera and action, returning trajectory points and the current frame"""
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameras_info)
    videoCapture = cv2.VideoCapture(pathVideo)

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
        frameUndistorted = cv2.resize(frameUndistorted, (800, 800))

        detections, center, confidence = applyModel(frameUndistorted, model)

        # First, predict next state for all active trackers
        for tracker in trackers:
            if tracker.active:
                tracker.predict()

        # Update trackers with new detections
        for detection in detections:
            matched_tracker = None
            min_distance = float('inf')

            for tracker in trackers:
                if not tracker.active:
                    continue
                if tracker.last_position is not None:
                    distance = np.linalg.norm(np.array(detection) - np.array(tracker.last_position))
                    if distance < DISTANCE_THRESHOLD and distance < min_distance:
                        min_distance = distance
                        matched_tracker = tracker

            if matched_tracker is None:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_tracker = ParticleFilterBallTracker(len(trackers), color)
                new_tracker.reset_tracker(detection)
                trackers.append(new_tracker)
            else:
                matched_tracker.update(detection)

        # Draw active trackers
        for tracker in trackers:
            if tracker.active:
                tracker.draw_particles(frameUndistorted)
                tracker.draw_estimated_position(frameUndistorted)
                tracker.draw_trajectory(frameUndistorted)

                # Collect trajectory points from active trackers
                if tracker.last_position is not None:
                    trajectory_points.append(tracker.last_position)

        # Save and return the current frame
        frame = frameUndistorted.copy()  # Get the current frame with ball tracking

    videoCapture.release()
    return trajectory_points, frame  # Return trajectory points and the processed frame

# Serve the ball tracking page
@app.route('/ball_tracking')
def ball_tracking():
    return render_template('ball_tracking.html', available_cameras=available_cameras)

# Route to handle ball tracking frame retrieval
@app.route('/get_ball_tracking_frame', methods=['GET'])
def get_ball_tracking_frame():
    action = int(request.args.get('action'))
    camera = int(request.args.get('camera'))

    if action not in ACTIONS or camera not in available_cameras:
        return jsonify(error="Invalid action or camera selected"), 400

    # Process the video for the selected action and camera
    trajectory_points, frame = testModel(camera, action)

    # Save the frame as a static image
    frame_path = os.path.join(PATH_STATIC, 'tracking_frame.png')
    cv2.imwrite(frame_path, frame)

    return jsonify(frame_src='static/tracking_frame.png')

@app.route('/get_ball_tracking_video')
def get_ball_tracking_video():
    camera = request.args.get('camera')
    action = request.args.get('action')

    # Logic to determine the correct video path
    video_src = f'static/videos/camera_{camera}_action_{action}.mp4'  # Adjust as needed

    return jsonify({'video_src': video_src})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/point_projection')
def point_projection():
    return render_template('point_projection.html', available_cameras=available_cameras)

# Other routes...

if __name__ == "__main__":
    app.run(debug=True)
