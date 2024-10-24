from flask import Flask, render_template, jsonify, request, send_file
import cv2
import numpy as np
from config import *
import random
import torch
from particleFilter import ParticleFilterBallTracker
from utils import *
from ultralytics import YOLO
import os

app = Flask(__name__, static_folder=PATH_STATIC)

available_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
interInfo = load_pickle(PATH_HOMOGRAPHY_MATRIX)
cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)
pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
model = YOLO(pathWeight, verbose=False)
DISTANCE_THRESHOLD = 200

selected_cameras = {
    "camera_src": 1,
    "camera_dst": 2
}


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
            
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    return detections, center_ret, confidence

def testModel(num_cam, action):
    """Process the video for the given camera and action, return trajectory points and the current frame"""
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameras_info)
    videoCapture = cv2.VideoCapture(pathVideo)

    START, END = ACTIONS[action]
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, START)

    trackers = []
    frame_count = 0
    trajectory_points = []  # To store the ball's positions (trajectory)

    while True:
        current_frame = int(videoCapture.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > END:
            break

        ret, frame = videoCapture.read()
        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)
        frameUndistorted = cv2.resize(frameUndistorted, (800, 800))  # Resize frame to fit

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
    return render_template('ball_tracking.html', available_cameras=available_cameras, css_path=PATH_CSS)

# Route to handle ball tracking frame retrieval
@app.route('/get_ball_tracking_frame', methods=['GET'])
def get_ball_tracking_frame():
    action = int(request.args.get('action'))
    camera = int(request.args.get('camera'))

    if action not in ACTIONS or camera not in available_cameras:
        return jsonify(error="Invalid action or camera selected"), 400

    # Process the video for the selected action and camera
    trajectory_points, frame = testModel(camera, action)  # Modify testModel to return frame

    # Get the current frame image with ball tracking
    frame_path = os.path.join(PATH_STATIC, 'tracking_frame.png')

    # Save the frame as a static image
    cv2.imwrite(frame_path, frame)

    return jsonify(frame_src='static/tracking_frame.png')



def ret_homography(camera_src, camera_dst):
    inter_camera_info = next((inter for inter in interInfo if inter.camera_number_1 == camera_src and inter.camera_number_2 == camera_dst), None)
    return inter_camera_info.homography

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/point_projection')
def point_projection():
    return render_template('point_projection.html', available_cameras=available_cameras, css_path=PATH_CSS)

@app.route('/set_cameras', methods=['POST'])
def set_cameras():
    global selected_cameras
    selected_cameras['camera_src'] = int(request.json['camera_src'])
    selected_cameras['camera_dst'] = int(request.json['camera_dst'])
    return jsonify(success=True)

@app.route('/get_images')
def get_images():
    for file_name in os.listdir(PATH_STATIC):
        file_path = os.path.join(PATH_STATIC, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    camera_src = selected_cameras['camera_src']
    camera_dst = selected_cameras['camera_dst']
    
    img_src = cv2.imread(os.path.join(PATH_FRAME_DISTORTED, f'cam_{camera_src}.png'))
    img_dst = cv2.imread(os.path.join(PATH_FRAME_DISTORTED, f'cam_{camera_dst}.png'))
    
    camera_info_1, _ = take_info_camera(camera_src, cameras_info)
    camera_info_2, _ = take_info_camera(camera_dst, cameras_info)

    img_src = undistorted(img_src, camera_info_1)
    img_dst = undistorted(img_dst, camera_info_2)

    if img_src is None or img_dst is None:
        return jsonify(error="Could not load images")

    success_src = cv2.imwrite(os.path.join(PATH_STATIC, 'src_img.png'), img_src)
    success_dst = cv2.imwrite(os.path.join(PATH_STATIC, 'dst_img.png'), img_dst)

    if not success_src or not success_dst:
        print("Could not save images")

    return jsonify(
        src_img='static/src_img.png', 
        dst_img='static/dst_img.png'
    )

@app.route('/project_point', methods=['POST'])
def project_point():
    data = request.json
    x = int(data['x'])
    y = int(data['y'])

    print(f"Received point: ({x}, {y})")

    camera_src = selected_cameras['camera_src']
    camera_dst = selected_cameras['camera_dst']
    
    homography = ret_homography(camera_src, camera_dst)
    
    if homography is None:
        return jsonify(error=f"No homography available for cameras {camera_src} and {camera_dst}")

    camera_info_1, _ = take_info_camera(camera_src, cameras_info)
    camera_info_2, _ = take_info_camera(camera_dst, cameras_info)

    point = np.array([[x + camera_info_1.roi[0], y + camera_info_1.roi[1]]], dtype=np.float32)
    
    point_transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography).reshape(-1, 2)
    
    img_src = cv2.imread(os.path.join(PATH_STATIC, 'src_img.png'))
    img_dst = cv2.imread(os.path.join(PATH_STATIC, 'dst_img.png'))

    cv2.circle(img_src, (x, y), 15, (0, 255, 0), -1)  # Draw circle on source image
    
    div = img_src.shape[1] / 15
    
    x_transformed = int(point_transformed[0][0] - camera_info_2.roi[0])
    y_transformed = int(point_transformed[0][1] - camera_info_2.roi[1])
            
    cv2.circle(img_dst, (x_transformed, y_transformed), int((img_dst.shape[1]/div + img_dst.shape[0]/div)/2), (0, 255, 0), -1)  # Draw circle on destination image

    # Save updated images
    cv2.imwrite(os.path.join(PATH_STATIC, 'src_img_updated.png'), img_src)
    cv2.imwrite(os.path.join(PATH_STATIC, 'dst_img_updated.png'), img_dst)
    
    # Return relative paths (static/...) instead of absolute paths
    return jsonify(
        src_img='static/src_img_updated.png',
        dst_img='static/dst_img_updated.png',
        x_transformed=x_transformed,
        y_transformed=y_transformed
    )



if __name__ == "__main__":
    app.run(debug=True)