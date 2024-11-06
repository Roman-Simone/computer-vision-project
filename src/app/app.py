from flask import Flask, render_template, jsonify, request, Response
import cv2
import numpy as np
import os
import sys
import torch
import threading
import random

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *
from utils.particleFilter import *
from ultralytics import YOLO

app = Flask(__name__, static_folder=PATH_STATIC)

available_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
interInfo = load_pickle(PATH_HOMOGRAPHY_MATRIX)
cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)
YOLO_INPUT_SIZE = 800

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
model = YOLO(pathWeight)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600),
    7: (5150, 5330)
}

selected_cameras = {
    "camera_src": 1,
    "camera_dst": 2
}

def ret_homography(camera_src, camera_dst):
    inter_camera_info = next((inter for inter in interInfo if inter.camera_number_1 == camera_src and inter.camera_number_2 == camera_dst), None)
    return inter_camera_info.homography

@app.route('/')
def index():
    return render_template('index.html', css_path=PATH_CSS)

@app.route('/point_projection')
def point_projection():
    return render_template('point_projection.html', available_cameras=available_cameras, css_path=PATH_CSS)

@app.route('/2D_ball_tracking')
def ball_tracking():
    return render_template('2D_ball_tracking.html', available_cameras=available_cameras, css_path=PATH_CSS)

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
    img_court = cv2.imread(os.path.join(PATH_COURT))

    camera_info_1, _ = take_info_camera(camera_src, cameras_info)
    camera_info_2, _ = take_info_camera(camera_dst, cameras_info)

    img_src = undistorted(img_src, camera_info_1)
    img_dst = undistorted(img_dst, camera_info_2)

    if img_src is None or img_dst is None or img_court is None:
        return jsonify(error="Could not load images")

    success_src = cv2.imwrite(os.path.join(PATH_STATIC, 'src_img.png'), img_src)
    success_dst = cv2.imwrite(os.path.join(PATH_STATIC, 'dst_img.png'), img_dst)
    success_court = cv2.imwrite(os.path.join(PATH_STATIC, 'court_img.jpg'), img_court)

    if not success_src or not success_dst or not success_court:
        print("Could not save images")

    return jsonify(
        src_img='static/src_img.png',
        dst_img='static/dst_img.png',
        court_img='static/court_img.jpg'
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
    homography_court = ret_homography(camera_src, 0)

    if homography is None:
        return jsonify(error=f"No homography available for cameras {camera_src} and {camera_dst}")

    camera_info_1, _ = take_info_camera(camera_src, cameras_info)
    camera_info_2, _ = take_info_camera(camera_dst, cameras_info)

    point = np.array([[x + camera_info_1.roi[0], y + camera_info_1.roi[1]]], dtype=np.float32)
    
    point_transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography).reshape(-1, 2)
    point_transformed_court = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography_court).reshape(-1, 2)
    
    img_src = cv2.imread(os.path.join(PATH_STATIC, 'src_img.png'))
    img_dst = cv2.imread(os.path.join(PATH_STATIC, 'dst_img.png'))
    img_court = cv2.imread(os.path.join(PATH_STATIC, 'court_img.jpg'))

    cv2.circle(img_src, (x, y), 15, (0, 255, 0), -1)  # Draw circle on source image
    
    div = img_src.shape[1] / 15
    
    x_transformed = int(point_transformed[0][0] - camera_info_2.roi[0])
    y_transformed = int(point_transformed[0][1] - camera_info_2.roi[1])
    x_transformed_court = int(point_transformed_court[0][0])
    y_transformed_court = int(point_transformed_court[0][1])
            
    cv2.circle(img_dst, (x_transformed, y_transformed), int((img_dst.shape[1]/div + img_dst.shape[0]/div)/2), (0, 255, 0), -1)  # Draw circle on destination image
    cv2.circle(img_court, (x_transformed_court, y_transformed_court), 15, (0, 255, 0), -1)  # Draw circle on court image

    # Save updated images
    cv2.imwrite(os.path.join(PATH_STATIC, 'src_img_updated.png'), img_src)
    cv2.imwrite(os.path.join(PATH_STATIC, 'dst_img_updated.png'), img_dst)
    cv2.imwrite(os.path.join(PATH_STATIC, 'court_img_updated.jpg'), img_court)
    
    # Return relative paths (static/...) instead of absolute paths
    return jsonify(
        src_img='static/src_img_updated.png',
        dst_img='static/dst_img_updated.png',
        court_img='static/court_img_updated.jpg',
        x_transformed=x_transformed,
        y_transformed=y_transformed,
        x_transformed_court=x_transformed_court,
        y_transformed_court=y_transformed_court
    )

def applyModel(frame, model):
    height, width = frame.shape[:2]
    
    # Resize for YOLO model
    frameResized = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    
    results = model.track(frameResized, verbose=False, device=device)
    
    center_ret = (-1, -1)
    confidence = -1
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Move tensor to CPU and convert to NumPy
        # Scale back to original size
        x1 = x1 * width / YOLO_INPUT_SIZE
        y1 = y1 * height / YOLO_INPUT_SIZE
        x2 = x2 * width / YOLO_INPUT_SIZE
        y2 = y2 * height / YOLO_INPUT_SIZE
        confidence = box.conf[0].cpu().numpy()  # Move tensor to CPU and convert to NumPy
        class_id = box.cls[0].cpu().numpy()  # Move tensor to CPU and convert to NumPy

        if class_id == 0 and confidence > 0.35:
            x_center = (x1 + x2) / 2    
            y_center = (y1 + y2) / 2
            center_ret = (int(x_center), int(y_center))
            detections.append(center_ret)
            cv2.circle(frame, center_ret, 3, (0, 255, 0), -1)

    return detections, center_ret, confidence

def testModel(num_cam, action):
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameras_info)
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
        detections, center_ret, confidence = applyModel(frameUndistorted, model)

        new_trackers = []
        for detection in detections:
            if detection:  # Check if detection is not empty
                matched = False
                # Ensure detection is a tuple of floats or integers
                detection = np.array(detection, dtype=np.float32)  # Convert to numpy array if needed
                for tracker in trackers:
                    if tracker.last_position is not None and tracker.last_position.any():
                        distance = np.linalg.norm(np.array(tracker.last_position) - detection)
                        if distance < DISTANCE_THRESHOLD:
                            tracker.update(detection)
                            matched = True
                            break
                if not matched:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    new_tracker = ParticleFilterBallTracker(len(trackers), color, frame_size)
                    new_tracker.update(detection)
                    new_trackers.append(new_tracker)
            else:
                print("No valid detection found.")

        trackers.extend(new_trackers)

        # Draw trackers and trajectories
        for tracker in trackers:
            if tracker.last_position is not None and tracker.last_position.any():
                cv2.circle(frame, tuple(tracker.last_position.astype(int)), 5, tracker.color, -1)
                tracker.predict()  # Predict the next position
                tracker.draw_particles(frame)  # Draw particles
                tracker.draw_estimated_position(frame)  # Draw estimated position
                tracker.draw_trajectory(frame)  # Draw trajectory

                # Append the last position to the trajectory points
                trajectory_points.append(tracker.last_position)

        # Draw the trajectory points
        for i in range(1, len(trajectory_points)):
            if trajectory_points[i - 1] is None or trajectory_points[i] is None:
                continue
            cv2.line(frame, tuple(trajectory_points[i - 1].astype(int)), tuple(trajectory_points[i].astype(int)), (0, 255, 0), 2)

        # Prepare frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_stream = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_stream + b'\r\n')

    videoCapture.release()

@app.route('/video_feed')
def video_feed():
    camera_number = request.args.get('camera', default=1, type=int)
    action_number = request.args.get('action', default=1, type=int)
    return Response(testModel(camera_number, action_number),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    data = request.json
    camera_number = int(data['camera_number'])
    action_number = int(data['action_number'])

    # Start a thread to run the video processing function
    thread = threading.Thread(target=testModel, args=(camera_number, action_number))
    thread.start()

    return jsonify(success=True)

if __name__ == "__main__":
    app.run(debug=True)