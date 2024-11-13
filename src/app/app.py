from flask import Flask, render_template, jsonify, request, Response
import cv2
import numpy as np
import os
import sys
import torch
import random
import importlib
import threading

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *
from ultralytics import YOLO

app = Flask(__name__, static_folder=PATH_STATIC)

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

selected_cameras = {
    "camera_src": 1,
    "camera_dst": 2
}

def ret_homography(camera_src, camera_dst):
    inter_camera_info = next((inter for inter in interInfo if inter.camera_number_1 == camera_src and inter.camera_number_2 == camera_dst), None)
    return inter_camera_info.homography

@app.route('/')
def index():
    return render_template('index.html', css_path=PATH_CSS, available_cameras=VALID_CAMERA_NUMBERS)

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

if __name__ == "__main__":
    app.run(debug=True)