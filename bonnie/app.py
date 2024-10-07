from flask import Flask, render_template, jsonify, request, send_file
import cv2
import numpy as np
from config import *
from utils import *
import os

app = Flask(__name__, static_folder=PATH_STATIC)

available_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
interInfo = load_pickle(PATH_HOMOGRAPHY_MATRIX)
cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)

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

@app.route('/ball_tracking')
def ball_tracking():
    return render_template('ball_tracking.html', available_cameras=available_cameras, css_path=PATH_CSS)

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

    img_src = cv2.imread(f"{PATH_FRAME_DISTORTED}/cam_{camera_src}.png")
    img_dst = cv2.imread(f"{PATH_FRAME_DISTORTED}/cam_{camera_dst}.png")
    
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

    return jsonify(src_img='static/src_img.png', dst_img='static/dst_img.png')

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
        
    print("Img shape: ", img_dst.shape)
    
    print("Circle radius: ", int((img_dst.shape[1]/div + img_dst.shape[0]/div)/2))
    
    cv2.circle(img_dst, (x_transformed, y_transformed), int((img_dst.shape[1]/div + img_dst.shape[0]/div)/2), (0, 255, 0), -1)  # Draw circle on destination image
    # cv2.circle(img_dst, (x_transformed, y_transformed), 15, (0, 255, 0), -1)  # Draw circle on destination image

    cv2.imwrite(os.path.join(PATH_STATIC, 'src_img_updated.png'), img_src)
    cv2.imwrite(os.path.join(PATH_STATIC, 'dst_img_updated.png'), img_dst)
    

    return jsonify(
        src_img='static/src_img_updated.png',
        dst_img='static/dst_img_updated.png',
        x_transformed=x_transformed,
        y_transformed=y_transformed
    )

if __name__ == "__main__":
    app.run(debug=True)