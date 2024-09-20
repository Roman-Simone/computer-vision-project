# Script to save 1 frame from each video and select the corners of the volleyball court

import os
import re
import cv2
import json
from utils import *
from cameraInfo import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
path_frames = os.path.join(parent_path, 'data/dataset/singleFrame')
path_videos = os.path.join(parent_path, 'data/dataset/video')
path_calibrationMTX = os.path.join(parent_path, 'data/calibrationMatrix/calibration.pkl')
path_court = os.path.join(parent_path, 'data/images/courts.jpg')
path_json = os.path.join(parent_path, 'data/world_points_all_cameras.json')

valid_camera_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
rightCamera = [5, 12, 13]
points = {
    (195,145): 0,
    (195,355): 0,
    (340,355): 0,
    (480,355): 0,
    (625,355): 0,
    (625,145): 0,
    (480,145): 0,
    (340,145): 0, 
    (75,75): 0,
    (75, 425): 0,
    (745, 425): 0,
    (745, 75): 0
}

worldPoints = {
    (-9, 4.5): (),
    (-9, -4.5): (),
    (-3, -4.5): (),
    (3, -4.5): (),
    (9, -4.5): (),
    (9, 4.5): (),
    (3, 4.5): (),
    (-3, 4.5): (),
    (-14, 7.5): (),
    (-14, -7.5): (),
    (14, -7.5): (),
    (14, 7.5): ()
}

# Define camera coordinates for specific cameras
camera_coordinates_dict = {
    1: [14.5, 17.7, 6.2],
    2: [0.0, 17.7, 6.2],
    3: [22.0, 10.0, 6.6],
    4: [-14.5, 17.7, 6.2],
    5: [22.0, -10.0, 5.8],
    6: [0.0, -10.0, 6.3],
    7: [-25.0, 0.0, 6.4],
    8: [-22.0, -10.0, 6.3],
    12: [-22.0, 10.0, 6.9],
    13: [22.0, 0.0, 7.0]
}

# Global variables
clicked_point = ()
all_world_points = {}  # Dictionary to store world-image points for all cameras

# Mouse callback function
def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")

def unifyImages(img1, img2, rightCameraFlag):
    # Get dimensions of the undistorted frame
    height1, width1, _ = img1.shape

    # Decide the scaling factor (e.g., 15% of the frame width)
    scale_factor = 0.15
    new_width = int(width1 * scale_factor)
    # Maintain aspect ratio
    aspect_ratio = img2.shape[1] / img2.shape[0]
    new_height = int(new_width / aspect_ratio)

    # Resize img2 to the new dimensions
    img2_resized = cv2.resize(img2, (new_width, new_height))

    # Get dimensions of the resized img2
    rows, cols, channels = img2_resized.shape

    # Define the position where the image will be placed
    x_margin = 50  # Margin from the edge
    y_offset = 50  # Vertical offset from the top

    if rightCameraFlag:
        # Place the image at the top-right corner
        x_offset = width1 - cols - x_margin
    else:
        # Place the image at the top-left corner
        x_offset = x_margin

    roi = img1[y_offset:y_offset+rows, x_offset:x_offset+cols]

    # Create a mask of the court
    img2gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Black-out the area of the image in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Extract the region
    img2_fg = cv2.bitwise_and(img2_resized, img2_resized, mask=mask)

    # Overlay the image on the ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[y_offset:y_offset+rows, x_offset:x_offset+cols] = dst

    return img1

def edit_image(image):
    # Thickness of the point
    point_thickness = 20

    for point in points:
        point_x = point[0]
        point_y = point[1]
        if points[point] == 0:
            point_color = (0, 170, 255) # yellow
        elif points[point] == 1:
            point_color = (0, 255, 0)
        elif points[point] == 2:
            point_color = (255, 0, 0)

        cv2.circle(image, (point_x, point_y), point_thickness, point_color, -1)

    return image

def takePoints(imageUndistorted, courtImg, rightCameraFlag):
    global clicked_point, points
    print("Select the corners of the court")
    window_name = 'Select Points'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    img_copy = imageUndistorted.copy()
    image_points = []

    for point in points:
        if points[point] == 0:
            while True:
                courtImgEdited = edit_image(courtImg.copy())
                image = unifyImages(img_copy.copy(), courtImgEdited, rightCameraFlag)

                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF

                if clicked_point:
                    # User clicked on the image so Update the point with the clicked coordinates
                    points[point] = 1
                    print(f"Point {point} selected at {clicked_point}")

                    for worldPoint in worldPoints:
                        if worldPoints[worldPoint] == ():
                            worldPoints[worldPoint] = clicked_point
                            break
                    # Save and Reset clicked_point
                    clicked_point = ()

                    break
                elif key == ord('s'):
                    # User wants to skip this point
                    points[point] = 2
                    print(f"Point {point} skipped.")
                    for worldPoint in worldPoints:
                        if worldPoints[worldPoint] == ():
                            worldPoints[worldPoint] = (0, 0)
                            break
                    break
                elif key == ord('q'):
                    print("Exiting...")
                    cv2.destroyWindow(window_name)
                    return None  # Indicate interruption

    # Reset the points
    for point in points:
        points[point] = 0

    retCoords = {}

    for worldPoint in worldPoints:
        if worldPoints[worldPoint] != (0, 0):
            retCoords[worldPoint] = worldPoints[worldPoint]
        worldPoints[worldPoint] = ()

    cv2.destroyWindow(window_name)

    return retCoords

def commonList(camera_number, world_image_coordinates):
    if world_image_coordinates:
        # Include 'camera_coordinates' in the data if available
        if camera_number in camera_coordinates_dict:
            camera_coords = camera_coordinates_dict[camera_number]
            data = {
                "camera_coordinates": camera_coords,
                "points": []
            }
        else:
            data = {
                "points": []
            }

        # Build the list of points
        for world_coord, image_coord in world_image_coordinates.items():
            data["points"].append({
                "world_coordinate": list(world_coord),
                "image_coordinate": list(image_coord)
            })

        # Add to the global all_world_points dictionary
        all_world_points[str(camera_number)] = data
    else:
        print(f"No valid points selected for camera {camera_number}. Data not saved.")

def saveFrames():
    videos = find_file_mp4(path_videos)
    camera_infos = load_pickle(path_calibrationMTX)

    for video in videos:
        camera_number_match = re.findall(r'\d+', video.replace(".mp4", ""))
        if not camera_number_match:
            continue
        camera_number = int(camera_number_match[0])
        if camera_number not in valid_camera_numbers:
            continue

        if camera_number in rightCamera:
            rightCameraFlag = True
        else:
            rightCameraFlag = False

        # Open the video
        camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)
        path_video = os.path.join(path_videos, video)
        video_capture = cv2.VideoCapture(path_video)

        # Read the first frame
        ret, frame = video_capture.read()
        if not ret:
            continue

        undistorted_frame = undistorted(frame, camera_info)
        undistorted_frame_copy = undistorted_frame.copy()

        courtImg = cv2.imread(path_court)

        cv2.imshow('image', undistorted_frame)
        key = cv2.waitKey(0)
        if key == ord('s'):
            frame_filename = os.path.join(path_frames, f"cam_{camera_number}.png")
            cv2.imwrite(frame_filename, undistorted_frame)
            cv2.destroyAllWindows()

            world_image_coordinates = takePoints(undistorted_frame_copy, courtImg, rightCameraFlag)

            print(world_image_coordinates)
            # Save worldPoints and imagePoints to the global dictionary
            commonList(camera_number, world_image_coordinates)
            break

        cv2.destroyAllWindows()
        video_capture.release()

    cv2.destroyAllWindows()
    # After processing all videos, save the combined JSON file
    if all_world_points:
        with open(path_json, 'w') as json_file:
            json.dump(all_world_points, json_file, indent=4)
        print(f"All world points saved to {path_json}")
    else:
        print("No valid points selected for any camera. No data saved.")

if __name__ == '__main__':
    saveFrames()
