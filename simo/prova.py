# Script to save one frame from each video and select the corners of the volleyball court

import os
import re
import cv2
import json
from utils import *
from cameraInfo import *

# Define paths
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
path_frames = os.path.join(parent_path, 'data/dataset/singleFrame')
path_videos = os.path.join(parent_path, 'data/dataset/video')
path_calibrationMTX = os.path.join(parent_path, 'data/calibrationMatrix/calibration.pkl')
path_court = os.path.join(parent_path, 'data/images/courts.jpg')

# Define the path for the combined JSON file
path_json = os.path.join(path_frames, 'world_points_all_cameras.json')

# Define valid cameras and right cameras
valid_camera_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
rightCamera = [5, 12, 13]

# Initialize points dictionaries
points_template = {
    (195, 145): 0,
    (195, 355): 0,
    (340, 355): 0,
    (480, 355): 0,
    (625, 355): 0,
    (625, 145): 0,
    (480, 145): 0,
    (340, 145): 0,
    (75, 75): 0,
    (75, 425): 0,
    (745, 425): 0,
    (745, 75): 0
}

worldPoints_template = [
    (-9, 4.5),
    (-9, -4.5),
    (-3, -4.5),
    (3, -4.5),
    (9, -4.5),
    (9, 4.5),
    (3, 4.5),
    (-3, 4.5),
    (-14, 7.5),
    (-14, -7.5),
    (14, -7.5),
    (14, 7.5)
]

# Global variables
clicked_point = ()
all_world_points = {}  # Dictionary to store world-image points for all cameras

# Mouse callback function to capture clicks
def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")

# Function to overlay the court image onto the main image
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
    rows, cols, _ = img2_resized.shape

    # Define the position where the image will be placed
    x_margin = 50  # Margin from the edge
    y_offset = 50  # Vertical offset from the top

    if rightCameraFlag:
        # Place the image at the top-right corner
        x_offset = width1 - cols - x_margin
    else:
        # Place the image at the top-left corner
        x_offset = x_margin

    # Ensure offsets are within the frame dimensions
    if x_offset < 0 or y_offset + rows > height1:
        raise ValueError("The resized image does not fit within the frame with the given offsets.")

    roi = img1[y_offset:y_offset+rows, x_offset:x_offset+cols]

    # Create a mask of the court image
    img2gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Black-out the area of the court in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Extract the region of the court image
    img2_fg = cv2.bitwise_and(img2_resized, img2_resized, mask=mask)

    # Overlay the court image on the ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[y_offset:y_offset+rows, x_offset:x_offset+cols] = dst

    return img1

# Function to draw points on the image
def edit_image(image, points):
    # Thickness of the point
    point_thickness = 5  # Adjusted for better visibility

    for point in points:
        point_x, point_y = point
        if points[point] == 0:
            point_color = (0, 170, 255)  # Yellow for unprocessed points
        elif points[point] == 1:
            point_color = (0, 255, 0)    # Green for selected points
        elif points[point] == 2:
            point_color = (255, 0, 0)    # Blue for skipped points

        cv2.circle(image, (point_x, point_y), point_thickness, point_color, -1)

    return image

# Function to interactively take points from the user
def takePoints(imageUndistorted, courtImg, rightCameraFlag, points):
    global clicked_point
    print("Select the corners of the court")
    window_name = 'Select Points'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    img_copy = imageUndistorted.copy()
    image_points = []

    for point in points:
        if points[point] == 0:
            while True:
                # Edit the court image with points
                courtImgEdited = edit_image(courtImg.copy(), points)
                # Overlay the court image onto the undistorted frame
                image = unifyImages(img_copy.copy(), courtImgEdited, rightCameraFlag)

                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF

                if clicked_point:
                    # User clicked on the image, update the point with the clicked coordinates
                    points[point] = 1
                    print(f"Point {point} selected at {clicked_point}")

                    # Append the clicked point to image_points
                    image_points.append(list(clicked_point))
                    clicked_point = ()

                    break
                elif key == ord('s'):
                    # User wants to skip this point
                    points[point] = 2
                    print(f"Point {point} skipped.")

                    # Append a placeholder (e.g., [0, 0]) to image_points
                    image_points.append([0, 0])

                    break
                elif key == ord('q'):
                    print("Exiting...")
                    cv2.destroyWindow(window_name)
                    return None  # Indicate that the process was interrupted

    cv2.destroyWindow(window_name)
    return image_points  # Return the list of image points

# Function to save the world-image points to a single JSON file
def save_worldPoints_to_json(camera_number, worldPoints_list, imagePoints_list):
    # Combine world and image points into a list of dictionaries
    data_list = []
    for world_coord, image_coord in zip(worldPoints_list, imagePoints_list):
        if image_coord != [0, 0]:  # Only save if image coordinate is not [0, 0]
            data_list.append({
                "world_coordinate": list(world_coord),
                "image_coordinate": image_coord
            })

    if data_list:
        # Add to the global all_world_points dictionary
        all_world_points[str(camera_number)] = data_list
    else:
        print(f"No valid points selected for camera {camera_number}. Data not saved.")

# Function to process videos and save frames
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

        # Create a fresh copy of points for each camera
        points = points_template.copy()
        worldPoints = worldPoints_template.copy()

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

        courtImg_with_points = edit_image(courtImg.copy(), points)

        undistorted_frame = unifyImages(undistorted_frame, courtImg_with_points, rightCameraFlag)

        cv2.imshow('image', undistorted_frame)
        key = cv2.waitKey(0)
        if key == ord('s'):
            frame_filename = os.path.join(path_frames, f"cam_{camera_number}.png")
            cv2.imwrite(frame_filename, undistorted_frame)

            image_points = takePoints(undistorted_frame_copy, courtImg, rightCameraFlag, points)
            if image_points is None:
                print("Point selection interrupted.")
                cv2.destroyAllWindows()
                video_capture.release()
                continue  # Skip to the next video

            # Save worldPoints and imagePoints to the global dictionary
            save_worldPoints_to_json(camera_number, worldPoints, image_points)

            print(f"Frame saved as {frame_filename}")

            cv2.destroyAllWindows()

        video_capture.release()

    # After processing all videos, save the combined JSON file
    if all_world_points:
        with open(path_json, 'w') as json_file:
            json.dump(all_world_points, json_file, indent=4)
        print(f"All world points saved to {path_json}")
    else:
        print("No valid points selected for any camera. No data saved.")

if __name__ == '__main__':
    saveFrames()
