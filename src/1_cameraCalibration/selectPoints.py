import os
import re
import cv2
import sys
import json
import numpy as np
from cameraInfo import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

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
    (75,110): 0,
    (75,210): 0,
    (200,210): 0,
    (210,220): 0,
    (210,290): 0,
    (200,300): 0,
    (75,300): 0,
    (75,390): 0,
    (75, 425): 0,
    (410, 425): 0,
    (745, 425): 0,
    (745,390): 0, 
    (745,300): 0, 
    (620,300): 0, 
    (610,290): 0, 
    (610,220): 0, 
    (620,210): 0, 
    (745,210): 0, 
    (745,110): 0,
    (745, 75): 0,
    (410, 75): 0,
    (410, 140): 0,
    (410, 210): 0,
    (410, 290): 0,
    (410, 360): 0
}

worldPoints = {
    (-9.0, 4.5, 0.0): (),
    (-9.0, -4.5, 0.0): (),
    (-3.0, -4.5, 0.0): (),
    (3.0, -4.5, 0.0): (),
    (9.0, -4.5, 0.0): (),
    (9.0, 4.5, 0.0): (),
    (3.0, 4.5, 0.0): (),
    (-3.0, 4.5, 0.0): (),
    (-14.0, 7.5, 0.0): (),
    (-14.0, 6.55, 0.0): (),
    (-14.0, 2.45, 0.0): (),
    (-8.2, 2.45, 0.0): (),
    (-8.2, 1.8, 0.0): (),
    (-8.2, -1.8, 0.0): (),
    (-8.2, -2.45, 0.0): (),
    (-14.0, -2.45, 0.0): (),
    (-14.0, -6.55, 0.0): (),
    (-14.0, -7.5, 0.0): (),
    (0.0, -7.5, 0.0): (),
    (14.0, -7.5, 0.0): (),
    (14.0, -6.55, 0.0): (),
    (14.0, -2.45, 0.0): (),
    (8.2, -2.45, 0.0): (),
    (8.2, -1.8, 0.0): (),
    (8.2, 1.8, 0.0): (),
    (8.2, 2.45, 0.0): (),
    (14.0, 2.45, 0.0): (),
    (14.0, 6.55, 0.0): (),
    (14.0, 7.5, 0.0): (),
    (0.0, 7.5, 0.0): (),
    (0.0, 4.5, 0.0): (),
    (0.0, 1.8, 0.0): (),
    (0.0, -1.8, 0.0): (),
    (0.0, -4.5, 0.0): ()
}

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

camera_coordinates_visual = {
    1: [790, 30],
    2: [410, 30],
    3: [790, 70],
    4: [35, 35],
    5: [790, 470],
    6: [410, 470],
    7: [40, 250],
    8: [40, 470],
    12: [40, 80],
    13: [790, 250]
}

clicked_point = ()
all_world_points = {}  # Dictionary to store world-image points for all cameras

def on_mouse(event, x, y, flags, param):
    """
    Mouse callback function to capture clicked points on an image.
    
    Parameters:
        event (int): type of mouse event (e.g., left button down).
        x (int): x-coordinate of the mouse event.
        y (int): y-coordinate of the mouse event.
        flags (int): flags for the mouse event.
        param (str): window name parameter to display the image.
    """
    global clicked_point, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")
        cv2.circle(img_copy, clicked_point, 5, (0, 0, 255), -1)
        cv2.putText(img_copy, f"{clicked_point}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow(param, img_copy)


def unifyImages(img1, img2, rightCameraFlag):
    """
    Merges a court image onto a frame image, aligning to the left or right based on the camera position.
    
    Parameters:
        img1 (numpy.ndarray): background frame image.
        img2 (numpy.ndarray): court image to overlay.
        rightCameraFlag (bool): whether the camera is on the right side.
        
    Returns:
        numpy.ndarray: merged image with the court overlay.
    """
    height1, width1, _ = img1.shape
    scale_factor = 0.15
    new_width = int(width1 * scale_factor)
    aspect_ratio = img2.shape[1] / img2.shape[0]
    new_height = int(new_width / aspect_ratio)
    img2_resized = cv2.resize(img2, (new_width, new_height))

    rows, cols, channels = img2_resized.shape
    x_margin = 50  
    y_offset = 50  

    x_offset = width1 - cols - x_margin if rightCameraFlag else x_margin
    roi = img1[y_offset:y_offset+rows, x_offset:x_offset+cols]

    img2gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2_resized, img2_resized, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    img1[y_offset:y_offset+rows, x_offset:x_offset+cols] = dst

    img1 = add_legend(img1, cols, rightCameraFlag)

    return img1


def add_legend(image, width_court, rightCameraFlag):
    """
    Adds a legend to the image with descriptive labels for points and camera position.
    
    Parameters:
        image (numpy.ndarray): image to annotate with the legend.
        width_court (int): width of the court image for positioning the legend.
        rightCameraFlag (bool): whether the camera is on the right side.
        
    Returns:
        numpy.ndarray: Annotated image with the legend.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color_text = (255, 255, 255)  
    thickness_text = 2
    first_y_offset = 20
    y_offset = 40

    x_offset = image.shape[1] - width_court - 400 if rightCameraFlag else width_court + 80

    legend_items = [
        {"text": "Corner to select", "color": (0, 170, 255)},  
        {"text": "Corner already selected", "color": (0, 255, 0)},  
        {"text": "Corner skipped", "color": (255, 0, 0)},  
        {"text": "Position of camera", "color": (139, 139, 0)}  
    ]

    for i, item in enumerate(legend_items):
        y_position = (i + 1) * y_offset + first_y_offset

        if "Triangolo" in item["text"]:
            triangle_side = 13
            triangle_points = np.array([
                [x_offset, y_position],
                [x_offset - triangle_side, y_position + triangle_side],
                [x_offset + triangle_side, y_position + triangle_side]
            ], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [triangle_points], color=item["color"])
        else:
            cv2.circle(image, (x_offset, y_position), 13, item["color"], -1)

        cv2.putText(image, item["text"], (x_offset + 30, y_position + 5), font, font_scale, color_text, thickness_text)

    return image


def edit_image(image, camera_number=1):
    """
    Annotates an image with points and camera position for visual guidance during point selection.
    
    Parameters:
        image (numpy.ndarray): image to annotate.
        camera_number (int): camera number to position the camera icon.
        
    Returns:
        numpy.ndarray: annotated image with points and camera position.
    """
    point_thickness = 20
    # width, height, _ = image.shape

    for camera in camera_coordinates_visual:
        if camera == camera_number:
            camera_x, camera_y = camera_coordinates_visual[camera]
            triangle_side = 20
            triangle_points = np.array([
                [camera_x, camera_y - triangle_side],
                [camera_x - triangle_side, camera_y + triangle_side],
                [camera_x + triangle_side, camera_y + triangle_side]
            ], np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(image, [triangle_points], color=(139, 139, 0))
            break

    for point, status in points.items():
        point_color = (0, 170, 255) if status == 0 else (0, 255, 0) if status == 1 else (255, 0, 0)
        cv2.circle(image, point, point_thickness, point_color, -1)
        if status == 0:
            break

    return image


def takePoints(imageUndistorted, courtImg, camera_number, rightCameraFlag, undistortedFlag=False, cameraInfo=None):
    """
    Allows the user to select points on an undistorted frame with visual court overlay and saves the selected coordinates.
    
    Parameters:
        imageUndistorted (numpy.ndarray): frame with undistortion applied.
        courtImg (numpy.ndarray): court image overlay.
        camera_number (int): camera number for labeling.
        rightCameraFlag (bool): whether the camera is on the right side.
        undistortedFlag (bool): flag for undistorted images.
        cameraInfo (CameraInfo): camera information with undistortion parameters.

    Returns:
        dict: dictionary of world points and selected image points.
    """
    global clicked_point, points, img_copy

    print("Select the corners of the court")
    window_name = f"Select Points camera {camera_number}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse, param=window_name)
    img_copy = imageUndistorted.copy()

    for point in points:
        if points[point] == 0:
            while True:
                courtImgEdited = edit_image(courtImg, camera_number)
                image = unifyImages(img_copy, courtImgEdited, rightCameraFlag)
                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF

                if clicked_point:
                    points[point] = 1
                    print(f"Point {point} selected at {clicked_point}")
                    if undistortedFlag:
                        clicked_point = (clicked_point[0] + cameraInfo.roi[0], clicked_point[1] + cameraInfo.roi[1])
                    for worldPoint in worldPoints:
                        if worldPoints[worldPoint] == ():
                            worldPoints[worldPoint] = clicked_point
                            break
                    clicked_point = ()
                    break
                elif key == ord('s'):
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
                    return

    for point in points:
        points[point] = 0

    retCoords = {worldPoint: coord for worldPoint, coord in worldPoints.items() if coord != (0, 0)}
    worldPoints.update({worldPoint: () for worldPoint in worldPoints})

    cv2.destroyWindow(window_name)
    return retCoords


def commonList(camera_number, world_image_coordinates):
    """
    Appends selected points and camera coordinates to a global dictionary for JSON export.
    
    Parameters:
        camera_number (int): camera number to include in the JSON data.
        world_image_coordinates (dict): mapping of world points to image coordinates.
    """
    data_list = [{"world_coordinate": list(coord), "image_coordinate": image_coord} for coord, image_coord in world_image_coordinates.items()]
    all_world_points[str(camera_number)] = {"camera_coordinates": camera_coordinates_dict.get(camera_number, []), "points": data_list}


def selectPointsAllCameras(undistortedFlag=False):
    """
    Loops through frames for all valid cameras, allowing the user to select points, and saves the results in JSON.
    
    Parameters:
        undistortedFlag (bool): if True, undistorts each frame before displaying.
    """
    frames = sorted(find_files(PATH_FRAME_DISTORTED))

    for frame in frames:
        camera_number = int(re.findall(r'\d+', frame.replace(".png", ""))[0])
        if camera_number not in valid_camera_numbers:
            continue
        
        rightCameraFlag = camera_number in rightCamera
        frameImg = cv2.imread(os.path.join(PATH_FRAME_DISTORTED, frame))
        courtImg = cv2.imread(PATH_COURT)

        if undistortedFlag:
            camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
            cameraInfo, _ = take_info_camera(camera_number, camerasInfo)
            frameImg = undistorted(frameImg, cameraInfo)
            world_image_coordinates = takePoints(frameImg, courtImg, camera_number, rightCameraFlag, undistortedFlag, cameraInfo)
        else:
            world_image_coordinates = takePoints(frameImg, courtImg, camera_number, rightCameraFlag)

        commonList(camera_number, world_image_coordinates)
        cv2.destroyAllWindows()

    pathToSave = PATH_JSON_UNDISTORTED if undistortedFlag else PATH_JSON_DISTORTED
    if all_world_points:
        with open(pathToSave, 'w') as json_file:
            json.dump(all_world_points, json_file, indent=4)
        print(f"All world points saved to {pathToSave}")
    else:
        print("No valid points selected for any camera. No data saved.")


def selectPointsCamera(camera_to_select, undistortedFlag=False):
    """
    Allows the user to select points for a single specified camera and saves the results.
    
    Parameters:
        camera_to_select (int or str): camera number to select points for, or 'bonus' for court-only points.
        undistortedFlag (bool): if True, undistorts the frame before displaying.
    """
    frames = find_files(PATH_FRAME_DISTORTED)

    if camera_to_select == "bonus":
        frameImg = cv2.imread(PATH_COURT)
        courtImg = cv2.imread(PATH_COURT)
        world_image_coordinates = takePoints(frameImg, courtImg, 0, False)
        update_json_file(0, world_image_coordinates, PATH_JSON_DISTORTED)
        update_json_file(0, world_image_coordinates, PATH_JSON_UNDISTORTED)
        return

    for frame in frames:
        camera_number = int(re.findall(r'\d+', frame.replace(".png", ""))[0])
        if camera_number not in VALID_CAMERA_NUMBERS or camera_number != camera_to_select:
            continue
        
        rightCameraFlag = camera_number in rightCamera
        frameImg = cv2.imread(os.path.join(PATH_FRAME_DISTORTED, frame))
        courtImg = cv2.imread(PATH_COURT)
        
        if undistortedFlag:
            camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
            cameraInfo, _ = take_info_camera(camera_number, camerasInfo)
            frameImg = undistorted(frameImg, cameraInfo)
            world_image_coordinates = takePoints(frameImg, courtImg, camera_number, rightCameraFlag, undistortedFlag, cameraInfo)
        else:
            world_image_coordinates = takePoints(frameImg, courtImg, camera_number, rightCameraFlag)
        
        pathToSave = PATH_JSON_UNDISTORTED if undistortedFlag else PATH_JSON_DISTORTED
        update_json_file(camera_number, world_image_coordinates, pathToSave)


def update_json_file(camera_number, world_image_coordinates, file_name):
    """
    Updates or creates a JSON file with the selected points for a specified camera.
    
    Parameters:
        camera_number (int): camera number to include in the JSON data.
        world_image_coordinates (dict): mapping of world points to image coordinates.
        file_name (str): path to the JSON file to update.
    """
    try:
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}

    camera_data = {"camera_coordinates": camera_coordinates_dict.get(camera_number, []), "points": [
        {"world_coordinate": list(world_coord), "image_coordinate": list(image_coord)} for world_coord, image_coord in world_image_coordinates.items()
    ]}
    data[str(camera_number)] = camera_data

    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data for camera {camera_number} updated successfully in {file_name}")


def addCourtCoordinates():
    """
    Maps world coordinates to court image coordinates for court visualization, saving them in JSON.
    """
    world_image_coordinates = {worldCord: courtCord for courtCord, worldCord in zip(points, worldPoints)}
    update_json_file(0, world_image_coordinates, PATH_JSON_DISTORTED)
    update_json_file(0, world_image_coordinates, PATH_JSON_UNDISTORTED)
    

if __name__ == "__main__":
    undistortedFlag = False
    user_input = input("Enter camera number to select points (1-8, 12-13) or press Enter for all cameras: ")
    if int(user_input) in VALID_CAMERA_NUMBERS:
        camera_number = int(user_input)
        selectPointsCamera(camera_number, undistortedFlag)
    else:
        selectPointsAllCameras(undistortedFlag)
    
    addCourtCoordinates()