import os
import re
import cv2
import sys
import json
import numpy as np
from cameraInfo import *
from utilsCameraCalibration.utilsSelectPoints import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

rightCamera = [5, 12, 13]

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


def take_points(imageUndistorted, courtImg, camera_number, rightCameraFlag, undistortedFlag=False, cameraInfo=None):
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
                image = unify_images(img_copy, courtImgEdited, rightCameraFlag)
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


def create_common_list(camera_number, world_image_coordinates):
    """
    Appends selected points and camera coordinates to a global dictionary for JSON export.
    
    Parameters:
        camera_number (int): camera number to include in the JSON data.
        world_image_coordinates (dict): mapping of world points to image coordinates.
    """
    
    data_list = [{"world_coordinate": list(coord), "image_coordinate": image_coord} for coord, image_coord in world_image_coordinates.items()]
    all_world_points[str(camera_number)] = {"camera_coordinates": camera_coordinates_real_world.get(camera_number, []), "points": data_list}


def select_points_all_cameras(undistortedFlag=False):
    """
    Loops through frames for all valid cameras, allowing the user to select points, and saves the results in JSON.
    
    Parameters:
        undistortedFlag (bool): if True, undistorts each frame before displaying.
    """
    
    frames = sorted(find_files(PATH_FRAME_DISTORTED))

    for frame in frames:
        camera_number = int(re.findall(r'\d+', frame.replace(".png", ""))[0])
        if camera_number not in VALID_CAMERA_NUMBERS:
            continue
        
        rightCameraFlag = camera_number in rightCamera
        frameImg = cv2.imread(os.path.join(PATH_FRAME_DISTORTED, frame))
        courtImg = cv2.imread(PATH_COURT)

        if undistortedFlag:
            camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
            cameraInfo, _ = take_info_camera(camera_number, camerasInfo)
            frameImg = undistorted(frameImg, cameraInfo)
            world_image_coordinates = take_points(frameImg, courtImg, camera_number, rightCameraFlag, undistortedFlag, cameraInfo)
        else:
            world_image_coordinates = take_points(frameImg, courtImg, camera_number, rightCameraFlag)

        create_common_list(camera_number, world_image_coordinates)
        cv2.destroyAllWindows()

    pathToSave = PATH_JSON_UNDISTORTED if undistortedFlag else PATH_JSON_DISTORTED
    
    if all_world_points:
        with open(pathToSave, 'w') as json_file:
            json.dump(all_world_points, json_file, indent=4)
        print(f"All world points saved to {pathToSave}")
    else:
        print("No valid points selected for any camera. No data saved.")


def select_points_camera(camera_to_select, undistortedFlag=False):
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
        world_image_coordinates = take_points(frameImg, courtImg, 0, False)
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
            world_image_coordinates = take_points(frameImg, courtImg, camera_number, rightCameraFlag, undistortedFlag, cameraInfo)
        else:
            world_image_coordinates = take_points(frameImg, courtImg, camera_number, rightCameraFlag)
        
        pathToSave = PATH_JSON_UNDISTORTED if undistortedFlag else PATH_JSON_DISTORTED
        update_json_file(camera_number, world_image_coordinates, pathToSave)


def addCourtCoordinates():
    """
    Maps world coordinates to court image coordinates for court visualization, saving them in JSON.
    """
    
    world_image_coordinates = {worldCord: courtCord for courtCord, worldCord in zip(points, worldPoints)}
    update_json_file(0, world_image_coordinates, PATH_JSON_DISTORTED)
    update_json_file(0, world_image_coordinates, PATH_JSON_UNDISTORTED)
    

if __name__ == "__main__":
    
    # undistortedFlag = False
    # select_points_all_cameras(undistortedFlag)

    undistortedFlag = False
    camera_number = 1
    select_points_camera(camera_number, undistortedFlag)
    
    addCourtCoordinates()