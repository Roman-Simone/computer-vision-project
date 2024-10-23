import os
import re
import cv2
import json
import numpy as np
from utils import *
from config import *


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


# Global variables
clicked_point = ()
all_world_points = {}  # Dictionary to store world-image points for all cameras


# Mouse callback function
def on_mouse(event, x, y, flags, param):
    global clicked_point, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        #print(rateoImages)
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")
        # Disegna un pallino rosso nel punto cliccato
        cv2.circle(img_copy, clicked_point, 5, (0, 0, 255), -1)

        # Mostra le coordinate accanto al punto cliccato
        cv2.putText(img_copy, f"{clicked_point}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #clicked_point = (int(x * rateoImages[0]), int(y * rateoImages[1]))
        #print(f"Clicked(resized) at: {clicked_point}")

        # Aggiorna la finestra di visualizzazione
        cv2.imshow(param, img_copy)


def unifyImages(img1, img2, rightCameraFlag):

    # Get dimensions of the undistorted frame
    height1, width1, _ = img1.shape

    # Decide the scaling factor (e.g., 20% of the frame width)
    scale_factor = 0.15
    new_width = int(width1 * scale_factor)
    # Maintain aspect ratio
    aspect_ratio = img2.shape[1] / img2.shape[0]
    new_height = int(new_width / aspect_ratio)

    # Resize img2 to the new dimensions
    img2_resized = cv2.resize(img2, (new_width, new_height))

    # Get dimensions of the resized img2
    rows, cols, channels = img2_resized.shape

    # Define the position where the court will be placed
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

    # Black-out the area of the logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Extract the region
    img2_fg = cv2.bitwise_and(img2_resized, img2_resized, mask=mask)

    # Overlay the logo on the ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[y_offset:y_offset+rows, x_offset:x_offset+cols] = dst

    _, width, _ = img2_resized.shape

    img1 = add_legend(img1, width, rightCameraFlag)

    return img1


def add_legend(image, width_court, rightCameraFlag):
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color_text = (255, 255, 255)  # White
    thickness_text = 2

    # Vertical spacing
    first_y_offset = 20
    y_offset = 40

    if rightCameraFlag:
        _, image_width, _ = image.shape
        x_offset = (image_width) - (width_court) - 400  # Right margin
    else:
        x_offset = width_court + 80  # Left margin

    # Legend items
    legend_items = [
        {"text": "Corner to select", "color": (0, 170, 255)},  # Yellow
        {"text": "Corner already selected", "color": (0, 255, 0)},  # Green
        {"text": "Corner skipped", "color": (255, 0, 0)},  # Blue
        {"text": "Position of camera", "color": (139, 139, 0)}  # Triangle color
    ]

    # Draw legend
    for i, item in enumerate(legend_items):
        y_position = (i + 1) * y_offset + first_y_offset

        # Draw symbol
        if "Triangolo" in item["text"]:
            triangle_side = 13
            triangle_points = np.array([
                [x_offset, y_position],
                [x_offset - triangle_side, y_position + triangle_side],
                [x_offset + triangle_side, y_position + triangle_side]
            ], np.int32)
            triangle_points = triangle_points.reshape((-1, 1, 2))
            cv2.fillPoly(image, [triangle_points], color=item["color"])
        else:
            cv2.circle(image, (x_offset, y_position), 13, item["color"], -1)

        # Add text
        cv2.putText(image, item["text"], (x_offset + 30, y_position + 5), font, font_scale, color_text, thickness_text)

    return image


def edit_image(image, camera_number=1):

    # Thickness of the point
    point_thickness = 20
    
    width, height, _ = image.shape

    for camera in camera_coordinates_visual:
        if camera == camera_number:
            camera_x = camera_coordinates_visual[camera][0]
            camera_y = camera_coordinates_visual[camera][1]

            # Draw triangle representing the camera position
            triangle_side = 20
            trianglePoints = np.array([
                [camera_x, camera_y - triangle_side],
                [camera_x - triangle_side, camera_y + triangle_side],
                [camera_x + triangle_side, camera_y + triangle_side]
            ], np.int32)

            trianglePoints = trianglePoints.reshape((-1, 1, 2))

            cv2.fillPoly(image, [trianglePoints], color=(139, 139, 0))  # Color: (139, 139, 0)
            break

    for point in points:
        point_x = point[0]
        point_y = point[1]
        if points[point] == 0:
            point_color = (0, 170, 255)  # Yellow
        elif points[point] == 1:
            point_color = (0, 255, 0)  # Green
        elif points[point] == 2:
            point_color = (255, 0, 0)  # Blue

        cv2.circle(image, (point_x, point_y), point_thickness, point_color, -1)

        if points[point] == 0:
            break

    return image

def takePoints(imageUndistorted, courtImg, camera_number, rightCameraFlag, undistortedFlag = False, cameraInfo = None):
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
                    # User clicked on the image
                    points[point] = 1
                    print(f"Point {point} selected at {clicked_point}")

                    if undistortedFlag:
                        clicked_point = (clicked_point[0] + cameraInfo.roi[0], clicked_point[1] + cameraInfo.roi[1])

                    for worldPoint in worldPoints:
                        if worldPoints[worldPoint] == ():
                            worldPoints[worldPoint] = clicked_point
                            break
                    # Save and reset clicked_point
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
                    return

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
    # Include 'camera_coordinates' in the data
    data_list = []
    for coordinate in world_image_coordinates:
        data_list.append({
            "world_coordinate": list(coordinate),
            "image_coordinate": world_image_coordinates[coordinate]
        })

    if data_list:
        # Add to the global all_world_points dictionary
        all_world_points[str(camera_number)] = {
            "camera_coordinates": camera_coordinates_dict.get(camera_number, []),
            "points": data_list
        }
    else:
        print(f"No valid points selected for camera {camera_number}. Data not saved.")


def selectPointsAllCameras(undistortedFlag = False):
    #global rateoImages
    frames = find_files(PATH_FRAME_DISTORTED)
    frames.sort()

    for frame in frames:
        print(frame)

        camera_number = re.findall(r'\d+', frame.replace(".png", ""))
        camera_number = int(camera_number[0])
        if camera_number not in valid_camera_numbers:
            continue
        
        if camera_number in rightCamera:
            rightCameraFlag = True
        else:
            rightCameraFlag = False

        frameImg = cv2.imread(os.path.join(PATH_FRAME_DISTORTED, frame))
        courtImg = cv2.imread(PATH_COURT)
        
        if undistortedFlag:
            camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
            cameraInfo, _ = take_info_camera(camera_number, camerasInfo)

            frameImg = undistorted(frameImg, cameraInfo)
            world_image_coordinates = takePoints(frameImg, courtImg, camera_number, rightCameraFlag, undistortedFlag, cameraInfo)

        else:
            world_image_coordinates = takePoints(frameImg, courtImg, camera_number, rightCameraFlag)

        # Save worldPoints and imagePoints to the global dictionary
        commonList(camera_number, world_image_coordinates)

        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

    pathToSave = PATH_JSON_DISTORTED
    if undistortedFlag:
        pathToSave = PATH_JSON_UNDISTORTED
    
    # After processing all videos, save the combined JSON file
    if all_world_points:
        with open(pathToSave, 'w') as json_file:
            json.dump(all_world_points, json_file, indent=4)
        print(f"All world points saved to {pathToSave}")
    else:
        print("No valid points selected for any camera. No data saved.")


def selectPointsCamera(camera_to_select, undistortedFlag = False):
    #global rateoImages

    frames = find_files(PATH_FRAME_DISTORTED)

    for frame in frames:
        print(frame)

        camera_number = re.findall(r'\d+', frame.replace(".png", ""))
        camera_number = int(camera_number[0])
        if camera_number not in VALID_CAMERA_NUMBERS or camera_number != camera_to_select:
            continue
        
        if camera_number in rightCamera:
            rightCameraFlag = True
        else:
            rightCameraFlag = False

        frameImg = cv2.imread(os.path.join(PATH_FRAME_DISTORTED, frame))
        courtImg = cv2.imread(PATH_COURT)
        
        if undistortedFlag:
            camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
            cameraInfo, _ = take_info_camera(camera_number, camerasInfo)

            frameImg = undistorted(frameImg, cameraInfo)
            world_image_coordinates = takePoints(frameImg, courtImg, camera_number, rightCameraFlag, undistortedFlag, cameraInfo)

        else:
            world_image_coordinates = takePoints(frameImg, courtImg, camera_number, rightCameraFlag)
        
        pathToSave = PATH_JSON_DISTORTED
        if undistortedFlag:
            pathToSave = PATH_JSON_UNDISTORTED

        update_json_file(camera_number, world_image_coordinates, pathToSave)
    
        cv2.destroyAllWindows()



def update_json_file(camera_number, world_image_coordinates, file_name):
    # Legge il file JSON esistente
    try:
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}

    # Prepara i nuovi dati per la camera selezionata
    camera_data = {
        "camera_coordinates": camera_coordinates_dict.get(camera_number, []),
        "points": []
    }

    # Unisci le coordinate mondiali e dell'immagine
    for world_coord, image_coord in world_image_coordinates.items():
        camera_data["points"].append({
            "world_coordinate": list(world_coord),
            "image_coordinate": list(image_coord)
        })

    # Aggiorna solo i dati della camera specificata nel file JSON
    data[str(camera_number)] = camera_data

    # Scrive i dati aggiornati nel file JSON mantenendo intatti gli altri dati
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"I dati per la camera {camera_number} sono stati aggiornati correttamente in {file_name}")




if __name__ == '__main__':
    # Select points for all cameras
    # undistortedFlag = False
    # selectPointsAllCameras(undistortedFlag)

    # Select points for a specific camera
    camera_to_select = 4
    undistortedFlag = False
    selectPointsCamera(camera_to_select, undistortedFlag)

