import cv2
import json
import numpy as np


"""
This file is used to store the points used in selectPoint.py 
"""

# Notable points of the volleyball and basketball court image 
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

# Real world coordinates of the notable points
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

# Cameras position in the real world
camera_coordinates_real_world = {
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

# Camera position in the court image 
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

    camera_data = {"camera_coordinates": camera_coordinates_real_world.get(camera_number, []), "points": [
        {"world_coordinate": list(world_coord), "image_coordinate": list(image_coord)} for world_coord, image_coord in world_image_coordinates.items()
    ]}
    data[str(camera_number)] = camera_data

    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data for camera {camera_number} updated successfully in {file_name}")


def unify_images(img1, img2, rightCameraFlag):
    """
    Merges a court image onto a frame image, aligning to the left or right based on the camera position.
    
    Parameters:
        img1 (numpy.ndarray): background frame image.
        img2 (numpy.ndarray): court image to overlay.
        rightCameraFlag (bool): whether the camera is on the right side.
        
    Returns:
        numpy.ndarray: merged image with the court overlay.
    """
    
    _, width1, _ = img1.shape
    scale_factor = 0.15
    new_width = int(width1 * scale_factor)
    aspect_ratio = img2.shape[1] / img2.shape[0]
    new_height = int(new_width / aspect_ratio)
    img2_resized = cv2.resize(img2, (new_width, new_height))

    rows, cols, _ = img2_resized.shape
    x_margin = 50  
    y_offset = 50  

    x_offset = width1 - cols - x_margin if rightCameraFlag else x_margin
    roi = img1[y_offset:y_offset+rows, x_offset:x_offset+cols]

    img2gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2_resized, img2_resized, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    img1[y_offset:y_offset+rows, x_offset:x_offset+cols] = dst

    img1 = add_legend(img1, cols, rightCameraFlag)

    return img1



