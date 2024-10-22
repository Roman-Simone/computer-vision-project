from config import *
from utils import *
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
import json

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600),
    7: (5150, 5330)
}

pathPickle = os.path.join(PATH_DETECTIONS, 'detections.pkl')
camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

cam = [take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS]

def main():
    try:
        action_id = int(input(f"Select an action from the available actions [1, 2, 3, 4, 5, 6, 7] : "))
        if action_id not in ACTIONS:
            print("Invalid action selected. Exiting.")
            exit()
    except ValueError:
        print("Invalid input. Please enter a number corresponding to the action.")
        exit()
    
    START, END = ACTIONS[action_id]
    
    

if __name__ == "__main__":
    main()
