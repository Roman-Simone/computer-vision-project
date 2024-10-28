import pickle
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *


pkl_file_path = os.path.join(PATH_DETECTIONS, 'all_detections.pkl')

def load_detections(file_path):
    """Load the detections from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def frames_with_few_detections(detections_data):
    """Process detections data to find frames with fewer than 2 detections."""
    frame_detection_count = {}  # Dictionary to count detections per frame
    frames_with_less_than_2_detections = []  # List to store frames with <2 detections
    count_fewer_detections = 0  # Counter for frames with <2 detections

    # Loop through each camera and action in the data
    for camera_id, actions in detections_data.items():
        for action_id, frames in actions.items():
            for frame_idx, detection_point in frames.items():
                # Initialize the frame count if it doesn't exist
                if frame_idx not in frame_detection_count:
                    frame_detection_count[frame_idx] = 0

                # Check if there's at least one detection point
                if detection_point is not None:
                    frame_detection_count[frame_idx] += 1  # Increment the count for this frame

    # Now identify frames with fewer than 2 detections
    for frame_idx, count in frame_detection_count.items():
        if count < 2:  # If fewer than 2 cameras detected a ball
            frames_with_less_than_2_detections.append(frame_idx)
            count_fewer_detections += 1

    return frames_with_less_than_2_detections, count_fewer_detections

if __name__ == '__main__':
    # Load detections from the pickle file
    if not os.path.exists(pkl_file_path):
        print("Pickle file not found!")
    else:
        detections_data = load_detections(pkl_file_path)
        
        # Get frames with fewer than 2 detections and the count
        frames_with_few_detections_list, total_count = frames_with_few_detections(detections_data)

        # Output the results
        print("Frames with fewer than 2 detections:")
        for frame_idx in frames_with_few_detections_list:
            print(f"Frame {frame_idx}")

        print(f"\nTotal frames with fewer than 2 detections: {total_count}")
