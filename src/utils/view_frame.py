import cv2
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

def show_frame(video_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return
    
    # Display the frame
    cv2.imshow(f'Frame {frame_number}', frame)
    
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    video_path = os.path.join(PATH_VIDEOS, "out6.mp4")  # Replace with your video path
    frame_number = 100  # Replace with your desired frame number
    show_frame(video_path, frame_number)
