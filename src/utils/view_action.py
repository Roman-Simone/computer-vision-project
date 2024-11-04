import os
import cv2
import sys

# Add the parent directory to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

# Now you can import the utils module from the parent directory
from utils.config import *

# Define valid camera numbers if they are not defined in config
VALID_CAMERA_NUMBERS = {1, 2, 3, 4, 5}  # Replace with your valid camera numbers

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990),
    6: (4450, 4600)
}

def show_frame_number(video_path, START, END, action_number):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening the video")
        return
    
    # Set the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    
    frame_idx = START
    paused = False  # This flag controls pause/play state

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            
            if not ret or frame_idx > END:
                break
            
            # Display the frame number
            text = f"Frame: {frame_idx}"
            position = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            color = (0, 255, 0)
            thickness = 2
            frame = cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
            
            # Update frame index after reading a frame
            frame_idx += 1

        # Show the frame with the frame number
        cv2.imshow(f"Action {action_number}", frame)
        
        key = cv2.waitKey(25) & 0xFF  # Wait for 25 ms for a key press
        
        if key == ord('q'):  # Quit on 'q' key
            break
        elif key == ord(' '):  # Toggle pause/play on spacebar press
            paused = not paused  # Toggle the paused flag

    cap.release()
    cv2.destroyAllWindows()

def main():
    try:
        # action_number = int(input("Enter the action number: "))
        # if action_number not in ACTIONS:
        #     print("Invalid action number.")
        #     return

        camera_number = 2

        action_number = int(input("Enter the action number: "))
        if action_number not in ACTIONS:
            print("Invalid camera number.")
            return

        video_path = os.path.join(PATH_VIDEOS, f'out{camera_number}.mp4')
        if not os.path.exists(video_path):
            print("Video file does not exist.")
            return

        START, END = ACTIONS[action_number]
        show_frame_number(video_path, START, END, action_number)

    except ValueError:
        print("Invalid input. Please enter numeric values.")

if __name__ == '__main__':
    main()
