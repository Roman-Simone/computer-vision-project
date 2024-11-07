import os
import cv2
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from config import *


def show_frame_number(video_path, START, END, action_number):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening the video")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    
    frame_idx = START
    paused = False  

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            
            if not ret or frame_idx > END:
                break
            
            text = f"Frame: {frame_idx}"
            position = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            color = (0, 255, 0)
            thickness = 2
            frame = cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
            
            frame_idx += 1

        cv2.imshow(f"Action {action_number}", frame)
        
        key = cv2.waitKey(25) & 0xFF  
        
        if key == ord('q'):  
            break
        elif key == ord(' '):  # toggle pause/play on spacebar press
            paused = not paused  

    cap.release()
    cv2.destroyAllWindows()

def main():
    try:

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
