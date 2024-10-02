from config import *
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np

selected_images = {}
first_camera_selected = None

def button_action(camera_num, remaining_cameras=None):
    global first_camera_selected
    
    if first_camera_selected is None:
        first_camera_selected = camera_num
        
        video_path_1 = path_videos + '/out' + str(camera_num) + '.mp4'
        cap1 = cv2.VideoCapture(video_path_1)
        ret1, img1 = cap1.read()
        if not ret1 or img1 is None:
            print("Error reading video file for camera", camera_num)
            cap1.release()
            return

        cap1.release()  

        selected_images[f"img{first_camera_selected}"] = img1
        
        root.withdraw()
        
        take_points(img1, camera_num, remaining_cameras)

    else:
        
        # here if in the second selection screen we select a camera (not "All camera views")
        
        
        video_path_2 = path_videos + '/out' + str(camera_num) + '.mp4'
        cap2 = cv2.VideoCapture(video_path_2)
        ret2, img2 = cap2.read()  
        if not ret2 or img2 is None:
            print("Error reading video file for camera", camera_num)
            cap2.release()
            return

        cap2.release()  

        selected_images[f"img{camera_num}"] = img2
        
        root.destroy()  

        show_side_by_side(camera_num)



def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        param['clicked_point'] = (x, y)
        # print(f"Mouse clicked at: {x}, {y}")

def take_points(img1, camera_number, remaining_cameras=None):
    clicked_point = {}
    # print(f"Select a point in the image for camera {camera_number}")
    window_name = f"Select Point Camera {camera_number}"

    img1_copy = img1.copy()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)  
        
    cv2.setMouseCallback(window_name, select_point, clicked_point)

    while True:
        cv2.imshow(window_name, img1_copy)
        key = cv2.waitKey(1) & 0xFF

        if 'clicked_point' in clicked_point:
            pt1 = clicked_point['clicked_point']
                        
            clicked_point.clear()
            cv2.destroyWindow(window_name)
            
            open_second_camera_selection(remaining_cameras)
            
            return None

        elif key == ord('q'):
            print("Exiting without selecting a point.")
            cv2.destroyWindow(window_name)
            return None

def open_second_camera_selection(remaining_cameras):
    
    second_window = tk.Toplevel()
    second_window.title("Point projection")
    second_window.geometry("650x400")

    label_message = tk.Label(second_window, text="Select the second camera to project the initial point onto.", font=("Arial", 12, "bold"))
    label_message.pack(pady=20)

    button_frame = tk.Frame(second_window)
    button_frame.pack()

    remaining_cameras = [cam for cam in camera_list if cam != first_camera_selected]
    max_columns = 3

    for i, camera_num in enumerate(remaining_cameras):
        btn = tk.Button(button_frame, text=f"Camera {camera_num}", width=10, height=2,
                        command=lambda num=camera_num: button_action(num, remaining_cameras=None))
        row = i // max_columns
        col = i % max_columns
        btn.grid(row=row, column=col, padx=10, pady=10)  

    total_rows = len(remaining_cameras) // max_columns + (1 if len(remaining_cameras) % max_columns > 0 else 0)
    btn_all_views = tk.Button(button_frame, text="All Camera Views", width=15, height=2, command=show_all_views)
    btn_all_views.grid(row=total_rows, column=0, columnspan=max_columns, padx=10, pady=10)  
    
    # second_window.mainloop()


def show_all_views():
    global first_camera_selected
    root.destroy()
    remaining_cameras = [cam for cam in camera_list if cam != first_camera_selected]
    
    first_video_path = path_videos + f'/out{first_camera_selected}.mp4'
    first_cap = cv2.VideoCapture(first_video_path)
    
    ret, first_img = first_cap.read()
    if not ret or first_img is None:
        print("Error reading video file for the first camera")
        first_cap.release()
        return
    
    first_cap.release()
    
    selected_images[f"img{first_camera_selected}"] = first_img

    for camera_num in remaining_cameras:
        video_path = path_videos + f'/out{camera_num}.mp4'
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        if not ret or img is None:
            print(f"Error reading video file for camera {camera_num}")
            cap.release()
            continue  

        cap.release()
        
        selected_images[f"img{camera_num}"] = img
        show_side_by_side(camera_num)

def show_side_by_side(camera_num):
    
    print(f"Showing side by side for camera {first_camera_selected} and {camera_num}")
    
    img_first = selected_images.get(f"img{first_camera_selected}")
    img_next = selected_images.get(f"img{camera_num}")

    if img_first is not None and img_next is not None:
        combined_img = np.hstack((img_first, img_next))
        
        window_name = f"Camera {first_camera_selected} [left] and {camera_num} [right]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1920, 1080)  
        
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        cv2.imshow(window_name, combined_img)
        
        # wait for the user to press 'n' to proceed to the next camera
        while True:
            key = cv2.waitKey(0)
            if key == ord('n'):  
                cv2.destroyAllWindows()  
                break  

        cv2.destroyAllWindows()
    else:
        print(f"Error: One or both images are missing for camera {first_camera_selected} and {camera_num}")


root = tk.Tk()
root.title("Camera selection")
root.geometry("650x300")

label_message = tk.Label(root, text="Select the camera to take the initial point from.", font=("Arial", 12, "bold"))
label_message.pack(pady=20)

camera_list = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

button_frame = tk.Frame(root)
button_frame.pack()

for i, camera_num in enumerate(camera_list):
    remaining_cameras = [cam for cam in camera_list if cam != camera_num]
    btn = tk.Button(button_frame, text=f"Camera {camera_num}", width=10, height=2,
                    command=lambda num=camera_num, rem=remaining_cameras: button_action(num, rem))
    row = i // 5
    col = i % 5
    btn.grid(row=row, column=col, padx=10, pady=10)

root.mainloop()
