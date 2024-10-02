import tkinter as tk
from tkinter import messagebox
from config import *
import cv2
import numpy as np

# Variabili globali per tracciare le immagini e le camere selezionate
selected_images = {}
first_camera_selected = None

# Funzione per gestire l'azione dei pulsanti
def button_action(camera_num, remaining_cameras=None):
    global first_camera_selected
    
    # Prima selezione della camera
    if first_camera_selected is None:
        first_camera_selected = camera_num
        # messagebox.showinfo("Camera selezionata", f"Hai selezionato Camera {camera_num}")

        video_path_1 = path_videos + '/out' + str(camera_num) + '.mp4'
        cap1 = cv2.VideoCapture(video_path_1)
        ret1, img1 = cap1.read()  # Lettura del frame
        if not ret1 or img1 is None:
            print("Error reading video file for camera", camera_num)
            cap1.release()
            return

        cap1.release()  # Rilascia il video

        # Salva l'immagine selezionata
        selected_images["img1"] = img1

        # Chiude la finestra principale
        root.withdraw()
        
        # Procede alla selezione del punto
        take_points(img1, camera_num, remaining_cameras)

    # Seconda selezione della camera
    else:
        # messagebox.showinfo("Second Camera selezionata", f"Hai selezionato Camera {camera_num}")

        video_path_2 = path_videos + '/out' + str(camera_num) + '.mp4'
        cap2 = cv2.VideoCapture(video_path_2)
        ret2, img2 = cap2.read()  # Lettura del frame
        if not ret2 or img2 is None:
            print("Error reading video file for camera", camera_num)
            cap2.release()
            return

        cap2.release()  # Rilascia il video

        # Salva la seconda immagine selezionata
        selected_images["img2"] = img2

        root.destroy()  # Chiude la finestra principale

        # Mostra le due immagini affiancate
        show_side_by_side()

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Se si clicca con il pulsante sinistro
        param['clicked_point'] = (x, y)
        print(f"Mouse clicked at: {x}, {y}")

def take_points(img1, camera_number, remaining_cameras=None):
    clicked_point = {}
    print(f"Select a point in the image for camera {camera_number}")
    window_name = f"Select Point Camera {camera_number}"

    img1_copy = img1.copy()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_point, clicked_point)

    while True:
        cv2.imshow(window_name, img1_copy)
        key = cv2.waitKey(1) & 0xFF

        if 'clicked_point' in clicked_point:
            pt1 = clicked_point['clicked_point']
            print(f"---> Point selected at {pt1}")
                        
            clicked_point.clear()
            cv2.destroyWindow(window_name)
            
            ############################################################
            ###### find correspondences with homography matrix #########
            ############################################################
            
            open_second_camera_selection(remaining_cameras)
            return None

        elif key == ord('q'):
            print("Exiting without selecting a point.")
            cv2.destroyWindow(window_name)
            return None

def open_second_camera_selection(remaining_cameras):

    second_window = tk.Toplevel()
    second_window.title("Selection of Second Camera")
    second_window.geometry("650x300")
    
    label_message = tk.Label(second_window, text="Select the second camera to project the initial point onto.", font=("Arial", 12, "bold"))
    label_message.pack(pady=20)

    button_frame = tk.Frame(second_window)
    button_frame.pack()

    for i, camera_num in enumerate(remaining_cameras):
        btn = tk.Button(button_frame, text=f"Camera {camera_num}", width=10, height=2,
                        command=lambda num=camera_num: button_action(num, remaining_cameras=None))
        row = i // 5  
        col = i % 5   
        btn.grid(row=row, column=col, padx=10, pady=10)  

def show_side_by_side():
    img1 = selected_images.get("img1")
    img2 = selected_images.get("img2")

    if img1 is not None and img2 is not None:
        combined_img = np.hstack((img1, img2))  
        cv2.imshow("Images Side by Side", combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

root = tk.Tk()
root.title("Selezione Camera")
root.geometry("650x300")

label_message = tk.Label(root, text="Select the camera to take the initial point from.", font=("Arial", 12, "bold"))
label_message.pack(pady=20)

camera_list = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

button_frame = tk.Frame(root)
button_frame.pack()

for i, camera_num in enumerate(camera_list):
    remaining_cameras = [cam for cam in camera_list if cam != camera_num]  # Camere rimanenti
    btn = tk.Button(button_frame, text=f"Camera {camera_num}", width=10, height=2,
                    command=lambda num=camera_num, rem=remaining_cameras: button_action(num, rem))
    row = i // 5
    col = i % 5
    btn.grid(row=row, column=col, padx=10, pady=10)

root.mainloop()
