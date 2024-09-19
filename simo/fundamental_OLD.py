import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt
from utils import undistorted

n_cam = 1

# Carica le informazioni di calibrazione della telecamera
with open('/home/bonnie/Desktop/computer vision/project/Computer_Vision_project/data/calibrationMatrix/calibration.pkl', 'rb') as file:
    camerasInfo = pickle.load(file)

# Carica il video e leggi il primo fotogramma
cap = cv2.VideoCapture('/home/bonnie/Desktop/computer vision/project/Computer_Vision_project/data/dataset/video/out'+ str(n_cam) +'.mp4')
ret, image = cap.read()
cap.release()

# Correggi la distorsione dell'immagine
image = undistorted(image, camera_info=camerasInfo[n_cam])

# Converti l'immagine in HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definisci i range di colore per il bianco
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

# Crea la maschera per il colore bianco
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Applica dilation per migliorare la maschera
kernel = np.ones((10, 10), np.uint8)
dilated_image = cv2.dilate(mask_white, kernel, iterations=1)

# Applica la maschera all'immagine originale
result = cv2.bitwise_and(image, image, mask=dilated_image)

# Converti l'immagine mascherata in scala di grigi
gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Applica il filtro di Canny per rilevare i bordi
edges = cv2.Canny(gray_result, 50, 150)

# Trova le linee rette usando la Trasformata di Hough
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Crea un'immagine di sfondo nero
black_background = np.zeros_like(image)

# Disegna le linee blu sull'immagine di sfondo nero
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(black_background, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Ridimensiona l'immagine
width = 1880
height = 990
resized_image = cv2.resize(black_background, (width, height))

# Mostra l'immagine con le linee blu su sfondo nero
cv2.imshow('Linee Blu su Sfondo Nero', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
