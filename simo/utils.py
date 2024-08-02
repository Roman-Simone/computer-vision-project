import os
import pickle
import pandas as pd
import numpy as np
import cv2


def crop_polygon(img, vertices):
    # Creare una maschera nera con le stesse dimensioni dell'immagine
    mask = np.zeros_like(img)
    vertices = np.array([vertices], dtype=np.int32)
    # Riempire il poligono con il bianco
    cv2.fillPoly(mask, [vertices], (255,) * img.shape[2])
    
    # Applicare la maschera all'immagine
    masked_img = cv2.bitwise_and(img, mask)
    
    # Trova i confini del rettangolo che contiene il poligono
    x, y, w, h = cv2.boundingRect(vertices)
    
    # Ritagliare l'immagine utilizzando i confini trovati
    cropped_img = masked_img[y:y+h, x:x+w]
    
    return cropped_img



# Funzione per convertire le stringhe 'x_y' in tuple (x, y)
def convert_to_tuple(coord_str):
    if coord_str == 'ALL':
        return 'ALL'
    x, y = map(float, coord_str.split('_'))
    return (x, y)

def save_pickle(camerasInfo, filename):
    with open(filename, 'wb') as file:
        pickle.dump(camerasInfo, file)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        camerasInfo = pickle.load(file)
    return camerasInfo


def trova_file_mp4(cartella):
    file_mp4 = []
    for file in os.listdir(cartella):
        if file.endswith(".mp4"):
            file_mp4.append(file)
    return file_mp4


