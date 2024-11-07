import pickle
import pprint
import os
import sys

from config import *

def read_and_save_pkl(file_path, output_file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        with open(output_file_path, 'w') as output_file:
            output_file.write(pprint.pformat(data, indent=4))
        print(data)

if __name__ == "__main__":
    file_path = os.path.join(PATH_3D_DETECTIONS_04, 'points_3d_action3.pkl')
    output_file_path = os.path.join(PATH_3D_DETECTIONS_04, 'points_3d_action3.txt')
    read_and_save_pkl(file_path, output_file_path)