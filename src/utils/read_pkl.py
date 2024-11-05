import pickle
import pprint
import os
import sys

# # Add the parent directory to the system path
# current_path = os.path.dirname(os.path.abspath(__file__))
# parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
# sys.path.append(parent_path)

from utils import *
from config import *

def read_and_save_pkl(file_path, output_file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        with open(output_file_path, 'w') as output_file:
            output_file.write(pprint.pformat(data, indent=4))

if __name__ == "__main__":
    file_path = os.path.join(PATH_3D_DETECTIONS_05, f'points_3d_action4.pkl')
    output_file_path = os.path.join(PATH_3D_DETECTIONS_05, f'points_3d_action4.txt')
    read_and_save_pkl(file_path, output_file_path)