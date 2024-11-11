import pickle
import pprint
import os
import sys

from config import *

def read_and_save_pkl(file_path, output_file_path):
    """
    Reads a pickle file, pretty-prints its content, and saves it to a text file.

    Args:
        file_path (str): the path to the input pickle file.
        output_file_path (str): the path to the output text file where the pretty-printed content will be saved.

    Raises:
        FileNotFoundError: if the input file does not exist.
        IOError: if there is an error reading from the input file or writing to the output file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        with open(output_file_path, 'w') as output_file:
            output_file.write(pprint.pformat(data, indent=4))
        print(data)

if __name__ == "__main__":
    file_path = os.path.join(PATH_3D_DETECTIONS_04, 'points_3d_action3.pkl')
    output_file_path = os.path.join(PATH_3D_DETECTIONS_04, 'points_3d_action3.txt')
    read_and_save_pkl(file_path, output_file_path)