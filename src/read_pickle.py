import pickle
from src.utils.config import *

def open_and_print_pkl_file(file_path, output_file_path): 
    with open(file_path, 'rb') as file: 
        content = pickle.load(file) 
        print(content)
        with open(output_file_path, 'w') as output_file:
            output_file.write(str(content))
            
pkl_file_path =  f'{PATH_DETECTIONS}/detections.pkl'
output_file_path = f'{PATH_DETECTIONS}/detections.txt'


open_and_print_pkl_file(pkl_file_path, output_file_path)

