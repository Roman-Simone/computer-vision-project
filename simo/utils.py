import os
import pickle


def salva_camerasInfo_pickle(camerasInfo, filename):
    with open(filename, 'wb') as file:
        pickle.dump(camerasInfo, file)


def load_calibration(filename):
    with open(filename, 'rb') as file:
        camerasInfo = pickle.load(file)
    return camerasInfo


def trova_file_mp4(cartella):
    file_mp4 = []
    for file in os.listdir(cartella):
        if file.endswith(".mp4"):
            file_mp4.append(file)
    return file_mp4