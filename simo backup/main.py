import sys
from intrinsic import *
from extrinsic import *
from homography import *


def menu():
    while True:
        print("\n--- Menu ---\n")
        print("Calibration Intrinsic:")
        print("1. Calibrate all cameras")
        print("2. Calibrate a single camera")
        print("3. Test calibration")
        print("\nCalibration Extrinsic:")
        print("5. Calculate extrinsic matrices")
        print("6. Test extrinsic matrices")
        print("\nHomography:")
        print("7. Calculate homography")
        print("8. Test homography")
        print("\n9. Exit:")

        
        scelta = input("\nScegli un'opzione: ")

        if scelta == '1':
            calibrateAllIntrinsic()
        elif scelta == '2':
            try:
                camera_number = int(input("Inserisci il numero della telecamera da calibrare: "))
                calibrateCameraIntrinsic(camera_number)
            except ValueError:
                print("Devi inserire un numero valido.")
        elif scelta == '3':
            test_calibration()
        elif scelta == '4':
            print("Uscita dal programma.")
            sys.exit()
        elif scelta == '5':
            findAllExtrinsics()
        elif scelta == '6':
            plotAllCameras()
        elif scelta == '7':
            calculateHomographyAllCameras()
        elif scelta == '8':
            testHomography()
        else:
            print("Scelta non valida. Riprova.")

if __name__ == '__main__':
    menu()
