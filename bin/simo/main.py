import sys
from intrinsic import *
from extrinsic import *
from homography import *
from selectPoints import *


def menu():
    while True:
        print("\n--- Menu ---\n")
        print("Intrinsic Calibration:")
        print("0. Calibrate all cameras")
        print("1. Calibrate a single camera")
        print("2. Test calibration")
        print("\nExtrinsic Calibration:")
        print("3. Calculate extrinsic matrices")
        print("4. Test extrinsic matrices")
        print("\nHomography:")
        print("5. Calculate homography")
        print("6. Test homography")
        print("\nSelect Points:")
        print("7. Select points")
        print("\n8. Exit:")

        
        choice = input("\nChoose an option: ")

        if choice == '0':
            calibrateAllIntrinsic()
        elif choice == '1':
            try:
                camera_number = int(input("Enter the camera number to calibrate: "))
                calibrateCameraIntrinsic(camera_number)
            except ValueError:
                print("You must enter a valid number.")
        elif choice == '2':
            test_calibration()
        elif choice == '3':
            findAllExtrinsics()
        elif choice == '4':
            plotAllCameras()
        elif choice == '5':
            calculateHomographyAllCameras()
        elif choice == '6':
            testHomography()

        elif choice == '7':
            try:
                camera_number = int(input("Enter the camera number to calibrate: "))
                calibrateCameraIntrinsic(camera_number)
            except ValueError:
                print("You must enter a valid number.")
            selectPointsCamera(camera_to_select, undistortedFlag = False)
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    menu()
