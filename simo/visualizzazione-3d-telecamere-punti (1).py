import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from config import *
from utils import *
import cv2

def calculate_extrinsics(camera_number):
    # Read the data
    coordinates_by_camera = read_json_file_and_structure_data(path_json)

    all_camera_coordinates = {}

    if str(camera_number) not in coordinates_by_camera:
        print(f"Camera {camera_number} not found in the dataset.")
        return None

    for camera_id, coords in coordinates_by_camera.items():
        if int(camera_id) == camera_number:
            world_points = np.array(coords["world_coordinates"], dtype=np.float32)
            image_points = np.array(coords["image_coordinates"], dtype=np.float32)

        # Collect all camera coordinates (assuming they are provided)
        if "camera_coordinates" in coords:
            cam_coords = np.array(coords["camera_coordinates"], dtype=np.float32)
            all_camera_coordinates[camera_id] = cam_coords

    print(f"World points for Camera {camera_number}:")
    print(world_points)
    print(f"Image points for Camera {camera_number}:")
    print(image_points)
    print(f"Camera coordinates for Camera {camera_number}:")
    print(all_camera_coordinates.get(str(camera_number)))

    # Load camera calibration data
    camera_infos = load_pickle(path_calibration_matrix)

    camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)

    if camera_info is None:
        print(f"Camera info for camera {camera_number} not found.")
        return None

    camera_matrix = camera_info.mtx
    distortion_coefficients = np.zeros((1, 5), dtype=np.float32)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        world_points, image_points, camera_matrix, distortion_coefficients
    )

    # Convert the rotation vector to a rotation matrix using Rodrigues
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)

    extrinsic_matrix = np.hstack((inverse_rotation_matrix, inverse_translation_vector))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

    return extrinsic_matrix


def plot_3d_data(extrinsic_matrix):

    # Crea una figura 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Carica i dati dal file JSON
    with open(path_json, 'r') as file:
        data = json.load(file)

    camera_position = extrinsic_matrix[:3, 3]

    # Plot camera location obtained from extrinsic matrix
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c="r", marker="o", label=f"Camera {camera_number}")

    # Plot camera direction obtained from extrinsic matrix
    direction_vector_size = 5
    camera_direction = extrinsic_matrix[:3, :3] @ np.array([0, 0, direction_vector_size]) + camera_position
    ax.plot(
        [camera_position[0], camera_direction[0]],
        [camera_position[1], camera_direction[1]],
        [camera_position[2], camera_direction[2]],
        c="g",
        label="Camera Direction",
    )


    

    # Colori per le telecamere e i punti
    camera_color = 'red'
    point_color = 'blue'

    # Visualizza le telecamere
    for camera_id, camera_data in data.items():
        camera_coords = camera_data['camera_coordinates']
        ax.scatter(*camera_coords, color=camera_color, s=100, label='Telecamere' if camera_id == '7' else '')
        ax.text(*camera_coords, f'Cam {camera_id}', fontsize=8)

    # Visualizza i punti nel mondo
    all_points = []
    first_point = True  # Variabile per controllare il primo punto
    for camera_data in data.values():
        for point in camera_data['points']:
            world_coord = point['world_coordinate']
            all_points.append(world_coord)
            # Aggiungi la label solo per il primo punto
            ax.scatter(*world_coord, color=point_color, s=100, label='Points' if first_point else '')
            first_point = False

    # Calcola il centro e il raggio della scena
    all_points = np.array(all_points)
    center = all_points.mean(axis=0)
    radius = np.max(np.linalg.norm(all_points - center, axis=1))

    # Imposta i limiti degli assi
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2], center[2] + radius)  # Assumiamo che il suolo sia a z=0

    # Etichette degli assi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Titolo
    ax.set_title('Visualizzazione 3D di Telecamere e Punti nel Mondo')

    # Legenda
    ax.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()



def display_extrinsic_matrix(extrinsic_matrix):
    
    if extrinsic_matrix is not None:
        print(f"\nExtrinsic Matrix for Camera {camera_number}:")
        print(extrinsic_matrix)
        
        # Estrai rotazione e traslazione dalla matrice degli estrinseci
        rotation = extrinsic_matrix[:3, :3]
        translation = extrinsic_matrix[:3, 3]
        
        print("\nRotation Matrix:")
        print(rotation)
        print("\nTranslation Vector:")
        print(translation)
        
        # Calcola gli angoli di Eulero dalla matrice di rotazione
        euler_angles = rotationMatrixToEulerAngles(rotation)
        print("\nEuler Angles (in degrees):")
        print(f"Roll: {np.degrees(euler_angles[0]):.2f}")
        print(f"Pitch: {np.degrees(euler_angles[1]):.2f}")
        print(f"Yaw: {np.degrees(euler_angles[2]):.2f}")
    else:
        print(f"Unable to calculate extrinsic matrix for Camera {camera_number}")

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

if __name__ == "__main__":
    camera_number = 5  # Puoi cambiare questo numero per selezionare una camera diversa
    extrinsic_matrix = calculate_extrinsics(camera_number)
    extrinsic_parameter = display_extrinsic_matrix(extrinsic_matrix)
    plot_3d_data(extrinsic_matrix)

