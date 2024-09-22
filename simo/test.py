import json

def add_z_to_world_coordinates(json_path):
    # Leggi i dati dal file JSON
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Itera su ogni telecamera nel JSON
    for camera in data:
        points_list = data[camera]['points']
        # Itera su ogni punto nella lista dei punti
        for point in points_list:
            world_coordinate = point['world_coordinate']
            # Se la coordinata del mondo ha solo due elementi, aggiungi z = 0.0
            if len(world_coordinate) == 2:
                world_coordinate.append(0.0)
                point['world_coordinate'] = world_coordinate
    
    # Scrivi i dati modificati di nuovo nel file JSON
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    # Specifica il percorso al tuo file JSON
    json_path = '/Users/simoneroman/Desktop/CV/Computer_Vision_project/data/world_points_all_cameras.json'
    add_z_to_world_coordinates(json_path)
    print(f"Le coordinate del mondo sono state aggiornate nel file {json_path}")
