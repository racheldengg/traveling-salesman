import os
import shutil

folder_path = './tsp_decoded'

def euclid_2d():
    destination_folder_eulid_2d = '../tsp_decoded/euclid_2d'
    destination_folder_geo_coordinates = '../tsp_decoded/geo_coordinates'
    destination_folder_att_distance = '../tsp_decoded/att_distance'
    destination_folder_upper_row = '../tsp_decoded/upper_row_matrix'
    destination_folder_full_matrix = '../tsp_decoded/full_matrix'
    destination_folder_lower_diag_matrix = '../tsp_decoded/lower_diagonal_matrix'
    destination_folder_upper_diag_row = '../tsp_decoded/upper_diag_row'
    destination_folder_ceil_2d = '../tsp_decoded/ceil_2d'

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                # Read the file line by line
                # print(file_path)
                for line in file:
                    # Check if the desired line exists
                    if 'EDGE_WEIGHT_TYPE : EUC_2D' in line or 'EDGE_WEIGHT_TYPE: EUC_2D' in line:
                        # Move the file to the destination folder
                        destination_path = os.path.join(destination_folder_eulid_2d, filename)
                        shutil.move(file_path, destination_path)
                        print(f"Moved file: {filename} to {destination_path}")
                        break
                    elif 'EDGE_WEIGHT_TYPE: GEO' in line:
                        destination_path = os.path.join(destination_folder_geo_coordinates, filename)
                        shutil.move(file_path, destination_path)
                        print(f"Moved file: {filename} to {destination_path}")
                        break
                    elif 'EDGE_WEIGHT_TYPE : ATT' in line:
                        destination_path = os.path.join(destination_folder_att_distance, filename)
                        shutil.move(file_path, destination_path)
                        print(f"Moved file: {filename} to {destination_path}")
                        break
                    elif 'EDGE_WEIGHT_FORMAT: UPPER_ROW' in line:
                        destination_path = os.path.join(destination_folder_upper_row, filename)
                        print(destination_path)
                        shutil.move(file_path, destination_path)
                        print(f"Moved file: {filename} to {destination_path}")
                        break
                    elif 'EDGE_WEIGHT_FORMAT: FULL_MATRIX' in line:
                        destination_path = os.path.join(destination_folder_full_matrix, filename)
                        print(destination_path)
                        shutil.move(file_path, destination_path)
                        print(f"Moved file: {filename} to {destination_path}")
                        break
                    elif 'EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW' in line or 'EDGE_WEIGHT_FORMAT : LOWER_DIAG_ROW' in line:
                        destination_path = os.path.join(destination_folder_lower_diag_matrix, filename)
                        print(destination_path)
                        shutil.move(file_path, destination_path)
                        print(f"Moved file: {filename} to {destination_path}")
                        break
                    elif 'EDGE_WEIGHT_FORMAT: UPPER_DIAG_ROW' in line:
                        destination_path = os.path.join(destination_folder_upper_diag_row, filename)
                        print(destination_path)
                        shutil.move(file_path, destination_path)
                        print(f"Moved file: {filename} to {destination_path}")
                        break
                    else:
                        destination_path = os.path.join(destination_folder_ceil_2d, filename)
                        print(destination_path)
                        shutil.move(file_path, destination_path)
                        print(f"Moved file: {filename} to {destination_path}")
                        break




                    


euclid_2d()