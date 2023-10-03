import os
import shutil

folder_path = '../txt_tsp_data'

def organize_files_by_edge_type():
    destination_folder_eulid_2d = '../txt_tsp_data/euclid_2d'
    destination_folder_geo_coordinates = '../txt_tsp_data/geo_coordinates'
    destination_folder_att_distance = '../txt_tsp_data/att_distance'
    destination_folder_upper_row = '../txt_tsp_data/upper_row_matrix'
    destination_folder_full_matrix = '../txt_tsp_data/full_matrix'
    destination_folder_lower_diag_matrix = '../txt_tsp_data/lower_diagonal_matrix'
    destination_folder_upper_diag_row = '../txt_tsp_data/upper_diag_row'
    destination_folder_ceil_2d = '../txt_tsp_data/ceil_2d'

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                # Read the file line by line
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

organize_files_by_edge_type()