from asyncio import gather
# import matplotlib.pyplot as plt
from tour_length_algorithms.coordinate_data import *
from tour_length_algorithms.matrix_data import *
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import gzip

def parse_matrix_data(file_path): 
    folder_name = file_path.split('/')[-2]
    start_flag = 'EDGE_WEIGHT_SECTION'
    end_flags = ['DISPLAY_DATA_SECTION', 'EOF']
    reading_data = False
    data = []

    if folder_name == 'full_matrix':
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == start_flag:
                    reading_data = True
                elif line in end_flags:
                    reading_data = False
                elif reading_data:
                    row = list(map(int, line.split()))
                    data.append(row)
        adjacency_matrix = np.array(data)
        return adjacency_matrix
    
    elif folder_name == 'lower_diagonal_matrix':
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                if line == start_flag:
                    reading_data = True
                elif line in end_flags:
                    reading_data = False
                elif reading_data:
                    row = list(map(int, line.split()))
                    data.append(row)
        num_cities = len(data)
        adjacency_matrix = np.zeros((num_cities, num_cities), dtype=int)

        for i in range(num_cities):
            for j in range(i + 1):
                adjacency_matrix[i, j] = data[i][j]

        
        adjacency_matrix += adjacency_matrix.T # Copy values to the upper triangle to make the matrix symmetric

        # print(adjacency_matrix) # Print the adjacency matrix
        return adjacency_matrix
    
    elif folder_name == 'upper_diag_row':
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == start_flag:
                    reading_data = True
                elif line in end_flags:
                    reading_data = False
                elif reading_data:
                    data.append(list(map(int, line.split())))

        num_cities = len(data) # Determine the size of the square matrix
        matrix_size = (num_cities, num_cities)

        adjacency_matrix = np.zeros(matrix_size, dtype=int)  # Create an empty matrix with zeros

        for i in range(num_cities): # Fill the upper diagonal of the matrix
            for j in range(i, num_cities):
                adjacency_matrix[i, j] = data[i][j - i]  # Adjust the column index

        adjacency_matrix += adjacency_matrix.T # Copy values to the lower triangle to make the matrix symmetric

        # print(adjacency_matrix) # Print the adjacency matrix
        return adjacency_matrix

    else:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                if line == start_flag:
                    reading_data = True
                elif line in end_flags:
                    reading_data = False
                elif reading_data:
                    line = list(map(int, line.split()))
                    # print([0] + line)
                    data.append([0]+ line)
        data = data + [[0]]
        num_cities = len(data) # Determine the size of the square matrix
        matrix_size = (num_cities, num_cities)

        
        adjacency_matrix = np.zeros(matrix_size, dtype=int) # Create an empty matrix with zeros

        
        for i in range(num_cities): # Fill the upper diagonal of the matrix
            for j in range(i, num_cities):
                adjacency_matrix[i, j] = data[i][j - i]  # Adjust the column index

        
        adjacency_matrix += adjacency_matrix.T # Copy values to the lower triangle to make the matrix symmetric

        # print(adjacency_matrix) # Print the adjacency matrix
        return adjacency_matrix

def get_matrix_length(tour_order, adj_matrix): # don't need tour_length_strategy because
    length = 0
    num_vertices = len(tour_order)

    for i in range(num_vertices - 1):
        from_vertex = tour_order[i]
        to_vertex = tour_order[i + 1]
        length += adj_matrix[from_vertex][to_vertex]
        print(length)

    # Add distance from last to first vertex to complete the tour
    length += adj_matrix[tour_order[-1]][tour_order[0]]
    return length

def parse_coordinate_data(file_path, x_values, y_values, node_labels):
    full_matrix_directory = os.path.dirname(os.path.dirname(file_path))
    folder_name = full_matrix_directory.split('/')[-1]

    with open(file_path, 'r') as file:
        end_file = len(file.readlines())
        file.seek(0)
        line = file.readline()
        while (line != 'NODE_COORD_SECTION\n'):
            line = file.readline()
        lines = file.readlines()
    for line in lines:
        try:
            parts = line.strip().split()
            node = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            x_values.append(x)
            y_values.append(y)
            node_labels.append(str(node))
        except:
            pass
    return

def get_coordinate_length(approx_algorithm, vertices_x, vertices_y, distance_metric, adj_mat_dist):
    ordered_vertices = approx_algorithm(vertices_x, vertices_y, distance_metric, adj_mat_dist)
    length_of_tour = tour_length(vertices_x, vertices_y, ordered_vertices, distance_metric)
    print(length_of_tour)
    return length_of_tour

def main(file_path, approx_algorithm):
    folder_name = file_path.split('/')[-2]
    # print(folder_name)
    if folder_name == 'euclid_2d':
        x_values = []
        y_values = []
        node_labels = []
        parse_coordinate_data(file_path, x_values, y_values, node_labels)
        optimal_distance = get_coordinate_length(approx_algorithm, x_values, y_values, euclidean_distance, euclidean_distance_np)
        return optimal_distance
    if folder_name == 'ceil_2D':
        x_values = []
        y_values = []
        node_labels = []
        parse_coordinate_data(file_path, x_values, y_values, node_labels)
        optimal_distance = get_coordinate_length(approx_algorithm, x_values, y_values, ceil2D_distance, ceil2D_distance_np)
        return optimal_distance
    if folder_name == 'att_distance':
        x_values = []
        y_values = []
        node_labels = []
        parse_coordinate_data(file_path, x_values, y_values, node_labels)
        optimal_distance = get_coordinate_length(approx_algorithm, x_values, y_values, att_distance, att_distance_np)
        return optimal_distance
    if folder_name == 'geo_coordinates':
        x_values = []
        y_values = []
        node_labels = []
        parse_coordinate_data(file_path, x_values, y_values, node_labels)
        optimal_distance = get_coordinate_length(approx_algorithm, x_values, y_values, geo_distance, geo_distance_np)
        return optimal_distance
    if folder_name == 'full_matrix' or folder_name == 'lower_diagonal_matrix' or folder_name == 'upper_diag_row' or folder_name == 'upper_row_matrix':
         adjacency_matrix = parse_matrix_data(file_path)
         tour_order = approx_algorithm(adjacency_matrix)
         optimal_distance = get_matrix_length(tour_order, adjacency_matrix)
         print(optimal_distance)
         return optimal_distance

def insert_to_database(db_name, file_path, approx_algorithm):
    filename = file_path.split('/')[-1]
    length = main(file_path, approx_algorithm)

    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()
    approx_algorithm = approx_algorithm.__name__.replace("_coordinates", "")
    print(length, filename, approx_algorithm)
    # Create a table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS graph_data_redone (filename TEXT, approx_algorithm TEXT, length REAL)''')
    cursor.execute('''INSERT INTO graph_data_redone (filename, approx_algorithm, length) VALUES (?, ?, ?)''',
                           (filename, approx_algorithm, int(length)))
    # graph_data_with_complexities
    connection.commit()
    connection.close()

def check_database_values(db_name):
    # Connect to the SQLite database
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    # Execute a SELECT query to retrieve the stored values
    cursor.execute('''SELECT * FROM graph_data_redone''')
    rows = cursor.fetchall()

    # Print the retrieved rows (filename and result)
    for row in rows:
        print(row)

    # Close the connection
    connection.close()

def insert_test_values_into_db():
    db_name = 'tsp.db'
    folder_path = ['/home/rachel/Desktop/traveling-salesman/tsp_decoded/full_matrix/', 
                '/home/rachel/Desktop/traveling-salesman/tsp_decoded/lower_diagonal_matrix/', 
                '/home/rachel/Desktop/traveling-salesman/tsp_decoded/upper_diag_row/', 
                '/home/rachel/Desktop/traveling-salesman/tsp_decoded/upper_row_matrix/'] # change with respect to algorithm 

    for folder in folder_path:
        file_list = file_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        for i in range(5):
            for file in file_list:
                print(file)
                file_path = folder + file
                insert_to_database(db_name, file_path, kruskal_dfs_matrix) # change algorithm
            check_database_values(db_name)

def insert_complexity_values_into_db(db_name):
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS algorithm_complexities (approx_algorithm TEXT, complexity TEXT)''')
    cursor.execute('''INSERT INTO algorithm_complexities (approx_algorithm, complexity) VALUES (?, ?)''', ('prim_dfs', 'O(n^2log(n))'))
    cursor.execute('''INSERT INTO algorithm_complexities (approx_algorithm, complexity) VALUES (?, ?)''', ('kruskal_dfs', 'O(n^2log(n))'))
    cursor.execute('''INSERT INTO algorithm_complexities (approx_algorithm, complexity) VALUES (?, ?)''', ('nearest_neighbor', 'O(n^2)'))
    cursor.execute('''INSERT INTO algorithm_complexities (approx_algorithm, complexity) VALUES (?, ?)''', ('farthest_insertion', 'O(n^2)'))
    cursor.execute('''INSERT INTO algorithm_complexities (approx_algorithm, complexity) VALUES (?, ?)''', ('nearest_insertion', 'O(n^2)'))
    connection.commit()
    connection.close()

def move_optimal_solution():
    source_folder = "/home/rachel/Desktop/traveling-salesman/tsp_extracted"  # Replace with the source folder path
    destination_folder = "/home/rachel/Desktop/traveling-salesman/tsp_decoded_optimal_solution"  # Replace with the destination folder path

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through the files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".opt.tour.gz"):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(source_path, destination_path)
            print(f"Moved {filename} to {destination_folder}")

def extract_optimal_solution(output_folder, source_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".opt.tour.gz"):
            file_path = os.path.join(source_folder, filename)
            output_file = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_file)
            
            with gzip.open(file_path, "rt") as f:
                content = f.read()
                
                with open(output_path, "w") as output:
                    output.write(content)

    print("Extraction complete. Files saved in", output_folder)


# output_folder = '/home/rachel/Desktop/traveling-salesman/tsp_optimal'
# source_folder = '/home/rachel/Desktop/traveling-salesman/tsp_decoded_optimal_solution'
# extract_optimal_solution(output_folder, source_folder)
# move_optimal_solution()
# insert_complexity_values_into_db('tsp.db')

#complete_data
#graph_data_with_properties
# preprocessed_data
#graph_data

db_name = 'tsp.db'
folder_path = '/home/rachel/Desktop/traveling-salesman/txt_tsp_data/euclid_2d/'
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for file in file_list:
    file_path = folder_path + file
    insert_to_database(db_name, file_path, kruskal_dfs_coordinates)
    insert_to_database(db_name, file_path, prim_dfs_coordinates)
    insert_to_database(db_name, file_path, farthest_insertion_coordinates)
    insert_to_database(db_name, file_path, nearest_insertion_coordinates)
    insert_to_database(db_name, file_path, nearest_neighbor_coordinates)

check_database_values(db_name)

