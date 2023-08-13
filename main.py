from asyncio import gather
# import matplotlib.pyplot as plt
from tsp_algorithms.coordinate_data import *
from tsp_algorithms.matrix_data import *
import os

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
            print(line)
            
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
         print('adjacency matrix')
         print(adjacency_matrix)
         tour_order = approx_algorithm(adjacency_matrix)
         optimal_distance = get_matrix_length(tour_order, adjacency_matrix)
         return optimal_distance


# parameters for main: path to file, approximation algorithm, type of distance to calculate, how to get the adjacency matrix
length = main('/Users/racheldeng/Desktop/traveling salesman/tsp_decoded/ceil_2D/dsj1000.tsp.txt', farthest_insertion_coordinates)
print(length)


# kruskal_mst_create_tsp_tour
# Total length of tour: 797562.3212599959
# Total time taken for algorithm: 106.60002422332764