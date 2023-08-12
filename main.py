from asyncio import gather
# import matplotlib.pyplot as plt
from tsp_algorithms.coordinate_data import *
from tsp_algorithms.matrix_data import *
import os


file_path = '/home/rachel/Desktop/traveling-salesman/tsp_decoded/full_matrix/bays29.tsp.txt'
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
        print(adjacency_matrix)
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

def gather_matrix_data(approx_algorithm, adjacency_matrix): # don't need tour_length_strategy because 
    list_of_vertices = approx_algorithm(adjacency_matrix)
    return list_of_vertices


# currently implemented tour length strategies: euclidean_distance, ceil2D_distance
def parse_coordinate_data(file_path, x_values, y_values, node_labels):
    file_path = '/home/rachel/Desktop/traveling-salesman/tsp_decoded/full_matrix/bays29.tsp.txt'
    full_matrix_directory = os.path.dirname(os.path.dirname(file_path))
    folder_name = full_matrix_directory.split('/')[-1]

    with open(file_path, 'r') as file:
        end_file = len(file.readlines())
        file.seek(0)
        lines = file.readlines()[6:end_file-1]
        
    for line in lines:
        parts = line.strip().split()
        node = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        x_values.append(x)
        y_values.append(y)
        node_labels.append(str(node))
    return

def gather_coordinate_data(approx_algorithm, vertices_x, vertices_y, tour_len_strategy):
    mst = approx_algorithm(vertices_x, vertices_y)
    length_of_tour = tour_length(vertices_x, vertices_y, mst, tour_len_strategy)
    print(length_of_tour)
    return length_of_tour

def main(file_path):
    folder_name = file_path.split('/')[-2]
    if folder_name == 'full_matrix' or folder_name == 'lower_diagonal_matrix' or folder_name == 'upper_diag_row' or folder_name == 'upper_row_matrix':
        adjacency_matrix = parse_matrix_data(file_path)
    else:
        x_values = []
        y_values = []
        node_labels = []
        parse_coordinate_data(file_path, x_values, y_values, node_labels)
        optimal_distance = gather_coordinate_data(prim_mst, x_values, y_values, euclidean_distance)
    # optimal_distance = prims_mst_create_tsp_tour(x_values, y_values)
    # optimal_distance = nearest_insertion_tsp(x_values, y_values)
    # optimal_distance = farthest_insertion_tsp(x_values, y_values)
    # optimal_distance = nearest_neighbor(x_values, y_values)
    # optimal_distance = kruskal_mst_create_tsp_tour(x_values, y_values)
    return optimal_distance


# parse_data('/home/rachel/Desktop/traveling-salesman/tsp_decoded/euclid_2d/d18512.tsp.txt')
# parse_data('/home/rachel/Desktop/traveling-salesman/tsp_decoded/euclid_2d/a280.tsp.txt')
adjacency_matrix = parse_matrix_data('/Users/racheldeng/Desktop/traveling salesman/tsp_decoded/upper_row_matrix/bayg29.tsp.txt')
ordered_vertices = gather_matrix_data(kruskal_dfs_matrix, adjacency_matrix)
length = tour_length_matrix(adjacency_matrix, ordered_vertices)
print(length)
# gather_matrix_data(nearest_neighbor_matrix, adjacency_matrix)

# kruskal_mst_create_tsp_tour
# Total length of tour: 797562.3212599959
# Total time taken for algorithm: 106.60002422332764