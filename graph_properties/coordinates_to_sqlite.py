import sys
sys.path.append('/home/rachel/Desktop/traveling-salesman')
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tour_length_algorithms.coordinate_data import *
from main import *
from graph_properties.matrix_to_sqlite import *

def adapt_coordinates_to_matrix(file_path):
    vertices_x = []
    vertices_y = []
    node_labels = []

    parse_coordinate_data(file_path, vertices_x, vertices_y, node_labels)
    num_vertices = len(vertices_x)
    
    # Create an adjacency matrix to store edge weights
    adj_matrix = np.zeros((num_vertices, num_vertices))
    vert_x = np.array(vertices_x)
    vert_y = np.array(vertices_y)

    x_mat = np.tile(np.array([vert_x]).transpose(), (1, num_vertices))
    y_mat = np.tile(np.array([vert_y]).transpose(), (1, num_vertices))

    x_mat_t = x_mat.transpose()
    y_mat_t = y_mat.transpose()

    folder_name = file_path.split('/')[-2]
    if folder_name == 'att_distance':
        adj_mat_dist = att_distance_np
    elif folder_name == 'ceil_2D':
        adj_mat_dist = ceil2D_distance_np
    elif folder_name == 'euclid_2d':
        adj_mat_dist = euclidean_distance_np
    else:
        adj_mat_dist = geo_distance_np
    
    adj_matrix = adj_mat_dist(x_mat, y_mat, x_mat_t, y_mat_t)
    return adj_matrix

def get_data_coordinates(file_path):

    adjacency_matrix = adapt_coordinates_to_matrix(file_path)

    optimal_k = get_optimal_k_value(adjacency_matrix)
    
    intercluster_variance = get_intercluster_variance(adjacency_matrix, optimal_k)

    intracluster_variance = get_intracluster_variance(adjacency_matrix, optimal_k)

    number_of_cities = len(adjacency_matrix)

    standard_deviation = calculate_standard_deviation(adjacency_matrix)


    # x = np.arange(1, len(weighted_similarity) + 1) # Plot the elbow method for visualization
    # plt.plot(x, weighted_similarity, marker='o')
    # plt.axvline(x=optimal_k, color='r', linestyle='--', label='Elbow Value')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Weighted Similarity')
    # plt.title('Elbow Method for Optimal Clusters (Weighted)')
    # plt.legend()
    # plt.show()
    db_name = '../tsp.db'
    insert_to_database(db_name, file_path, optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation)
    print(f"Optimal K: {optimal_k}, Intercluster Variance: {intercluster_variance}, Intracluster Variance: {intracluster_variance}, Number of Cities: {number_of_cities}, Standard Deviation: {standard_deviation}")
    return optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation

def insert_to_database(db_name, file_path, optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation):
    filename = file_path.split('/')[-1]
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS graph_properties (filename TEXT PRIMARY KEY, optimal_k REAL, intercluster_variance REAL, number_of_cities REAL, standard_deviation REAL)''')
    cursor.execute('''INSERT OR REPLACE INTO graph_properties (filename, optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation) VALUES (?, ?, ?, ?, ?, ?)''',
                           (filename, int(optimal_k), int(intercluster_variance), int(intracluster_variance), int(number_of_cities), int(standard_deviation)))

    connection.commit()
    connection.close()

file_path = '/home/rachel/Desktop/traveling-salesman/txt_tsp_data/ceil_2D/dsj1000.tsp.txt'
db_name = '../tsp.db'
get_data_coordinates(file_path)
