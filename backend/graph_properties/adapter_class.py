import numpy as np
from main import *

class to_matrix_adapter:
    def __init__(self, file_path):
        self.file_path = file_path

    def adapt_array_data(file_path):
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

    