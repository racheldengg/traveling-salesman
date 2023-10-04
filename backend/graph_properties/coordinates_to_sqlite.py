import sys
sys.path.append('../')
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tour_length_algorithms import coordinate_data
from main import *
from graph_properties.matrix_to_sqlite import *
from adapter_class import to_matrix_adapter
from matrix_to_sqlite import matrix_data_properties

class coordinate_data_properties:
    def adapt_coordinates_to_matrix(self, file_path):
        adj_matrix = to_matrix_adapter.adapt_array_data(file_path)
        return adj_matrix

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

    def get_data_and_insert(self, file_path):
        adjacency_matrix = self.adapt_coordinates_to_matrix(file_path)
        optimal_k = matrix_data_properties.get_optimal_k_value(adjacency_matrix)
        intercluster_variance = matrix_data_properties.get_intercluster_variance(adjacency_matrix, optimal_k)
        intracluster_variance = matrix_data_properties.get_intracluster_variance(adjacency_matrix, optimal_k)
        number_of_cities = len(adjacency_matrix)
        standard_deviation = matrix_data_properties.calculate_standard_deviation(adjacency_matrix)

        db_name = '../tsp.db'

        self.insert_to_database(db_name, file_path, optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation)
        print(f"Optimal K: {optimal_k}, Intercluster Variance: {intercluster_variance}, Intracluster Variance: {intracluster_variance}, Number of Cities: {number_of_cities}, Standard Deviation: {standard_deviation}")
        return optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation


file_path = '/home/rachel/Desktop/tsp-part-2/traveling-salesman/backend/data/txt_tsp_data/ceil_2D/dsj1000.tsp.txt'
db_name = '../tsp.db'
data_properties = coordinate_data_properties()
data_properties.get_data_and_insert(file_path)
