from asyncio import gather
import matplotlib.pyplot as plt
from tsp_algorithms.euclid_2d import *

# go thorugh euclid_2d and gather data for name of dataset, # of points, standard deviation of clusters, number of clusters, sparsity and density of graph
# ^^ all things that cannot change
# store in tuples
# run the algorithm more times to get varying data for runtime and distance


def parse_data(file_path):
    x_values = []
    y_values = []
    node_labels = []

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
    # optimal_distance = prims_mst_create_tsp_tour(x_values, y_values)
    optimal_distance = nearest_neighbor(x_values, y_values)
    return optimal_distance

## test run held_karp
# parse_data('/home/rachel/Desktop/traveling-salesman/tsp_decoded/euclid_2d/d18512.tsp.txt')
parse_data('/home/rachel/Desktop/traveling-salesman/tsp_decoded/euclid_2d/a280.tsp.txt')