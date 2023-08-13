import sys
sys.path.append('/home/rachel/Desktop/traveling-salesman')
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tsp_algorithms.coordinate_data import *
from main import *

def function1(file_path):
    file_path_array = file_path.split('/')
    distance_calculation = file_path_array[-2]
    x_values = []
    y_values = []
    node_labels = []
    parse_coordinate_data(file_path, x_values, y_values, node_labels)
    if distance_calculation == 'euclid_2D':
        print('euclid2D')
        optimal_k = euclid_data_k_value(x_values, y_values)
    elif distance_calculation == 'ceil_2D':
        print('ceil2D')
        optimal_k = other_data_k_value(x_values, y_values, ceil2D_distance)
    elif distance_calculation == 'att_distance':
        print('att')
        optimal_k = other_data_k_value(x_values, y_values, att_distance)
    elif distance_calculation == 'geo_coordinates':
        print('geocoordinates')
        optimal_k = other_data_k_value(x_values, y_values, geo_distance)
    print(optimal_k)
    return optimal_k

def euclid_data_k_value(x_coordinates, y_coordinates):
    distortions = []
    max_k = len(x_coordinates)  # Adjust this if needed

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(np.column_stack((x_coordinates, y_coordinates)))
        distortions.append(kmeans.inertia_)
    
    # Calculate the rate of change in distortions
    distortion_diffs = np.diff(distortions)
    
    # Find the "elbow point" index
    elbow_index = np.argmin(distortion_diffs) + 1
    
    # plt.plot(range(1, max_k + 1), distortions, marker='o')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Distortion')
    # plt.title('Elbow Method for Optimal K')
    # plt.axvline(x=elbow_index, color='r', linestyle='--', label='Elbow Point')
    # plt.legend()
    # plt.show()
    return elbow_index

def other_data_k_value(x_values, y_values, distance_metric):
    coords = list(zip(x_values, y_values))
    distortions = []
    
    print(len(coords))
    for k in range(1, len(coords) + 1):
        print(k)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(coords)
        
        distortion = 0
        for i, label in enumerate(kmeans.labels_):
            center = kmeans.cluster_centers_[label]
            distortion += distance_metric(coords[i][0], coords[i][1], center[0], center[1])
        distortions.append(distortion)
    
    optimal_k_index = np.argmin(np.diff(distortions)) + 1
    optimal_k = optimal_k_index + 1
    
    # plt.plot(range(1, len(distortions) + 1), distortions, marker='o')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Distortion')
    # plt.title('Elbow Method for Optimal K')
    # plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K: {optimal_k}')
    # plt.legend()
    # plt.show()
    
    return optimal_k



file_name = '/home/rachel/Desktop/traveling-salesman/tsp_decoded/ceil_2D/dsj1000.tsp.txt'
function1(file_name)