import random
import numpy as np
import math
# trying to implement nearest neighbor
# takes in arrays of respective x and y coordinates
def nearest_neighbor(x_coordinates, y_coordinates):
    visited = []
    tour = []
    total_number_cities = len(x_coordinates)
    all_cities = set(range(0, total_number_cities))

    # fix the starting city
    starting_city = random.randint(0, total_number_cities-1)
    tour.append(starting_city)
    visited.append(starting_city)

    # get the tour and distance
    unvisited_cities = all_cities - set(visited)
    total_distance = 0
    while len(unvisited_cities) > 0: # while the number of unvisited cities is not 0
        current_city = tour[-1]
        min_distance = np.inf
        nearest_city = None
        
        for city in unvisited_cities:
            distance = np.sqrt((x_coordinates[current_city] - x_coordinates[city]) ** 2 + (y_coordinates[current_city] - y_coordinates[city]) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_city = city
        tour.append(nearest_city)
        visited.append(nearest_city)
        total_distance += min_distance
        unvisited_cities = all_cities - set(visited)
    # i have total distance, tour, average distance of each cluster, number of cities
    # get data for: total distance, number of cities, how spread out the cities are from each other, how spread out the clusters are from each other

    # get average distance between each point
    sum_distances = 0
    for i in range(total_number_cities):
        x1, y1 = x_coordinates[i], y_coordinates[i]

        for j in range(i + 1, total_number_cities):
            x2, y2 = x_coordinates[j], y_coordinates[j]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            sum_distances += distance
    
    average_distance_btw_points = sum_distances / (total_number_cities * (total_number_cities - 1) / 2)
    print(total_distance)
    return tour, distance, total_number_cities, average_distance_btw_points