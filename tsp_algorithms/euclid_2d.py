import random
import numpy as np
import math
import time
import sys
import itertools

# trying to implement nearest neighbor
# takes in arrays of respective x and y coordinates
def nearest_neighbor(x_coordinates, y_coordinates):
    start_time = time.time()
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
        print(f"Number of unvisited cities left: {len(unvisited_cities)}")
    # i have total distance, tour, average distance of each cluster, number of cities
    # get data for: total distance, number of cities, how spread out the cities are from each other, how spread out the clusters are from each other, number of clusters

    # get average distance between each point
    sum_distances = 0
    for i in range(total_number_cities):
        x1, y1 = x_coordinates[i], y_coordinates[i]

        for j in range(i + 1, total_number_cities):
            x2, y2 = x_coordinates[j], y_coordinates[j]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            sum_distances += distance
    
    average_distance_btw_points = sum_distances / (total_number_cities * (total_number_cities - 1) / 2)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Total distance: {total_distance}")
    return tour, distance, total_number_cities, average_distance_btw_points



# using Prim's algorithm for mst heuristic


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def prim_mst(vertices_x, vertices_y):
    num_vertices = len(vertices_x)
    mst = [None] * num_vertices
    mst[0] = 0  # Start the MST with vertex 0
    selected = [False] * num_vertices
    selected[0] = True
    edges = []

    for i in range(1, num_vertices):
        edges.append((0, i, euclidean_distance(vertices_x[0], vertices_y[0], vertices_x[i], vertices_y[i])))

    mst_edges = []

    while len(edges) > 0:
        edges.sort(key=lambda edge: edge[2])
        while True:
            u, v, weight = edges.pop(0)
            if selected[v] is False:
                break

        mst[v] = u
        selected[v] = True
        mst_edges.append((u, v, weight))
        print(f"Length of MST: {len(mst_edges)}")

        edges.clear()  # Clear the edges array before adding new edges

        for i in range(num_vertices):
            if selected[i] is False:
                edges.append((v, i, euclidean_distance(vertices_x[v], vertices_y[v], vertices_x[i], vertices_y[i])))

    return mst_edges
    

def dfs_preorder(mst_edges, start_vertex=0):
    num_vertices = max(max(u, v) for u, v, _ in mst_edges) + 1
    graph = [[] for _ in range(num_vertices)]
    length = 0

    for u, v, weight in mst_edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))

    stack = [(start_vertex, None, None)]  # (vertex, parent, edge_weight)
    visited = set()
    preorder_edges = []

    while stack:
        vertex, parent, edge_weight = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            if parent is not None:
                preorder_edges.append((parent, vertex, edge_weight))
                length += edge_weight
            for v, weight in graph[vertex]:
                stack.append((v, vertex, weight))

    return preorder_edges, length

def prims_mst_create_tsp_tour(vertices_x, vertices_y):
    start_time = time.time()
    mst = prim_mst(vertices_x, vertices_y)
    print("MST found")
    preorder_tour, length = dfs_preorder(mst, 0)
    last_vertex = preorder_tour[-1][1]
    last_to_first_distance = euclidean_distance(vertices_x[0], vertices_y[0], vertices_x[last_vertex], vertices_y[last_vertex])
    preorder_tour.append((last_vertex, 0, last_to_first_distance))
    end_time = time.time()
    length += last_to_first_distance
    print(f"Total length of tour: {length}")
    print(f"Total time taken for algorithm: {end_time-start_time}")
    return