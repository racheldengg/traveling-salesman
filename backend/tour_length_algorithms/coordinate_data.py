import random
import numpy as np
import math
import time
import heapq
import sys
import itertools
from collections import deque

# distance for euclidean coordinates
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# distance for euclidean coordinates rounded up to nearest integer
def ceil2D_distance(x1, y1, x2, y2):
    return np.ceil(np.sqrt((x1 - x2)**2 + (y1 - y2)**2))

#distance for att_distance, aka pseudo-euclidean distance
def att_distance(x1, y1, x2, y2):
    x_ij = x1 - x2
    y_ij = y1 - y2
    r_ij = np.sqrt(((x_ij*x_ij) + (y_ij * y_ij))/10)
    t_ij = np.round(r_ij)
    if (t_ij < r_ij):
        distance = t_ij + 1
    else:
        distance = t_ij
    return distance

# distance for geographical coordinates
def geo_distance(x1, y1, x2, y2):
    earth_radius = 6731

    x1 = math.radians(x1)
    y1 = math.radians(y1)
    x2 = math.radians(x2)
    y2 = math.radians(y2)

    dlat = x2 - x1
    dlon = y2 - y1
    a = math.sin(dlat / 2)**2 + math.cos(x1) * math.cos(x2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    return distance

# get tour length for nearest and farthest insertion
def tour_length(vertices_x, vertices_y, tour, distance_calculation):
    total_length = 0
    for i in range(len(tour)-1):
        total_length += distance_calculation(vertices_x[tour[i]], vertices_y[tour[i]],
                                           vertices_x[tour[i+1]], vertices_y[tour[i+1]])    

    total_length += distance_calculation(vertices_x[tour[-1]], vertices_y[tour[-1]],
                                       vertices_x[tour[0]], vertices_y[tour[0]])
    return total_length


def euclidean_distance_np(x_mat, y_mat, x_mat_t, y_mat_t):
    return np.sqrt(np.add(np.square(np.subtract(x_mat, x_mat_t)), np.square(np.subtract(y_mat, y_mat_t))))

def ceil2D_distance_np(x_mat, y_mat, x_mat_t, y_mat_t):
    return np.ceil(euclidean_distance_np(x_mat, y_mat, x_mat_t, y_mat_t))

def att_distance_np(x_mat, y_mat, x_mat_t, y_mat_t):
    return np.ceil((1/10) * euclidean_distance(x_mat, y_mat, x_mat_t, y_mat_t))

def geo_distance_np(x_mat, y_mat, x_mat_t, y_mat_t):
    diff_x = np.subtract(np.radians(x_mat), np.radians(x_mat_t))
    diff_y = np.subtract(np.radians(y_mat), np.radians(y_mat_t))

    a = np.add(
        np.square(np.sin((1/2) * diff_x)),
        np.multiply(
            np.cos(np.radians(x_mat_t)),
            np.multiply(
                np.cos(np.radians(x_mat)),
                np.square(
                    np.sin((1/2) * diff_y)
                )
            )
        )
    )
    # print(a)

    # print(a.shape)
    # print(np.min(a))
    c = 2 * np.arctan2(
        np.sqrt(a), 
        np.sqrt
        (np.subtract
         (np.ones_like(a), a)))
    return 6731 * c


# trying to implement nearest neighbor
def nearest_neighbor_coordinates(vertices_x, vertices_y, distance_metric, adj_mat_dist):
    # start_time = time.time()
    visited = []
    tour = []
    total_number_cities = len(vertices_x)
    all_cities = set(range(0, total_number_cities))

    # fix the starting city
    starting_city = random.randint(0, total_number_cities-1)
    tour.append(starting_city)
    visited.append(starting_city)

    # get the tour and distance
    unvisited_cities = all_cities - set(visited)
    total_distance = 0
    while len(unvisited_cities) > 0:
        current_city = tour[-1]
        min_distance = np.inf
        nearest_city = None
        
        for city in unvisited_cities:
            distance = distance_metric(vertices_x[current_city], vertices_y[current_city], vertices_x[city], vertices_y[city])
            if distance < min_distance:
                min_distance = distance
                nearest_city = city
        tour.append(nearest_city)
        visited.append(nearest_city)
        total_distance += min_distance
        unvisited_cities = all_cities - set(visited)

    # length = tour_length(x_coordinates, y_coordinates, tour, distance_metric)
    return tour

# nearest insertion
def nearest_insertion_coordinates(vertices_x, vertices_y, distance_metric, adj_mat_dist):
    num_vertices = len(vertices_x)
    unvisited = set(range(num_vertices))
    random_numbers = random.sample(range(num_vertices), 2)
    # Initialize the tour with the first two cities
    tour = np.array([random_numbers[0], random_numbers[1]])
    unvisited.discard(random_numbers[0])
    unvisited.discard(random_numbers[1])

    while unvisited:
        min_increase = np.inf
        best_insertion = None
        
        for new_vertex in unvisited:
            min_distance = np.inf
            best_edge = None
            
            for i in range(len(tour)):
                distance = distance_metric(vertices_x[new_vertex], vertices_y[new_vertex],
                                              vertices_x[tour[i]], vertices_y[tour[i]])
                if distance < min_distance:
                    min_distance = distance
                    best_edge = i
            
            # Calculate the increase in tour length
            increase = min_distance + distance_metric(vertices_x[new_vertex], vertices_y[new_vertex],
                                                          vertices_x[tour[(best_edge+1) % len(tour)]],
                                                          vertices_y[tour[(best_edge+1) % len(tour)]])
            
            if increase < min_increase:
                min_increase = increase
                best_insertion = (best_edge + 1) % len(tour), new_vertex
        
        # Insert the new vertex into the tour using array slicing
        tour = np.insert(tour, best_insertion[0], best_insertion[1])
        unvisited.discard(best_insertion[1])

    # tour_len = tour_length(vertices_x, vertices_y, tour, distance_metric)
    return tour
    
# farthest insertion
def farthest_insertion_coordinates(vertices_x, vertices_y, distance_metric, adj_mat_dist):
    num_vertices = len(vertices_x)
    unvisited = set(range(num_vertices))
    
    # Find the two cities that are farthest apart
    max_distance = -np.inf
    city1, city2 = None, None
    print('finding two farthest cities...')

    vert_x = np.array(vertices_x)
    vert_y = np.array(vertices_y)

    x_mat = np.tile(np.array([vert_x]).transpose(), (1, num_vertices))
    y_mat = np.tile(np.array([vert_y]).transpose(), (1, num_vertices))

    x_mat_t = x_mat.transpose()
    y_mat_t = y_mat.transpose()

    dist_matrix = adj_mat_dist(x_mat, y_mat, x_mat_t, y_mat_t)

    max_index = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)


    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            distance = distance_metric(vertices_x[i], vertices_y[i], vertices_x[j], vertices_y[j])
            if distance > max_distance:
                max_distance = distance
                city1, city2 = i, j
    
    tour = np.array([city1, city2], dtype=np.int32)
    print(f"found two farthest cities, {city1} and {city2}")
    unvisited.discard(city1)
    unvisited.discard(city2)

    k = 0
    while unvisited:
        farthest_vertex = max(unvisited, key=lambda v: min(distance_metric(vertices_x[v], vertices_y[v],
                                                                              vertices_x[t], vertices_y[t])
                                                            for t in tour))
        
        # Find the edge in the tour where inserting the farthest vertex results in the smallest increase
        min_increase = np.inf
        best_edge = None
        
        for i in range(len(tour)):
            increase = distance_metric(vertices_x[farthest_vertex], vertices_y[farthest_vertex],
                                          vertices_x[tour[i]], vertices_y[tour[i]]) + \
                       distance_metric(vertices_x[farthest_vertex], vertices_y[farthest_vertex],
                                          vertices_x[tour[(i+1) % len(tour)]], vertices_y[tour[(i+1) % len(tour)]]) - \
                       distance_metric(vertices_x[tour[i]], vertices_y[tour[i]],
                                          vertices_x[tour[(i+1) % len(tour)]], vertices_y[tour[(i+1) % len(tour)]])
            
            if increase < min_increase:
                min_increase = increase
                best_edge = i
        
        # Insert the farthest vertex into the tour
        insert_position = (best_edge + 1) % len(tour)
        tour = np.insert(tour, insert_position, farthest_vertex)

        k += 1
        print(f"Inserted vertex {farthest_vertex}", k, len(tour))
        unvisited.discard(farthest_vertex)

    # tour_len = tour_length(vertices_x, vertices_y, tour, distance_metric)
    # print(f"Length of tour: {tour_len}")
    
    return tour


# Prim's and Kruskal's
def dfs_preorder(mst_edges):
    start_vertex = random.randint(0, len(mst_edges))
    visited = set()
    result = []
    stack = deque([start_vertex])

    while stack:
        vertex = stack.pop()
        
        if vertex not in visited:
            result.append(vertex)
            visited.add(vertex)
            
            for edge in mst_edges:
                if edge[0] == vertex and edge[1] not in visited:
                    stack.append(edge[1])
                elif edge[1] == vertex and edge[0] not in visited:
                    stack.append(edge[0])
    print(result)
    return result

# using Prim's algorithm for mst heuristic
def prim_dfs_coordinates(vertices_x, vertices_y, distance_metric, adj_mat_dist):
    num_vertices = len(vertices_x)
    mst = [None] * num_vertices
    mst[0] = 0  # Start the MST with vertex 0
    selected = [False] * num_vertices
    selected[0] = True
    edges = []

    for i in range(1, num_vertices):
        edges.append((0, i, distance_metric(vertices_x[0], vertices_y[0], vertices_x[i], vertices_y[i])))
        print(f"Number of edges added to edge list: {len(edges)}")

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
                edges.append((v, i, distance_metric(vertices_x[v], vertices_y[v], vertices_x[i], vertices_y[i])))
    preordered_mst_edges = dfs_preorder(mst_edges)
    return preordered_mst_edges
    
# using Kruskal's algorithm for mst heuristic
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)

    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1

def kruskal_dfs_coordinates(vertices_x, vertices_y, distance_metric, adj_mat_dist):
    num_vertices = len(vertices_x)
    mst_edges = []

    parent = [i for i in range(num_vertices)]
    rank = [0] * num_vertices

    print("Running Kruskal's algorithm...")

    # Create an adjacency matrix to store edge weights
    adj_matrix = np.zeros((num_vertices, num_vertices))
    vert_x = np.array(vertices_x)
    vert_y = np.array(vertices_y)

    x_mat = np.tile(np.array([vert_x]).transpose(), (1, num_vertices))
    y_mat = np.tile(np.array([vert_y]).transpose(), (1, num_vertices))

    x_mat_t = x_mat.transpose()
    y_mat_t = y_mat.transpose()

    adj_matrix = adj_mat_dist(x_mat, y_mat, x_mat_t, y_mat_t)
    print(adj_matrix)

    x_ind_mat = np.tile(np.array([np.arange(0, num_vertices, step=1)]).transpose(), (1, num_vertices))
    y_ind_mat = np.tile(np.arange(0, num_vertices, step=1), (num_vertices, 1))
    
    import numpy.lib.recfunctions as rfn

    x_mat = None 
    y_mat = None 
    x_mat_t = None
    y_mat_t = None 
    vert_x = None 
    vert_y = None

    a_f = adj_matrix.flatten()
    x_f = x_ind_mat.flatten()
    y_f = y_ind_mat.flatten()
    
    adj_matrix = None 
    x_ind_mat = None 
    y_ind_mat = None 
    
    # print('merging sloth')

    flattened = np.zeros(num_vertices ** 2, dtype={
        'names': ('dist', 'u', 'v'),
        'formats': ('f8', 'i8', 'i8')
    })
    
    flattened['dist'] = a_f
    flattened['u'] = x_f
    flattened['v'] = y_f

    flattened.sort(order='dist')
    total = num_vertices ** 2
    
    for i in range(len(flattened)):
        if i % 10000 == 0:
            print((i / total) * 100, '% ', len(mst_edges))
        edge = flattened[i]
        weight = edge['dist']
        u = edge['u']
        v = edge['v']
        if find(parent, u) != find(parent, v):
            mst_edges.append((u, v, weight))
            union(parent, rank, u, v)
            print(f"Added edge: ({u}, {v}) with weight {weight}")

    print("Kruskal's algorithm completed.")
    preordered_mst_edges = dfs_preorder(mst_edges)
    return preordered_mst_edges