import random
import numpy as np
import math
import time
import heapq
import sys
import itertools

# get data for: total distance, number of cities, how spread out the cities are from each other, how spread out the clusters are from each other, number of clusters

# trying to implement nearest neighbor
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
    while len(unvisited_cities) > 0:
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

    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Total distance: {total_distance}")
    return tour, distance, elapsed_time

# nearest insertion
def tour_length(vertices_x, vertices_y, tour):
    total_length = 0
    for i in range(len(tour) - 1):
        total_length += euclidean_distance(vertices_x[tour[i]], vertices_y[tour[i]],
                                           vertices_x[tour[i]], vertices_y[tour[i+1]])
        print(f"Total Length: {vertices_x[tour[i]]}, {vertices_y[tour[i]]}, {vertices_x[tour[i]]}, {vertices_y[tour[i+1]]}, {euclidean_distance(vertices_x[tour[i]], vertices_y[tour[i]], vertices_x[tour[i]], vertices_y[tour[i+1]])}, {total_length}")
    

    total_length += euclidean_distance(vertices_x[tour[-1]], vertices_y[tour[-1]],
                                       vertices_x[tour[0]], vertices_y[tour[0]])
    print(f"Total Length: {total_length}")
    return total_length

def nearest_insertion_tsp(vertices_x, vertices_y):
    num_vertices = len(vertices_x)
    unvisited = set(range(num_vertices))
    
    # Initialize the tour with the first two cities
    tour = np.array([0, 1])
    unvisited.discard(0)
    unvisited.discard(1)

    while unvisited:
        min_increase = np.inf
        best_insertion = None
        
        for new_vertex in unvisited:
            min_distance = np.inf
            best_edge = None
            
            for i in range(len(tour)):
                distance = euclidean_distance(vertices_x[new_vertex], vertices_y[new_vertex],
                                              vertices_x[tour[i]], vertices_y[tour[i]])
                if distance < min_distance:
                    min_distance = distance
                    best_edge = i
            
            # Calculate the increase in tour length
            increase = min_distance + euclidean_distance(vertices_x[new_vertex], vertices_y[new_vertex],
                                                          vertices_x[tour[(best_edge+1) % len(tour)]],
                                                          vertices_y[tour[(best_edge+1) % len(tour)]])
            
            if increase < min_increase:
                min_increase = increase
                best_insertion = (best_edge + 1) % len(tour), new_vertex
        
        # Insert the new vertex into the tour using array slicing
        tour = np.insert(tour, best_insertion[0], best_insertion[1])
        print(f"Length of tour: {len(tour)}")
        unvisited.discard(best_insertion[1])

    tour_len = tour_length(vertices_x, vertices_y, tour)
    return tour, tour_len

# farthest insertion
def farthest_insertion_tsp(vertices_x, vertices_y):
    num_vertices = len(vertices_x)
    unvisited = set(range(num_vertices))
    
    # Find the two cities that are farthest apart
    max_distance = -np.inf
    city1, city2 = None, None
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            distance = euclidean_distance(vertices_x[i], vertices_y[i], vertices_x[j], vertices_y[j])
            if distance > max_distance:
                max_distance = distance
                city1, city2 = i, j
    
    # Initialize the tour with the two farthest cities
    tour = [city1, city2]
    unvisited.discard(city1)
    unvisited.discard(city2)

    while unvisited:
        farthest_vertex = max(unvisited, key=lambda v: min(euclidean_distance(vertices_x[v], vertices_y[v],
                                                                              vertices_x[t], vertices_y[t])
                                                            for t in tour))
        
        # Find the edge in the tour where inserting the farthest vertex results in the smallest increase
        min_increase = np.inf
        best_edge = None
        
        for i in range(len(tour)):
            increase = euclidean_distance(vertices_x[farthest_vertex], vertices_y[farthest_vertex],
                                          vertices_x[tour[i]], vertices_y[tour[i]]) + \
                       euclidean_distance(vertices_x[farthest_vertex], vertices_y[farthest_vertex],
                                          vertices_x[tour[(i+1) % len(tour)]], vertices_y[tour[(i+1) % len(tour)]]) - \
                       euclidean_distance(vertices_x[tour[i]], vertices_y[tour[i]],
                                          vertices_x[tour[(i+1) % len(tour)]], vertices_y[tour[(i+1) % len(tour)]])
            
            if increase < min_increase:
                min_increase = increase
                best_edge = i
        
        # Insert the farthest vertex into the tour
        insert_position = (best_edge + 1) % len(tour)
        tour.insert(insert_position, farthest_vertex)
        unvisited.discard(farthest_vertex)

    print(tour)
    tour_len = tour_length(vertices_x, vertices_y, tour)
    
    return tour, tour_len

# using Prim's algorithm for mst heuristic
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def prim_mst(vertices_x, vertices_y):
    num_vertices = len(vertices_x)
    mst = [None] * num_vertices
    mst[0] = 0  # Start the MST with vertex 0
    selected = [False] * num_vertices
    selected[0] = True
    edges = []

    for i in range(1, num_vertices):
        edges.append((0, i, euclidean_distance(vertices_x[0], vertices_y[0], vertices_x[i], vertices_y[i])))
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

def kruskal_mst(vertices_x, vertices_y):
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

    adj_matrix = np.sqrt(np.add(np.square(np.subtract(x_mat, x_mat_t)), np.square(np.subtract(y_mat, y_mat_t))))
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
    
    print('merging sloth')

    flattened = np.zeros(num_vertices ** 2, dtype={
        'names': ('dist', 'u', 'v'),
        'formats': ('f8', 'i8', 'i8')
    })
    
    flattened['dist'] = a_f
    flattened['u'] = x_f
    flattened['v'] = y_f

    np.save('sloth.npz', flattened)
    print('sloth got flattened')
    flattened.sort(order='dist')
    print('sloth got sorted')
    
    np.save('sloth2.npy', flattened)

    print('loading')
    # flattened = np.load('sloth2.npy')
    print('loaded', len(flattened), num_vertices ** 2)

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
    return mst_edges

def kruskal_mst_create_tsp_tour(vertices_x, vertices_y):
    start_time = time.time()
    mst = kruskal_mst(vertices_x, vertices_y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time}")