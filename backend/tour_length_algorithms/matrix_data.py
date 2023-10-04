import numpy as np
import os
import random

#calculate length of tour
def tour_length_matrix(adj_matrix, tour_order):
    length = 0
    num_vertices = len(tour_order)

    for i in range(num_vertices - 1):
        from_vertex = tour_order[i]
        to_vertex = tour_order[i + 1]
        length += adj_matrix[from_vertex][to_vertex]

    # Add distance from last to first vertex to complete the tour
    length += adj_matrix[tour_order[-1]][tour_order[0]]
    return length

# nearest neighbor
def nearest_neighbor_matrix(adjacency_matrix):
    num_vertices = len(adjacency_matrix)
    visited = [False] * num_vertices
    tour = []
    
    # Start from vertex 0
    current_vertex = random.randint(0, num_vertices - 1)
    tour.append(current_vertex)
    visited[current_vertex] = True
    
    while len(tour) < num_vertices:
        nearest_vertex = None
        min_distance = float('inf')
        
        # Find the nearest unvisited vertex
        for vertex in range(num_vertices):
            if not visited[vertex] and adjacency_matrix[current_vertex][vertex] < min_distance:
                nearest_vertex = vertex
                min_distance = adjacency_matrix[current_vertex][vertex]
        
        if nearest_vertex is not None:
            tour.append(nearest_vertex)
            visited[nearest_vertex] = True
            current_vertex = nearest_vertex
        else:
            break
    
    # Complete the tour by returning to the starting vertex
    tour.append(tour[0])
    return tour

def find_nearest_vertex(adj_matrix, tour, visited):
    num_vertices = len(adj_matrix)
    min_distance = float('inf')
    nearest_vertex = None
    
    for vertex in range(num_vertices):
        if not visited[vertex]:
            distance = adj_matrix[tour[-1]][vertex]
            if distance < min_distance:
                nearest_vertex = vertex
                min_distance = distance
                
    return nearest_vertex

# nearest insertion
def nearest_insertion_matrix(adjacency_matrix):
    print(adjacency_matrix)
    num_vertices = len(adjacency_matrix)
    visited = [False] * num_vertices
    tour = []
    
    # Choose a random starting vertex
    start_vertex = random.randint(0, num_vertices - 1)
    tour.append(start_vertex)
    visited[start_vertex] = True
    
    while len(tour) < num_vertices:
        next_vertex = find_nearest_vertex(adjacency_matrix, tour, visited)
        
        # Insert the next_vertex at the optimal position in the tour
        min_insert_cost = float('inf')
        optimal_position = 0
        
        for i in range(len(tour)):
            current_cost = adjacency_matrix[tour[i]][next_vertex] + adjacency_matrix[next_vertex][tour[(i + 1) % len(tour)]]
            if current_cost < min_insert_cost:
                min_insert_cost = current_cost
                optimal_position = i
        
        tour.insert((optimal_position + 1) % len(tour), next_vertex)
        visited[next_vertex] = True
    print(tour)
    return tour

# farthest insertion
def find_farthest_vertex(adj_matrix, tour, visited):
    num_vertices = len(adj_matrix)
    max_distance = 0
    farthest_vertex = None
    
    for vertex in range(num_vertices):
        if not visited[vertex]:
            distance = adj_matrix[tour[-1]][vertex]
            if distance > max_distance:
                farthest_vertex = vertex
                max_distance = distance
                
    return farthest_vertex

def farthest_insertion_matrix(adjacency_matrix):
    num_vertices = len(adjacency_matrix)
    unvisited = list(range(num_vertices))
    start_vertex = random.choice(unvisited)  # Start with a random vertex
    tour = [unvisited.pop(unvisited.index(start_vertex))]
    
    while unvisited:
        farthest_vertex = -1
        max_distance = -1
        
        for v in unvisited:
            min_distance = float('inf')
            for tour_vertex in tour:
                if adjacency_matrix[tour_vertex][v] < min_distance:
                    min_distance = adjacency_matrix[tour_vertex][v]
            
            if min_distance > max_distance:
                max_distance = min_distance
                farthest_vertex = v
        
        best_insertion = -1
        min_increase = float('inf')
        
        for i, tour_vertex in enumerate(tour):
            if i == 0:
                prev_vertex = tour[-1]
            else:
                prev_vertex = tour[i - 1]
            increase = adjacency_matrix[prev_vertex][farthest_vertex] + adjacency_matrix[farthest_vertex][tour_vertex] - adjacency_matrix[prev_vertex][tour_vertex]
            if increase < min_increase:
                min_increase = increase
                best_insertion = i
        
        tour.insert(best_insertion, farthest_vertex)
        unvisited.remove(farthest_vertex)
    
    return tour

# prim's and kruskal's
def dfs(graph, start_vertex, visited):
    num_vertices = len(graph)
    stack = [start_vertex]
    tour = [start_vertex]
    visited[start_vertex] = True
    
    while stack:
        current_vertex = stack[-1]
        found_unvisited_neighbor = False
        
        for neighbor in graph[current_vertex]:
            if not visited[neighbor]:
                stack.append(neighbor)
                tour.append(neighbor)
                visited[neighbor] = True
                found_unvisited_neighbor = True
                break
        
        if not found_unvisited_neighbor:
            stack.pop()
    
    return tour

# prim's heuristic
def prim_mst(adj_matrix):
    num_vertices = len(adj_matrix)
    selected = [False] * num_vertices
    selected[0] = True
    mst_edges = []
    
    for _ in range(num_vertices - 1):
        min_weight = float('inf')
        u, v = -1, -1
        
        for i in range(num_vertices):
            if selected[i]:
                for j in range(num_vertices):
                    if not selected[j] and adj_matrix[i][j] < min_weight:
                        min_weight = adj_matrix[i][j]
                        u, v = i, j
                        
        mst_edges.append((u, v))
        selected[v] = True
        
    return mst_edges

def prim_dfs_matrix(adj_matrix):
    num_vertices = len(adj_matrix)
    mst_edges = prim_mst(adj_matrix)
    
    graph = [[] for _ in range(num_vertices)]
    for u, v in mst_edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = [False] * num_vertices
    start_vertex = random.randint(0, num_vertices - 1)
    tour = dfs(graph, start_vertex, visited)
    tour.append(tour[0])  # Complete the tour
    print(tour)
    return tour

# kruskal's heuristic
def find(parent, vertex):
    if parent[vertex] != vertex:
        parent[vertex] = find(parent, parent[vertex])
    return parent[vertex]

def union(parent, rank, u, v):
    root_u = find(parent, u)
    root_v = find(parent, v)
    
    if root_u != root_v:
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        elif rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_v] = root_u
            rank[root_u] += 1

def kruskal_mst(adj_matrix):
    num_vertices = len(adj_matrix)
    edges = []
    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            edges.append((u, v, adj_matrix[u][v]))
            
    edges.sort(key=lambda x: x[2])
    
    parent = list(range(num_vertices))
    rank = [0] * num_vertices
    mst_edges = []
    
    for u, v, weight in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst_edges.append((u, v))
            
    return mst_edges

def kruskal_dfs_matrix(adj_matrix):
    num_vertices = len(adj_matrix)
    mst_edges = kruskal_mst(adj_matrix)
    
    graph = [[] for _ in range(num_vertices)]
    for u, v in mst_edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Choose a random starting vertex for DFS
    start_vertex = random.randint(0, num_vertices - 1)
    visited = [False] * num_vertices
    tour = dfs(graph, start_vertex, visited)
    tour.append(tour[0])  # Complete the tour
    
    print(tour)
    return tour