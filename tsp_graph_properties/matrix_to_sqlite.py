from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/rachel/Desktop/traveling-salesman')
from main import *
import networkx as nx
import sqlite3
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def get_optimal_k_value(adjacency_matrix):
    similarity_matrix = np.exp(-adjacency_matrix)

    # Range of k values to consider
    k_values = range(2, 8)

    # Initialize a list to store silhouette scores
    silhouette_scores = []

    # Loop through different k values
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(similarity_matrix)
        silhouette_scores.append(silhouette_score(similarity_matrix, labels))

    # Plot the silhouette scores
    # plt.plot(k_values, silhouette_scores, marker='o')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Score for Different k Values (K-Means)')
    # plt.show()

    # Find the index of the maximum silhouette score
    optimal_k_index = np.argmax(silhouette_scores)

    # Calculate the corresponding optimal k value
    optimal_k = k_values[optimal_k_index]
    return optimal_k

# Load adjacency matrix from file
def get_data_matrices(file_path):
    adjacency_matrix = parse_matrix_data(file_path) # Calculate the weighted similarity measure (sum of weights for each node)
    
    optimal_k = get_optimal_k_value(adjacency_matrix)

    intracluster_variance = get_intracluster_variance (adjacency_matrix, optimal_k)
    
    intercluster_variance = get_intercluster_variance(adjacency_matrix, optimal_k)

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
    
    return optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation

def calculate_standard_deviation(adjacency_matrix):
    distances = []
    
    # Iterate through the upper triangular part of the adjacency matrix
    for i in range(len(adjacency_matrix)):
        for j in range(i + 1, len(adjacency_matrix)):
            distance = adjacency_matrix[i][j]
            distances.append(distance)
    
    # Calculate the mean distance
    mean_distance = np.mean(distances)
    
    # Calculate squared differences and average squared differences
    squared_diffs = [(distance - mean_distance) ** 2 for distance in distances]
    avg_squared_diff = np.mean(squared_diffs)
    
    # Calculate and return the standard deviation
    standard_deviation = np.sqrt(avg_squared_diff)
    return standard_deviation

def get_intercluster_variance(adjacency_matrix, k):
    num_nodes = adjacency_matrix.shape[0]
    cluster_assignments = np.random.randint(0, k, num_nodes)  # Replace with your clustering algorithm
    
    intercluster_distances = []

    for i in range(k):
        for j in range(i + 1, k):
            nodes_in_cluster_i = np.where(cluster_assignments == i)[0]
            nodes_in_cluster_j = np.where(cluster_assignments == j)[0]
            
            distances = []

            for node_i in nodes_in_cluster_i:
                for node_j in nodes_in_cluster_j:
                    if node_i < node_j:  # Avoid redundant calculations
                        distance = adjacency_matrix[node_i, node_j]
                        distances.append(distance)
            
            intercluster_distances.extend(distances)

    if len(intercluster_distances) > 0:
        intercluster_variance = np.var(intercluster_distances)
    else:
        intercluster_variance = 0.0
    
    return intercluster_variance

def get_intracluster_variance(adjacency_matrix, k):
    # Create a graph from the adjacency matrix
    graph = nx.Graph(adjacency_matrix)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    node_labels = kmeans.fit_predict(adjacency_matrix)
    
    # Create a dictionary to store nodes in each cluster
    clusters = {i: [] for i in range(k)}
    for i, label in enumerate(node_labels):
        clusters[label].append(i)
    
    # Calculate cluster centroids
    centroids = []
    for cluster_nodes in clusters.values():
        cluster_subgraph = graph.subgraph(cluster_nodes)
        centroid = np.mean(np.array(cluster_subgraph.nodes), axis=0)
        centroids.append(centroid)
    
    # Calculate intra-cluster variance
    intra_cluster_variance = 0
    for label, cluster_nodes in clusters.items():
        cluster_subgraph = graph.subgraph(cluster_nodes)
        centroid = centroids[label]
        for node in cluster_subgraph.nodes:
            distance = np.linalg.norm(node - centroid) ** 2
            intra_cluster_variance += distance
    
    # Divide by total number of nodes
    total_nodes = len(graph.nodes)
    intra_cluster_variance /= total_nodes
    
    return intra_cluster_variance

def process_folder_and_store_results(folder_path, db_name):
    # Connect to the SQLite database
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS graph_properties (filename TEXT PRIMARY KEY, optimal_k REAL, intercluster_variance REAL, intracluster_variance REAL, number_of_cities REAL, standard_deviation REAL)''')

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation = get_data_matrices(file_path)

            # Insert the filename and optimal_k_value into the database
            cursor.execute('''INSERT OR REPLACE INTO graph_properties (filename, optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation) VALUES (?, ?, ?, ?, ?, ?)''',
                           (filename, int(optimal_k), int(intercluster_variance), int(intracluster_variance), int(number_of_cities), int(standard_deviation)))

    # Commit the changes and close the connection
    connection.commit()
    connection.close()

def check_database_values(db_name):
    # Connect to the SQLite database
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    # Execute a SELECT query to retrieve the stored values
    cursor.execute('''SELECT * FROM graph_properties''')
    rows = cursor.fetchall()

    # Print the retrieved rows (filename and result)
    for row in rows:
        print(f"Filename: {row[0]}, Optimal K Value: {row[1]}, Intercluster Variance: {row[2]}, Intracluster Variance: {row[3]}, Number of Cities: {row[4]}, Standard Deviation: {row[5]}")

    # Close the connection
    connection.close()

# visualize_adjacency_matrix(file_path)
# get_number_of_k_clusters(file_path)
# process_folder_and_store_results('/home/rachel/Desktop/traveling-salesman/tsp_decoded/full_matrix', '../tsp.db')
# process_folder_and_store_results('/home/rachel/Desktop/traveling-salesman/tsp_decoded/lower_diagonal_matrix', '../tsp.db')
# process_folder_and_store_results('/home/rachel/Desktop/traveling-salesman/tsp_decoded/upper_diag_row', '../tsp.db')
# process_folder_and_store_results('/home/rachel/Desktop/traveling-salesman/tsp_decoded/upper_row_matrix', '../tsp.db')

# check_database_values('../tsp.db')
# check(file_path)







































# function for visualizing the elbow method more traditionally
def check(file_path):
    adj_matrix = parse_matrix_data(file_path)
    print(adj_matrix)
    # Calculate sum of squared distances (inertia) for different K values
    inertias = []
    max_clusters = 10
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(adj_matrix)
        inertias.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.xticks(np.arange(1, max_clusters + 1))
    plt.grid(True)
    plt.show()

# function for visualizing the adjacency matrix
def visualize_adjacency_matrix(file_path):
    adjacency_matrix = parse_matrix_data(file_path)
    G = nx.Graph()
    num_nodes = adjacency_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = adjacency_matrix[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    
    # Draw the graph with edge weights as labels
    pos = nx.spring_layout(G)  # Position nodes using spring layout algorithm
    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Weighted Graph Visualization from Adjacency Matrix")
    plt.show()