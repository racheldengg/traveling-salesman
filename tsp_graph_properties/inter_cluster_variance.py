from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load adjacency matrix from file
adj_matrix = parse_adjacency_matrix('your_file.txt')

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