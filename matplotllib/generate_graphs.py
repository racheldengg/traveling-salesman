import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

def clustering_vs_length():
    connection = sqlite3.connect('tsp.db')
    cursor = connection.cursor()

    cursor.execute("SELECT * from clustering_vs_length")
    results = cursor.fetchall()


    algorithm_name =[]
    cluster_ratio = []
    algorithm_length = []
    for row in results:
        algorithm_name.append(row[0])
        cluster_ratio.append(row[1])
        algorithm_length.append(row[2])
    connection.close()

    unique_algorithm_names = np.unique(algorithm_name)
    color_cycle = cycle(plt.cm.tab10.colors)

    color_dict = {name: next(color_cycle) for name in unique_algorithm_names}


    # Create scatter plot
    plt.figure(figsize=(10, 6))

    for name in unique_algorithm_names:
        mask = np.array(algorithm_name) == name
        plt.scatter(np.array(cluster_ratio)[mask], np.array(algorithm_length)[mask], label=name, color=color_dict[name])

    plt.xlabel('Cluster Ratio')
    plt.ylabel('Algorithm Length')
    plt.title('Scatter Plot of Cluster Ratio vs Algorithm Length')
    plt.legend()
    plt.grid(True)
    plt.show()

clustering_vs_length()