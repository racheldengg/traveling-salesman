from asyncio import gather
import matplotlib.pyplot as plt
from tsp_algorithms import nearest_neighbor

def plot_coordinates(file_path):
    x_values = []
    y_values = []
    node_labels = []

    with open(file_path, 'r') as file:
        lines = file.readlines()[6:286]
    for line in lines:
        parts = line.strip().split()
        node = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        x_values.append(x)
        y_values.append(y)
        node_labels.append(str(node))
    nn_solution = nearest_neighbor(x_values, y_values)[0]
    x_ordered = [x_values[i] for i in nn_solution]
    y_ordered = [y_values[i] for i in nn_solution]

    fig, ax = plt.subplots()
    ax.plot(x_ordered, y_ordered, marker='s', color='black', markersize=6)

    for i, point in enumerate(nn_solution):
        bbox_props = dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3')
        ax.text(x_ordered[i], y_ordered[i], str(i+1),
            color='white', fontsize=6, ha='center', va='center', bbox=bbox_props)

    plt.show()


file_path = './tsp_decoded/euclid_2d/a280.tsp.txt'
plot_coordinates(file_path)