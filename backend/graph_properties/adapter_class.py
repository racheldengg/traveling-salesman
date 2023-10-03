
class coordinate_to_matrix_adapter:
    def __init__(self, file2):
        self.file2 = file2

    def process_data(self, file_path):
        adjacency_matrix = self.coordinates_to_sqlite.adapt_coordinates_to_matrix(file_path)
        optimal_k, intercluster_variance, intracluster_variance, number_of_cities, standard_deviation = get_data_matrices(adjacency_matrix)
        print(f"Optimal K: {optimal_k}, Intercluster Variance: {intercluster_variance}, Intracluster Variance: {intracluster_variance}, Number of Cities: {number_of_cities}, Standard Deviation: {standard_deviation}")