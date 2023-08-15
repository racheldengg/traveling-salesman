import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import sys

# Connect to SQLite database
conn = sqlite3.connect('tsp.db')

# Load data from tables into Pandas DataFrames
preprocessed_data = pd.read_sql_query("SELECT * FROM preprocessed_data", conn)
test = pd.read_sql_query("SELECT approx_algorithm FROM preprocessed_data", conn)
# Close the database connection
conn.close()

# Merge data based on tsp_instance

# Preprocessing, feature selection/engineering, and splitting (X and y)
# Your preprocessing, feature selection/engineering, and splitting code here
# Make sure your X contains the input features and y contains the difference in tour lengths


df = pd.DataFrame(preprocessed_data)

y = df['approx_algorithm']
encoded_y = pd.get_dummies(y, prefix='approx_algorithm')
encoded_y_columns = encoded_y.columns

df['approx_algorithm_complexity'] = df['approx_algorithm_complexity'].map({
    'O(n^2log(n))': 2,
    'O(n^2)': 1
})


df['intra_to_inter_ratio'] =  df['intracluster_variance'] / df['intercluster_variance']
df['length_difference'] = df['approx_algorithm_length'] - df['optimal_tour_length']



X = df.drop(['approx_algorithm', 'tsp_instance', 'approx_algorithm_length', 'approx_algorithm_complexity', 'intercluster_variance', 'intracluster_variance', 'optimal_tour_length', 'length_difference'], axis=1)
print(X.columns)
y = encoded_y
print(y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Data
new_data = {
    'optimal_k': [4.0],
    'number_of_cities': [175.0],
    'standard_deviation': [65.0],
    'intra_to_inter_ratio': [715.0/4294.0],
    # 'length_difference': [10.0]
}
# Make predictions using the trained model for new instances
new_instance_features = pd.DataFrame(new_data)  # Replace with new instance features

# Predict the differences for each approximation algorithm
predicted_differences = model.predict(new_instance_features)

print(predicted_differences)

recommended_algorithm_index = np.argmin(predicted_differences)

# List of approximation algorithms
approximation_algorithms = ['approx_algorithm_farthest_insertion', 'approx_algorithm_kruskal_dfs', 'approx_algorithm_nearest_insertion', 'approx_algorithm_nearest_neighbor', 'approx_algorithm_prim_dfs']

# Get the recommended algorithm using the index
recommended_algorithm = approximation_algorithms[recommended_algorithm_index]

print("Recommended Approximation Algorithm:", recommended_algorithm)