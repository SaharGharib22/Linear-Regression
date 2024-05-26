# Multi-task Elastic-Net:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Define file path and read the dataset from CSV
file_path = r"G:\python_project\Regression\dataset.csv"
df = pd.read_csv(file_path)

# Create synthetic multiple target variables based on the original target variable
df['y1'] = df['y'] + np.random.normal(0, 1, df.shape[0])
df['y2'] = df['y'] * 2 + np.random.normal(0, 1, df.shape[0])

# Split the dataset into features and multiple target variables
X = df[['x']]
Y = df[['y', 'y1', 'y2']]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Multi-task Elastic-Net model
multi_task_elastic_net = MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust alpha and l1_ratio as needed
multi_task_elastic_net.fit(X_train_scaled, Y_train)

# Make predictions
Y_pred_train = multi_task_elastic_net.predict(X_train_scaled)
Y_pred_test = multi_task_elastic_net.predict(X_test_scaled)

# Evaluate the model for each task
for i, target in enumerate(['y', 'y1', 'y2']):
    train_mse = mean_squared_error(Y_train.iloc[:, i], Y_pred_train[:, i])
    test_mse = mean_squared_error(Y_test.iloc[:, i], Y_pred_test[:, i])
    train_r2 = r2_score(Y_train.iloc[:, i], Y_pred_train[:, i])
    test_r2 = r2_score(Y_test.iloc[:, i], Y_pred_test[:, i])

    print(f'Task {target} - Training MSE: {train_mse:.2f}')
    print(f'Task {target} - Testing MSE: {test_mse:.2f}')
    print(f'Task {target} - Training R^2: {train_r2:.2f}')
    print(f'Task {target} - Testing R^2: {test_r2:.2f}')
    print()

# Plot the results
plt.figure(figsize=(18, 6))

# Plot training data
for i, target in enumerate(['y', 'y1', 'y2']):
    plt.subplot(2, 3, i + 1)
    plt.scatter(X_train, Y_train.iloc[:, i], color='blue', label='Training data')
    plt.plot(X_train, Y_pred_train[:, i], color='red', linewidth=2, label='Multi-task Elastic-Net fit')
    plt.title(f'Training Data and Multi-task Elastic-Net Regression Fit for {target}')
    plt.xlabel('X')
    plt.ylabel(target)
    plt.legend()

# Plot testing data
for i, target in enumerate(['y', 'y1', 'y2']):
    plt.subplot(2, 3, i + 4)
    plt.scatter(X_test, Y_test.iloc[:, i], color='blue', label='Testing data')
    plt.plot(X_test, Y_pred_test[:, i], color='red', linewidth=2, label='Multi-task Elastic-Net fit')
    plt.title(f'Testing Data and Multi-task Elastic-Net Regression Fit for {target}')
    plt.xlabel('X')
    plt.ylabel(target)
    plt.legend()

plt.tight_layout()
plt.show()

# Display model coefficients
print("Model coefficients:", multi_task_elastic_net.coef_)
print("Model intercept:", multi_task_elastic_net.intercept_)
