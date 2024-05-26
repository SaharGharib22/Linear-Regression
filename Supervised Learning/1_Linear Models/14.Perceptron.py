# Perceptron:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

# Define file path and read the dataset from CSV
file_path = r"G:\python_project\Regression\dataset.csv"
df = pd.read_csv(file_path)

# Split the dataset into features and target variable
X = df[['x']]
y = df['y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert continuous labels to binary labels
threshold = y_train.mean()
y_train_binary = np.where(y_train > threshold, 1, 0)
y_test_binary = np.where(y_test > threshold, 1, 0)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Perceptron model
perceptron = Perceptron()
perceptron.fit(X_train_scaled, y_train_binary)

# Make predictions
y_pred_train = perceptron.predict(X_train_scaled)
y_pred_test = perceptron.predict(X_test_scaled)

# Evaluate the model
train_accuracy = accuracy_score(y_train_binary, y_pred_train)
test_accuracy = accuracy_score(y_test_binary, y_pred_test)
conf_matrix_train = confusion_matrix(y_train_binary, y_pred_train)
conf_matrix_test = confusion_matrix(y_test_binary, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Confusion Matrix (Training):\n", conf_matrix_train)
print("Confusion Matrix (Testing):\n", conf_matrix_test)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_train_scaled, y_train_binary, color='blue', label='Training data')
plt.scatter(X_test_scaled, y_test_binary, color='red', label='Testing data')
plt.plot(X_train_scaled, perceptron.predict(X_train_scaled), color='green', linewidth=2, label='Decision boundary')
plt.title('Perceptron Decision Boundary')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
