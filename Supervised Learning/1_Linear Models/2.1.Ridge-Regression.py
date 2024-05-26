# Ridge Regression:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Ensure reproducibility
np.random.seed(42)

# Define file path and read the dataset from CSV
file_path = r"G:\python_project\Regression\dataset.csv"
df = pd.read_csv(file_path)

# Split the dataset
X = df[['x']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Ridge regression model
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = ridge_reg.predict(X_train_scaled)
y_pred_test = ridge_reg.predict(X_test_scaled)

# Ensure predictions have the same length as true values
assert len(y_train) == len(y_pred_train), "Inconsistent number of samples in training set"
assert len(y_test) == len(y_pred_test), "Inconsistent number of samples in test set"

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f'Training MSE: {train_mse:.2f}')
print(f'Testing MSE: {test_mse:.2f}')

# Visualize the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.7)
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Train Data: Actual vs Predicted y')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.7, color='orange')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Test Data: Actual vs Predicted y')

plt.tight_layout()
plt.show()

# Plot the regression line
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='Ridge Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ridge Regression')
plt.legend()
plt.show()
