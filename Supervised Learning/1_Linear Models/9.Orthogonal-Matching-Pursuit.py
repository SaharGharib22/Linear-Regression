# Orthogonal Matching Pursuit:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error, r2_score

# Define file path and read the dataset from CSV
file_path = r"G:\python_project\Regression\dataset.csv"
df = pd.read_csv(file_path)

# Split the dataset into features and target variable
X = df[['x']]
y = df['y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get the number of features
n_features = X_train_scaled.shape[1]

# Specify the maximum number of non-zero coefficients (atoms)
max_nonzero_coefs = min(n_features, 5)  # Adjust as needed

# Train the Orthogonal Matching Pursuit (OMP) model
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max_nonzero_coefs)
omp.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = omp.predict(X_train_scaled)
y_pred_test = omp.predict(X_test_scaled)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f'Training MSE: {train_mse:.2f}')
print(f'Testing MSE: {test_mse:.2f}')
print(f'Training R^2: {train_r2:.2f}')
print(f'Testing R^2: {test_r2:.2f}')

# Plot the results
plt.figure(figsize=(12, 6))

# Plot training data
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='OMP fit')
plt.title('Training Data and OMP Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Plot testing data
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Testing data')
plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='OMP fit')
plt.title('Testing Data and OMP Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()

# Display model coefficients
print("Model coefficients:", omp.coef_)
print("Model intercept:", omp.intercept_)
