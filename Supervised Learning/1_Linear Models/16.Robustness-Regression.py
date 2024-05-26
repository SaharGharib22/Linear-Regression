# Robustness Regression: Outliers and Modeling Errors

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r"G:\python_project\Regression\dataset.csv"
df = pd.read_csv(file_path)

# Split features and target variable
X = df[['x']]
y = df['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RANSACRegressor
ransac = RANSACRegressor(random_state=42)

# Fit the model
ransac.fit(X_train, y_train)

# Make predictions
y_pred_train = ransac.predict(X_train)
y_pred_test = ransac.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

# Plot the results
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Testing data')
plt.plot(X_train, y_pred_train, color='green', linewidth=2, label='RANSAC Regression')
plt.title('RANSAC Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
