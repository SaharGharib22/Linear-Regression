# Passive Aggressive Algorithms:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r"G:\python_project\Regression\dataset.csv"
df = pd.read_csv(file_path)

# Split features and target variable
X = df.drop(columns=['y'])
y = df['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Passive Aggressive Regressor
pa_regressor = PassiveAggressiveRegressor(max_iter=1000, random_state=42)

# Train the model
pa_regressor.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = pa_regressor.predict(X_train_scaled)
y_pred_test = pa_regressor.predict(X_test_scaled)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
