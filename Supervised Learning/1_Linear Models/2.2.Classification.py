# Classification:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Create synthetic dataset for classification:
np.random.seed(0)
X = np.random.rand(100, 1) * 2                 # Create 100 instances with a feature
y = (X.flatten() > 1).astype(int)              # Convert to two classes: 0 and 1

# Create a dataframe:
df = pd.DataFrame(data={'x': X.flatten(), 'y': y})

# Checking the distribution of classes:
print("Class distribution:")
print(df['y'].value_counts())

# Divide the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['x']], df['y'], test_size=0.2,
                                                    random_state=42, stratify=df['y'])

# Standardization of features:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Logistic Regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_scaled, y_train)

# Predict:
y_pred_train = log_reg.predict(X_train_scaled)
y_pred_test = log_reg.predict(X_test_scaled)

# Model evaluation:
print("Training Classification Report:\n", classification_report(y_train, y_pred_train))
print("Testing Classification Report:\n", classification_report(y_test, y_pred_test))

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')

# Confusion matrix:
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Display the confusion matrix:
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
