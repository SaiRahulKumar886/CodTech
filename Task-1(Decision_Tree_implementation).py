# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load dataset
iris = load_iris()

# Convert to DataFrame
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print(X.head())
# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Create Decision Tree classifier
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model
model.fit(X_train, y_train)
# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
plt.figure(figsize=(15,10))

plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)

plt.title("Decision Tree Visualization")
plt.show()