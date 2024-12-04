import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Random seed for reproducibility
seed = 42

# Load the dataset
iris_df = pd.read_csv('C:/Users/ricky/Downloads/Iris.csv')

# Drop the 'Id' column as it's not relevant for training
X = iris_df.drop(columns=['Id', 'Species'])  # Features
y = iris_df['Species']  # Target variable

# Split data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

# Create an instance of the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")  # Display accuracy

# Save the trained model to disk
joblib.dump(clf, "rf_model.sav")
print("Model saved to 'rf_model.sav'")
