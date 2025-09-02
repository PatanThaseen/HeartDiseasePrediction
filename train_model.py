import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv("data/heart.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)))
]

# Final estimator
final_estimator = LogisticRegression()

# Create and train stacked model
stacked_model = StackingClassifier(estimators=base_models, final_estimator=final_estimator)
stacked_model.fit(X_train, y_train)

# Save model
joblib.dump(stacked_model, "model/stacked_model.pkl")

print("✅ Model trained and saved to model/stacked_model.pkl")
