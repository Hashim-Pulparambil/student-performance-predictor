import pandas as pd
import joblib

from src.feature_engineering import create_performance_column
from src.train import split_data
from src.models.logistic_pipeline import train_logistic_pipeline

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Feature Engineering
df = create_performance_column(df)

# Split data
X_train, X_test, y_train, y_test = split_data(df)

# Train model
model = train_logistic_pipeline(X_train, y_train)

# Save model
joblib.dump(model, "models/best_model.pkl")

print("✅ Model saved successfully!")