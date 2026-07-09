import pandas as pd

from src.feature_engineering import create_performance_column
from src.preprocessing import encode_data
from src.train import split_data
from src.model import train_decision_tree
from src.evaluate import evaluate_model

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Feature Engineering
df = create_performance_column(df)

# Encode categorical data
df = encode_data(df)

# Split data
X_train, X_test, y_train, y_test = split_data(df)

# Train Decision Tree
model = train_decision_tree(X_train, y_train)

# Evaluate model
evaluate_model(model, X_test, y_test)