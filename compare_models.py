import pandas as pd

from src.feature_engineering import create_performance_column
from src.preprocessing import encode_data
from src.train import split_data
from src.evaluate import evaluate_model

from src.models.logistic import train_logistic
from src.models.decision_tree import train_decision_tree
from src.models.random_forest import train_random_forest
# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Preprocessing
df = create_performance_column(df)
df = encode_data(df)

# Split
X_train, X_test, y_train, y_test = split_data(df)

print("=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

logistic = train_logistic(X_train, y_train)
evaluate_model(logistic, X_test, y_test)

tree = train_decision_tree(X_train, y_train)
evaluate_model(tree, X_test, y_test)

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

forest = train_random_forest(X_train, y_train)
evaluate_model(forest, X_test, y_test)