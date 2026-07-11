import pandas as pd

from src.feature_engineering import create_performance_column
from src.train import split_data
from src.evaluate import evaluate_model

from src.models.logistic_pipeline import train_logistic_pipeline
from src.models.decision_tree_pipeline import train_decision_tree_pipeline
from src.models.random_forest_pipeline import train_random_forest_pipeline

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Feature Engineering
df = create_performance_column(df)

# Split
X_train, X_test, y_train, y_test = split_data(df)

print("=" * 50)
print("LOGISTIC REGRESSION (PIPELINE)")
print("=" * 50)

logistic = train_logistic_pipeline(X_train, y_train)
logistic_acc = evaluate_model(logistic, X_test, y_test)

print("\n" + "=" * 50)
print("DECISION TREE")
print("=" * 50)

tree = train_decision_tree_pipeline(X_train, y_train)
tree_acc = evaluate_model(tree, X_test, y_test)

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

forest = train_random_forest_pipeline(X_train, y_train)
forest_acc = evaluate_model(forest, X_test, y_test)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_acc,
        tree_acc,
        forest_acc
    ]
})

print("\n")
print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(results)

results.to_csv("model_results.csv", index=False)