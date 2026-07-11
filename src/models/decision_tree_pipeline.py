from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import create_preprocessor


def train_decision_tree_pipeline(X_train, y_train):

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline