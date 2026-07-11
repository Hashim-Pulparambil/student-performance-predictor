from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.pipeline import create_preprocessor


def train_random_forest_pipeline(X_train, y_train):

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    return pipeline