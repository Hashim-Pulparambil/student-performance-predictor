from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.pipeline import create_preprocessor


def train_logistic_pipeline(X_train, y_train):

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline