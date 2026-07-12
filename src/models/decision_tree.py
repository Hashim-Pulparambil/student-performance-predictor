from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    )

    model.fit(X_train, y_train)

    return model
print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

forest = train_random_forest(X_train, y_train)
evaluate_model(forest, X_test, y_test)