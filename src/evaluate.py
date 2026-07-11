from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("=" * 50)
    print("Accuracy")
    print("=" * 50)
    print(f"{accuracy:.3f}")

    print("\n")

    print("=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_test, predictions))

    print("=" * 50)
    print("Confusion Matrix")
    print("=" * 50)
    print(confusion_matrix(y_test, predictions))

    return accuracy