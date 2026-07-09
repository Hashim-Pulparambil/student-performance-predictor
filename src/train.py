from sklearn.model_selection import train_test_split


def split_data(df):

    X = df.drop(columns=[
        "Performance",
        "average_score",
        "math score",
        "reading score",
        "writing score"
    ])

    y = df["Performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test