from sklearn.preprocessing import LabelEncoder


def encode_data(df):
    """
    Encode categorical columns into numeric values.
    """

    label_encoder = LabelEncoder()

    categorical_columns = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "Performance"
    ]

    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    return df