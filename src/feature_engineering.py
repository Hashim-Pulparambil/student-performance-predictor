import pandas as pd


def create_performance_column(df):

    df["average_score"] = (
        df["math score"] +
        df["reading score"] +
        df["writing score"]
    ) / 3

    def performance(avg):
        if avg >= 80:
            return "Excellent"
        elif avg >= 60:
            return "Average"
        else:
            return "Needs Improvement"

    df["Performance"] = df["average_score"].apply(performance)

    return df