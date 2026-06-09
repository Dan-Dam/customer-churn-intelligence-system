import pandas as pd


FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure",
    "AOV",
    "Purchase_Frequency",
    "Profit"
]


def prepare_features(df):

    """
    Centralized enterprise feature engineering pipeline.
    """

    df = df.copy()

    df["AOV"] = (
        df["Monetary"]
        /
        df["Frequency"].replace(0, 1)
    )

    df["Purchase_Frequency"] = (
        df["Frequency"]
        /
        df["Tenure"].replace(0, 1)
    )

    df["Profit"] = (
        df["Monetary"] * 0.2
    )

    return df[FEATURE_COLUMNS]