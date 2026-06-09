# ===============================
# 🧹 Data Preprocessing & RFM Pipeline
# ===============================

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataset:
    - Convert date columns
    - Remove missing values
    - Handle infinite values
    """

    df = df.copy()

    # ----------------------------------
    # Convert Dates
    # ----------------------------------
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    # ----------------------------------
    # Remove Invalid Values
    # ----------------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def create_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level RFM table
    """

    # ----------------------------------
    # Reference Date
    # ----------------------------------
    reference_date = df["Order Date"].max()

    # ----------------------------------
    # Core RFM Metrics
    # ----------------------------------
    rfm = df.groupby("Customer ID").agg({
        "Order Date": lambda x: (
            reference_date - x.max()
        ).days,

        "Order ID": "nunique",

        "Sales": "sum",

        "Profit": "sum"
    }).reset_index()

    rfm.columns = [
        "Customer ID",
        "Recency",
        "Frequency",
        "Monetary",
        "Profit"
    ]

    # ----------------------------------
    # Tenure Calculation
    # ----------------------------------
    tenure = df.groupby("Customer ID")["Order Date"].agg(
        ["min", "max"]
    ).reset_index()

    tenure["Tenure"] = (
        tenure["max"] - tenure["min"]
    ).dt.days

    # ----------------------------------
    # Merge Tenure Into RFM
    # ----------------------------------
    rfm = rfm.merge(
        tenure[["Customer ID", "Tenure"]],
        on="Customer ID",
        how="left"
    )

    return rfm


def add_features(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features used for modeling
    """

    rfm = rfm.copy()

    # ----------------------------------
    # Prevent Division By Zero
    # ----------------------------------
    frequency_safe = rfm["Frequency"].replace(0, 1)
    tenure_safe = rfm["Tenure"].replace(0, 1)

    # ----------------------------------
    # Feature Engineering
    # ----------------------------------

    # Average Order Value
    rfm["AOV"] = (
        rfm["Monetary"] / frequency_safe
    )

    # Purchase Frequency Rate
    rfm["Purchase_Frequency"] = (
        rfm["Frequency"] / tenure_safe
    )

    # Profit Proxy
    rfm["Profit"] = (
        rfm["Monetary"] * 0.2
    )

    # ----------------------------------
    # Clean Invalid Values
    # ----------------------------------
    rfm.replace([np.inf, -np.inf], 0, inplace=True)
    rfm.fillna(0, inplace=True)

    return rfm


def create_target(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic churn label
    using business-rule logic
    """

    rfm = rfm.copy()

    rfm["Churn"] = (
        (rfm["Recency"] > 90) &
        (rfm["Frequency"] < 3) &
        (rfm["Monetary"] < 500)
    ).astype(int)

    return rfm