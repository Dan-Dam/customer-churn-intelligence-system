# ===============================
# 🤖 Model Training Script
# ===============================

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# Load Data
# ----------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "superstore.csv")

df = pd.read_csv(DATA_PATH)

# ----------------------------------
# Preprocess Dates
# ----------------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"])

# ----------------------------------
# Create Reference Date
# ----------------------------------
reference_date = df["Order Date"].max()

# ----------------------------------
# RFM Feature Engineering
# ----------------------------------
rfm = df.groupby("Customer ID").agg({
    "Order Date": lambda x: (reference_date - x.max()).days,  # Recency
    "Customer ID": "count",                                  # Frequency
    "Sales": "sum"                                           # Monetary
}).rename(columns={
    "Order Date": "Recency",
    "Customer ID": "Frequency",
    "Sales": "Monetary"
}).reset_index()

# ----------------------------------
# Tenure (customer lifespan)
# ----------------------------------
tenure = df.groupby("Customer ID")["Order Date"].agg(["min", "max"]).reset_index()
tenure["Tenure"] = (tenure["max"] - tenure["min"]).dt.days

rfm = rfm.merge(tenure[["Customer ID", "Tenure"]], on="Customer ID")

# ----------------------------------
# Additional Features
# ----------------------------------
rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"].replace(0, 1)
rfm["Purchase_Frequency"] = rfm["Frequency"] / rfm["Tenure"].replace(0, 1)
rfm["Profit"] = rfm["Monetary"] * 0.2  # proxy

# ----------------------------------
# Feature Columns (MUST MATCH APP)
# ----------------------------------
FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure",
    "AOV",
    "Purchase_Frequency",
    "Profit"
]

X = rfm[FEATURES]

# ----------------------------------
# Create Synthetic Churn Label
# More Graduated Risk Logic
# ----------------------------------

rfm["Churn"] = (
    (
        (rfm["Recency"] > 60) &
        (rfm["Frequency"] < 8)
    )
    |
    (
        rfm["Recency"] > 90
    )
).astype(int)

y = rfm["Churn"]

# ----------------------------------
# Train/Test Split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# Scaling
# ----------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------
# Train Model
# ----------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ----------------------------------
# Save Model + Scaler
# ----------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("✅ Model retrained and saved successfully")