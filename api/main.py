# ===============================
# 🚀 FastAPI - Churn Prediction API
# ===============================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict

import sys
import os

# ----------------------------------
# Path Setup (import src modules)
# ----------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_churn
from src.utils import FEATURE_COLUMNS

# ----------------------------------
# Initialize API
# ----------------------------------
app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Predict customer churn risk using ML and decision intelligence",
    version="1.0.0"
)

# ----------------------------------
# Request Schema (Input Validation)
# ----------------------------------
class CustomerData(BaseModel):
    Recency: float = Field(..., ge=0)
    Frequency: float = Field(..., ge=0)
    Monetary: float = Field(..., ge=0)
    Tenure: float = Field(..., ge=0)
    AOV: float = Field(..., ge=0)
    Purchase_Frequency: float = Field(..., ge=0)
    Profit: float = Field(..., ge=0)


# ----------------------------------
# Health Check Endpoint
# ----------------------------------
@app.get("/")
def root():
    return {
        "message": "Customer Churn Intelligence API is running",
        "status": "success"
    }


# ----------------------------------
# Prediction Endpoint
# ----------------------------------
@app.post("/predict")
def predict(data: CustomerData, threshold: float = 0.6) -> Dict:
    """
    Predict churn probability and customer priority
    """

    try:
        result = predict_churn(data.dict(), threshold)
        return {
            "status": "success",
            "data": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")