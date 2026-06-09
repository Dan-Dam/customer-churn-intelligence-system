# ===============================
# 🚀 FastAPI - Churn Prediction API
# Enterprise Production Version
# ===============================

import os
import sys
import time
import logging

from datetime import datetime
from typing import Dict, Any

from fastapi import (
    FastAPI,
    HTTPException,
    Query
)

from pydantic import (
    BaseModel,
    Field
)

# ==================================
# LOGGING
# ==================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==================================
# PATH SETUP
# ==================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.predict import predict_churn

# ==================================
# FASTAPI INITIALIZATION
# ==================================

app = FastAPI(

    title="Customer Churn Intelligence API",

    description="""
    Enterprise-grade AI API for customer churn prediction,
    explainable AI, retention intelligence,
    and decision support.
    """,

    version="3.0.0",

    contact={
        "name": "Daniel Damilola Amosun",
        "organization": "InferaIQ"
    },

    tags_metadata=[

        {
            "name": "Health",
            "description":
            "System monitoring endpoints"
        },

        {
            "name": "Prediction",
            "description":
            "Churn prediction endpoints"
        }
    ]
)

# ==================================
# STARTUP EVENT
# ==================================

@app.on_event("startup")
async def startup_event():

    logger.info(
        "Customer Churn Intelligence API started successfully."
    )

# ==================================
# REQUEST SCHEMA
# ==================================

class CustomerData(BaseModel):

    Recency: float = Field(
        ...,
        ge=0,
        description="Days since last purchase"
    )

    Frequency: float = Field(
        ...,
        ge=0,
        description="Number of purchases"
    )

    Monetary: float = Field(
        ...,
        ge=0,
        description="Total customer spend"
    )

    Tenure: float = Field(
        ...,
        ge=0,
        description="Customer lifespan in days"
    )

# ==================================
# RESPONSE SCHEMA
# ==================================

class PredictionResponse(BaseModel):

    status: str

    timestamp: str

    latency_seconds: float

    data: Dict[str, Any]

# ==================================
# HEALTH CHECK
# ==================================

@app.get(
    "/",
    tags=["Health"]
)

def root():

    return {

        "status": "success",

        "message":
        "Customer Churn Intelligence API is running",

        "version": "3.0.0",

        "timestamp":
        datetime.utcnow().isoformat()
    }

# ==================================
# PREDICTION ENDPOINT
# ==================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)

def predict(

    data: CustomerData,

    threshold: float = Query(
        0.60,
        ge=0.0,
        le=1.0,
        description="Decision threshold"
    )

):

    """
    Predict customer churn probability,
    priority level,
    and explainability insights.
    """

    start_time = time.time()

    try:

        # ==========================
        # INPUT PAYLOAD
        # ==========================

        payload = data.model_dump()

        logger.info(
            f"Prediction request received: {payload}"
        )

        # ==========================
        # RUN PREDICTION
        # ==========================

        result = predict_churn(
            payload,
            threshold
        )

        # ==========================
        # LATENCY
        # ==========================

        latency = round(
            time.time() - start_time,
            4
        )

        logger.info(
            f"Prediction completed "
            f"| Probability={result['probability']:.4f}"
        )

        # ==========================
        # RESPONSE
        # ==========================

        return {

            "status": "success",

            "timestamp":
            datetime.utcnow().isoformat(),

            "latency_seconds":
            latency,

            "data":
            result
        }

    except ValueError as e:

        logger.error(
            f"Validation Error: {e}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        logger.error(
            f"Internal Server Error: {e}"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Internal Server Error. "
                "Prediction failed."
            )
        )