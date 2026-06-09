# ===============================
# 🤖 PREDICTION ENGINE
# Enterprise Production Version
# ===============================

import os
import logging
import warnings

from typing import Dict, Any

import joblib
import numpy as np
import shap
import pandas as pd

from src.features import prepare_features

from src.utils import (
    FEATURE_COLUMNS,
    validate_input,
    assign_priority
)

warnings.filterwarnings("ignore")

# ==================================
# LOGGING
# ==================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ==================================
# BASE PATHS
# ==================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

# ==================================
# VALIDATE ARTIFACTS
# ==================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.exists(SCALER_PATH):

    raise FileNotFoundError(
        f"Scaler file not found: {SCALER_PATH}"
    )

# ==================================
# LOAD ARTIFACTS
# ==================================

logger.info("Loading model artifacts...")

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

logger.info("Model artifacts loaded successfully.")

# ==================================
# SHAP EXPLAINER
# ==================================

explainer = None

try:

    explainer = shap.TreeExplainer(model)

    logger.info(
        "SHAP TreeExplainer initialized."
    )

except Exception as e:

    logger.warning(
        f"TreeExplainer failed: {e}"
    )

    try:

        explainer = shap.Explainer(model)

        logger.info(
            "Generic SHAP Explainer initialized."
        )

    except Exception as ex:

        logger.error(
            f"SHAP initialization failed: {ex}"
        )

        explainer = None

# ==================================
# PREDICTION FUNCTION
# ==================================

def predict_churn(
    data: Dict[str, Any],
    threshold: float = 0.45
) -> Dict[str, Any]:

    """
    Predict customer churn probability
    using hybrid ML + business logic.
    """

    try:

        # ==========================
        # COPY INPUT (SAFE)
        # ==========================

        payload = data.copy()

        # ==========================
        # FEATURE ENGINEERING
        # ==========================

        input_df = pd.DataFrame([payload])

        feature_df = prepare_features(
            input_df
        )

        # Add engineered features back to payload
        # for validation and business rules

        payload.update(
            feature_df.iloc[0].to_dict()
        )

        # ==========================
        # VALIDATE INPUT
        # ==========================

        validate_input(payload)

        # ==========================
        # FEATURE ORDER
        # ==========================

        feature_vector = [

            float(payload[col])

            for col in FEATURE_COLUMNS
        ]

        features = np.array(
            [feature_vector],
            dtype=np.float64
        )

        # ==========================
        # NUMERIC SAFETY
        # ==========================

        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ==========================
        # SCALE FEATURES
        # ==========================

        features_scaled = (
            scaler.transform(features)
        )

        # ==========================
        # MODEL PREDICTION
        # ==========================

        model_prob = float(

            model.predict_proba(
                features_scaled
            )[0][1]
        )

        prob = model_prob

        # ==========================
        # BUSINESS RULES
        # ==========================

        business_adjustment = False

        if (
            payload["Recency"] > 180
            and
            payload["Frequency"] <= 2
        ):

            prob = max(prob, 0.75)

            business_adjustment = True

        if (
            payload["Frequency"] == 0
            and
            payload["Monetary"] == 0
        ):

            prob = max(prob, 0.85)

            business_adjustment = True

        if (
            payload["Tenure"] < 30
            and
            payload["Recency"] > 20
        ):

            prob = max(prob, 0.65)

            business_adjustment = True

        # ==========================
        # PROBABILITY SAFETY
        # ==========================

        prob = float(
            np.clip(prob, 0.0, 1.0)
        )

        # ==========================
        # DECISION LAYER
        # ==========================

        prediction = int(
            prob > threshold
        )

        priority = assign_priority(prob)

        # ==========================
        # SHAP EXPLAINABILITY
        # ==========================

        explanations = {}

        base_value = None

        if explainer is not None:

            try:

                shap_output = explainer(
                    features_scaled
                )

                # ----------------------
                # SHAP Values
                # ----------------------

                if hasattr(
                    shap_output,
                    "values"
                ):

                    shap_values = (
                        shap_output.values
                    )

                else:

                    shap_values = shap_output

                # Binary handling
                if (
                    isinstance(
                        shap_values,
                        list
                    )
                ):

                    shap_values = (
                        shap_values[1]
                    )

                shap_values = (
                    shap_values[0]
                )

                explanations = {

                    col: float(val)

                    for col, val in zip(
                        FEATURE_COLUMNS,
                        shap_values
                    )
                }

                # ----------------------
                # Base Value
                # ----------------------

                if hasattr(
                    shap_output,
                    "base_values"
                ):

                    base_value = float(
                        np.array(
                            shap_output.base_values
                        ).flatten()[0]
                    )

                elif hasattr(
                    explainer,
                    "expected_value"
                ):

                    expected = (
                        explainer.expected_value
                    )

                    if isinstance(
                        expected,
                        (
                            list,
                            np.ndarray
                        )
                    ):

                        base_value = float(
                            np.array(
                                expected
                            ).flatten()[-1]
                        )

                    else:

                        base_value = float(
                            expected
                        )

            except Exception as e:

                logger.error(
                    f"SHAP Error: {e}"
                )

                explanations = {}

                base_value = None

        # ==========================
        # RESPONSE
        # ==========================

        response = {

            "probability": prob,

            "model_probability":
            model_prob,

            "prediction":
            prediction,

            "priority":
            priority,

            "threshold":
            threshold,

            "business_adjustment":
            business_adjustment,

            "explanations":
            explanations,

            "base_value":
            base_value,

            "features":
            FEATURE_COLUMNS,

            "model_note":
            (
                "Hybrid ML + "
                "Business Rules + "
                "SHAP Explainability"
            )
        }

        logger.info(
            f"Prediction generated "
            f"| Probability={prob:.4f}"
        )

        return response

    except Exception as e:

        logger.error(
            f"Prediction Engine Error: {e}"
        )

        raise RuntimeError(
            f"Prediction failed: {e}"
        )