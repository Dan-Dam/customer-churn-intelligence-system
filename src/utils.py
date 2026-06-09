# ===============================
# ⚙️ Utility Functions & Schema
# Enterprise Production Version
# ===============================

import math
import logging

from typing import Dict, Any

# ==================================
# LOGGING
# ==================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ==================================
# FEATURE SCHEMA
# ==================================

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure",
    "AOV",
    "Purchase_Frequency",
    "Profit"
]

# ==================================
# INPUT VALIDATION
# ==================================

def validate_input(
    data: Dict[str, Any]
) -> bool:
    """
    Validate input payload
    for churn prediction.
    """

    try:

        # ==========================
        # CHECK MISSING FIELDS
        # ==========================

        missing_fields = [

            field

            for field in FEATURE_COLUMNS

            if field not in data
        ]

        if missing_fields:

            raise ValueError(
                f"Missing fields: {missing_fields}"
            )

        # ==========================
        # VALIDATE NUMERIC VALUES
        # ==========================

        for field in FEATURE_COLUMNS:

            value = data[field]

            # ----------------------
            # Type Conversion
            # ----------------------

            try:

                numeric_value = float(value)

            except Exception:

                raise TypeError(
                    f"{field} must be numeric."
                )

            # ----------------------
            # NaN Check
            # ----------------------

            if math.isnan(
                numeric_value
            ):

                raise ValueError(
                    f"{field} contains NaN."
                )

            # ----------------------
            # Infinite Check
            # ----------------------

            if math.isinf(
                numeric_value
            ):

                raise ValueError(
                    f"{field} contains infinite value."
                )

            # ----------------------
            # Negative Check
            # ----------------------

            if numeric_value < 0:

                raise ValueError(
                    f"{field} cannot be negative."
                )

        logger.info(
            "Input validation successful."
        )

        return True

    except Exception as e:

        logger.error(
            f"Validation Error: {e}"
        )

        raise

# ==================================
# PRIORITY ENGINE
# ==================================

def assign_priority(
    prob: float
) -> str:
    """
    Strategic business priority
    classification based on
    churn probability.
    """

    try:

        # ==========================
        # SAFETY CLIPPING
        # ==========================

        prob = max(
            0.0,
            min(
                float(prob),
                1.0
            )
        )

        # ==========================
        # PRIORITY LOGIC
        # ==========================

        if prob >= 0.85:

            return "🔥 Critical"

        elif prob >= 0.70:

            return "⚠️ High"

        elif prob >= 0.50:

            return "⚡ Medium"

        else:

            return "✅ Low"

    except Exception as e:

        logger.error(
            f"Priority Assignment Error: {e}"
        )

        return "Unknown"

# ==================================
# FEATURE VECTOR CHECK
# ==================================

def validate_feature_order(
    feature_list: list
) -> bool:
    """
    Ensure feature ordering matches
    training schema exactly.
    """

    if feature_list != FEATURE_COLUMNS:

        raise ValueError(
            "Feature ordering mismatch detected."
        )

    return True