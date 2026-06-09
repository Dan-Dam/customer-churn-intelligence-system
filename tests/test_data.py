import os
import pandas as pd

LOG_FILE = "logs/predictions_log.csv"

def test_log_file_exists():
    assert os.path.exists(LOG_FILE)


def test_log_structure():
    df = pd.read_csv(LOG_FILE)

    required_columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "Tenure",
        "probability"
    ]

    for col in required_columns:
        assert col in df.columns