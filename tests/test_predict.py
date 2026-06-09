from src.predict import predict_churn

def test_prediction_runs():
    data = {
        "Recency": 10,
        "Frequency": 5,
        "Monetary": 500,
        "Tenure": 50
    }

    result = predict_churn(data)

    assert "probability" in result
    assert 0 <= result["probability"] <= 1


def test_high_risk_case():
    data = {
        "Recency": 200,
        "Frequency": 1,
        "Monetary": 50,
        "Tenure": 300
    }

    result = predict_churn(data)

    assert result["probability"] > 0.5


def test_low_risk_case():
    data = {
        "Recency": 5,
        "Frequency": 50,
        "Monetary": 2000,
        "Tenure": 365
    }

    result = predict_churn(data)

    assert result["probability"] < 0.5


def test_zero_input():
    data = {
        "Recency": 0,
        "Frequency": 0,
        "Monetary": 0,
        "Tenure": 0
    }

    result = predict_churn(data)

    assert result["probability"] >= 0