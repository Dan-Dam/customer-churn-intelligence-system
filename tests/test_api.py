import requests

API_URL = "http://localhost:8000/predict"

def test_api_prediction():
    payload = {
        "Recency": 10,
        "Frequency": 5,
        "Monetary": 500,
        "Tenure": 50
    }

    response = requests.post(API_URL, json=payload)

    assert response.status_code == 200

    data = response.json()["data"]

    assert "probability" in data