from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

def test_health_check():
	response = client.get("/")
	assert response.status_code == 200
	assert "status" in response.json()

def test_prediction_endpoint():
    payload = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_REGISTRATION": -4000,
        "DAYS_ID_PUBLISH": -1500,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.7
    }
    response = client.post("/predict?threshold=0.35", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "decision" in data
    assert data["decision"] in ["accepted", "rejected"]