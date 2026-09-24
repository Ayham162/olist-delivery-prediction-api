"""End-to-end through the actual HTTP layer (FastAPI's TestClient — no real
socket, but the full app: routing, pydantic validation, exception handlers).
Deferred from §6 until app/main.py existed; every case here was first
verified manually against a live `uvicorn` process before being written
down, including both 422 paths."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "logistic_regression"
    assert body["threshold"] == 0.71
    assert body["feature_count"] > 0


def test_predict_valid_order(canonical_order):
    response = client.post("/predict", json=_json_safe(canonical_order))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"late", "probability", "model_version"}
    assert isinstance(body["late"], bool)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_missing_fields_returns_422_not_500():
    response = client.post("/predict", json={"n_items": 1})
    assert response.status_code == 422
    # FastAPI's own pydantic-error shape, not our DataValidationError shape
    assert isinstance(response.json()["detail"], list)


def test_predict_bad_state_code_returns_422_with_failure_detail(canonical_order):
    bad_order = _json_safe(dict(canonical_order, customer_state="ZZ"))
    response = client.post("/predict", json=bad_order)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "data validation failed"
    assert any(f["column"] == "customer_state" for f in body["failures"])


def test_predict_batch(canonical_order):
    response = client.post("/predict/batch", json={"orders": [_json_safe(canonical_order), _json_safe(canonical_order)]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2


def _json_safe(order: dict) -> dict:
    """canonical_order's timestamps are already ISO strings, but datetime
    objects (if a test builds one directly) don't survive json= as-is."""
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in order.items()}
