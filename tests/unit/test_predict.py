from inference_service.predict import get_model, predict_one
from inference_service.schemas import OrderInput

from ..conftest import CANONICAL_ORDER_EXPECTED_PROBABILITY


def test_model_loads():
    model = get_model()
    assert hasattr(model, "predict_proba")


def test_predict_one_shape_and_types(canonical_order):
    result = predict_one(OrderInput(**canonical_order).model_dump())

    assert set(result.keys()) == {"late", "probability", "model_version"}
    assert isinstance(result["late"], bool)
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["model_version"], str)


def test_predict_one_matches_verified_notebook_output(canonical_order):
    """Pipeline output matches notebook output, as a permanent automated
    assertion instead of a one-off script. The expected value matches a
    from-scratch reconstruction of pipeline.ipynb's own transform, max abs
    diff 0.0."""
    result = predict_one(OrderInput(**canonical_order).model_dump())
    assert abs(result["probability"] - CANONICAL_ORDER_EXPECTED_PROBABILITY) < 1e-9
