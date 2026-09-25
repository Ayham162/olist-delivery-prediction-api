import pandas as pd
import pytest

from inference_service.validation import DataValidationError, validate_orders


def test_valid_order_passes():
    df = pd.DataFrame([{
        "customer_state": "SP", "seller_state": "RJ", "main_payment_type": "credit_card",
    }])
    validate_orders(df)  # should not raise


def test_invalid_state_code_rejected():
    df = pd.DataFrame([{
        "customer_state": "ZZ", "seller_state": "RJ", "main_payment_type": "credit_card",
    }])
    with pytest.raises(DataValidationError) as exc_info:
        validate_orders(df)

    failures = exc_info.value.failures
    assert any(f["column"] == "customer_state" and "ZZ" in f["unexpected_values"] for f in failures)


def test_invalid_payment_type_rejected():
    df = pd.DataFrame([{
        "customer_state": "SP", "seller_state": "RJ", "main_payment_type": "bitcoin",
    }])
    with pytest.raises(DataValidationError):
        validate_orders(df)
