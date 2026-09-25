import pandas as pd
import pytest

from inference_service.validation import DataValidationError, validate_orders


def test_valid_order_passes():
    df = pd.DataFrame(
        [
            {
                "customer_state": "SP",
                "seller_state": "RJ",
                "main_payment_type": "credit_card",
            }
        ]
    )
    validate_orders(df)  # should not raise


def test_invalid_state_code_rejected():
    df = pd.DataFrame(
        [
            {
                "customer_state": "ZZ",
                "seller_state": "RJ",
                "main_payment_type": "credit_card",
            }
        ]
    )
    with pytest.raises(DataValidationError) as exc_info:
        validate_orders(df)

    failures = exc_info.value.failures
    assert any(
        f["column"] == "customer_state" and "ZZ" in f["unexpected_values"]
        for f in failures
    )


def test_invalid_payment_type_rejected():
    df = pd.DataFrame(
        [
            {
                "customer_state": "SP",
                "seller_state": "RJ",
                "main_payment_type": "bitcoin",
            }
        ]
    )
    with pytest.raises(DataValidationError):
        validate_orders(df)


def test_batch_with_too_many_missing_values_rejected():
    """10% null in customer_state on a 10-row batch, against the 5%-tolerance
    missing-rate expectation (MISSING_RATE_MOSTLY=0.95) — must reject the
    whole batch, not silently let the nulls through."""
    rows = [
        {
            "customer_state": "SP",
            "seller_state": "RJ",
            "main_payment_type": "credit_card",
        }
        for _ in range(9)
    ]
    rows.append(
        {
            "customer_state": None,
            "seller_state": "RJ",
            "main_payment_type": "credit_card",
        }
    )
    df = pd.DataFrame(rows)

    with pytest.raises(DataValidationError) as exc_info:
        validate_orders(df)

    failures = exc_info.value.failures
    assert any(
        f["expectation"] == "expect_column_values_to_not_be_null"
        and f["column"] == "customer_state"
        for f in failures
    )


def test_batch_within_missing_rate_tolerance_passes():
    """1 null out of 20 rows (5% missing) sits right at the tolerance edge -
    should still pass, not reject on the first hint of any missing data."""
    rows = [
        {
            "customer_state": "SP",
            "seller_state": "RJ",
            "main_payment_type": "credit_card",
        }
        for _ in range(19)
    ]
    rows.append(
        {
            "customer_state": None,
            "seller_state": "RJ",
            "main_payment_type": "credit_card",
        }
    )
    df = pd.DataFrame(rows)

    validate_orders(df)  # should not raise
