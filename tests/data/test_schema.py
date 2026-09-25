"""Schema/quality checks on the training-time data itself. Distinct from
tests/unit (which tests the service's own code) and validation.py (which
gates live requests). This is about the data pipeline.ipynb actually
produced."""

import pandas as pd

from inference_service.config import PROJECT_ROOT
from inference_service.features import RAW_FEATURE_COLS

TEST_CSV = PROJECT_ROOT / "notebooks" / "artifacts" / "03_test.csv"

# Columns that are only known after the fact (delivery outcome) or are the
# target itself. Must never end up in the feature set the model consumes,
# or the model would be trained/served on information it can't have at
# prediction time (Stage 4 explicitly drops these; this test guards that).
LEAKAGE_COLUMNS = {
    "late",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_approved_at",
    "order_estimated_delivery_date",
}


def test_test_csv_has_expected_columns_and_label():
    df = pd.read_csv(TEST_CSV)
    assert "late" in df.columns
    assert df["late"].dropna().isin([0, 1]).all()

    required = [
        "n_items",
        "total_price",
        "customer_state",
        "seller_state",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    ]
    for col in required:
        assert col in df.columns, f"missing expected column: {col}"


def test_no_leakage_columns_in_model_input():
    assert LEAKAGE_COLUMNS.isdisjoint(set(RAW_FEATURE_COLS))


def test_no_unexpected_nulls_in_required_columns():
    """Every one of these columns has a 0% null rate in 03_test.csv. This
    locks that in so a future pipeline.ipynb re-run can't silently introduce
    missing values Stage 4 never accounted for."""
    df = pd.read_csv(TEST_CSV)
    required_non_null = [
        "n_items",
        "total_price",
        "total_freight_value",
        "customer_state",
        "seller_state",
        "main_payment_type",
        "max_installments",
        "n_payment_methods",
    ]
    for col in required_non_null:
        assert df[col].isna().mean() == 0, f"unexpected nulls in {col}"


def test_numeric_columns_within_business_rule_ranges():
    """Bounds are business rules (an order can't have zero items, a price
    can't be negative), not just "whatever the current data happens to
    show." They still catch a real data-quality regression even if the
    observed values change on a future pipeline.ipynb re-run."""
    df = pd.read_csv(TEST_CSV)
    assert (df["n_items"] >= 1).all()
    assert (df["max_installments"] >= 1).all()
    assert (df["n_payment_methods"] >= 1).all()
    assert (df["total_price"] >= 0).all()
    assert (df["total_freight_value"] >= 0).all()
    assert (df["total_weight_g"] >= 0).all()
