"""Schema/quality checks on the training-time data itself — distinct from
tests/unit (which tests our code) and validation.py (which gates live
requests). This is about the data pipeline.ipynb actually produced."""
import pandas as pd

from inference_service.config import PROJECT_ROOT
from inference_service.features import RAW_FEATURE_COLS

TEST_CSV = PROJECT_ROOT / "notebooks" / "artifacts" / "03_test.csv"

# columns that are only known after the fact (delivery outcome) or are the
# target itself — must never end up in the feature set the model consumes,
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
        "n_items", "total_price", "customer_state", "seller_state",
        "order_purchase_timestamp", "order_estimated_delivery_date",
    ]
    for col in required:
        assert col in df.columns, f"missing expected column: {col}"


def test_no_leakage_columns_in_model_input():
    assert LEAKAGE_COLUMNS.isdisjoint(set(RAW_FEATURE_COLS))
