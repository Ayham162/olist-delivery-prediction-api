"""Shared fixtures. CANONICAL_ORDER is the exact order used throughout manual
verification during development (predict.predict_one gave probability
0.6113450641808283 for it, cross-checked against a from-scratch
reconstruction of pipeline.ipynb's own transform with max abs diff 0.0) —
reused here so that verification is now a permanent, automated test instead
of a one-off scratch script."""

import pytest

CANONICAL_ORDER = dict(
    n_items=1,
    n_distinct_products=1,
    n_distinct_sellers=1,
    total_price=120.5,
    total_freight_value=15.0,
    avg_item_price=120.5,
    total_weight_g=500,
    max_installments=3,
    n_payment_methods=1,
    main_payment_type="credit_card",
    customer_state="SP",
    customer_zip_code_prefix=1310,
    seller_state="RJ",
    seller_zip_code_prefix=20040,
    order_purchase_timestamp="2018-05-10T14:22:00",
    order_estimated_delivery_date="2018-05-25T00:00:00",
)

CANONICAL_ORDER_EXPECTED_PROBABILITY = 0.6113450641808283


@pytest.fixture
def canonical_order():
    return dict(CANONICAL_ORDER)
