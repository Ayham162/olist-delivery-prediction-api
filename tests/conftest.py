"""Shared fixtures. CANONICAL_ORDER is a fixed reference order: predict.predict_one
gives probability 0.6113450641808283 for it, matching a from-scratch
reconstruction of pipeline.ipynb's own transform to a max abs diff of 0.0.
Reused here so that check is a permanent, automated test instead of a
one-off scratch script."""

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
