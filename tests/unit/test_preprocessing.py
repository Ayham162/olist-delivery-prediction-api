"""Every expected value here was verified independently before being hardcoded
(pd.Timestamp().day_name(), plain timedelta subtraction, a known city-pair
distance) — not eyeballed from the function's own output."""

import pandas as pd

from inference_service.preprocessing import add_derived_features, haversine_km


def test_derived_features_same_state():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [pd.Timestamp("2018-05-10 14:22:00")],
            "order_estimated_delivery_date": [pd.Timestamp("2018-05-25 00:00:00")],
            "customer_state": ["SP"],
            "seller_state": ["SP"],
        }
    )
    result = add_derived_features(df)
    assert result.loc[0, "purchase_dow"] == "Thursday"
    assert result.loc[0, "purchase_month"] == 5
    assert result.loc[0, "promised_days"] == 14
    assert result.loc[0, "same_state"] == 1


def test_derived_features_different_state():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [pd.Timestamp("2018-01-01")],
            "order_estimated_delivery_date": [pd.Timestamp("2018-01-10")],
            "customer_state": ["SP"],
            "seller_state": ["RJ"],
        }
    )
    result = add_derived_features(df)
    assert result.loc[0, "promised_days"] == 9
    assert result.loc[0, "same_state"] == 0


def test_haversine_km_known_city_pair():
    # Sao Paulo <-> Rio de Janeiro city centers, real-world distance ~360km
    sp_lat, sp_lng = -23.5505, -46.6333
    rj_lat, rj_lng = -22.9068, -43.1729
    dist = haversine_km(sp_lat, sp_lng, rj_lat, rj_lng)
    assert 350 < dist < 370


def test_haversine_km_same_point_is_zero():
    assert haversine_km(-23.5, -46.6, -23.5, -46.6) == 0
