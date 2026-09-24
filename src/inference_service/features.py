"""Assembles raw orders into the exact feature matrix model.joblib expects,
using the already-fitted preprocessor.joblib (.transform only — never
.fit/.fit_transform at inference, that would silently invalidate the encoders
the model was trained against)."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List

import joblib
import pandas as pd

from inference_service.config import get_config, resolve_path
from inference_service.preprocessing import add_derived_features, add_distance

# Same order/names as pipeline.ipynb cell 17 — this is what the fitted
# ColumnTransformer was fit on, so it's what it must be given at inference.
NUMERIC_FEATURES = [
    "n_items", "n_distinct_products", "n_distinct_sellers", "total_price",
    "total_freight_value", "avg_item_price", "total_weight_g",
    "max_installments", "n_payment_methods", "promised_days", "same_state",
    "distance_km", "purchase_month",
]
CATEGORICAL_FEATURES = [
    "customer_state", "seller_state", "main_payment_type", "purchase_dow",
]
RAW_FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@lru_cache(maxsize=1)
def get_preprocessor():
    cfg = get_config()
    return joblib.load(resolve_path(cfg.model.preprocessor_path))


@lru_cache(maxsize=1)
def get_feature_list() -> List[str]:
    cfg = get_config()
    with open(resolve_path(cfg.model.feature_list_path), encoding="utf-8") as f:
        return json.load(f)


def orders_to_frame(orders: List[Dict]) -> pd.DataFrame:
    """Raw order dicts (already pydantic-validated) -> a DataFrame with parsed
    timestamps, ready for the Stage-4 derivations."""
    df = pd.DataFrame(orders)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    return df


def derive_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Already-framed raw orders (see orders_to_frame) -> the numeric matrix
    the model expects, column-for-column identical to feature_list.json.
    Reproduces pipeline.ipynb Stage 4 end to end (add_derived_features ->
    add_distance -> preprocessor.transform). Split out from build_feature_matrix
    so callers can validate the raw df (validation.py) in between parsing and
    transforming — GE needs the raw fields, not the derived/encoded ones."""
    df = add_derived_features(df)
    df = add_distance(df)

    preprocessor = get_preprocessor()
    X = preprocessor.transform(df[RAW_FEATURE_COLS])
    X_df = pd.DataFrame(X, columns=preprocessor.get_feature_names_out(), index=df.index)

    return X_df[get_feature_list()]


def build_feature_matrix(orders: List[Dict]) -> pd.DataFrame:
    """Convenience wrapper: raw order dicts straight to the feature matrix,
    with no validation gate in between (use predict.predict_batch for the
    full validated path)."""
    return derive_and_transform(orders_to_frame(orders))
