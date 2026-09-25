"""Feature derivations mirror pipeline.ipynb Stage 4 exactly: same column
names, same formulas, same zip_geoloc join. A live request then produces
the same numbers the notebook produced for the same raw order.
Any change here must be made in the notebook too, or the two silently drift."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from inference_service.config import get_config, resolve_path


@lru_cache(maxsize=1)
def get_zip_geoloc() -> pd.DataFrame:
    """Zip-code prefix -> mean lat/lng lookup (pipeline.ipynb cell 7's output).
    Loaded once per process and cached, not once per request."""
    cfg = get_config()
    return pd.read_csv(resolve_path(cfg.data.zip_geoloc_path))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors pipeline.ipynb cell 14 (add_derived_features)."""
    df = df.copy()
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.day_name()
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["promised_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lng points, in km.
    Mirrors pipeline.ipynb cell 15 (haversine_km)."""
    r = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors pipeline.ipynb cell 15 (add_distance): join each side's zip
    prefix to its lat/lng via the same lookup used at training time, then
    haversine between them. A zip prefix absent from zip_geoloc.csv (never
    seen in the training data) leaves distance_km as NaN here. The fitted
    preprocessor's median imputer, the same one used at training time,
    handles that downstream, so no separate fallback is needed here."""
    zip_geoloc = get_zip_geoloc()
    df = df.merge(
        zip_geoloc.rename(
            columns={
                "zip_code_prefix": "customer_zip_code_prefix",
                "lat": "customer_lat",
                "lng": "customer_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )
    df = df.merge(
        zip_geoloc.rename(
            columns={
                "zip_code_prefix": "seller_zip_code_prefix",
                "lat": "seller_lat",
                "lng": "seller_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )
    df["distance_km"] = haversine_km(
        df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"]
    )
    return df.drop(columns=["customer_lat", "customer_lng", "seller_lat", "seller_lng"])
