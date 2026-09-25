"""Loads model.joblib + the tuned threshold once, and turns raw orders into
predictions. The only module that touches the model object; never re-fits."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Dict, List

import joblib

from inference_service.config import get_config, resolve_path
from inference_service.features import derive_and_transform, orders_to_frame
from inference_service.logger import get_logger
from inference_service.validation import validate_orders

logger = get_logger(__name__)

# MLflow's own defaults (120s timeout, 7 retries with exponential backoff) mean
# an unreachable tracking server hangs the service for several MINUTES before
# the joblib fallback ever kicks in. An unreachable server took over 90s
# across 5 retries before this was added. The fallback only helps if it
# triggers fast, so setdefault lets a real deployment still override these
# via the environment. Needs to run before mlflow itself is imported (a
# local import inside _load_model_and_source, not at module level), so it's
# placed here, after this module's own imports, rather than before them.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")


@lru_cache(maxsize=1)
def _load_model_and_source():
    """Tries the MLflow registry first (models:/<name>@<alias>, an alias,
    not a "stage": stages are deprecated as of MLflow 2.9+). Falls back to
    the local joblib path if the tracking server is unreachable, the
    registered model/alias doesn't exist yet, or MLFLOW_TRACKING_URI was
    never set (config.yaml's ${MLFLOW_TRACKING_URI} then stays a literal
    unexpanded placeholder, treated the same as "not configured"). This
    fallback is documented, not silent: it logs at WARNING so it's visible
    which source actually served a model."""
    cfg = get_config()
    registry_cfg = cfg.model.registry
    tracking_uri = getattr(registry_cfg, "tracking_uri", None)

    if tracking_uri and "$" not in str(tracking_uri):
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            uri = f"models:/{registry_cfg.name}@{registry_cfg.alias}"
            model = mlflow.sklearn.load_model(uri)
            logger.info("loaded model from MLflow registry: %s", uri)
            return model, f"mlflow:{registry_cfg.name}@{registry_cfg.alias}"
        except Exception as exc:
            logger.warning(
                "MLflow registry load failed (%s), falling back to local joblib path",
                exc,
            )

    model = joblib.load(resolve_path(cfg.model.path))
    logger.info("loaded model from local joblib path: %s", cfg.model.path)
    return model, "joblib:local"


def get_model():
    return _load_model_and_source()[0]


def get_model_source() -> str:
    """'mlflow:<name>@<alias>' or 'joblib:local': which of the two actually
    served the loaded model. Surfaced on GET /model/info, not stuffed into
    every prediction's model_version, to keep that field a stable,
    predictable shape."""
    return _load_model_and_source()[1]


@lru_cache(maxsize=1)
def get_run_config() -> Dict:
    cfg = get_config()
    with open(resolve_path(cfg.model.run_config_path), encoding="utf-8") as f:
        return json.load(f)


def model_version() -> str:
    rc = get_run_config()
    return f"{rc['model_type']}@threshold={rc['threshold']}"


def predict_batch(orders: List[Dict]) -> List[Dict]:
    """orders: raw order dicts, already pydantic-validated by schemas.OrderInput
    for structure/types/ranges. Raises validation.DataValidationError if any
    order fails the Great Expectations suite (category-membership checks
    pydantic can't express). That's a second, independent gate, not a
    duplicate of pydantic's job. Returns one {late, probability, model_version}
    dict per order, same order as the input. Logs one line per order (input,
    output, model version: the per-order request/response pair the task asks
    for) plus one aggregate line for the call (count, total latency, positive
    rate), since per-order latency isn't meaningful inside a single vectorized
    batch transform/predict_proba call. The full input logged here also goes
    into the DB's prediction_logs table: that's for querying later against
    ground truth, this is for tracing a specific request now."""
    start = time.perf_counter()

    df = orders_to_frame(orders)
    validate_orders(df)
    X = derive_and_transform(df)
    model = get_model()
    threshold = get_run_config()["threshold"]

    proba = model.predict_proba(X)[:, 1]
    late = proba >= threshold
    version = model_version()

    results = [
        {"late": bool(is_late), "probability": float(p), "model_version": version}
        for is_late, p in zip(late, proba)
    ]

    for order, result in zip(orders, results):
        logger.info("prediction | input=%s | output=%s", order, result)

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "predicted %d order(s) in %.1fms | positive_rate=%.3f | model=%s",
        len(orders),
        latency_ms,
        float(late.mean()),
        version,
    )

    return results


def predict_one(order: Dict) -> Dict:
    return predict_batch([order])[0]
