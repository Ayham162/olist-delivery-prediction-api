"""One-off maintenance script: registers the already-trained model.joblib
(training stays in pipeline.ipynb for this task, per the brief) into MLflow's
tracking store and model registry. Run this once whenever a new model.joblib
is chosen; the service (predict.py) loads whatever this script last pointed
the "production" alias at.

Usage: python scripts/register_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402 - must follow the sys.path.insert above

import joblib  # noqa: E402
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from inference_service.config import get_config, resolve_path  # noqa: E402

EXPERIMENT_NAME = "olist-late-delivery"
REGISTERED_MODEL_NAME = "olist-late-delivery"
ALIAS = "production"


def main() -> None:
    cfg = get_config()
    if getattr(cfg.model.registry, "tracking_uri", None):
        mlflow.set_tracking_uri(cfg.model.registry.tracking_uri)

    model = joblib.load(resolve_path(cfg.model.path))
    with open(resolve_path(cfg.model.run_config_path), encoding="utf-8") as f:
        run_config = json.load(f)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"register-{run_config['model_type']}") as run:
        mlflow.log_params(run_config["params"])
        mlflow.log_params(run_config.get("fixed_hyperparameters", {}))
        mlflow.log_param("threshold", run_config["threshold"])
        mlflow.log_metrics(run_config.get("notebook_test_metrics", {}))

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",  # matches joblib/pickle, not skops
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    client = MlflowClient()
    # the just-registered version is the latest for this name
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, latest.version)

    print(f"run_id={run_id}")
    print(
        f"registered {REGISTERED_MODEL_NAME} version {latest.version}, "
        f"alias '{ALIAS}' -> models:/{REGISTERED_MODEL_NAME}@{ALIAS}"
    )


if __name__ == "__main__":
    main()
