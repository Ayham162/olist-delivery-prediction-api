"""FastAPI service for the Olist late-delivery model. Thin HTTP layer over
src/inference_service: no business/ML logic lives here, just request and
response handling, exception-to-HTTP-status translation, and startup wiring."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from inference_service import db
from inference_service.logger import get_logger
from inference_service.predict import (
    get_model,
    get_model_source,
    get_run_config,
    model_version,
    predict_batch,
)
from inference_service.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderInput,
    PredictionResponse,
)
from inference_service.validation import DataValidationError

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force model/preprocessor/zip_geoloc load at startup, not lazily on the
    # first request. A broken config/missing artifact should fail fast at
    # boot with a clear log line, not surface as a confusing 500 on whoever
    # happens to send the first request.
    get_model()
    db.ensure_table()  # no-op if DB isn't configured/reachable, see db.py
    logger.info(
        "service started | model_source=%s | model_version=%s",
        get_model_source(),
        model_version(),
    )
    yield


app = FastAPI(
    title="Olist Late-Delivery Inference Service",
    description="Predicts whether a new order will arrive after its estimated "
    "delivery date, with a probability.",
    lifespan=lifespan,
)

# Request count, latency histograms, and status-code/error breakdown at
# GET /metrics, in Prometheus's own format. Not hand-rolled: this library
# already does it correctly (instrument() adds a middleware, expose() adds
# the /metrics route that serves what it collected).
Instrumentator().instrument(app).expose(app)


@app.exception_handler(DataValidationError)
async def data_validation_error_handler(request: Request, exc: DataValidationError):
    """DataValidationError (the Great Expectations gate in validation.py)
    isn't a type FastAPI knows about on its own. Without this handler it
    would surface as an unhandled 500. pydantic field errors (bad
    types/missing fields) don't
    need a handler here: FastAPI already turns those into 422s automatically
    for any route parameter typed as a pydantic model."""
    logger.warning("rejected request at %s: %s", request.url.path, exc.failures)
    return JSONResponse(
        status_code=422,
        content={"detail": "data validation failed", "failures": exc.failures},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Anything else (a genuinely unexpected bug, a missing file, ...) gets
    logged with the full exception here and returns a clean 500 body.
    The client never sees a raw traceback."""
    logger.exception("unhandled error at %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except Exception:
        logger.exception("health check: model failed to load")
        return {"status": "degraded", "model_loaded": False}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    rc = get_run_config()
    return {
        "model_type": rc["model_type"],
        "threshold": rc["threshold"],
        "model_version": model_version(),
        "model_source": get_model_source(),
        "feature_count": len(rc["feature_list"]),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    order_dict = order.model_dump()
    start = time.perf_counter()
    result = predict_batch([order_dict])[0]
    db.log_prediction(order_dict, result, (time.perf_counter() - start) * 1000)
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_route(request: BatchPredictionRequest):
    order_dicts = [o.model_dump() for o in request.orders]
    start = time.perf_counter()
    results = predict_batch(order_dicts)
    latency_ms = (time.perf_counter() - start) * 1000
    for order_dict, result in zip(order_dicts, results):
        db.log_prediction(order_dict, result, latency_ms)
    return {"predictions": results}
