"""Request/response contracts for the service. Structural and range validation
lives here (pydantic, checked at the API boundary before anything touches the
model). Category-membership and data-quality checks (allowed state codes,
missing-rate thresholds) belong to the Great Expectations suite — deliberately
not duplicated here, see TASK3_CHECKLIST.md step 4."""
from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    """One new order, as it looks before delivery. These are exactly the raw
    fields pipeline.ipynb Stage 1 aggregates from Postgres for an existing
    order — a brand-new order has no DB row yet, so the caller supplies them
    directly (see README's "What the inference service actually needs")."""

    n_items: int = Field(ge=1, description="line items on the order")
    n_distinct_products: int = Field(ge=1)
    n_distinct_sellers: int = Field(ge=1)
    total_price: float = Field(ge=0)
    total_freight_value: float = Field(ge=0)
    avg_item_price: float = Field(ge=0)
    total_weight_g: float = Field(ge=0)
    max_installments: int = Field(ge=1)
    n_payment_methods: int = Field(ge=1)
    main_payment_type: str = Field(min_length=1)
    customer_state: str = Field(min_length=2, max_length=2, description="2-letter Brazilian state code")
    customer_zip_code_prefix: int = Field(ge=0)
    seller_state: str = Field(min_length=2, max_length=2)
    seller_zip_code_prefix: int = Field(ge=0)
    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime


class PredictionResponse(BaseModel):
    late: bool = Field(description="predicted late-delivery outcome")
    probability: float = Field(ge=0, le=1, description="P(late)")
    model_version: str


class BatchPredictionRequest(BaseModel):
    orders: List[OrderInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
