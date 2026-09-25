"""Great Expectations suite for incoming orders, deliberately scoped to what
pydantic's type system can't express well: allowed categorical values (real
Brazilian state codes, known payment-method types) and a tolerated missing
rate on a batch request (pydantic's required-field check is strict per-row,
not "at most 5% of this batch may be missing X"). Structural/range checks and
strict per-row null rejection are schemas.OrderInput's job instead, and are
not duplicated here.

On failure: reject (raise DataValidationError -> the API returns 422 with the
failed-expectation detail), not flag-and-continue or silently default. A
wrong guess at customer_state shouldn't produce a confident-looking
prediction on bad data."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import great_expectations as gx
import pandas as pd
from great_expectations.expectations import (
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToNotBeNull,
)

from inference_service.logger import get_logger

logger = get_logger(__name__)

# ISO 3166-2:BR state codes: the real-world set of valid codes, not just the
# subset that happened to appear in training data. A first-ever order from a
# seller in a state absent from training is still a legitimate order; the
# fitted OneHotEncoder already handles an unseen *category* gracefully
# (handle_unknown="ignore"). This check is a different question: "is this a
# real code at all," e.g. catching a typo'd "ZZ".
BRAZIL_STATE_CODES = [
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
]

# Closed vocabulary defined by the payment platform itself. Unlike state
# codes, there's no legitimate value outside this set (from feature_list.json).
PAYMENT_TYPES = ["boleto", "credit_card", "debit_card", "voucher", "unknown"]


class DataValidationError(Exception):
    """Raised when an incoming order fails the expectation suite. The API
    layer catches this and returns 422 with the failure detail."""

    def __init__(self, failures: List[Dict]):
        self.failures = failures
        super().__init__(f"{len(failures)} expectation(s) failed: {failures}")


@lru_cache(maxsize=1)
def _get_context():
    """gx.get_context(mode="ephemeral") isn't a process-wide singleton. Each
    call makes a new isolated in-memory context, so a datasource registered
    on one context is invisible to another; that breaks batch.validate() when
    the batch definition and the suite come from two separately-created
    contexts. Cache the context itself, once, and derive everything else
    from this same instance."""
    return gx.get_context(mode="ephemeral")


@lru_cache(maxsize=1)
def _get_batch_definition():
    """Reused across requests. get_batch(batch_parameters=...) binds fresh
    data each call, so it never sticks to the first DataFrame."""
    context = _get_context()
    data_source = context.data_sources.add_pandas("pandas")
    asset = data_source.add_dataframe_asset(name="orders")
    return asset.add_batch_definition_whole_dataframe("batch")


# Tolerated missing rate on a batch (predict/batch) request. On a single-row
# /predict request this is equivalent to "must be present" (pydantic already
# guarantees that at the API boundary anyway), but on a real batch it allows
# up to 5% missing without rejecting the whole batch outright. Scoped to the
# same 3 columns GE already governs above, not pydantic's required fields:
# a null there can never reach here through the normal API path (FastAPI's
# automatic 422 catches it first), so re-checking non-nullness on those would
# be pure duplication, not an added guarantee.
MISSING_RATE_MOSTLY = 0.95


@lru_cache(maxsize=1)
def _get_suite():
    context = _get_context()
    suite = context.suites.add(gx.ExpectationSuite(name="order_input_suite"))
    for column, value_set in (
        ("customer_state", BRAZIL_STATE_CODES),
        ("seller_state", BRAZIL_STATE_CODES),
        ("main_payment_type", PAYMENT_TYPES),
    ):
        suite.add_expectation(
            ExpectColumnValuesToNotBeNull(column=column, mostly=MISSING_RATE_MOSTLY)
        )
        suite.add_expectation(
            ExpectColumnValuesToBeInSet(column=column, value_set=value_set)
        )
    return suite


def validate_orders(df: pd.DataFrame) -> None:
    """Raises DataValidationError if any row fails the suite; returns None on
    success. Callers only need pass/fail + failure detail, not raw GX internals."""
    batch = _get_batch_definition().get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(_get_suite())

    if not result.success:
        failures = [
            {
                "expectation": r.expectation_config.type,
                "column": r.expectation_config.kwargs.get("column"),
                "unexpected_values": r.result.get("partial_unexpected_list", []),
            }
            for r in result.results
            if not r.success
        ]
        logger.warning("order validation failed: %s", failures)
        raise DataValidationError(failures)
