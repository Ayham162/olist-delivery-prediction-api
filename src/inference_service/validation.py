"""Great Expectations suite for incoming orders — deliberately scoped to what
pydantic's type system can't express well: allowed categorical values (real
Brazilian state codes, known payment-method types). Structural/range checks
(required fields present, numeric bounds) are schemas.OrderInput's job and
are not duplicated here — see TASK3_CHECKLIST.md §4 for the split rationale.

On failure: reject (raise DataValidationError -> the API returns 422 with the
failed-expectation detail), not flag-and-continue or silently default. A
wrong guess at customer_state shouldn't produce a confident-looking
prediction on bad data."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import great_expectations as gx
import pandas as pd
from great_expectations.expectations import ExpectColumnValuesToBeInSet

from inference_service.logger import get_logger

logger = get_logger(__name__)

# ISO 3166-2:BR state codes — the real-world set of valid codes, not just the
# subset that happened to appear in training data. A first-ever order from a
# seller in a state absent from training is still a legitimate order; the
# fitted OneHotEncoder already handles an unseen *category* gracefully
# (handle_unknown="ignore"). This check is a different question: "is this a
# real code at all," e.g. catching a typo'd "ZZ".
BRAZIL_STATE_CODES = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
]

# Closed vocabulary defined by the payment platform itself — unlike state
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
    """A fresh gx.get_context(mode="ephemeral") call creates a new, isolated
    in-memory context each time — it is NOT a process-wide singleton. A
    datasource registered on one context is invisible to another, which broke
    batch.validate() when the batch definition and the suite came from two
    separately-created contexts. Cache the context itself, once, and derive
    everything else from this same instance."""
    return gx.get_context(mode="ephemeral")


@lru_cache(maxsize=1)
def _get_batch_definition():
    """Reused across requests — safe: get_batch(batch_parameters=...) binds
    fresh data each call, verified it doesn't stick to the first DataFrame."""
    context = _get_context()
    data_source = context.data_sources.add_pandas("pandas")
    asset = data_source.add_dataframe_asset(name="orders")
    return asset.add_batch_definition_whole_dataframe("batch")


@lru_cache(maxsize=1)
def _get_suite():
    context = _get_context()
    suite = context.suites.add(gx.ExpectationSuite(name="order_input_suite"))
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="customer_state", value_set=BRAZIL_STATE_CODES)
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="seller_state", value_set=BRAZIL_STATE_CODES)
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="main_payment_type", value_set=PAYMENT_TYPES)
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
