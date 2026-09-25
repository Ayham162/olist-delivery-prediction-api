import json

from inference_service.config import get_config, resolve_path
from inference_service.features import build_feature_matrix


def test_feature_matrix_columns_match_feature_list_json(canonical_order):
    cfg = get_config()
    with open(resolve_path(cfg.model.feature_list_path), encoding="utf-8") as f:
        expected_columns = json.load(f)

    X = build_feature_matrix([canonical_order])

    assert list(X.columns) == expected_columns


def test_feature_matrix_one_row_per_order(canonical_order):
    X = build_feature_matrix([canonical_order, canonical_order])
    assert len(X) == 2
