from pathlib import Path
import sys
from typing import Any

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from supply_chain_opt.data.preprocessor import ML_FEATURES

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Customer Segment",
    "Category Name",
    "Department Name",
    "Type",
]

NUMERIC_FEATURES = [
    "Order Item Quantity",
    "Order Item Total",
    "Product Price",
    "Order Item Discount Rate",
    "order_month",
    "order_quarter",
    "route_historical_delay_rate",
]

# Guards against silent drift from the approved feature set in data/README.md.
assert set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) == set(ML_FEATURES)


def build_preprocessor() -> ColumnTransformer:
    """OneHotEncoder on categoricals (unseen categories ignored), passthrough on numerics."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_xgboost_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Preprocessor + XGBClassifier; params overridable for grid search."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", XGBClassifier(**(params or {}))),
        ]
    )


def build_lightgbm_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Preprocessor + LGBMClassifier; params overridable for grid search."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LGBMClassifier(**(params or {}))),
        ]
    )
