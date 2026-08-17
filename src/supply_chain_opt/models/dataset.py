from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
from sklearn.model_selection import train_test_split

from supply_chain_opt.config import settings
from supply_chain_opt.data.preprocessor import TARGET


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the processed ML feature parquet built by preprocessor.build_ml_dataset."""
    return pd.read_parquet(path or settings.ml_dataset_path)


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on TARGET, returning X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
