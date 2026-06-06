from pathlib import Path
import sys

# Makes the package importable when run directly: python src/supply_chain_opt/data/loader.py
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from supply_chain_opt.config import settings


def load_raw() -> pd.DataFrame:
    """Load the raw DataCo CSV."""
    df = pd.read_csv(settings.raw_data_path, encoding="latin-1")
    df.columns = df.columns.str.strip() #remove whitespace from column names
    return df


def print_shape(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


def print_dtypes(df: pd.DataFrame) -> None:
    print("\nColumn names and data types:")
    print(df.dtypes.to_string())


def print_summary(df: pd.DataFrame) -> None:
    print("\nNumeric summary:")
    print(df.describe().T.to_string())
    print("\nCategorical summary:")
    print(df.describe(include="str").T.to_string())


def print_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = (
        pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
    )
    print(f"\nMissing values: {len(report)} of {df.shape[1]} columns affected")
    if not report.empty:
        print(report.to_string())


if __name__ == "__main__":
    df = load_raw()
    print_shape(df)
    print_dtypes(df)
    print_summary(df)
    print_missing(df)
