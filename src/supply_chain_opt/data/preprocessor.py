from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from supply_chain_opt.config import settings
from supply_chain_opt.data.loader import load_raw

_LEAKAGE_COLS = [
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
    "Sales per customer",
]

_DROP_COLS = [
    # Redundant numeric shadow of Shipping Mode (1-to-1 mapping confirmed in EDA section 8.8)
    "Days for shipment (scheduled)",
    # PII / masked
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    # Redundant IDs
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Order Item Cardprod Id",
    "Product Card Id",
    # Redundant categoricals
    "Category Id",
    "Department Id",
    "Product Category Id",
    # Useless
    "Product Description",
    "Product Image",
    "Product Status",
    # Customer registration address — not the delivery destination
    "Customer City",
    "Customer Country",
    "Customer State",
    "Customer Street",
    "Customer Zipcode",
    # >95% missing
    "Order Zipcode",
    # Duplicate of kept columns
    "Order Item Product Price",
    "Order Item Discount",
    "Sales",
    # High cardinality, covered by Category Name / Department Name
    "Product Name",
]

ML_FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Customer Segment",
    "Category Name",
    "Department Name",
    "Type",
    "Order Item Quantity",
    "Order Item Total",
    "Product Price",
    "Order Item Discount Rate",
    "order_month",
    "order_quarter",
]

TARGET = "Late_delivery_risk"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all EDA-decided transformations: drop columns, extract date features, fix precision."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    dates = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
    df["order_month"] = dates.dt.month
    df["order_quarter"] = dates.dt.quarter

    to_drop = _LEAKAGE_COLS + _DROP_COLS + ["order date (DateOrders)"]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    df["Order Item Discount Rate"] = df["Order Item Discount Rate"].round(2)

    return df


def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 14 approved ML features and target. Encoding happens at training time."""
    return df[ML_FEATURES + [TARGET]].copy()


def build_cflp_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build facility node and demand node tables for the CFLP solver."""
    facility_nodes = (
        df[["Latitude", "Longitude"]]
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    demand_nodes = (
        df.groupby("Order Region", as_index=False)
        .agg(
            demand=("Order Item Total", "sum"),
            n_orders=("Order Item Total", "count"),
        )
    )

    return {"facility_nodes": facility_nodes, "demand_nodes": demand_nodes}


def run(save: bool = True) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load raw data, clean it, build both output datasets, and optionally persist to disk."""
    df_raw = load_raw()
    df_clean = clean(df_raw)

    ml = build_ml_dataset(df_clean)
    cflp = build_cflp_dataset(df_clean)

    if save:
        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        ml.to_parquet(settings.ml_dataset_path, index=False)
        cflp["facility_nodes"].to_parquet(settings.cflp_facility_path, index=False)
        cflp["demand_nodes"].to_parquet(settings.cflp_demand_path, index=False)

        print(f"ML dataset     : {ml.shape[0]:,} rows x {ml.shape[1]} cols  ->  {settings.ml_dataset_path.name}")
        print(f"Facility nodes : {len(cflp['facility_nodes']):,} unique locations  ->  {settings.cflp_facility_path.name}")
        print(f"Demand nodes   : {len(cflp['demand_nodes']):,} regions  ->  {settings.cflp_demand_path.name}")

    return ml, cflp


if __name__ == "__main__":
    run(save=True)
