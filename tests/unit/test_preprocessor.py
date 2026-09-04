from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import pytest

from supply_chain_opt.config import settings
from supply_chain_opt.data import preprocessor
from supply_chain_opt.data.preprocessor import ML_FEATURES, TARGET, _LEAKAGE_COLS


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    return pd.read_csv(settings.raw_data_path, encoding="latin-1", nrows=1000)


@pytest.fixture(scope="module")
def clean_sample(raw_sample: pd.DataFrame) -> pd.DataFrame:
    return preprocessor.clean(raw_sample)


@pytest.fixture(scope="module")
def ml_sample(clean_sample: pd.DataFrame) -> pd.DataFrame:
    return preprocessor.build_ml_dataset(clean_sample)


@pytest.fixture(scope="module")
def cflp_sample(clean_sample: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return preprocessor.build_cflp_dataset(clean_sample)


class TestClean:
    def test_no_leakage_columns(self, clean_sample: pd.DataFrame) -> None:
        for col in _LEAKAGE_COLS:
            assert col not in clean_sample.columns, f"Leakage column still present: {col}"

    def test_source_date_column_dropped(self, clean_sample: pd.DataFrame) -> None:
        assert "order date (DateOrders)" not in clean_sample.columns

    def test_order_month_present_and_valid(self, clean_sample: pd.DataFrame) -> None:
        assert "order_month" in clean_sample.columns
        assert clean_sample["order_month"].between(1, 12).all()

    def test_order_quarter_present_and_valid(self, clean_sample: pd.DataFrame) -> None:
        assert "order_quarter" in clean_sample.columns
        assert clean_sample["order_quarter"].between(1, 4).all()

    def test_discount_rate_precision(self, clean_sample: pd.DataFrame) -> None:
        residual = (
            clean_sample["Order Item Discount Rate"]
            - clean_sample["Order Item Discount Rate"].round(2)
        ).abs().max()
        assert residual < 1e-10

    def test_row_count_unchanged(self, raw_sample: pd.DataFrame, clean_sample: pd.DataFrame) -> None:
        assert len(clean_sample) == len(raw_sample)


class TestBuildMlDataset:
    def test_all_features_present(self, ml_sample: pd.DataFrame) -> None:
        for col in ML_FEATURES:
            assert col in ml_sample.columns, f"Missing approved feature: {col}"

    def test_target_present(self, ml_sample: pd.DataFrame) -> None:
        assert TARGET in ml_sample.columns

    def test_exact_column_count(self, ml_sample: pd.DataFrame) -> None:
        assert ml_sample.shape[1] == len(ML_FEATURES) + 1

    def test_no_leakage_columns(self, ml_sample: pd.DataFrame) -> None:
        for col in _LEAKAGE_COLS:
            assert col not in ml_sample.columns, f"Leakage column in ML output: {col}"

    def test_target_is_binary(self, ml_sample: pd.DataFrame) -> None:
        assert set(ml_sample[TARGET].unique()).issubset({0, 1})


class TestBuildCflpDataset:
    def test_facility_nodes_columns(self, cflp_sample: dict) -> None:
        assert list(cflp_sample["facility_nodes"].columns) == ["Latitude", "Longitude"]

    def test_facility_nodes_no_nulls(self, cflp_sample: dict) -> None:
        assert cflp_sample["facility_nodes"].isnull().sum().sum() == 0

    def test_facility_nodes_unique(self, cflp_sample: dict) -> None:
        fn = cflp_sample["facility_nodes"]
        assert fn.duplicated().sum() == 0

    def test_demand_nodes_has_required_columns(self, cflp_sample: dict) -> None:
        dn = cflp_sample["demand_nodes"]
        assert "Order Region" in dn.columns
        assert "demand" in dn.columns
        assert "n_orders" in dn.columns

    def test_demand_nodes_positive_demand(self, cflp_sample: dict) -> None:
        assert (cflp_sample["demand_nodes"]["demand"] > 0).all()

    def test_demand_nodes_one_row_per_region(self, cflp_sample: dict) -> None:
        dn = cflp_sample["demand_nodes"]
        assert dn["Order Region"].nunique() == len(dn)
