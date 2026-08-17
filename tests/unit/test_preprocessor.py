from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
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


class TestRouteHistoricalDelayRate:
    """Pins down the leakage-safety claims in data/README.md with synthetic lanes.

    Each row's rate must depend only on same-lane rows strictly earlier in
    `_order_date` — never on its own label, never on a later row's label.
    """

    @staticmethod
    def _synthetic_orders(n: int, n_lanes: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "_order_date": pd.Timestamp("2016-01-01")
                + pd.to_timedelta(np.arange(n) * 37, unit="h"),
                "Market": rng.choice(["LATAM", "Europe"], size=n),
                "Order Region": rng.choice(
                    [f"Region {i}" for i in range(n_lanes)], size=n
                ),
                TARGET: rng.integers(0, 2, size=n),
            }
        )

    def test_hand_computed_three_row_lane(self) -> None:
        df = pd.DataFrame(
            {
                "_order_date": pd.to_datetime(
                    ["2016-01-01", "2016-01-02", "2016-01-03"]
                ),
                "Market": ["LATAM"] * 3,
                "Order Region": ["Caribbean"] * 3,
                TARGET: [1, 0, 1],
            }
        )
        result = preprocessor.add_route_historical_delay_rate(df, smoothing_k=20.0)
        rates = result["route_historical_delay_rate"].to_numpy()

        # Row 0: no prior orders at all -> falls back to the 0.5 anchor.
        assert rates[0] == pytest.approx(0.5)
        # Row 1: one prior order dataset-wide (row 0, label=1), so the
        # point-in-time global rate is 1.0; blended with the identical
        # one-prior-order lane rate: (1 + 20*1.0) / (1 + 20) = 1.0.
        assert rates[1] == pytest.approx(1.0)
        # Row 2: two prior orders (labels=1,0) both lane- and dataset-wide,
        # so global_rate = 1/2 = 0.5: (1 + 20*0.5) / (2 + 20) = 0.5.
        assert rates[2] == pytest.approx(0.5)

    def test_row_rate_unaffected_by_own_label(self) -> None:
        df = self._synthetic_orders(n=40, n_lanes=3, seed=1)
        flipped = df.copy()
        flipped.loc[10, TARGET] = 1 - flipped.loc[10, TARGET]

        result = preprocessor.add_route_historical_delay_rate(df.copy())
        result_flipped = preprocessor.add_route_historical_delay_rate(flipped)

        # Locate row 10 in each (function re-sorts and resets the index).
        original_row = df.loc[10]
        idx = result.index[
            (result["_order_date"] == original_row["_order_date"])
            & (result["Market"] == original_row["Market"])
            & (result["Order Region"] == original_row["Order Region"])
        ][0]
        idx_flipped = result_flipped.index[
            (result_flipped["_order_date"] == original_row["_order_date"])
            & (result_flipped["Market"] == original_row["Market"])
            & (result_flipped["Order Region"] == original_row["Order Region"])
        ][0]

        assert result.loc[idx, "route_historical_delay_rate"] == pytest.approx(
            result_flipped.loc[idx_flipped, "route_historical_delay_rate"]
        )

    def test_row_rate_unaffected_by_later_labels(self) -> None:
        df = self._synthetic_orders(n=40, n_lanes=3, seed=2)
        cutoff = df["_order_date"].sort_values().iloc[20]

        mutated = df.copy()
        later_mask = mutated["_order_date"] > cutoff
        mutated.loc[later_mask, TARGET] = 1 - mutated.loc[later_mask, TARGET]

        result = preprocessor.add_route_historical_delay_rate(df.copy())
        result_mutated = preprocessor.add_route_historical_delay_rate(mutated)

        earlier_or_equal = result["_order_date"] <= cutoff
        pd.testing.assert_series_equal(
            result.loc[earlier_or_equal, "route_historical_delay_rate"].reset_index(
                drop=True
            ),
            result_mutated.loc[
                earlier_or_equal, "route_historical_delay_rate"
            ].reset_index(drop=True),
        )

    def test_first_ever_order_gets_neutral_prior(self) -> None:
        df = self._synthetic_orders(n=15, n_lanes=2, seed=3)
        result = preprocessor.add_route_historical_delay_rate(df)
        first_row = result.sort_values("_order_date").iloc[0]
        assert first_row["route_historical_delay_rate"] == pytest.approx(0.5)

    def test_rates_bounded_between_zero_and_one(self) -> None:
        df = self._synthetic_orders(n=100, n_lanes=5, seed=4)
        result = preprocessor.add_route_historical_delay_rate(df)
        rates = result["route_historical_delay_rate"]
        assert ((rates >= 0.0) & (rates <= 1.0)).all()


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
