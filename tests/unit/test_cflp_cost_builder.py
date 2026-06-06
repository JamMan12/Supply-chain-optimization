from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt.config import settings
from supply_chain_opt.optimizer.cost_builder import (
    add_region_centroids,
    cluster_facilities,
    compute_capacities,
    compute_fixed_costs,
    haversine_matrix,
)


@pytest.fixture(scope="module")
def raw_facilities() -> pd.DataFrame:
    return pd.read_parquet(settings.cflp_facility_path)


@pytest.fixture(scope="module")
def raw_demand() -> pd.DataFrame:
    return pd.read_parquet(settings.cflp_demand_path)


@pytest.fixture(scope="module")
def clustered(raw_facilities: pd.DataFrame) -> pd.DataFrame:
    return cluster_facilities(raw_facilities, n_clusters=5, random_state=42)


@pytest.fixture(scope="module")
def enriched(raw_demand: pd.DataFrame) -> pd.DataFrame:
    return add_region_centroids(raw_demand)


class TestClusterFacilities:
    def test_output_row_count(self, clustered: pd.DataFrame) -> None:
        assert len(clustered) == 5

    def test_output_columns(self, clustered: pd.DataFrame) -> None:
        assert list(clustered.columns) == ["lat", "lon", "n_points"]

    def test_no_nulls(self, clustered: pd.DataFrame) -> None:
        assert clustered.isnull().sum().sum() == 0

    def test_n_points_sum(
        self, raw_facilities: pd.DataFrame, clustered: pd.DataFrame
    ) -> None:
        assert clustered["n_points"].sum() == len(raw_facilities)

    def test_deterministic(self, raw_facilities: pd.DataFrame) -> None:
        a = cluster_facilities(raw_facilities, n_clusters=5, random_state=42)
        b = cluster_facilities(raw_facilities, n_clusters=5, random_state=42)
        pd.testing.assert_frame_equal(a, b)


class TestAddRegionCentroids:
    def test_row_count_unchanged(
        self, raw_demand: pd.DataFrame, enriched: pd.DataFrame
    ) -> None:
        assert len(enriched) == len(raw_demand)

    def test_lat_lon_columns_added(self, enriched: pd.DataFrame) -> None:
        assert "lat" in enriched.columns
        assert "lon" in enriched.columns

    def test_lat_range_valid(self, enriched: pd.DataFrame) -> None:
        assert enriched["lat"].between(-90, 90).all()

    def test_lon_range_valid(self, enriched: pd.DataFrame) -> None:
        assert enriched["lon"].between(-180, 180).all()

    def test_unknown_region_raises(self) -> None:
        bad_df = pd.DataFrame({
            "Order Region": ["Definitely Not A Region"],
            "demand": [1.0],
            "n_orders": [1],
        })
        with pytest.raises(KeyError):
            add_region_centroids(bad_df)


class TestHaversineMatrix:
    def test_shape(self, clustered: pd.DataFrame, enriched: pd.DataFrame) -> None:
        matrix = haversine_matrix(clustered, enriched)
        assert matrix.shape == (len(clustered), len(enriched))

    def test_non_negative(self, clustered: pd.DataFrame, enriched: pd.DataFrame) -> None:
        matrix = haversine_matrix(clustered, enriched)
        assert (matrix >= 0).all()

    def test_known_distance_london_to_nyc(self) -> None:
        # London (51.51, -0.13) → NYC (40.71, -74.01) ≈ 5,570 km
        fac = pd.DataFrame({"lat": [51.51], "lon": [-0.13]})
        dem = pd.DataFrame({"lat": [40.71], "lon": [-74.01]})
        dist = haversine_matrix(fac, dem)[0, 0]
        assert abs(dist - 5570) < 50

    def test_zero_distance_same_point(self) -> None:
        fac = pd.DataFrame({"lat": [48.69], "lon": [9.14]})
        dem = pd.DataFrame({"lat": [48.69], "lon": [9.14]})
        dist = haversine_matrix(fac, dem)[0, 0]
        assert dist < 1e-6


class TestComputeCapacities:
    def test_shape(self, clustered: pd.DataFrame) -> None:
        caps = compute_capacities(clustered, total_demand=1_000_000.0, capacity_multiplier=0.3)
        assert caps.shape == (len(clustered),)

    def test_uniform_value(self, clustered: pd.DataFrame) -> None:
        total = 1_000_000.0
        mult = 0.3
        caps = compute_capacities(clustered, total_demand=total, capacity_multiplier=mult)
        assert np.allclose(caps, total * mult)


class TestComputeFixedCosts:
    def test_shape(self, clustered: pd.DataFrame) -> None:
        costs = compute_fixed_costs(clustered, fixed_cost_per_facility=1e9)
        assert costs.shape == (len(clustered),)

    def test_uniform_value(self, clustered: pd.DataFrame) -> None:
        costs = compute_fixed_costs(clustered, fixed_cost_per_facility=1e9)
        assert np.allclose(costs, 1e9)
