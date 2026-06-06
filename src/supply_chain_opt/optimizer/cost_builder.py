from dataclasses import dataclass
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from supply_chain_opt.config import Settings, settings
from supply_chain_opt.optimizer.region_centroids import REGION_CENTROIDS


@dataclass
class CFLPData:
    facility_df: pd.DataFrame   # columns: lat, lon, n_points
    demand_df: pd.DataFrame     # columns: Order Region, demand, n_orders, lat, lon
    cost_matrix: np.ndarray     # shape (n_facilities, n_demand), haversine km
    capacities: np.ndarray      # shape (n_facilities,)
    fixed_costs: np.ndarray     # shape (n_facilities,)
    demands: np.ndarray         # shape (n_demand,)
    n_facilities: int
    n_demand: int


def cluster_facilities(
    df: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Reduce raw facility lat/lon points to n_clusters representative centers via KMeans."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(df[["Latitude", "Longitude"]].values)
    centers = km.cluster_centers_

    counts = np.bincount(labels, minlength=n_clusters)
    return pd.DataFrame({
        "lat": centers[:, 0],
        "lon": centers[:, 1],
        "n_points": counts,
    })


def add_region_centroids(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Attach lat/lon centroid coordinates to each demand region row."""
    missing = set(demand_df["Order Region"]) - set(REGION_CENTROIDS)
    if missing:
        raise KeyError(f"Regions missing from centroid lookup: {missing}")

    df = demand_df.copy()
    df["lat"] = df["Order Region"].map(lambda r: REGION_CENTROIDS[r][0])
    df["lon"] = df["Order Region"].map(lambda r: REGION_CENTROIDS[r][1])
    return df


def haversine_matrix(
    facility_df: pd.DataFrame,
    demand_df: pd.DataFrame,
) -> np.ndarray:
    """Compute great-circle distances (km) between every facility-demand pair.

    Returns array of shape (n_facilities, n_demand).
    """
    R = 6371.0

    fac_lat = np.radians(facility_df["lat"].values)
    fac_lon = np.radians(facility_df["lon"].values)
    dem_lat = np.radians(demand_df["lat"].values)
    dem_lon = np.radians(demand_df["lon"].values)

    dlat = dem_lat[np.newaxis, :] - fac_lat[:, np.newaxis]
    dlon = dem_lon[np.newaxis, :] - fac_lon[:, np.newaxis]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(fac_lat[:, np.newaxis]) * np.cos(dem_lat[np.newaxis, :]) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_capacities(
    facility_df: pd.DataFrame,
    total_demand: float,
    capacity_multiplier: float,
) -> np.ndarray:
    """Return uniform capacity array: multiplier * total_demand for every facility."""
    return np.full(len(facility_df), capacity_multiplier * total_demand)


def compute_fixed_costs(
    facility_df: pd.DataFrame,
    fixed_cost_per_facility: float,
) -> np.ndarray:
    """Return uniform fixed-cost array for every facility."""
    return np.full(len(facility_df), fixed_cost_per_facility)


def build_problem_data(cfg: Settings = settings) -> CFLPData:
    """Load processed parquets, cluster facilities, enrich demand nodes, build cost matrix."""
    raw_facilities = pd.read_parquet(cfg.cflp_facility_path)
    raw_demand = pd.read_parquet(cfg.cflp_demand_path)

    facility_df = cluster_facilities(raw_facilities, cfg.cflp_n_facility_clusters)
    demand_df = add_region_centroids(raw_demand)

    cost_matrix = haversine_matrix(facility_df, demand_df)
    demands = demand_df["demand"].values
    total_demand = float(demands.sum())

    return CFLPData(
        facility_df=facility_df,
        demand_df=demand_df,
        cost_matrix=cost_matrix,
        capacities=compute_capacities(facility_df, total_demand, cfg.cflp_capacity_multiplier),
        fixed_costs=compute_fixed_costs(facility_df, cfg.cflp_fixed_cost_per_facility),
        demands=demands,
        n_facilities=len(facility_df),
        n_demand=len(demand_df),
    )
