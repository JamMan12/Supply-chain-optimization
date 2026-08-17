from dataclasses import dataclass, field
from pathlib import Path

# Resolves to the project root (SupplyChainOpt/) regardless of working directory.
# TODO: replace with pydantic_settings.BaseSettings once pydantic-settings is installed.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    raw_data_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
    )
    processed_data_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed"
    )
    models_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models"
    )
    ml_dataset_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed" / "ml_features.parquet"
    )
    cflp_facility_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed" / "cflp_facility_nodes.parquet"
    )
    cflp_demand_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed" / "cflp_demand_nodes.parquet"
    )

    # Feature engineering
    route_history_smoothing_k: float = 20.0

    # Facility clustering
    cflp_n_facility_clusters: int = 20

    # Problem parameters
    cflp_max_open_facilities: int = 5
    cflp_capacity_multiplier: float = 0.3
    cflp_fixed_cost_per_facility: float = 1e9
    cflp_cost_per_km: float = 1.0

    # Solver engine
    cflp_solver_backend: str = "HIGHS"
    cflp_solver_time_limit_s: int = 60

    # Output
    cflp_results_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed" / "cflp_solution.json"
    )

    # ML — Phase 2 delay classifier
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    delay_classifier_xgboost_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "delay_classifier_xgboost.pkl"
    )
    delay_classifier_lightgbm_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "delay_classifier_lightgbm.pkl"
    )
    delay_classifier_figures_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "figures"
    )
    delay_classifier_comparison_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "comparison_report.json"
    )


settings = Settings()
