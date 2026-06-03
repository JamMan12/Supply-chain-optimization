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


settings = Settings()
