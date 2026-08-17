from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import joblib
import lightgbm
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from supply_chain_opt.config import settings
from supply_chain_opt.data.preprocessor import ML_FEATURES
from supply_chain_opt.models.dataset import load_dataset, split_dataset
from supply_chain_opt.models.evaluate import compare_models, evaluate_model, plot_roc_pr_calibration
from supply_chain_opt.models.pipeline import build_lightgbm_pipeline, build_xgboost_pipeline

_XGB_PARAM_GRID = {
    "clf__max_depth": [5, 7, 9],
    "clf__learning_rate": [0.05, 0.1],
    "clf__n_estimators": [200, 400],
    "clf__min_child_weight": [1, 5],
}

_LGBM_PARAM_GRID = {
    "clf__num_leaves": [31, 63, 127],
    "clf__learning_rate": [0.05, 0.1],
    "clf__n_estimators": [200, 400],
    "clf__min_child_samples": [10, 30],
}


@dataclass
class TrainResult:
    model_name: str
    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_best_score: float


def tune_model(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any], float]:
    """GridSearchCV over param_grid scored on roc_auc, refit on the full training set."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, refit=True)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> TrainResult:
    """Tune and fit the XGBoost pipeline."""
    pipeline = build_xgboost_pipeline({"random_state": random_state})
    best_pipeline, best_params, best_score = tune_model(
        pipeline, _XGB_PARAM_GRID, X_train, y_train, cv_folds, random_state
    )
    return TrainResult("xgboost", best_pipeline, best_params, best_score)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> TrainResult:
    """Tune and fit the LightGBM pipeline."""
    pipeline = build_lightgbm_pipeline({"random_state": random_state, "verbose": -1})
    best_pipeline, best_params, best_score = tune_model(
        pipeline, _LGBM_PARAM_GRID, X_train, y_train, cv_folds, random_state
    )
    return TrainResult("lightgbm", best_pipeline, best_params, best_score)


def save_model(result: TrainResult, path: Path, metadata: dict[str, Any]) -> None:
    """Persist the fitted pipeline via joblib and write a sibling *.metadata.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, path)
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))


def run(save: bool = True) -> dict[str, TrainResult]:
    """Load data, split, tune+fit both model families, persist both, print a comparison summary."""
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df, settings.test_size, settings.random_state)

    xgb_result = train_xgboost(X_train, y_train, settings.cv_folds, settings.random_state)
    lgbm_result = train_lightgbm(X_train, y_train, settings.cv_folds, settings.random_state)
    results = {"xgboost": xgb_result, "lightgbm": lgbm_result}

    metrics = [
        evaluate_model(xgb_result.pipeline, "xgboost", X_test, y_test),
        evaluate_model(lgbm_result.pipeline, "lightgbm", X_test, y_test),
    ]

    print(f"{'Model':<10} {'CV ROC-AUC':>12} {'Test ROC-AUC':>14} {'Brier':>10}")
    for result, metric in zip(results.values(), metrics):
        print(
            f"{result.model_name:<10} {result.cv_best_score:>12.4f} "
            f"{metric.roc_auc:>14.4f} {metric.brier_score:>10.4f}"
        )

    if save:
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        trained_at = datetime.now(timezone.utc).isoformat()
        common_meta = {
            "feature_names": ML_FEATURES,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "random_state": settings.random_state,
            "trained_at": trained_at,
            "sklearn_version": sklearn.__version__,
        }

        save_model(
            xgb_result,
            settings.delay_classifier_xgboost_path,
            {
                **common_meta,
                "model_family": "xgboost",
                "best_params": xgb_result.best_params,
                "cv_best_score": xgb_result.cv_best_score,
                "xgboost_version": xgboost.__version__,
            },
        )
        save_model(
            lgbm_result,
            settings.delay_classifier_lightgbm_path,
            {
                **common_meta,
                "model_family": "lightgbm",
                "best_params": lgbm_result.best_params,
                "cv_best_score": lgbm_result.cv_best_score,
                "lightgbm_version": lightgbm.__version__,
            },
        )

        plot_roc_pr_calibration(metrics, settings.delay_classifier_figures_dir)

        comparison_df = compare_models(metrics)
        recommended_model = comparison_df.sort_values("roc_auc", ascending=False)["model_name"].iloc[0]
        settings.delay_classifier_comparison_path.write_text(
            json.dumps(
                {
                    "models": comparison_df.to_dict(orient="records"),
                    "recommended_model": recommended_model,
                },
                indent=2,
            )
        )
        print(f"\nArtifacts written to {settings.models_dir}")
        print(f"Recommended model: {recommended_model}")

    return results


if __name__ == "__main__":
    run(save=True)
