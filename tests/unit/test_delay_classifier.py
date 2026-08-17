from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import joblib
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from supply_chain_opt.config import settings
from supply_chain_opt.data import preprocessor
from supply_chain_opt.data.loader import load_raw
from supply_chain_opt.models import dataset, evaluate, pipeline, train

_TINY_XGB_GRID = {"clf__max_depth": [3], "clf__n_estimators": [50]}
_TINY_LGBM_GRID = {"clf__num_leaves": [15], "clf__n_estimators": [50]}
_TEST_RANDOM_STATE = 42


@pytest.fixture(scope="module")
def ml_sample() -> pd.DataFrame:
    raw = pd.read_csv(settings.raw_data_path, encoding="latin-1", nrows=2000)
    clean = preprocessor.clean(raw)
    return preprocessor.build_ml_dataset(clean)


@pytest.fixture(scope="module")
def split_sample(ml_sample: pd.DataFrame):
    return dataset.split_dataset(ml_sample, test_size=0.2, random_state=_TEST_RANDOM_STATE)


@pytest.fixture(scope="module")
def fitted_xgb_result(split_sample):
    X_train, _, y_train, _ = split_sample
    xgb_pipeline = pipeline.build_xgboost_pipeline({"random_state": _TEST_RANDOM_STATE})
    best_pipeline, best_params, best_score = train.tune_model(
        xgb_pipeline, _TINY_XGB_GRID, X_train, y_train, cv_folds=2, random_state=_TEST_RANDOM_STATE
    )
    return train.TrainResult("xgboost", best_pipeline, best_params, best_score)


@pytest.fixture(scope="module")
def fitted_lgbm_result(split_sample):
    X_train, _, y_train, _ = split_sample
    lgbm_pipeline = pipeline.build_lightgbm_pipeline({"random_state": _TEST_RANDOM_STATE, "verbose": -1})
    best_pipeline, best_params, best_score = train.tune_model(
        lgbm_pipeline, _TINY_LGBM_GRID, X_train, y_train, cv_folds=2, random_state=_TEST_RANDOM_STATE
    )
    return train.TrainResult("lightgbm", best_pipeline, best_params, best_score)


class TestDatasetSplit:
    def test_row_count_preserved(self, ml_sample: pd.DataFrame, split_sample) -> None:
        X_train, X_test, y_train, y_test = split_sample
        assert len(X_train) + len(X_test) == len(ml_sample)
        assert len(y_train) + len(y_test) == len(ml_sample)

    def test_class_proportions_preserved(self, ml_sample: pd.DataFrame, split_sample) -> None:
        _, _, y_train, y_test = split_sample
        full_rate = ml_sample[preprocessor.TARGET].mean()
        assert abs(y_train.mean() - full_rate) < 0.05
        assert abs(y_test.mean() - full_rate) < 0.05

    def test_no_index_overlap(self, split_sample) -> None:
        X_train, X_test, _, _ = split_sample
        assert set(X_train.index).isdisjoint(set(X_test.index))


class TestPipelineBuild:
    def test_xgboost_pipeline_has_clf_step(self) -> None:
        assert "clf" in pipeline.build_xgboost_pipeline().named_steps

    def test_lightgbm_pipeline_has_clf_step(self) -> None:
        assert "clf" in pipeline.build_lightgbm_pipeline().named_steps

    def test_xgboost_fits_on_full_categorical_cardinality(self, split_sample) -> None:
        X_train, _, y_train, _ = split_sample
        pipeline.build_xgboost_pipeline({"random_state": _TEST_RANDOM_STATE}).fit(X_train, y_train)

    def test_lightgbm_fits_on_full_categorical_cardinality(self, split_sample) -> None:
        X_train, _, y_train, _ = split_sample
        pipeline.build_lightgbm_pipeline({"random_state": _TEST_RANDOM_STATE, "verbose": -1}).fit(X_train, y_train)


class TestTrainAndPredict:
    @pytest.mark.parametrize("result_fixture", ["fitted_xgb_result", "fitted_lgbm_result"])
    def test_predict_proba_shape_and_range(self, result_fixture, split_sample, request) -> None:
        result = request.getfixturevalue(result_fixture)
        _, X_test, _, _ = split_sample
        proba = result.pipeline.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert (proba >= 0).all() and (proba <= 1).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("result_fixture", ["fitted_xgb_result", "fitted_lgbm_result"])
    def test_beats_trivial_baseline(self, result_fixture, split_sample, request) -> None:
        result = request.getfixturevalue(result_fixture)
        _, X_test, _, y_test = split_sample
        y_proba = result.pipeline.predict_proba(X_test)[:, 1]
        assert roc_auc_score(y_test, y_proba) > 0.5


class TestSaveLoad:
    @pytest.mark.parametrize("result_fixture", ["fitted_xgb_result", "fitted_lgbm_result"])
    def test_round_trip_predictions_match(self, result_fixture, split_sample, request, tmp_path) -> None:
        result = request.getfixturevalue(result_fixture)
        _, X_test, _, _ = split_sample

        model_path = tmp_path / f"{result.model_name}.pkl"
        train.save_model(result, model_path, {"model_family": result.model_name})

        reloaded = joblib.load(model_path)
        original_proba = result.pipeline.predict_proba(X_test)
        reloaded_proba = reloaded.predict_proba(X_test)
        assert np.allclose(original_proba, reloaded_proba)

    def test_metadata_file_has_expected_keys(self, fitted_xgb_result, tmp_path) -> None:
        model_path = tmp_path / "xgboost.pkl"
        metadata = {
            "feature_names": preprocessor.ML_FEATURES,
            "model_family": "xgboost",
            "best_params": fitted_xgb_result.best_params,
            "trained_at": "2026-01-01T00:00:00+00:00",
            "random_state": _TEST_RANDOM_STATE,
        }
        train.save_model(fitted_xgb_result, model_path, metadata)

        metadata_path = model_path.with_suffix(".metadata.json")
        assert metadata_path.exists()
        saved = json.loads(metadata_path.read_text())
        for key in ("feature_names", "model_family", "best_params", "trained_at", "random_state"):
            assert key in saved


class TestEvaluate:
    @pytest.mark.parametrize("result_fixture", ["fitted_xgb_result", "fitted_lgbm_result"])
    def test_eval_metrics_well_formed(self, result_fixture, split_sample, request) -> None:
        result = request.getfixturevalue(result_fixture)
        _, X_test, _, y_test = split_sample

        metrics = evaluate.evaluate_model(result.pipeline, result.model_name, X_test, y_test)

        assert 0.0 <= metrics.roc_auc <= 1.0
        assert len(metrics.precision) == len(metrics.recall)
        assert len(metrics.calibration_prob_true) == len(metrics.calibration_prob_pred)
        assert ((metrics.calibration_prob_true >= 0) & (metrics.calibration_prob_true <= 1)).all()
        assert ((metrics.calibration_prob_pred >= 0) & (metrics.calibration_prob_pred <= 1)).all()
