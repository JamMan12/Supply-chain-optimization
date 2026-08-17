from dataclasses import dataclass
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline


@dataclass
class EvalMetrics:
    model_name: str
    roc_auc: float
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    pr_thresholds: np.ndarray
    calibration_prob_true: np.ndarray
    calibration_prob_pred: np.ndarray
    brier_score: float


def evaluate_model(
    pipeline: Pipeline,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_calibration_bins: int = 10,
) -> EvalMetrics:
    """Score X_test via predict_proba, computing AUC-ROC, PR curve, calibration curve, and Brier score."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_calibration_bins)

    return EvalMetrics(
        model_name=model_name,
        roc_auc=roc_auc_score(y_test, y_proba),
        roc_fpr=fpr,
        roc_tpr=tpr,
        precision=precision,
        recall=recall,
        pr_thresholds=pr_thresholds,
        calibration_prob_true=prob_true,
        calibration_prob_pred=prob_pred,
        brier_score=brier_score_loss(y_test, y_proba),
    )


def plot_roc_pr_calibration(metrics_list: list[EvalMetrics], output_dir: Path) -> None:
    """Save one PNG per metric type (roc, pr, calibration), one line per model, to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for m in metrics_list:
        ax.plot(m.roc_fpr, m.roc_tpr, label=f"{m.model_name} (AUC={m.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.savefig(output_dir / "roc_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for m in metrics_list:
        ax.plot(m.recall, m.precision, label=m.model_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.savefig(output_dir / "pr_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for m in metrics_list:
        ax.plot(m.calibration_prob_pred, m.calibration_prob_true, marker="o", label=m.model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.savefig(output_dir / "calibration_curve.png")
    plt.close(fig)


def compare_models(metrics_list: list[EvalMetrics]) -> pd.DataFrame:
    """Tabulate roc_auc and brier_score side by side for model selection."""
    return pd.DataFrame(
        [{"model_name": m.model_name, "roc_auc": m.roc_auc, "brier_score": m.brier_score} for m in metrics_list]
    )
