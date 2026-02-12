"""
SHAP Model Interpretability

Computes SHAP values and generates interpretability plots for XGBoost models.
"""

from typing import List, Tuple
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Compute SHAP values for the given model and data.

    Args:
        model: Trained XGBoost model.
        X: Feature DataFrame.
        max_samples: Maximum samples to use (for performance). If X has more
            rows, randomly sample.

    Returns:
        Tuple of:
        - shap_values: 2D numpy array of SHAP values (n_samples x n_features)
        - feature_names: List of feature names
        - X_sample: The (possibly sampled) DataFrame used for computation
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed. Run: pip install shap")
        raise

    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
        logger.info(
            "Sampled %d rows from %d for SHAP computation.", max_samples, len(X)
        )
    else:
        X_sample = X.copy()

    feature_names = list(X_sample.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Binary classification: shap_values may be a list of two arrays.
    # Take index [1] for the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(
        "Computed SHAP values: %d samples x %d features.",
        shap_values.shape[0],
        shap_values.shape[1],
    )

    return shap_values, feature_names, X_sample


def shap_summary_df(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Create a summary DataFrame of SHAP values.

    Returns:
        DataFrame with columns: Feature, Mean_Abs_SHAP, Rank
        Sorted by Mean_Abs_SHAP descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs,
    })
    df = df.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    return df


def save_shap_plots(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_dir: str,
) -> List[str]:
    """
    Save SHAP summary plots as PNG files.

    Args:
        shap_values: SHAP values array.
        X_sample: Feature DataFrame used for SHAP computation.
        output_dir: Directory to save plots.

    Returns:
        List of saved file paths.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed. Run: pip install shap")
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed. Run: pip install matplotlib")
        return []

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []

    # Bar plot of mean |SHAP| values
    bar_path = str(out_path / "shap_summary_bar.png")
    try:
        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(bar_path)
        logger.info("Saved SHAP bar plot: %s", bar_path)
    except Exception:
        logger.exception("Failed to save SHAP bar plot.")

    # Beeswarm plot
    beeswarm_path = str(out_path / "shap_beeswarm.png")
    try:
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(beeswarm_path)
        logger.info("Saved SHAP beeswarm plot: %s", beeswarm_path)
    except Exception:
        logger.exception("Failed to save SHAP beeswarm plot.")

    return saved_paths
