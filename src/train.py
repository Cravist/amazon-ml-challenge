"""
Model training pipeline for entity resolution.

Trains a gradient-boosted decision tree classifier (LightGBM) on pairwise similarity
features to distinguish true matching entity pairs from non-matches, validating against
a held-out split using F0.5 scoring.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_MODEL_PATH,
    EVAL_F_BETA,
    MODELS_DIR,
    RANDOM_SEED,
    VALIDATION_SPLIT_SIZE,
    seed_everything,
)

logger = logging.getLogger(__name__)


def load_training_data(
    features_path: Path | str,
    target_col: str = "is_match",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load precomputed pairwise features and target matching labels.

    Args:
        features_path: File path to features file (TSV, CSV, or Parquet).
        target_col: Column name corresponding to ground-truth binary match label.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the binary Series.
    """
    # TODO: Load features file, separate target_col from predictor columns,
    # drop identifier columns (record_id_1, record_id_2) from X, and return (X, y).
    raise NotImplementedError("load_training_data is not yet implemented.")


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = VALIDATION_SPLIT_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and validation sets with stratified sampling.

    Args:
        X: Feature matrix.
        y: Binary ground-truth labels.
        val_size: Fraction of data allocated to the validation split.
        seed: Random seed for deterministic splitting.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val).
    """
    # TODO: Perform stratified train_test_split using RANDOM_SEED.
    raise NotImplementedError("split_train_validation is not yet implemented.")


def get_default_lgbm_params() -> dict[str, Any]:
    """
    Return baseline LightGBM hyperparameters tuned for tabular entity matching.

    Returns:
        Dictionary of LightGBM hyperparameter key-value pairs.
    """
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
        "n_jobs": -1,
    }


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier on pairwise entity resolution features.

    Supports early stopping on the validation set when provided.

    Args:
        X_train: Training feature matrix.
        y_train: Training binary labels.
        X_val: Optional validation feature matrix.
        y_val: Optional validation binary labels.
        hyperparams: Optional dictionary overriding default model hyperparameters.

    Returns:
        Fitted LightGBM classifier instance.
    """
    # TODO: Merge hyperparams with get_default_lgbm_params, initialize LGBMClassifier,
    # fit on training data with early stopping callbacks if validation data is present,
    # and return trained model.
    raise NotImplementedError("train_lightgbm_model is not yet implemented.")


def evaluate_model_performance(
    model: lgb.LGBMClassifier,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    threshold: float = 0.5,
    beta: float = EVAL_F_BETA,
) -> dict[str, float]:
    """
    Evaluate trained model performance against validation split using F0.5 scoring.

    Args:
        model: Trained LightGBM classifier.
        X_eval: Validation features.
        y_eval: Ground truth validation labels.
        threshold: Classification probability threshold.
        beta: Weighting factor for F-beta score (0.5 prioritizes precision).

    Returns:
        Dictionary containing precision, recall, f_beta, and roc_auc metrics.
    """
    # TODO: Predict probabilities, apply threshold, compute fbeta_score(beta=beta),
    # precision, recall, and roc_auc, returning a metrics dictionary.
    raise NotImplementedError("evaluate_model_performance is not yet implemented.")


def save_model_artifact(
    model: lgb.LGBMClassifier,
    output_path: Path = DEFAULT_MODEL_PATH,
    feature_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Persist trained model artifact and training metadata to disk using joblib.

    Args:
        model: Trained LightGBM model.
        output_path: Destination path for serialized file (.joblib / .pkl).
        feature_names: List of feature names model was trained on.
        metadata: Additional run metadata (metrics, thresholds, hyperparameters).
    """
    # TODO: Ensure output directory exists, assemble artifact dictionary
    # (model, feature_names, metadata), and dump using joblib.
    raise NotImplementedError("save_model_artifact is not yet implemented.")


def run_training_pipeline(
    features_path: Path,
    model_output_path: Path = DEFAULT_MODEL_PATH,
    val_size: float = VALIDATION_SPLIT_SIZE,
    seed: int = RANDOM_SEED,
) -> lgb.LGBMClassifier:
    """
    Orchestrate full training cycle: data loading, splitting, training, and artifact persistence.

    Args:
        features_path: Path to engineered training features.
        model_output_path: Destination path for trained model artifact.
        val_size: Validation split fraction.
        seed: Random seed for reproducibility.

    Returns:
        Trained LGBMClassifier instance.
    """
    # TODO: Coordinate pipeline steps and return trained model.
    raise NotImplementedError("run_training_pipeline is not yet implemented.")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for model training.

    Args:
        args: Optional list of argument strings. Defaults to sys.argv[1:].

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train LightGBM binary classifier for entity resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        required=True,
        help="Path to precomputed training features file (TSV/CSV/Parquet).",
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Destination path to save trained model artifact.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=VALIDATION_SPLIT_SIZE,
        help="Fraction of training data reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for model training and data splitting.",
    )
    return parser.parse_args(args)


def main() -> None:
    """Entrypoint CLI execution for train.py."""
    args = parse_args()
    seed_everything(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Starting entity resolution model training...")
    logger.info("Features source: %s", args.features_path)
    logger.info("Target model path: %s", args.model_output_path)

    # TODO: Run training pipeline via run_training_pipeline.
    raise NotImplementedError("CLI execution for train.py is not yet implemented.")


if __name__ == "__main__":
    main()
