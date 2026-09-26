"""
Configuration constants and global path definitions for the entity resolution pipeline.

Ensures reproducibility across stages by enforcing a single source of truth for
random seeds, directory locations, filenames, and default hyperparameters.
"""

from pathlib import Path
import os
import random
import numpy as np

# ==============================================================================
# Global Reproducibility
# ==============================================================================
RANDOM_SEED: int = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """
    Seed random number generators across Python and NumPy for reproducible runs.

    Args:
        seed: Integer random seed value. Defaults to RANDOM_SEED (42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ==============================================================================
# Project Root & Directory Paths
# ==============================================================================
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Dataset splits (TSVs)
DATASET_DIR: Path = DATA_DIR / "dataset"
TRAIN_DATA_DIR: Path = DATASET_DIR / "train"
TEST_DATA_DIR: Path = DATASET_DIR / "test"

# Fallback flat split directories under data/
FALLBACK_TRAIN_DATA_DIR: Path = DATA_DIR / "train"
FALLBACK_TEST_DATA_DIR: Path = DATA_DIR / "test"

# Canonical Output File Paths
CANDIDATE_PAIRS_TSV: Path = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_TSV: Path = OUTPUT_DIR / "matching_results.tsv"

# Model Checkpoint Artifacts
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "lgbm_entity_resolution.joblib"


def get_split_dir(split: str) -> Path:
    """
    Resolve the input directory path for a given split ('train' or 'test').

    Checks both `data/dataset/<split>` and `data/<split>`. If neither exists,
    returns `data/dataset/<split>`.

    Args:
        split: The dataset split name ('train' or 'test').

    Returns:
        Path: Path object pointing to the split directory.

    Raises:
        ValueError: If split is not 'train' or 'test'.
    """
    split = split.lower().strip()
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")

    preferred = TRAIN_DATA_DIR if split == "train" else TEST_DATA_DIR
    fallback = FALLBACK_TRAIN_DATA_DIR if split == "train" else FALLBACK_TEST_DATA_DIR

    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    return preferred


# ==============================================================================
# Blocking & Feature Generation Constants
# ==============================================================================
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 3)
TFIDF_MAX_FEATURES: int = 50_000
TFIDF_MIN_DF: int = 2
DEFAULT_K_NEIGHBORS: int = 20
DEFAULT_MAX_DISTANCE: float = 0.6  # Cosine distance cutoff for candidate generation

# ==============================================================================
# Modeling & Evaluation Constants
# ==============================================================================
EVAL_F_BETA: float = 0.5  # Prioritize precision over recall in entity resolution
DEFAULT_DECISION_THRESHOLD: float = 0.5
VALIDATION_SPLIT_SIZE: float = 0.2
