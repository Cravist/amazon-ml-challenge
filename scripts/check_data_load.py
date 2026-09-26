"""
Data loading and schema validation script.

Validates that all 7 TSV files (train_source1/2/3, train_ground_truth,
test_source1/2/3) load cleanly using path constants anchored to the project root,
satisfy exact schema requirements, and have valid entity_id prefixes.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path so imports work regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_DATA_DIR, TEST_DATA_DIR, TRAIN_DIR, TEST_DIR


EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def check_source_file(
    file_path: Path, expected_prefix: str, split_name: str
) -> pd.DataFrame:
    """
    Load and validate a source TSV file (source1, source2, or source3).
    """
    print("=" * 80)
    print(f"CHECKING: {file_path.name} (Split: {split_name})")
    print(f"Path    : {file_path}")
    print("=" * 80)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Please verify that the dataset exists at {file_path.parent}."
        )

    # 1. Load TSV with explicit sep="\t" and dtype=str
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    row_count = len(df)
    columns = list(df.columns)

    # 2. Print file info
    print(f"File Name : {file_path.name}")
    print(f"Row Count : {row_count:,d}")
    print(f"Columns   : {columns}")

    # Assert exact columns
    assert columns == EXPECTED_SOURCE_COLUMNS, (
        f"Schema validation failed for '{file_path.name}'!\n"
        f"  Found columns   : {columns}\n"
        f"  Expected columns: {EXPECTED_SOURCE_COLUMNS}"
    )
    print("  ✓ Column schema matches exactly.")

    # 3. Print first 3 rows
    print("\n--- First 3 Rows ---")
    for idx, row in df.head(3).iterrows():
        print(f"Row {idx}:")
        print(f"  entity_id       : {row['entity_id']}")
        print(f"  business_name   : {row['business_name']}")
        print(f"  business_address: {row['business_address']}")
        print(f"  country         : {row['country']}")
    print("--------------------")

    # 4. Check entity_id prefix
    entity_ids = df["entity_id"].astype(str)
    invalid_mask = ~entity_ids.str.startswith(expected_prefix)
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        sample_invalid = entity_ids[invalid_mask].head(10).tolist()
        raise AssertionError(
            f"Prefix check failed for '{file_path.name}'!\n"
            f"Expected all entity_id values to start with '{expected_prefix}'.\n"
            f"Found {invalid_count:,d} invalid IDs out of {row_count:,d} total rows.\n"
            f"Sample invalid IDs: {sample_invalid}"
        )

    print(f"  ✓ All {row_count:,d} entity IDs correctly start with prefix '{expected_prefix}'.\n")
    return df


def check_ground_truth_file(file_path: Path) -> pd.DataFrame:
    """
    Load and validate train_ground_truth.tsv.
    """
    print("=" * 80)
    print(f"CHECKING: {file_path.name} (Ground Truth)")
    print(f"Path    : {file_path}")
    print("=" * 80)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Please verify that train_ground_truth.tsv exists at {file_path.parent}."
        )

    # 1. Load TSV with explicit sep="\t" and dtype=str
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    row_count = len(df)
    columns = list(df.columns)

    # 2. Print file info
    print(f"File Name : {file_path.name}")
    print(f"Row Count : {row_count:,d}")
    print(f"Columns   : {columns}")

    # Assert exact columns
    assert columns == EXPECTED_GROUND_TRUTH_COLUMNS, (
        f"Schema validation failed for '{file_path.name}'!\n"
        f"  Found columns   : {columns}\n"
        f"  Expected columns: {EXPECTED_GROUND_TRUTH_COLUMNS}"
    )
    print("  ✓ Column schema matches exactly.")

    # 3. Print first 3 rows
    print("\n--- First 3 Rows ---")
    for idx, row in df.head(3).iterrows():
        print(f"Row {idx}:")
        print(f"  source1_entity_id : {row['source1_entity_id']}")
        print(f"  matched_entity_ids: {row['matched_entity_ids']}")
    print("--------------------")

    # Verify source1_entity_id prefix starts with S1-
    s1_ids = df["source1_entity_id"].astype(str)
    invalid_mask = ~s1_ids.str.startswith("S1-")
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        sample_invalid = s1_ids[invalid_mask].head(10).tolist()
        raise AssertionError(
            f"Prefix check failed for '{file_path.name}' source1_entity_id column!\n"
            f"Expected all source1_entity_id values to start with 'S1-'.\n"
            f"Found {invalid_count:,d} invalid IDs out of {row_count:,d} total rows.\n"
            f"Sample invalid IDs: {sample_invalid}"
        )

    print(f"  ✓ All {row_count:,d} source1_entity_id values correctly start with prefix 'S1-'.\n")
    return df


def main() -> None:
    print("\n" + "=" * 80)
    print("VERIFYING DATASET PATH CONSTANTS AND INTEGRITY")
    print("=" * 80)
    print(f"Project Root   : {PROJECT_ROOT}")
    print(f"TRAIN_DATA_DIR : {TRAIN_DATA_DIR}")
    print(f"TEST_DATA_DIR  : {TEST_DATA_DIR}")
    print(f"TRAIN_DIR      : {TRAIN_DIR}")
    print(f"TEST_DIR       : {TEST_DIR}\n")

    # Source files configuration: (filename, split_dir, prefix, split_name)
    source_files = [
        ("train_source1.tsv", TRAIN_DATA_DIR, "S1-", "train"),
        ("train_source2.tsv", TRAIN_DATA_DIR, "S2-", "train"),
        ("train_source3.tsv", TRAIN_DATA_DIR, "S3-", "train"),
        ("test_source1.tsv", TEST_DATA_DIR, "S1-", "test"),
        ("test_source2.tsv", TEST_DATA_DIR, "S2-", "test"),
        ("test_source3.tsv", TEST_DATA_DIR, "S3-", "test"),
    ]

    # Validate 6 source files
    for filename, directory, prefix, split_name in source_files:
        filepath = directory / filename
        check_source_file(filepath, expected_prefix=prefix, split_name=split_name)

    # Validate train_ground_truth.tsv
    gt_path = TRAIN_DATA_DIR / "train_ground_truth.tsv"
    check_ground_truth_file(gt_path)

    # Final unambiguous success banner
    print("=" * 80)
    print("ALL CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
