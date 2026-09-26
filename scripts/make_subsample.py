"""
Stratified Subsampling Script for Fast Local ML Pipeline Iteration.

Creates a representative, stratified sample of the training dataset:
1. Samples ~20,000 source1_entity_id records preserving the exact match-count
   distribution from EDA (~5.6% singletons, ~5.4% 1-match, ~17% 2-match, ~72% 3+ match).
2. Pulls all ground-truth matched source2 and source3 records referenced by the sampled entities.
3. Pulls a random sample of additional unmatched source2 and source3 records (~5x the matched count)
   to ensure realistic negative noise for blocking and feature generation.
4. Outputs all 4 TSVs to data/dataset/train_sample/ maintaining exact original schemas.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RANDOM_SEED,
    TRAIN_DATA_DIR,
    TRAIN_SAMPLE_DATA_DIR,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified subsample of training data for fast iteration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20_000,
        help="Number of source1 entities to sample.",
    )
    parser.add_argument(
        "--noise-factor",
        type=float,
        default=5.0,
        help="Multiplier of unmatched records relative to matched records for source2 and source3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible stratified sampling.",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=TRAIN_DATA_DIR,
        help="Source directory containing full train TSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TRAIN_SAMPLE_DATA_DIR,
        help="Destination directory for subsampled TSV files.",
    )
    return parser.parse_args()


def make_subsample(
    src_dir: Path,
    out_dir: Path,
    n_samples: int = 20_000,
    noise_factor: float = 5.0,
    seed: int = RANDOM_SEED,
) -> None:
    t0 = time.time()
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING STRATIFIED TRAINING SUBSAMPLE")
    print("=" * 80)
    print(f"Source Directory : {src_dir}")
    print(f"Output Directory : {out_dir}")
    print(f"Target S1 Count  : {n_samples:,d}")
    print(f"Noise Factor     : {noise_factor:.1f}x unmatched records")
    print(f"Random Seed      : {seed}\n")

    # 1. Load Ground Truth and Source 1
    print("Step 1: Loading train_ground_truth.tsv and train_source1.tsv...")
    gt_df = pl.read_csv(
        src_dir / "train_ground_truth.tsv",
        separator="\t",
        infer_schema_length=0,
    )
    s1_df = pl.read_csv(
        src_dir / "train_source1.tsv",
        separator="\t",
        infer_schema_length=0,
    )

    print(f"  • Full Source 1 Rows     : {len(s1_df):,d}")
    print(f"  • Full Ground Truth Rows : {len(gt_df):,d}")

    # Compute match count per source1_entity_id
    # Split matched_entity_ids on comma; if empty or null/none -> 0 matches
    gt_s1_ids = gt_df["source1_entity_id"].to_list()
    gt_matched_strs = gt_df["matched_entity_ids"].to_list()

    match_counts: dict[str, int] = {}
    null_vals = {"", "none", "nan", "null", "<null>", "<blank>"}

    for s1_id, matched_str in zip(gt_s1_ids, gt_matched_strs):
        s1_clean = str(s1_id).strip()
        m_clean = str(matched_str).strip()
        if not m_clean or m_clean.lower() in null_vals:
            match_counts[s1_clean] = 0
        else:
            items = [item.strip() for item in m_clean.split(",") if item.strip()]
            match_counts[s1_clean] = len(items)

    all_s1_ids = s1_df["entity_id"].to_list()
    # Default to 0 if not present in GT
    s1_match_counts = np.array([match_counts.get(s, 0) for s in all_s1_ids])

    # Assign strata: 0 (singletons), 1 (1 match), 2 (2 matches), 3 (3+ matches)
    strata = np.where(
        s1_match_counts == 0,
        0,
        np.where(
            s1_match_counts == 1,
            1,
            np.where(s1_match_counts == 2, 2, 3),
        ),
    )

    strata_labels = {
        0: "0 matches (singletons)",
        1: "1 match",
        2: "2 matches",
        3: "3+ matches",
    }

    full_counts = {k: int((strata == k).sum()) for k in range(4)}
    full_total = len(all_s1_ids)

    print("\n[Full Dataset Match Distribution]")
    for k in range(4):
        cnt = full_counts[k]
        pct = (cnt / full_total) * 100
        print(f"  • {strata_labels[k]:<25}: {cnt:>10,d} ({pct:>5.2f}%)")

    # Stratified sampling of S1 IDs
    sampled_indices: list[int] = []
    target_sampled_counts: dict[int, int] = {}

    for k in range(4):
        k_indices = np.where(strata == k)[0]
        # Target count proportional to full dataset stratum weight
        target_k = int(round(n_samples * (full_counts[k] / full_total)))
        target_k = max(1, min(target_k, len(k_indices)))
        target_sampled_counts[k] = target_k

    # Adjust rounding differences to match exact n_samples
    diff = n_samples - sum(target_sampled_counts.values())
    target_sampled_counts[3] += diff

    for k in range(4):
        k_indices = np.where(strata == k)[0]
        chosen = rng.choice(k_indices, size=target_sampled_counts[k], replace=False)
        sampled_indices.extend(chosen)

    sampled_indices = np.array(sampled_indices)
    rng.shuffle(sampled_indices)

    sampled_s1_ids = [all_s1_ids[i] for i in sampled_indices]
    sampled_s1_set = set(sampled_s1_ids)

    # Filter Source 1 and Ground Truth
    sample_s1_df = s1_df.filter(pl.col("entity_id").is_in(sampled_s1_set))
    sample_gt_df = gt_df.filter(pl.col("source1_entity_id").is_in(sampled_s1_set))

    # Print sample distribution verification
    sample_match_counts = [match_counts.get(s, 0) for s in sampled_s1_ids]
    sample_strata = np.where(
        np.array(sample_match_counts) == 0,
        0,
        np.where(
            np.array(sample_match_counts) == 1,
            1,
            np.where(np.array(sample_match_counts) == 2, 2, 3),
        ),
    )

    print(f"\n[Subsampled Dataset Match Distribution ({len(sampled_s1_ids):,d} entities)]")
    for k in range(4):
        cnt = int((sample_strata == k).sum())
        pct = (cnt / len(sampled_s1_ids)) * 100
        full_pct = (full_counts[k] / full_total) * 100
        print(f"  • {strata_labels[k]:<25}: {cnt:>6,d} ({pct:>5.2f}% vs Full: {full_pct:>5.2f}%)")

    # Step 2: Extract all referenced ground truth target entity IDs
    print("\nStep 2: Extracting referenced ground-truth matches for Source 2 and Source 3...")
    matched_s2_ids: set[str] = set()
    matched_s3_ids: set[str] = set()

    for m_str in sample_gt_df["matched_entity_ids"].to_list():
        m_str = str(m_str).strip()
        if not m_str or m_str.lower() in null_vals:
            continue
        for item in m_str.split(","):
            item = item.strip()
            if item.startswith("S2-"):
                matched_s2_ids.add(item)
            elif item.startswith("S3-"):
                matched_s3_ids.add(item)

    print(f"  • True Matched Source 2 IDs : {len(matched_s2_ids):,d}")
    print(f"  • True Matched Source 3 IDs : {len(matched_s3_ids):,d}")

    # Step 3: Load, Filter, and Sample Source 2
    print("\nStep 3: Sampling Source 2 (Matched + 5x Unmatched Noise)...")
    s2_df = pl.read_csv(
        src_dir / "train_source2.tsv",
        separator="\t",
        infer_schema_length=0,
    )
    s2_matched_df = s2_df.filter(pl.col("entity_id").is_in(matched_s2_ids))
    s2_unmatched_df = s2_df.filter(~pl.col("entity_id").is_in(matched_s2_ids))

    n_s2_unmatched = int(round(len(matched_s2_ids) * noise_factor))
    n_s2_unmatched = min(n_s2_unmatched, len(s2_unmatched_df))

    s2_unmatched_indices = rng.choice(len(s2_unmatched_df), size=n_s2_unmatched, replace=False)
    s2_unmatched_sampled_df = s2_unmatched_df[s2_unmatched_indices]

    sample_s2_df = pl.concat([s2_matched_df, s2_unmatched_sampled_df]).sample(
        fraction=1.0, shuffle=True, seed=seed
    )
    print(f"  • Matched S2 Rows    : {len(s2_matched_df):,d}")
    print(f"  • Unmatched S2 Noise : {len(s2_unmatched_sampled_df):,d}")
    print(f"  • Total Subsample S2 : {len(sample_s2_df):,d}")

    # Step 4: Load, Filter, and Sample Source 3
    print("\nStep 4: Sampling Source 3 (Matched + 5x Unmatched Noise)...")
    s3_df = pl.read_csv(
        src_dir / "train_source3.tsv",
        separator="\t",
        infer_schema_length=0,
    )
    s3_matched_df = s3_df.filter(pl.col("entity_id").is_in(matched_s3_ids))
    s3_unmatched_df = s3_df.filter(~pl.col("entity_id").is_in(matched_s3_ids))

    n_s3_unmatched = int(round(len(matched_s3_ids) * noise_factor))
    n_s3_unmatched = min(n_s3_unmatched, len(s3_unmatched_df))

    s3_unmatched_indices = rng.choice(len(s3_unmatched_df), size=n_s3_unmatched, replace=False)
    s3_unmatched_sampled_df = s3_unmatched_df[s3_unmatched_indices]

    sample_s3_df = pl.concat([s3_matched_df, s3_unmatched_sampled_df]).sample(
        fraction=1.0, shuffle=True, seed=seed
    )
    print(f"  • Matched S3 Rows    : {len(s3_matched_df):,d}")
    print(f"  • Unmatched S3 Noise : {len(s3_unmatched_sampled_df):,d}")
    print(f"  • Total Subsample S3 : {len(sample_s3_df):,d}")

    # Step 5: Write all 4 subsampled files
    print("\nStep 5: Writing subsampled TSVs to data/dataset/train_sample/...")
    sample_s1_df.write_csv(out_dir / "train_source1.tsv", separator="\t")
    sample_s2_df.write_csv(out_dir / "train_source2.tsv", separator="\t")
    sample_s3_df.write_csv(out_dir / "train_source3.tsv", separator="\t")
    sample_gt_df.write_csv(out_dir / "train_ground_truth.tsv", separator="\t")

    t1 = time.time()

    print("\n" + "=" * 80)
    print("STRATIFIED SUBSAMPLE COMPLETE")
    print("=" * 80)
    print(f"Created in {t1 - t0:.2f}s:")
    print(f"  1. train_source1.tsv      : {len(sample_s1_df):>10,d} rows")
    print(f"  2. train_source2.tsv      : {len(sample_s2_df):>10,d} rows")
    print(f"  3. train_source3.tsv      : {len(sample_s3_df):>10,d} rows")
    print(f"  4. train_ground_truth.tsv : {len(sample_gt_df):>10,d} rows")
    print(f"Target location             : {out_dir}")
    print("=" * 80)


def main() -> None:
    args = parse_args()
    make_subsample(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        noise_factor=args.noise_factor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
