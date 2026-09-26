"""
Exploratory Data Analysis (EDA) script for entity resolution training data.

Loads train_source1.tsv, train_source2.tsv, train_source3.tsv, and train_ground_truth.tsv
using explicit tab delimiters, computing:
1. Row counts per source file, and per country within each source.
2. Distribution of number of matches per source1_entity_id in train_ground_truth.tsv
   (singletons/0 matches, 1, 2, 3+ matches) reported as counts and percentages.
3. Breakdown of ground-truth matches originating from source2 vs source3 vs both.
4. 25 random matched pairs printed side-by-side (business_name and business_address)
   for visual inspection of noise, abbreviations, formatting, and typos.
5. String length distributions for business_name and business_address, overall and by country.
6. Exact duplicate business_name + country combinations within each single source.

Saves visualization histograms and summary figures to output/eda/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to sys.path if run directly as python src/eda.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import OUTPUT_DIR, RANDOM_SEED, seed_everything
except ImportError:
    OUTPUT_DIR = PROJECT_ROOT / "output"
    RANDOM_SEED = 42

    def seed_everything(seed: int = RANDOM_SEED) -> None:
        np.random.seed(seed)

logger = logging.getLogger("eda")


# ==============================================================================
# File Loading & Schema Detection Helpers
# ==============================================================================

def resolve_train_data_dir(explicit_dir: Path | None = None) -> Path:
    """
    Search standard candidate locations for dataset/train TSV files.

    Candidate paths checked:
    1. Explicit path passed via CLI
    2. PROJECT_ROOT / "dataset" / "train"
    3. PROJECT_ROOT / "data" / "dataset" / "train"
    4. PROJECT_ROOT / "data" / "train"
    5. Path("dataset/train")
    6. Path("data/dataset/train")
    """
    candidates: list[Path] = []
    if explicit_dir is not None:
        candidates.append(Path(explicit_dir).resolve())

    candidates.extend([
        PROJECT_ROOT / "dataset" / "train",
        PROJECT_ROOT / "data" / "dataset" / "train",
        PROJECT_ROOT / "data" / "train",
        Path.cwd() / "dataset" / "train",
        Path.cwd() / "data" / "dataset" / "train",
        Path.cwd() / "data" / "train",
    ])

    for cand in candidates:
        if cand.exists() and (cand / "train_source1.tsv").exists():
            return cand

    # If not found, return the first valid or candidate path
    return candidates[0] if candidates else PROJECT_ROOT / "dataset" / "train"


def load_tsv_file(file_path: Path, description: str = "") -> pd.DataFrame:
    """
    Load a TSV file with explicit tab separation (sep="\\t").

    Args:
        file_path: Path to the TSV file.
        description: Informative label for logging.

    Returns:
        pd.DataFrame loaded from TSV.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Expected {description} file at: {file_path}\n"
            f"Please verify that the file exists."
        )

    logger.info("Loading %s from: %s", description or file_path.name, file_path)
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    return df


# Exact expected schemas from problem specification
EXPECTED_SOURCE_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)


def validate_source_dataframe(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Validate that a source DataFrame contains the exact expected schema.

    Expected columns: entity_id, business_name, business_address, country.

    Args:
        df: Input DataFrame loaded from TSV.
        filename: Name of the source file for error reporting.

    Returns:
        pd.DataFrame with exact schema and cleaned string values.

    Raises:
        AssertionError: If any expected column is missing from df.
    """
    missing = [col for col in EXPECTED_SOURCE_COLUMNS if col not in df.columns]
    assert not missing, (
        f"Schema validation error in {filename}: missing expected column(s) {missing}.\n"
        f"Expected exact columns: {list(EXPECTED_SOURCE_COLUMNS)}\n"
        f"Found columns: {list(df.columns)}"
    )

    out = df[list(EXPECTED_SOURCE_COLUMNS)].copy()
    out["entity_id"] = out["entity_id"].astype(str).str.strip()
    out["business_name"] = out["business_name"].astype(str).str.strip()
    out["business_address"] = out["business_address"].astype(str).str.strip()
    out["country"] = out["country"].astype(str).str.strip().str.upper()
    return out


# ==============================================================================
# EDA Tasks
# ==============================================================================

def analyze_row_counts_and_countries(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame
) -> pd.DataFrame:
    """
    Task 1: Row counts per source file, and per country within each source.
    """
    print("\n" + "=" * 80)
    print("1. ROW COUNTS & COUNTRY DISTRIBUTION")
    print("=" * 80)

    counts = {
        "Source 1 (train_source1.tsv)": len(s1),
        "Source 2 (train_source2.tsv)": len(s2),
        "Source 3 (train_source3.tsv)": len(s3),
    }

    print("\n[Total Record Counts]")
    for src, cnt in counts.items():
        print(f"  • {src:<32}: {cnt:>10,d} rows")

    # Country breakdown per source
    c1 = s1["country"].value_counts(dropna=False).rename("Source 1")
    c2 = s2["country"].value_counts(dropna=False).rename("Source 2")
    c3 = s3["country"].value_counts(dropna=False).rename("Source 3")

    country_df = pd.concat([c1, c2, c3], axis=1).fillna(0).astype(int)
    country_df["Total"] = country_df.sum(axis=1)
    country_df = country_df.sort_values(by="Total", ascending=False)

    print("\n[Country Breakdown Across Sources]")
    print(f"{'Country':<12} | {'Source 1':>12} | {'Source 2':>12} | {'Source 3':>12} | {'Total':>12}")
    print("-" * 70)
    for country, row in country_df.iterrows():
        c_label = str(country) if str(country).strip() else "(Blank/Missing)"
        print(
            f"{c_label:<12} | {row['Source 1']:>12,d} | {row['Source 2']:>12,d} | "
            f"{row['Source 3']:>12,d} | {row['Total']:>12,d}"
        )

    return country_df


EXPECTED_GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    "source1_entity_id",
    "matched_entity_ids",
)


def parse_ground_truth(
    gt_df: pd.DataFrame, filename: str = "train_ground_truth.tsv"
) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse train_ground_truth.tsv using exact expected column names:
    'source1_entity_id' and 'matched_entity_ids'.

    The 'matched_entity_ids' column contains a single comma-separated string
    of matched entity IDs prefixed with S2- or S3- (e.g., 'S2_101, S3_205').

    Args:
        gt_df: Ground truth DataFrame loaded from TSV.
        filename: Name of the ground truth file for error reporting.

    Returns:
        Tuple of (standardized_pairs_df, list_of_all_s1_ids_in_gt),
        where standardized_pairs_df has columns:
        ['source1_entity_id', 'target_source', 'target_entity_id'].

    Raises:
        AssertionError: If either source1_entity_id or matched_entity_ids is missing.
    """
    missing = [col for col in EXPECTED_GROUND_TRUTH_COLUMNS if col not in gt_df.columns]
    assert not missing, (
        f"Schema validation error in {filename}: missing expected ground-truth column(s) {missing}.\n"
        f"Expected exact columns: {list(EXPECTED_GROUND_TRUTH_COLUMNS)}\n"
        f"Found columns: {list(gt_df.columns)}"
    )

    s1_vals = gt_df["source1_entity_id"].astype(str).str.strip().values
    raw_matched_vals = gt_df["matched_entity_ids"].astype(str).str.strip().values

    pairs_s1: list[str] = []
    pairs_src: list[str] = []
    pairs_tgt: list[str] = []

    for s1_val, raw_matched in zip(s1_vals, raw_matched_vals):
        if not s1_val or not raw_matched or raw_matched.lower() in {"none", "nan", "null", ""}:
            continue

        matched_items = raw_matched.split(",")
        for item in matched_items:
            item = item.strip()
            if not item:
                continue

            # Determine source origin by S2- / S3- prefix
            item_lower = item.lower()
            if item_lower.startswith("s2") or "source2" in item_lower:
                target_source = "source2"
            elif item_lower.startswith("s3") or "source3" in item_lower:
                target_source = "source3"
            else:
                target_source = "unknown"

            pairs_s1.append(s1_val)
            pairs_src.append(target_source)
            pairs_tgt.append(item)

    standardized_pairs = pd.DataFrame({
        "source1_entity_id": pairs_s1,
        "target_source": pairs_src,
        "target_entity_id": pairs_tgt,
    })
    all_s1_in_gt = [s for s in dict.fromkeys(s1_vals) if s]
    return standardized_pairs, all_s1_in_gt




def analyze_match_distribution(
    s1_df: pd.DataFrame, gt_pairs: pd.DataFrame
) -> pd.Series:
    """
    Task 2: Distribution of number of matches per source1_entity_id.
    Includes singletons (0 matches), 1, 2, 3+ matches, reporting percentages.
    """
    print("\n" + "=" * 80)
    print("2. DISTRIBUTION OF MATCHES PER SOURCE1 ENTITY ID (BASELINE REWARD)")
    print("=" * 80)

    all_s1_ids = pd.Series(s1_df["entity_id"].unique(), name="entity_id")
    total_s1_count = len(all_s1_ids)

    # Count occurrences in ground-truth pairs
    if not gt_pairs.empty:
        match_counts_per_entity = (
            gt_pairs.groupby("source1_entity_id")["target_entity_id"]
            .nunique()
            .reindex(all_s1_ids, fill_value=0)
        )
    else:
        match_counts_per_entity = pd.Series(0, index=all_s1_ids)

    # Categorize into 0 (singletons), 1, 2, 3+
    def bucket_matches(val: int) -> str:
        if val == 0:
            return "0 matches (singletons)"
        elif val == 1:
            return "1 match"
        elif val == 2:
            return "2 matches"
        else:
            return "3+ matches"

    bucketed = match_counts_per_entity.map(bucket_matches)
    ordered_buckets = ["0 matches (singletons)", "1 match", "2 matches", "3+ matches"]
    counts = bucketed.value_counts().reindex(ordered_buckets, fill_value=0)
    percentages = (counts / total_s1_count) * 100.0

    print(f"\nTotal unique Source 1 entities: {total_s1_count:,d}\n")
    print(f"{'Match Category':<25} | {'Count':>12} | {'Percentage':>12}")
    print("-" * 55)
    for cat in ordered_buckets:
        cnt = counts[cat]
        pct = percentages[cat]
        print(f"{cat:<25} | {cnt:>12,d} | {pct:>11.2f}%")

    singleton_pct = percentages.get("0 matches (singletons)", 0.0)
    print("-" * 55)
    print(
        f"\n💡 Baseline Reward Insight:\n"
        f"   {singleton_pct:.2f}% of Source 1 entities have NO matching counterpart in the dataset.\n"
        f"   Predicting 'no match' (abstention) yields an inherent baseline accuracy of {singleton_pct:.2f}%."
    )

    return match_counts_per_entity


def analyze_match_source_breakdown(
    s1_df: pd.DataFrame, gt_pairs: pd.DataFrame
) -> dict[str, int]:
    """
    Task 3: Breakdown of ground-truth matches originating from source2 vs source3 vs both.
    """
    print("\n" + "=" * 80)
    print("3. GROUND-TRUTH MATCH SOURCE BREAKDOWN (SOURCE 2 vs SOURCE 3 vs BOTH)")
    print("=" * 80)

    all_s1_ids = set(s1_df["entity_id"].unique())
    total_s1 = len(all_s1_ids)

    if gt_pairs.empty:
        print("Ground truth pairs table is empty.")
        return {}

    # Pivot / aggregate target sources per source1_entity_id
    src_by_s1 = gt_pairs.groupby("source1_entity_id")["target_source"].unique().to_dict()

    s2_only = 0
    s3_only = 0
    both = 0
    neither = 0

    for s1_id in all_s1_ids:
        srcs = set(src_by_s1.get(s1_id, []))
        has_s2 = "source2" in srcs
        has_s3 = "source3" in srcs

        if has_s2 and has_s3:
            both += 1
        elif has_s2:
            s2_only += 1
        elif has_s3:
            s3_only += 1
        else:
            neither += 1

    matched_total = s2_only + s3_only + both

    print(f"\n[Entity-Level Breakdown across {total_s1:,d} Source 1 Entities]")
    print(f"{'Source Category':<25} | {'Source 1 Entities':>18} | {'% of S1 Population':>20}")
    print("-" * 69)
    print(f"{'Matched to Source 2 ONLY':<25} | {s2_only:>18,d} | {(s2_only / total_s1 * 100):>19.2f}%")
    print(f"{'Matched to Source 3 ONLY':<25} | {s3_only:>18,d} | {(s3_only / total_s1 * 100):>19.2f}%")
    print(f"{'Matched to BOTH S2 & S3':<25} | {both:>18,d} | {(both / total_s1 * 100):>19.2f}%")
    print(f"{'No Match (Neither)':<25} | {neither:>18,d} | {(neither / total_s1 * 100):>19.2f}%")
    print("-" * 69)
    print(f"{'Total Entities with Matches':<25} | {matched_total:>18,d} | {(matched_total / total_s1 * 100):>19.2f}%")

    # Pair counts
    pair_counts = gt_pairs["target_source"].value_counts().to_dict()
    print("\n[Pair-Level Matches Breakdown]")
    for src, cnt in pair_counts.items():
        print(f"  • Matched pairs targeting {src:<10}: {cnt:>10,d}")

    return {"s2_only": s2_only, "s3_only": s3_only, "both": both, "neither": neither}


def sample_and_print_matched_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_pairs: pd.DataFrame,
    n_samples: int = 25,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Task 4: Sample 25 random matched pairs and display business_name and business_address
    side by side for visual inspection of noise patterns.
    """
    print("\n" + "=" * 80)
    print(f"4. INSPECTION OF {n_samples} RANDOM MATCHED PAIRS (SIDE-BY-SIDE NOISE ANALYSIS)")
    print("=" * 80)

    if gt_pairs.empty:
        print("No ground truth pairs available to sample.")
        return pd.DataFrame()

    # Index sources by entity_id for quick lookups
    s1_idx = s1_df.set_index("entity_id")
    s2_idx = s2_df.set_index("entity_id")
    s3_idx = s3_df.set_index("entity_id")

    sample_pool = gt_pairs.sample(
        n=min(n_samples, len(gt_pairs)), random_state=seed
    ).reset_index(drop=True)

    records: list[dict] = []
    for i, row in sample_pool.iterrows():
        s1_id = row["source1_entity_id"]
        tgt_src = row["target_source"]
        tgt_id = row["target_entity_id"]

        if s1_id not in s1_idx.index:
            raise KeyError(
                f"Data integrity error: source1_entity_id '{s1_id}' in train_ground_truth.tsv "
                f"was not found in train_source1.tsv by exact entity_id match."
            )
        s1_entry = s1_idx.loc[s1_id]
        s1_row = s1_entry.iloc[0] if isinstance(s1_entry, pd.DataFrame) else s1_entry

        if tgt_src == "source2":
            if tgt_id not in s2_idx.index:
                raise KeyError(
                    f"Data integrity error: matched_entity_ids value '{tgt_id}' (source2) in "
                    f"train_ground_truth.tsv was not found in train_source2.tsv by exact entity_id match."
                )
            tgt_entry = s2_idx.loc[tgt_id]
            tgt_row = tgt_entry.iloc[0] if isinstance(tgt_entry, pd.DataFrame) else tgt_entry
        elif tgt_src == "source3":
            if tgt_id not in s3_idx.index:
                raise KeyError(
                    f"Data integrity error: matched_entity_ids value '{tgt_id}' (source3) in "
                    f"train_ground_truth.tsv was not found in train_source3.tsv by exact entity_id match."
                )
            tgt_entry = s3_idx.loc[tgt_id]
            tgt_row = tgt_entry.iloc[0] if isinstance(tgt_entry, pd.DataFrame) else tgt_entry
        else:
            raise ValueError(
                f"Unrecognized target source '{tgt_src}' for matched entity ID '{tgt_id}'. "
                f"Expected entity ID to begin with S2- or S3- prefix."
            )

        records.append({
            "pair_num": i + 1,
            "s1_id": s1_id,
            "target_source": tgt_src,
            "target_id": tgt_id,
            "s1_name": str(s1_row["business_name"]),
            "target_name": str(tgt_row["business_name"]),
            "s1_addr": str(s1_row["business_address"]),
            "target_addr": str(tgt_row["business_address"]),
            "country": str(s1_row["country"]) or str(tgt_row["country"]),
        })

    # Pretty print side by side
    sep_line = "-" * 80
    for r in records:
        print(f"\n[Pair #{r['pair_num']:02d}] Source 1 ({r['s1_id']}) ⟷ {r['target_source'].upper()} ({r['target_id']}) | Country: {r['country']}")
        print(f"  NAME S1   : {r['s1_name']}")
        print(f"  NAME TGT  : {r['target_name']}")
        print(f"  ADDR S1   : {r['s1_addr']}")
        print(f"  ADDR TGT  : {r['target_addr']}")
        print(sep_line)

    return pd.DataFrame(records)


def analyze_string_length_distributions(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Task 5: String length distributions for business_name and business_address,
    overall and by country.
    """
    print("\n" + "=" * 80)
    print("5. STRING LENGTH DISTRIBUTIONS (OVERALL & BY COUNTRY)")
    print("=" * 80)

    # Combine all records with source tag
    s1_tagged = s1_df.assign(source="source1")
    s2_tagged = s2_df.assign(source="source2")
    s3_tagged = s3_df.assign(source="source3")
    all_records = pd.concat([s1_tagged, s2_tagged, s3_tagged], ignore_index=True)

    all_records["name_len"] = all_records["business_name"].astype(str).str.len()
    all_records["addr_len"] = all_records["business_address"].astype(str).str.len()

    print("\n[Overall Summary Statistics - business_name Length]")
    print(all_records["name_len"].describe().to_frame("Name Length Stats").T.to_string())

    print("\n[Overall Summary Statistics - business_address Length]")
    print(all_records["addr_len"].describe().to_frame("Address Length Stats").T.to_string())

    # By country statistics
    print("\n[Mean & Median String Lengths by Country]")
    country_stats = (
        all_records.groupby("country")
        .agg(
            record_count=("entity_id", "count"),
            name_mean=("name_len", "mean"),
            name_median=("name_len", "median"),
            addr_mean=("addr_len", "mean"),
            addr_median=("addr_len", "median"),
        )
        .sort_values(by="record_count", ascending=False)
    )

    print(
        f"{'Country':<10} | {'Records':>10} | {'Name Mean':>10} | {'Name Med':>10} | "
        f"{'Addr Mean':>10} | {'Addr Med':>10}"
    )
    print("-" * 75)
    for country, r in country_stats.iterrows():
        c_label = str(country) if str(country).strip() else "(Blank)"
        print(
            f"{c_label:<10} | {int(r['record_count']):>10,d} | {r['name_mean']:>10.1f} | "
            f"{r['name_median']:>10.1f} | {r['addr_mean']:>10.1f} | {r['addr_median']:>10.1f}"
        )

    return all_records, country_stats


def check_exact_duplicate_combos(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame
) -> None:
    """
    Task 6: Check for exact duplicate business_name + country combos within a single source
    (helps gauge how much exact-match blocking alone would catch or collide).
    """
    print("\n" + "=" * 80)
    print("6. EXACT DUPLICATE (BUSINESS_NAME + COUNTRY) WITHIN SINGLE SOURCES")
    print("=" * 80)

    sources = [
        ("Source 1", s1_df),
        ("Source 2", s2_df),
        ("Source 3", s3_df),
    ]

    for name, df in sources:
        total_rows = len(df)
        combos = df[["business_name", "country"]].astype(str)

        # Raw exact duplicates
        raw_dupes = combos.duplicated(subset=["business_name", "country"], keep=False)
        num_raw_dupe_rows = raw_dupes.sum()

        # Normalized duplicates (casefolded & stripped whitespace)
        norm_name = df["business_name"].astype(str).str.strip().str.casefold()
        norm_country = df["country"].astype(str).str.strip().str.upper()
        norm_df = pd.DataFrame({"norm_name": norm_name, "norm_country": norm_country})
        norm_dupes = norm_df.duplicated(subset=["norm_name", "norm_country"], keep=False)
        num_norm_dupe_rows = norm_dupes.sum()

        raw_unique_dupe_groups = combos[raw_dupes].drop_duplicates().shape[0]

        print(f"\n[{name} - Total: {total_rows:,d} records]")
        print(f"  • Exact (Raw) Duplicates       : {num_raw_dupe_rows:>7,d} rows ({num_raw_dupe_rows / total_rows * 100:>5.2f}%) across {raw_unique_dupe_groups:,d} distinct (name, country) pairs")
        print(f"  • Case-Insensitive Duplicates  : {num_norm_dupe_rows:>7,d} rows ({num_norm_dupe_rows / total_rows * 100:>5.2f}%)")

        # Top 5 most frequent duplicate name+country combos
        if num_raw_dupe_rows > 0:
            top_combos = (
                combos[raw_dupes]
                .groupby(["business_name", "country"])
                .size()
                .sort_values(ascending=False)
                .head(5)
            )
            print("  • Top Repeated (Name, Country) combos:")
            for (b_name, c_code), freq in top_combos.items():
                print(f"      - '{b_name}' ({c_code}): appears {freq} times")


# ==============================================================================
# Visualization / Plotting Routines
# ==============================================================================

def generate_eda_plots(
    all_records: pd.DataFrame,
    match_counts: pd.Series,
    match_sources: dict[str, int],
    country_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate and save EDA histograms and summary charts to output/eda/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving EDA visualization plots to: %s", output_dir)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. String Length Distributions Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Business Name Length
    for src in ["source1", "source2", "source3"]:
        subset = all_records[all_records["source"] == src]["name_len"]
        axes[0].hist(subset, bins=40, alpha=0.5, label=src, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Business Name Character Length Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Characters")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].set_xlim(0, all_records["name_len"].quantile(0.99) + 10)

    # Business Address Length
    for src in ["source1", "source2", "source3"]:
        subset = all_records[all_records["source"] == src]["addr_len"]
        axes[1].hist(subset, bins=40, alpha=0.5, label=src, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Business Address Character Length Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Characters")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].set_xlim(0, all_records["addr_len"].quantile(0.99) + 10)

    plt.tight_layout()
    plot1_path = output_dir / "string_length_distributions.png"
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved plot: {plot1_path.name}")

    # 2. Match Count per Source 1 Entity ID
    fig, ax = plt.subplots(figsize=(8, 5))
    counts_map = {
        "0 matches\n(Singletons)": (match_counts == 0).sum(),
        "1 match": (match_counts == 1).sum(),
        "2 matches": (match_counts == 2).sum(),
        "3+ matches": (match_counts >= 3).sum(),
    }
    total = len(match_counts)
    labels = list(counts_map.keys())
    values = list(counts_map.values())
    pcts = [v / total * 100 for v in values]

    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], edgecolor="black")
    ax.set_title("Ground-Truth Matches per Source 1 Entity ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Source 1 Entities")

    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.annotate(
            f"{height:,d}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plot2_path = output_dir / "matches_per_entity_distribution.png"
    fig.savefig(plot2_path, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved plot: {plot2_path.name}")

    # 3. Ground Truth Match Source Breakdown
    if match_sources:
        fig, ax = plt.subplots(figsize=(8, 5))
        src_labels = ["Source 2 Only", "Source 3 Only", "Both (S2 & S3)", "No Match (Neither)"]
        src_vals = [
            match_sources.get("s2_only", 0),
            match_sources.get("s3_only", 0),
            match_sources.get("both", 0),
            match_sources.get("neither", 0),
        ]
        s_pcts = [v / total * 100 for v in src_vals]

        bars = ax.bar(src_labels, src_vals, color=["#4C72B0", "#DD8452", "#55A868", "#937860"], edgecolor="black")
        ax.set_title("Ground-Truth Match Provenance (S2 vs S3 vs Both)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count of Source 1 Entities")

        for bar, pct in zip(bars, s_pcts):
            height = bar.get_height()
            ax.annotate(
                f"{height:,d}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plot3_path = output_dir / "match_sources_breakdown.png"
        fig.savefig(plot3_path, dpi=300)
        plt.close(fig)
        print(f"  ✓ Saved plot: {plot3_path.name}")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for eda.py.
    """
    parser = argparse.ArgumentParser(
        description="Run Exploratory Data Analysis (EDA) on entity resolution dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to directory containing train_source1.tsv, train_source2.tsv, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "eda",
        help="Path to directory where EDA plots and summary artifacts will be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Number of matched pairs to sample for noise inspection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args(args)


def main() -> None:
    """
    Main orchestration routine for entity resolution EDA.
    """
    args = parse_args()
    seed_everything(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_dir = resolve_train_data_dir(args.data_dir)
    output_dir = args.output_dir.resolve()

    s1_file = data_dir / "train_source1.tsv"
    s2_file = data_dir / "train_source2.tsv"
    s3_file = data_dir / "train_source3.tsv"
    gt_file = data_dir / "train_ground_truth.tsv"

    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    print(f"Dataset Train Directory : {data_dir}")
    print(f"Output Directory        : {output_dir}")
    print(f"Random Seed             : {args.seed}")

    # Check for file existence
    missing_files = [f.name for f in [s1_file, s2_file, s3_file, gt_file] if not f.exists()]
    if missing_files:
        print(f"\n❌ Error: Missing required TSV file(s) in {data_dir}:")
        for mf in missing_files:
            print(f"   • {mf}")
        print("\nPlease ensure train_source1.tsv, train_source2.tsv, train_source3.tsv,")
        print("and train_ground_truth.tsv are placed in dataset/train/ (or specify --data-dir).")
        sys.exit(1)

    # 1. Load TSVs with sep="\t" explicitly
    df_s1_raw = load_tsv_file(s1_file, "train_source1.tsv")
    df_s2_raw = load_tsv_file(s2_file, "train_source2.tsv")
    df_s3_raw = load_tsv_file(s3_file, "train_source3.tsv")
    df_gt_raw = load_tsv_file(gt_file, "train_ground_truth.tsv")

    # Validate schemas with exact column assertions
    s1_std = validate_source_dataframe(df_s1_raw, "train_source1.tsv")
    s2_std = validate_source_dataframe(df_s2_raw, "train_source2.tsv")
    s3_std = validate_source_dataframe(df_s3_raw, "train_source3.tsv")
    gt_pairs, _ = parse_ground_truth(df_gt_raw, "train_ground_truth.tsv")

    # Execute EDA analysis tasks
    country_df = analyze_row_counts_and_countries(s1_std, s2_std, s3_std)
    match_counts = analyze_match_distribution(s1_std, gt_pairs)
    match_sources = analyze_match_source_breakdown(s1_std, gt_pairs)
    sample_and_print_matched_pairs(s1_std, s2_std, s3_std, gt_pairs, n_samples=args.sample_size, seed=args.seed)
    all_records, _ = analyze_string_length_distributions(s1_std, s2_std, s3_std)
    check_exact_duplicate_combos(s1_std, s2_std, s3_std)

    # Generate and save publication-quality plots
    print("\n" + "=" * 80)
    print("7. GENERATING & SAVING EDA PLOTS")
    print("=" * 80)
    generate_eda_plots(all_records, match_counts, match_sources, country_df, output_dir)

    print("\n" + "=" * 80)
    print("EDA COMPLETE. All outputs and plots saved to:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
