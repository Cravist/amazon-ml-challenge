"""
Text and address normalization utilities for entity resolution.

Provides general-purpose string normalization routines for business entity records
across international domains (US, India, France, etc.) without hardcoding country-specific
parsing rules:
1. normalize_name: Standardizes company names, stripping punctuation (preserving internal hyphens),
   expanding common business abbreviations (corp <-> corporation, pvt <-> private, & <-> and),
   and normalizing whitespace.
2. extract_legal_suffix: Splits off trailing corporate legal designations (e.g. 'inc', 'llc',
   'pvt ltd', 'gmbh', 'sarl') as a distinct feature tuple: (core_name, suffix_or_None).
3. normalize_address: Cleans street addresses, standardizing abbreviations (rd <-> road,
   st <-> street, ave <-> avenue, apt <-> apartment) while preserving landmark phrases
   ('near', 'opposite') and internal hyphens without structured geocoding.
4. tokenize: Simple whitespace tokenization on normalized strings for Jaccard/token-set features.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Sequence

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from src.config import RANDOM_SEED, get_split_dir, seed_everything

logger = logging.getLogger(__name__)

# ==============================================================================
# Abbreviation Dictionaries (General-purpose for US, India, France, etc.)
# ==============================================================================

# Mapping from common business abbreviations to canonical full expansions
NAME_ABBREVIATIONS: dict[str, str] = {
    # Corporate / Legal forms
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "cie": "compagnie",
    # Common industry terms
    "mfg": "manufacturing",
    "univ": "university",
    "intl": "international",
    "assoc": "association",
    "assn": "association",
    "tech": "technology",
    "svcs": "services",
    "svc": "service",
    "serv": "services",
    "dept": "department",
    "div": "division",
    "grp": "group",
    "ind": "industries",
    "inds": "industries",
    "ent": "enterprise",
    "mgmt": "management",
    "comm": "commercial",
    "natl": "national",
    "fed": "federal",
    "distrib": "distribution",
    "lab": "laboratory",
    "labs": "laboratories",
    "sys": "systems",
    "med": "medical",
    "hlth": "health",
    "pharma": "pharmaceuticals",
}

# Trailing legal suffix regex pattern (multi-word patterns evaluated first)
LEGAL_SUFFIX_PATTERN: re.Pattern = re.compile(
    r"(?:[\s,./-]+)"
    r"("
    # Multi-word suffixes
    r"pvt\.?\s+ltd\.?|"
    r"private\s+limited|"
    r"pvt\.?\s+limited|"
    r"co\.?\s+ltd\.?|"
    r"company\s+limited|"
    r"public\s+limited(?:\s+company)?|"
    # Single-word suffixes (US / UK / International)
    r"corp(?:oration)?\.?|"
    r"inc(?:orporated)?\.?|"
    r"l\.?l\.?c\.?|"
    r"l\.?l\.?p\.?|"
    r"ltd\.?|"
    r"limited|"
    r"pvt\.?|"
    r"private|"
    r"plc\.?|"
    r"co(?:mpany)?\.?|"
    # Germany / Continental Europe
    r"gmbh\.?|"
    r"ag\.?|"
    # France / Francophone Europe
    r"s\.?a\.?r\.?l\.?|"
    r"s\.?a\.?s\.?u?\.?|"
    r"s\.?a\.?|"
    r"eurl\.?|"
    r"sci\.?|"
    r"snc\.?|"
    r"cie\.?|"
    r"compagnie|"
    # Other common EU forms
    r"se\.?|"
    r"b\.?v\.?|"
    r"n\.?v\.?|"
    r"s\.?p\.?a\.?|"
    r"s\.?r\.?l\.?"
    r")\.?$",
    re.IGNORECASE,
)

# Canonical mapping for extracted legal suffixes
LEGAL_SUFFIX_CANONICAL: dict[str, str] = {
    "private limited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "company limited": "co ltd",
    "co ltd": "co ltd",
    "public limited company": "plc",
    "public limited": "plc",
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "compagnie": "cie",
}

# Address abbreviation mappings (road types, units, spatial landmark cues)
ADDRESS_ABBREVIATIONS: dict[str, str] = {
    # Road types (US / UK / India / France)
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bld": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
    "way": "way",
    "sq": "square",
    "pl": "place",
    "all": "allee",
    "imp": "impasse",
    "rte": "route",
    "cres": "crescent",
    "ter": "terrace",
    "r": "rue",  # French street abbreviation
    "rue": "rue",
    # Unit / Sub-building designators
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "bat": "batiment",
    "dept": "department",
    "rm": "room",
    "no": "number",
    # Landmark / Spatial relation indicators (kept intact for matching)
    "nr": "near",
    "opp": "opposite",
    "adj": "adjacent",
    "sec": "sector",
    "ph": "phase",
}


# ==============================================================================
# Null & Placeholder Strings
# ==============================================================================
NULL_PLACEHOLDERS: set[str] = {
    "",
    "null",
    "<null>",
    "nan",
    "<nan>",
    "none",
    "<none>",
    "blank",
    "<blank>",
    "n/a",
    "na",
    "undefined",
}


def strip_outer_junk(text: str) -> str:
    """
    Strip leading and trailing junk punctuation and symbol runs (e.g. '--', '<<', '>>', '**')
    while strictly preserving word characters across all Unicode scripts (including Indic
    combining marks/matras like 'ी', 'ो', 'ु', etc.).
    """
    start = 0
    n = len(text)
    while start < n:
        cat = unicodedata.category(text[start])
        if cat[0] in ("L", "M", "N") or text[start] == "_":
            break
        start += 1

    end = n
    while end > start:
        cat = unicodedata.category(text[end - 1])
        if cat[0] in ("L", "M", "N") or text[end - 1] == "_":
            break
        end -= 1

    return text[start:end]


def clean_text(text: str | None, preserve_hyphens: bool = True) -> str:
    """
    Standardize raw text:
    1. Check for null/placeholder strings ("", "<blank>", "<null>", "nan", etc.)
    2. Unicode NFKD decomposition (strips Latin combining accents/diacritics while preserving Indic scripts)
    3. Lowercase transformation
    4. Strip leading and trailing junk punctuation and symbol runs (e.g. "--", "<<", ">>", "**")
    5. Collapse dotted acronyms BEFORE general punctuation removal: S.A. -> sa, L.L.C. -> llc
    6. Ampersand expansion (& -> 'and')
    7. Unicode-aware punctuation stripping (preserving word characters across all scripts and internal hyphens)
    8. Whitespace collapsing and null placeholder check

    Args:
        text: Raw input string or None.
        preserve_hyphens: Whether to preserve internal hyphens flanked by word characters.

    Returns:
        Cleaned, normalized lowercase string with single whitespace separation.
    """
    if not text:
        return ""

    raw = str(text).strip()
    if raw.lower() in NULL_PLACEHOLDERS:
        return ""

    # 1. Unicode NFKD decomposition:
    # Strip Latin/European combining diacritical marks (U+0300 - U+036F)
    # while preserving Indic combining vowel signs/marks (U+0900 - U+0DFF, etc.)
    decomp = unicodedata.normalize("NFKD", raw)
    text_val = "".join(c for c in decomp if not (0x0300 <= ord(c) <= 0x036F))

    # 2. Lowercase transformation
    text_val = text_val.lower()

    # 3. Strip leading/trailing junk punctuation and symbol runs (e.g. "--", "<<", ">>", "**")
    text_val = strip_outer_junk(text_val)

    # 4. Collapse dotted acronyms BEFORE general punctuation removal so they don't
    #    fragment into individual letters: S.A. -> sa, L.L.C. -> llc, S.A.R.L. -> sarl
    text_val = re.sub(
        r"\b[a-z](?:\.[a-z])+\.?",
        lambda m: m.group(0).replace(".", ""),
        text_val,
    )

    # 5. Expand ampersands
    text_val = re.sub(r"&", " and ", text_val)

    # 6. Unicode-aware character filtering (preserves letters L, marks M, numbers N, whitespace Z, and hyphens)
    res: list[str] = []
    for c in text_val:
        cat = unicodedata.category(c)
        if cat[0] in ("L", "M", "N") or cat[0] == "Z" or (preserve_hyphens and c == "-"):
            res.append(c)
        else:
            res.append(" ")
    s = "".join(res)

    # 7. Preserve internal hyphens only if flanked on both sides by word characters
    if preserve_hyphens and "-" in s:
        chars = list(s)
        n = len(chars)
        for i, ch in enumerate(chars):
            if ch == "-":
                left_ok = (i > 0 and (unicodedata.category(chars[i - 1])[0] in ("L", "M", "N") or chars[i - 1] == "_"))
                right_ok = (i < n - 1 and (unicodedata.category(chars[i + 1])[0] in ("L", "M", "N") or chars[i + 1] == "_"))
                if not (left_ok and right_ok):
                    chars[i] = " "
        s = "".join(chars)

    # 8. Collapse multiple whitespaces and final placeholder check
    out = re.sub(r"\s+", " ", s).strip()
    if out.lower() in NULL_PLACEHOLDERS:
        return ""
    return out


def normalize_name(name: str | None) -> str:
    """
    Normalize business / entity name:
    - Lowercase, strip punctuation except internal hyphens.
    - Expand common business abbreviations both directions where safe:
      corp <-> corporation, inc <-> incorporated, ltd <-> limited,
      pvt <-> private, co <-> company, & <-> and, etc.
    - Normalize whitespace.

    Args:
        name: Raw entity name string.

    Returns:
        Normalized canonical name string.
    """
    cleaned = clean_text(name, preserve_hyphens=True)
    if not cleaned:
        return ""

    tokens = cleaned.split()
    normalized_tokens: list[str] = []

    for token in tokens:
        # If token contains an internal hyphen (e.g. high-tech), normalize parts if needed
        if "-" in token:
            parts = token.split("-")
            expanded_parts = [NAME_ABBREVIATIONS.get(p, p) for p in parts]
            normalized_tokens.append("-".join(expanded_parts))
        else:
            normalized_tokens.append(NAME_ABBREVIATIONS.get(token, token))

    return " ".join(normalized_tokens)


def extract_legal_suffix(name: str | None) -> tuple[str, str | None]:
    """
    Split off a trailing legal-entity suffix (corp, inc, llc, ltd, pvt ltd, gmbh, sarl, etc.)
    if present, returning (core_name, suffix_or_None).

    Keeps suffix extraction separate from normalized name since suffix presence/absence
    is a valuable feature for pairwise matching rather than noise to discard.

    Args:
        name: Raw or partially normalized business entity name.

    Returns:
        tuple[str, str | None]: (core_name, suffix_or_None).
        core_name has trailing punctuation/whitespace stripped.
        suffix is canonicalized to lowercase without periods (e.g., 'pvt ltd', 'inc', 'llc').
        If no suffix is found, returns (name.strip(), None).
    """
    if not name:
        return ("", None)

    raw_str = str(name).strip()
    if raw_str.lower() in NULL_PLACEHOLDERS:
        return ("", None)

    # Strip leading/trailing junk punctuation
    raw_str = strip_outer_junk(raw_str).strip()
    if not raw_str:
        return ("", None)

    match = LEGAL_SUFFIX_PATTERN.search(raw_str)
    if not match:
        return (raw_str, None)

    raw_suffix = match.group(1).strip()
    core_name = raw_str[: match.start()].rstrip(" ,.-/")
    core_name = strip_outer_junk(core_name).strip()

    # If removing suffix leaves an empty core name, treat entire name as core name
    if not core_name:
        return (raw_str, None)

    # Standardize suffix (lowercase, remove internal dots, normalize spaces)
    clean_sfx = re.sub(r"\.", "", raw_suffix).lower()
    clean_sfx = re.sub(r"\s+", " ", clean_sfx).strip()
    canonical_suffix = LEGAL_SUFFIX_CANONICAL.get(clean_sfx, clean_sfx)

    return (core_name, canonical_suffix)


def normalize_address(address: str | None) -> str:
    """
    Normalize physical street addresses without external geocoding:
    - Lowercase, normalize common abbreviations (rd <-> road, st <-> street,
      ave <-> avenue, apt <-> apartment).
    - Landmark phrases ('near', 'opposite') and spatial indicators are kept as-is
      to assist pairwise entity matching.
    - Normalize whitespace and punctuation, preserving internal hyphens.

    Args:
        address: Raw address string.

    Returns:
        Cleaned, normalized address string.
    """
    if not address:
        return ""

    # 1. Clean unicode, lowercase, expand ampersands, strip non-internal punctuation
    cleaned = clean_text(address, preserve_hyphens=True)
    if not cleaned:
        return ""

    # 2. Expand common road, unit, and landmark abbreviations
    tokens = cleaned.split()
    normalized_tokens: list[str] = []

    for token in tokens:
        if "-" in token:
            parts = token.split("-")
            expanded_parts = [ADDRESS_ABBREVIATIONS.get(p, p) for p in parts]
            normalized_tokens.append("-".join(expanded_parts))
        else:
            normalized_tokens.append(ADDRESS_ABBREVIATIONS.get(token, token))

    return " ".join(normalized_tokens)


def tokenize(s: str | None) -> list[str]:
    """
    Simple whitespace tokenizer on normalized strings, used for Jaccard and token-set features.

    Args:
        s: Input normalized string.

    Returns:
        List of non-empty string tokens.
    """
    if not s:
        return []
    return s.strip().split()


# Alias for backward compatibility
normalize_business_name = normalize_name


# ==============================================================================
# DataFrame Batch Normalization
# ==============================================================================

def normalize_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    address_col: str = "business_address",
    output_name_col: str = "norm_name",
    output_address_col: str = "norm_address",
    output_suffix_col: str = "legal_suffix",
) -> pd.DataFrame:
    """
    Apply normalization routines across business entity records in a pandas DataFrame.

    Adds:
        - output_name_col: Normalized business name
        - output_address_col: Normalized street address
        - output_suffix_col: Extracted trailing legal entity suffix (or None)

    Args:
        df: Input DataFrame containing entity records.
        name_col: Column name containing business names.
        address_col: Column name containing street addresses.
        output_name_col: Destination column for normalized names.
        output_address_col: Destination column for normalized addresses.
        output_suffix_col: Destination column for extracted legal suffixes.

    Returns:
        Enriched DataFrame with normalized columns.
    """
    out = df.copy()

    if name_col in out.columns:
        out[output_name_col] = out[name_col].astype(str).map(normalize_name)
        extracted = out[name_col].astype(str).map(extract_legal_suffix)
        out[output_suffix_col] = [sfx for _, sfx in extracted]

    if address_col in out.columns:
        out[output_address_col] = out[address_col].astype(str).map(normalize_address)

    return out


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for normalize.py.
    """
    parser = argparse.ArgumentParser(
        description="Run text and address normalization on entity dataset splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split to normalize ('train' or 'test').",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional explicit path to input TSV file. If not set, resolves via --split.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write normalized output file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic behavior.",
    )
    return parser.parse_args(args)


def main() -> None:
    """Entrypoint CLI execution for normalize.py."""
    args = parse_args()
    seed_everything(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_dir = get_split_dir(args.split) if args.input_path is None else args.input_path
    logger.info("Executing normalization on split '%s' from %s", args.split, input_dir)

    # If input is a directory, normalize all source TSVs found
    if input_dir.is_dir():
        for source_tsv in sorted(input_dir.glob("*.tsv")):
            logger.info("Normalizing file: %s", source_tsv.name)
            df = pd.read_csv(source_tsv, sep="\t", dtype=str, keep_default_na=False)
            norm_df = normalize_dataframe(df)
            out_file = (
                args.output_path
                if args.output_path
                else source_tsv.parent / f"normalized_{source_tsv.name}"
            )
            norm_df.to_csv(out_file, sep="\t", index=False)
            logger.info("Saved normalized records to: %s", out_file)
    elif input_dir.is_file():
        df = pd.read_csv(input_dir, sep="\t", dtype=str, keep_default_na=False)
        norm_df = normalize_dataframe(df)
        out_file = (
            args.output_path
            if args.output_path
            else input_dir.parent / f"normalized_{input_dir.name}"
        )
        norm_df.to_csv(out_file, sep="\t", index=False)
        logger.info("Saved normalized records to: %s", out_file)


if __name__ == "__main__":
    main()
