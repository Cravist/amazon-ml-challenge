"""
Unit tests for text and address normalization utilities.

Covers key noise patterns from the entity resolution problem statement:
- Corp vs Corporation
- Pvt vs Private (e.g. Pvt Ltd vs Private Limited)
- & vs and
- Rd vs Road (and St vs Street, Ave vs Avenue)
- Landmark references (Near/Nr, Opposite/Opp)
- Internal hyphen preservation (e.g., Wal-Mart, Saint-Denis)
- Legal suffix extraction (core_name, suffix)
- French names, accents, and address abbreviations (e.g., bd vs boulevard, r vs rue)
- Whitespace tokenization
"""

import pytest

from src.normalize import (
    clean_text,
    extract_legal_suffix,
    normalize_address,
    normalize_name,
    tokenize,
)


def test_corp_vs_corporation():
    """Verify that Corp, Corp., and Corporation normalize to identical string."""
    name1 = "Acme Corp"
    name2 = "Acme Corporation"
    name3 = "Acme Corp."

    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    norm3 = normalize_name(name3)

    assert norm1 == "acme corporation"
    assert norm1 == norm2 == norm3


def test_pvt_vs_private():
    """Verify Indian corporate suffix variants (Pvt Ltd vs Private Limited)."""
    name1 = "Reliance Industries Pvt Ltd"
    name2 = "Reliance Industries Private Limited"
    name3 = "Reliance Industries Pvt. Ltd."

    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    norm3 = normalize_name(name3)

    assert norm1 == "reliance industries private limited"
    assert norm1 == norm2 == norm3


def test_ampersand_vs_and():
    """Verify ampersands are expanded to 'and' and match textual 'and'."""
    name1 = "Barnes & Noble Inc."
    name2 = "Barnes and Noble Incorporated"
    name3 = "AT&T Corp"

    assert normalize_name(name1) == "barnes and noble incorporated"
    assert normalize_name(name1) == normalize_name(name2)
    assert normalize_name(name3) == "at and t corporation"


def test_internal_hyphens_preserved():
    """Verify internal hyphens are preserved while stripping external punctuation."""
    name1 = "Wal-Mart Stores Inc."
    name2 = "Saint-Denis Logistics Ltd."
    name3 = "-Acme-Corp-"  # Outer hyphens stripped; inner hyphen between tokens preserved

    assert normalize_name(name1) == "wal-mart stores incorporated"
    assert normalize_name(name2) == "saint-denis logistics limited"
    # Outer hyphens are stripped, but the remaining hyphen between 'acme' and 'corp'
    # is flanked by alphanumeric characters and is therefore treated as internal.
    assert normalize_name(name3) == "acme-corporation"


def test_extract_legal_suffix():
    """Verify extraction of trailing corporate legal suffixes into (core_name, suffix)."""
    # US / UK
    core1, sfx1 = extract_legal_suffix("Acme Corp.")
    assert core1 == "Acme"
    assert sfx1 == "corp"

    core2, sfx2 = extract_legal_suffix("Acme Corporation")
    assert core2 == "Acme"
    assert sfx2 == "corp"

    core3, sfx3 = extract_legal_suffix("Apex Technologies LLC")
    assert core3 == "Apex Technologies"
    assert sfx3 == "llc"

    # India
    core4, sfx4 = extract_legal_suffix("Tata Motors Pvt. Ltd.")
    assert core4 == "Tata Motors"
    assert sfx4 == "pvt ltd"

    core5, sfx5 = extract_legal_suffix("Tata Motors Private Limited")
    assert core5 == "Tata Motors"
    assert sfx5 == "pvt ltd"

    # France / Continental Europe
    core6, sfx6 = extract_legal_suffix("Carrefour S.A.")
    assert core6 == "Carrefour"
    assert sfx6 == "sa"

    core7, sfx7 = extract_legal_suffix("Renault SAS")
    assert core7 == "Renault"
    assert sfx7 == "sas"

    core8, sfx8 = extract_legal_suffix("Siemens GmbH")
    assert core8 == "Siemens"
    assert sfx8 == "gmbh"

    # No suffix present
    core9, sfx9 = extract_legal_suffix("Blue Star Enterprises")
    assert core9 == "Blue Star Enterprises"
    assert sfx9 is None


def test_rd_vs_road():
    """Verify Rd and Road normalize identically in addresses."""
    addr1 = "123 MG Rd."
    addr2 = "123 MG Road"

    norm1 = normalize_address(addr1)
    norm2 = normalize_address(addr2)

    assert norm1 == "123 mg road"
    assert norm1 == norm2


def test_st_ave_apt_abbreviations():
    """Verify common US street and apartment abbreviations normalize identically."""
    addr1 = "500 5th Ave., Apt. 4B"
    addr2 = "500 5th Avenue, Apartment 4B"
    addr3 = "100 North Main St., Suite 200"
    addr4 = "100 North Main Street, Ste 200"

    assert normalize_address(addr1) == "500 5th avenue apartment 4b"
    assert normalize_address(addr1) == normalize_address(addr2)
    assert normalize_address(addr3) == "100 north main street suite 200"
    assert normalize_address(addr3) == normalize_address(addr4)


def test_landmark_references_preserved():
    """Verify landmark phrases (near/nr, opposite/opp) are retained for spatial matching."""
    addr1 = "Opp. City Hospital, Nr Metro Station"
    addr2 = "Opposite City Hospital, Near Metro Station"
    addr3 = "Behind Grand Mall, Sector 15"

    norm1 = normalize_address(addr1)
    norm2 = normalize_address(addr2)
    norm3 = normalize_address(addr3)

    assert norm1 == "opposite city hospital near metro station"
    assert norm1 == norm2
    assert norm3 == "behind grand mall sector 15"


def test_french_names_accents_and_addresses():
    """Verify French names, accents, and boulevard/rue abbreviations normalize properly."""
    # Accents stripped in names
    name1 = "Société Générale S.A."
    name2 = "Societe Generale SA"
    assert normalize_name(name1) == "societe generale sa"
    assert normalize_name(name1) == normalize_name(name2)

    # Address with accents, boulevard (bd), and internal hyphens (Saint-Germain)
    addr1 = "12, Boulevard Saint-Germain"
    addr2 = "12, bd Saint-Germain"
    assert normalize_address(addr1) == "12 boulevard saint-germain"
    assert normalize_address(addr1) == normalize_address(addr2)

    # French street (rue vs r)
    addr3 = "5, rue de la Paix"
    addr4 = "5, r de la Paix"
    assert normalize_address(addr3) == "5 rue de la paix"
    assert normalize_address(addr3) == normalize_address(addr4)


def test_tokenize():
    """Verify whitespace tokenization produces clean token lists."""
    tokens1 = tokenize("acme corporation limited")
    assert tokens1 == ["acme", "corporation", "limited"]

    tokens2 = tokenize("   ")
    assert tokens2 == []

    tokens3 = tokenize(None)
    assert tokens3 == []
