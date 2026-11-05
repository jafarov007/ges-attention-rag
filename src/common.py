"""Shared paths, constants and text helpers used by every pipeline step."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT_DATA = OUT / "data"
OUT_LOGS = OUT / "logs"
OUT_TABLES = OUT / "tables"
OUT_PRED = OUT / "predictions"
OUT_MODELS = OUT / "models"
FIG = ROOT / "docs" / "figures"
PROTOCOL = ROOT / "protocol"

for _d in (OUT_DATA, OUT_LOGS, OUT_TABLES, OUT_PRED, OUT_MODELS, FIG, PROTOCOL):
    _d.mkdir(parents=True, exist_ok=True)

# Expected raw facts (checked by assertions/tests; taken from the audit, not assumed)
EXPECTED_COUNTS = {2022: 20, 2023: 11, 2024: 46, 2025: 76}
EXPECTED_TOTALS = {2022: 3834.8, 2023: 984.7, 2024: 3887.0, 2025: 8418.0}

SPLIT_SEED = 2026
INNER_SEED = 0
FINAL_SEEDS = (17, 42, 2026)
ABLATION_SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


def tr_lower(s: str) -> str:
    """Turkish-aware lowercasing (str.lower() maps 'İ' to 'i̇' and 'I' to 'i')."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def basic_clean(s) -> str:
    """NFC, non-breaking spaces -> space, collapse whitespace, strip."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# Year-specific naming templates. They encode the file year, so they are removed
# from the model text (see docs/02_veri_temizleme_on_isleme.md, rule R-NAME-02).
BOILERPLATE_PATTERNS = [
    r"güneş\s+enerji\s+santrali\s*\(ges\)",
    r"güneş\s+enerji\s+santrali",
    r"\(ges\)",
    r"\bges\b",
]


def strip_boilerplate(name: str) -> str:
    low = tr_lower(name)
    spans = []
    for pat in BOILERPLATE_PATTERNS:
        for m in re.finditer(pat, low):
            spans.append(m.span())
    if not spans:
        return name
    keep = [True] * len(name)
    for a, b in spans:
        for i in range(a, min(b, len(name))):
            keep[i] = False
    out = "".join(ch for ch, k in zip(name, keep) if k)
    out = re.sub(r"\s*-\s*", " - ", out)  # "Parkı- X" vs "Parkı - X"
    out = re.sub(r"\s+", " ", out).strip(" -")
    return out


def tokenize_tr(text: str) -> list[str]:
    """Lowercase Turkish word tokenizer; keeps digits (stage / sub-unit numbers)."""
    return re.findall(r"[0-9a-zçğıöşü]+", tr_lower(text))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
