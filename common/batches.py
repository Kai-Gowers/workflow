#!/usr/bin/env python3
"""
Shared loader for the per-batch files in ``data/batches/``.

Each batch is one JSON file: ``data/batches/<kind>_batch_<N>.json``, e.g.
``monolayer_batch_1.json`` or ``bilayer_batch_2.json``, holding:

    {"kind": "monolayer", "batch_number": 1, "materials": [...], "count": 8}
    {"kind": "bilayer",   "batch_number": 1, "bilayers":  [...], "count": 15}

Adding a new batch is just dropping a new file into the directory -- nothing
reads the directory listing order except display, and batch numbers do not
need to be contiguous.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
BATCHES_DIR = WORKFLOW_ROOT / "data" / "batches"

_ITEMS_KEY = {"monolayer": "materials", "bilayer": "bilayers"}


def batches_dir() -> Path:
    return BATCHES_DIR


def items_key(kind: str) -> str:
    if kind not in _ITEMS_KEY:
        raise ValueError(f"Unknown batch kind: {kind!r} (expected 'monolayer' or 'bilayer')")
    return _ITEMS_KEY[kind]


def batch_items(batch: dict) -> List[str]:
    """Return the material/bilayer name list for a batch dict, regardless of kind."""
    return list(batch.get(items_key(batch["kind"]), []))


def iter_batches(kind: str) -> List[dict]:
    """All batches of the given kind ('monolayer' or 'bilayer'), sorted by batch_number."""
    items_key(kind)  # validates kind
    if not BATCHES_DIR.exists():
        raise FileNotFoundError(
            f"Batches directory not found: {BATCHES_DIR}. "
            "Run scripts/batch_management/create_batches.py first."
        )
    batches = []
    for path in sorted(BATCHES_DIR.glob(f"{kind}_batch_*.json")):
        with open(path, "r") as f:
            batches.append(json.load(f))
    batches.sort(key=lambda b: b["batch_number"])
    return batches


def load_batch(kind: str, batch_number: int) -> dict:
    """Load a single batch by kind + number, raising a clear error if not found."""
    batches = iter_batches(kind)
    for b in batches:
        if b.get("batch_number") == batch_number:
            return b
    available = [b["batch_number"] for b in batches]
    raise ValueError(
        f"{kind.capitalize()} batch {batch_number} not found. "
        f"Available batches: {available}"
    )


def write_batch(kind: str, batch: dict) -> Path:
    """Write a single batch dict to its canonical file in data/batches/."""
    items_key(kind)  # validates kind
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    path = BATCHES_DIR / f"{kind}_batch_{batch['batch_number']}.json"
    with open(path, "w") as f:
        json.dump(batch, f, indent=2)
        f.write("\n")
    return path
