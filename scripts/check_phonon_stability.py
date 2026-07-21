#!/usr/bin/env python3
"""
Parse FINAL_RESULTS/*/band.yaml and report min/max phonon frequencies.
Usage:
    python3 scripts/check_phonon_stability.py                 # all results
    python3 scripts/check_phonon_stability.py --batch 1       # bilayer batch 1 only
    python3 scripts/check_phonon_stability.py --monolayer     # monolayer results only
    python3 scripts/check_phonon_stability.py --bilayer       # bilayer results only
"""
import argparse
import glob
import json
from pathlib import Path

import yaml

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
FINAL_RESULTS = WORKFLOW_ROOT / "FINAL_RESULTS"
BATCHES_FILE  = WORKFLOW_ROOT / "data" / "batches.json"

STABLE_THRESHOLD    = -0.10   # THz — fully stable
BORDERLINE_THRESHOLD = -0.50  # THz — small Γ acoustic dip, acceptable


def classify(min_freq):
    if min_freq > STABLE_THRESHOLD:
        return "STABLE"
    if min_freq > BORDERLINE_THRESHOLD:
        return "borderline"
    if min_freq > -1.5:
        return "WEAK imaginary"
    return "UNSTABLE"


def parse_band_yaml(path):
    data = yaml.safe_load(open(path))
    freqs = [b["frequency"] for pt in data["phonon"] for b in pt["band"]]
    return min(freqs), max(freqs)


def get_batch_materials(batch_num, kind="bilayer"):
    if not BATCHES_FILE.exists():
        return None
    batches = json.loads(BATCHES_FILE.read_text())
    key = f"{kind}_batches"
    for b in batches.get(key, []):
        if b["batch_number"] == batch_num:
            if kind == "bilayer":
                return b["bilayers"]
            return b["materials"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=None, help="Bilayer batch number to filter")
    parser.add_argument("--monolayer", action="store_true")
    parser.add_argument("--bilayer",   action="store_true")
    args = parser.parse_args()

    all_yamls = sorted(FINAL_RESULTS.glob("*/band.yaml"))

    # Determine target set
    if args.batch is not None:
        names = get_batch_materials(args.batch, "bilayer")
        if names is None:
            print(f"Batch {args.batch} not found in batches.json")
            return
        target = set(names)
        all_yamls = [p for p in all_yamls if p.parent.name in target]
        print(f"Bilayer batch {args.batch} — {len(target)} materials, {len(all_yamls)} with results\n")
    elif args.monolayer:
        # monolayer names have no stacking suffix
        all_yamls = [p for p in all_yamls if "_bilayer_" not in p.parent.name
                     and "_3R" not in p.parent.name and "_2H" not in p.parent.name
                     and "_AB" not in p.parent.name and "_BA" not in p.parent.name
                     and "_TM_H" not in p.parent.name]
    elif args.bilayer:
        all_yamls = [p for p in all_yamls if any(s in p.parent.name
                     for s in ("_bilayer_", "_3R", "_2H", "_AB", "_BA", "_TM_H"))]

    if not all_yamls:
        print("No band.yaml files found for the given filter.")
        return

    results = []
    errors = []
    for p in all_yamls:
        try:
            mn, mx = parse_band_yaml(p)
            results.append((p.parent.name, mn, mx, classify(mn)))
        except Exception as e:
            errors.append((p.parent.name, str(e)))

    stable     = [r for r in results if r[3] == "STABLE"]
    borderline = [r for r in results if r[3] == "borderline"]
    weak       = [r for r in results if r[3] == "WEAK imaginary"]
    unstable   = [r for r in results if r[3] == "UNSTABLE"]

    print(f"{'Material':<35}  {'Min (THz)':>10}  {'Max (THz)':>10}  Status")
    print("-" * 75)
    for name, mn, mx, status in results:
        print(f"{name:<35}  {mn:>10.4f}  {mx:>10.4f}  {status}")

    print()
    print(f"Summary: {len(stable)} stable | {len(borderline)} borderline | "
          f"{len(weak)} weak | {len(unstable)} unstable  (total {len(results)})")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, e in errors:
            print(f"  {name}: {e}")


if __name__ == "__main__":
    main()
