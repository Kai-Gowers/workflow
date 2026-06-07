#!/usr/bin/env python3
"""
Post-process phonopy results for a specific relaxation batch.

This mirrors the relaxation batches defined in workflow/data/batches.json.
For each material/bilayer in the selected batch, this script:

  - Locates the corresponding phonopy staticpoint directory
    (phonopy_monolayer_examples/<name>_staticpoint or
     phonopy_bilayer_examples/<name>_staticpoint)
  - Runs the same pipeline as postprocess_results.process_staticpoint_dir:
      * collect disp-*/vasprun.xml
      * phonopy --vasp -f ...
      * write band.conf
      * phonopy -p band.conf --save
      * copy band.pdf, band.yaml, FORCE_SETS → FINAL_RESULTS/<name>/

Usage examples:

  # Monolayer phonopy post-processing for relaxation batch 1
  python3 postprocess_batch.py --monolayer --batch 1

  # Bilayer phonopy post-processing for relaxation batch 2
  python3 postprocess_batch.py --bilayer --batch 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).parent.parent


def _load_batches(batch_file: str = "batches.json") -> dict:
    """Load batch information from workflow/data/batches.json."""
    batch_path = WORKFLOW_ROOT / "data" / batch_file
    if not batch_path.exists():
        raise FileNotFoundError(
            f"Batch file not found: {batch_path}. "
            "Run scripts/batch_management/create_batches.py first."
        )

    with open(batch_path, "r") as f:
        return json.load(f)


def _import_postprocess():
    """Import the phonopy postprocess helpers."""
    sys.path.insert(0, str(Path(__file__).parent))
    import postprocess_results as pp  # type: ignore[import]

    return pp


def postprocess_monolayer_batch(batch_number: int,
                                dim: str = "3 3 1"):
    """Post-process phonopy results for all monolayers in the given batch."""
    batches = _load_batches()
    mono_batches = batches.get("monolayer_batches", [])
    total_batches = batches.get("total_monolayer_batches", len(mono_batches))

    batch = None
    for b in mono_batches:
        if b.get("batch_number") == batch_number:
            batch = b
            break

    if batch is None:
        raise ValueError(
            f"Monolayer batch {batch_number} not found. "
            f"Available batches: 1-{total_batches}"
        )

    materials = list(batch.get("materials", []))
    if not materials:
        print(f"Warning: Monolayer batch {batch_number} has no materials", file=sys.stderr)
        return []

    static_root = WORKFLOW_ROOT / "phonopy_monolayer_examples"
    pp = _import_postprocess()

    print(f"\n{'=' * 60}")
    print(f"Phonopy post-process: Monolayer Batch {batch_number}")
    print(f"{'=' * 60}")
    print(f"Materials in batch: {len(materials)}")
    print(f"Staticpoint base dir: {static_root}")
    print(f"{'=' * 60}\n")

    results = []
    for idx, mat in enumerate(materials, 1):
        static_dir = static_root / f"{mat}_staticpoint"
        print(f"[{idx}/{len(materials)}] {static_dir.name}")
        if not static_dir.exists():
            print(
                f"  ✗ Skipping: staticpoint directory not found at {static_dir}",
                file=sys.stderr,
            )
            results.append({"success": False, "staticpoint_dir": static_dir})
            continue

        res = pp.process_staticpoint_dir(static_dir, "monolayer", dim=dim)
        results.append(res)

    return results


def postprocess_bilayer_batch(batch_number: int,
                              dim: str = "3 3 1"):
    """Post-process phonopy results for all bilayers in the given batch."""
    batches = _load_batches()
    bi_batches = batches.get("bilayer_batches", [])
    total_batches = batches.get("total_bilayer_batches", len(bi_batches))

    batch = None
    for b in bi_batches:
        if b.get("batch_number") == batch_number:
            batch = b
            break

    if batch is None:
        raise ValueError(
            f"Bilayer batch {batch_number} not found. "
            f"Available batches: 1-{total_batches}"
        )

    bilayers = list(batch.get("bilayers", []))
    if not bilayers:
        print(f"Warning: Bilayer batch {batch_number} has no bilayers", file=sys.stderr)
        return []

    static_root = WORKFLOW_ROOT / "phonopy_bilayer_examples"
    pp = _import_postprocess()

    print(f"\n{'=' * 60}")
    print(f"Phonopy post-process: Bilayer Batch {batch_number}")
    print(f"{'=' * 60}")
    print(f"Bilayers in batch: {len(bilayers)}")
    print(f"Staticpoint base dir: {static_root}")
    print(f"{'=' * 60}\n")

    results = []
    for idx, name in enumerate(bilayers, 1):
        static_dir = static_root / f"{name}_staticpoint"
        print(f"[{idx}/{len(bilayers)}] {static_dir.name}")
        if not static_dir.exists():
            print(
                f"  ✗ Skipping: staticpoint directory not found at {static_dir}",
                file=sys.stderr,
            )
            results.append({"success": False, "staticpoint_dir": static_dir})
            continue

        res = pp.process_staticpoint_dir(static_dir, "bilayer", dim=dim)
        results.append(res)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process phonopy results for a specific relaxation batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monolayer batch 1
  python3 postprocess_batch.py --monolayer --batch 1

  # Bilayer batch 2
  python3 postprocess_batch.py --bilayer --batch 2
        """,
    )

    parser.add_argument(
        "--monolayer",
        action="store_true",
        help="Post-process monolayer batch (monolayer_batches in data/batches.json)",
    )
    parser.add_argument(
        "--bilayer",
        action="store_true",
        help="Post-process bilayer batch (bilayer_batches in data/batches.json)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        metavar="N",
        help="Batch number to post-process (1-based)",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default="3 3 1",
        help='DIM value for band.conf (default: "3 3 1")',
    )

    args = parser.parse_args()

    if not args.monolayer and not args.bilayer:
        parser.error("Must specify --monolayer and/or --bilayer")

    results = []

    if args.monolayer:
        results.extend(
            postprocess_monolayer_batch(
                batch_number=args.batch,
                dim=args.dim,
            )
        )

    if args.bilayer:
        results.extend(
            postprocess_bilayer_batch(
                batch_number=args.batch,
                dim=args.dim,
            )
        )

    ok = sum(1 for r in results if r.get("success"))
    print(f"\n{'=' * 60}")
    print("Batch phonopy post-process summary")
    print(f"{'=' * 60}")
    print(f"Total staticpoint dirs: {len(results)}")
    print(f"Successful: {ok}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

