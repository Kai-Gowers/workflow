#!/usr/bin/env python3
"""
Post-process phonopy results for a specific relaxation batch.

This mirrors the relaxation batches defined in workflow/data/batches/.
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
import sys
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "common"))

from batches import load_batch


def _import_postprocess():
    """Import the phonopy postprocess helpers."""
    sys.path.insert(0, str(Path(__file__).parent))
    import postprocess_results as pp  # type: ignore[import]

    return pp


def postprocess_monolayer_batch(batch_number: int,
                                dim: str = "4 4 1"):
    """Post-process phonopy results for all monolayers in the given batch."""
    batch = load_batch("monolayer", batch_number)
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
                              dim: str = "4 4 1"):
    """Post-process phonopy results for all bilayers in the given batch."""
    batch = load_batch("bilayer", batch_number)
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
        help="Post-process monolayer batch (data/batches/monolayer_batch_<N>.json)",
    )
    parser.add_argument(
        "--bilayer",
        action="store_true",
        help="Post-process bilayer batch (data/batches/bilayer_batch_<N>.json)",
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
        default="4 4 1",
        help='DIM value for band.conf (default: "4 4 1")',
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

