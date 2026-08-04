#!/usr/bin/env python3
"""
Submit phonopy static-point/displacement jobs for a specific relaxation batch.

This mirrors the relaxation batches defined in data/batches/, but instead
of creating/relaxing structures, it:

  - Takes the corresponding relaxed examples in monolayer_examples/ or
    bilayer_examples/
  - Runs the phonopy static-point + displacement pipeline from
    prepare_and_submit.py
  - Optionally submits the displacement jobs with sbatch

Typical usage:

  # Monolayer phonopy for batch 1
  python3 submit_batch.py --monolayer --batch 1

  # Bilayer phonopy for batch 2
  python3 submit_batch.py --bilayer --batch 2

  # Dry run (just print what would be done)
  python3 submit_batch.py --bilayer --batch 2 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "common"))

from batches import load_batch


def _import_phonopy_prepare():
    """
    Import the phonopy prepare/submit helpers.

    We import the module and use its process_* helpers directly.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    import prepare_and_submit as prep  # type: ignore[import]

    return prep


def _filter_by_symmetry_manifest(names: list[str], *, is_bilayer: bool = False) -> list[str]:
    from materials_project_api import load_symmetry_eligible_set, parse_bilayer_components

    eligible = load_symmetry_eligible_set()
    if eligible is None:
        return names
    kept = []
    for name in names:
        if is_bilayer:
            parts = parse_bilayer_components(name)
            if parts and all(p in eligible for p in parts):
                kept.append(name)
            elif parts:
                print(f"  Skipping {name}: not symmetry-eligible", file=sys.stderr)
            else:
                kept.append(name)
        elif name in eligible:
            kept.append(name)
        else:
            print(f"  Skipping {name}: not in symmetry_eligible_materials.json", file=sys.stderr)
    return kept


def submit_monolayer_batch(batch_number: int,
                           supercell_dim: str = "4 4 1",
                           submit: bool = True,
                           dry_run: bool = False):
    """Run phonopy pipeline for all monolayers in the given batch."""
    batch = load_batch("monolayer", batch_number)
    materials = _filter_by_symmetry_manifest(list(batch.get("materials", [])))
    if not materials:
        print(f"Warning: Monolayer batch {batch_number} has no materials", file=sys.stderr)
        return []

    example_root = WORKFLOW_ROOT / "monolayer_examples"
    prep = _import_phonopy_prepare()

    print(f"\n{'=' * 60}")
    print(f"Phonopy: Monolayer Batch {batch_number}")
    print(f"{'=' * 60}")
    print(f"Materials in batch: {len(materials)}")
    print(f"Examples base dir: {example_root}")
    print(f"{'=' * 60}\n")

    results = []
    for idx, mat in enumerate(materials, 1):
        example_dir = example_root / mat
        print(f"[{idx}/{len(materials)}] {mat}")
        if not example_dir.exists():
            print(
                f"  ✗ Skipping: relaxed example not found at {example_dir}",
                file=sys.stderr,
            )
            results.append({"success": False, "error": f"missing {example_dir}"})
            continue

        res = prep.process_monolayer(
            example_dir,
            supercell_dim=supercell_dim,
            submit=submit,
            dry_run=dry_run,
        )
        results.append(res)

    return results


def submit_bilayer_batch(batch_number: int,
                         supercell_dim: str = "4 4 1",
                         submit: bool = True,
                         dry_run: bool = False):
    """Run phonopy pipeline for all bilayers in the given batch."""
    batch = load_batch("bilayer", batch_number)
    bilayers = _filter_by_symmetry_manifest(list(batch.get("bilayers", [])), is_bilayer=True)
    if not bilayers:
        print(f"Warning: Bilayer batch {batch_number} has no bilayers", file=sys.stderr)
        return []

    example_root = WORKFLOW_ROOT / "bilayer_examples"
    prep = _import_phonopy_prepare()

    print(f"\n{'=' * 60}")
    print(f"Phonopy: Bilayer Batch {batch_number}")
    print(f"{'=' * 60}")
    print(f"Bilayers in batch: {len(bilayers)}")
    print(f"Examples base dir: {example_root}")
    print(f"{'=' * 60}\n")

    results = []
    for idx, name in enumerate(bilayers, 1):
        example_dir = example_root / name
        print(f"[{idx}/{len(bilayers)}] {name}")
        if not example_dir.exists():
            print(
                f"  ✗ Skipping: relaxed bilayer example not found at {example_dir}",
                file=sys.stderr,
            )
            results.append({"success": False, "error": f"missing {example_dir}"})
            continue

        res = prep.process_bilayer(
            example_dir,
            supercell_dim=supercell_dim,
            submit=submit,
            dry_run=dry_run,
        )
        results.append(res)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit phonopy static-point/displacement jobs for a specific batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monolayer phonopy for relaxation batch 1
  python3 submit_batch.py --monolayer --batch 1

  # Bilayer phonopy for relaxation batch 2
  python3 submit_batch.py --bilayer --batch 2

  # Dry run (no submission, just show actions)
  python3 submit_batch.py --bilayer --batch 2 --dry-run
        """,
    )

    parser.add_argument(
        "--monolayer",
        action="store_true",
        help="Process monolayer batch (data/batches/monolayer_batch_<N>.json)",
    )
    parser.add_argument(
        "--bilayer",
        action="store_true",
        help="Process bilayer batch (data/batches/bilayer_batch_<N>.json)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        metavar="N",
        help="Batch number to process (1-based)",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default="4 4 1",
        help='Supercell dimensions for phonopy (default: "4 4 1")',
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Set up displacement folders but do not submit jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )

    args = parser.parse_args()

    if not args.monolayer and not args.bilayer:
        parser.error("Must specify --monolayer and/or --bilayer")

    submit = not args.no_submit

    results = []

    if args.monolayer:
        results.extend(
            submit_monolayer_batch(
                batch_number=args.batch,
                supercell_dim=args.dim,
                submit=submit,
                dry_run=args.dry_run,
            )
        )

    if args.bilayer:
        results.extend(
            submit_bilayer_batch(
                batch_number=args.batch,
                supercell_dim=args.dim,
                submit=submit,
                dry_run=args.dry_run,
            )
        )

    # Simple summary
    ok = sum(1 for r in results if r.get("success"))
    print(f"\n{'=' * 60}")
    print("Batch phonopy summary")
    print(f"{'=' * 60}")
    print(f"Total entries: {len(results)}")
    print(f"Successful: {ok}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

