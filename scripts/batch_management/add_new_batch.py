#!/usr/bin/env python3
"""
Incrementally batch newly added materials from common/materials_list.txt.

Workflow:
  1. Add new material name(s) to common/materials_list.txt.
  2. Run this script.

It will:
  - Diff materials_list.txt against the union of all existing monolayer
    batches in data/batches/ to find materials that aren't batched yet.
  - Apply the same hexagonal P6_3/mmc symmetry filter used elsewhere in the
    project (create_batches.py, generate_bilayer_combinations.py) -- new
    materials without a P6_3/mmc Materials Project entry are reported and
    dropped, not silently skipped.
  - Write a new monolayer batch (data/batches/monolayer_batch_<N>.json) with
    the newly eligible materials.
  - Generate bilayer combinations involving at least one new material (new-
    with-new and new-with-old), respecting the existing 10% lattice-mismatch
    tolerance and per-pair-type stacking rules. Old-with-old pairs are never
    regenerated, even if a lattice-constant override now makes a previously
    incompatible old pair compatible -- those are already batched.
  - Append the new bilayer names to relaxation/bilayer/bilayer_combinations.txt
    (kept as the authoritative full list) and write new bilayer batch
    file(s) (data/batches/bilayer_batch_<N>.json), chunked at --bilayer-batch-size.

Safe to re-run: already-batched materials/bilayers are always excluded, so
running this again with no new materials in the list is a no-op.

Usage:
  python3 scripts/batch_management/add_new_batch.py
  python3 scripts/batch_management/add_new_batch.py --dry-run
  python3 scripts/batch_management/add_new_batch.py --no-symmetry-filter
"""

import sys
from itertools import combinations, combinations_with_replacement
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "relaxation" / "monolayer"))
sys.path.insert(0, str(WORKFLOW_ROOT / "relaxation" / "bilayer"))
sys.path.insert(0, str(WORKFLOW_ROOT / "common"))

from batches import iter_batches, batch_items, write_batch  # noqa: E402
from materials_project_api import filter_materials, get_material_symmetry, get_api_key  # noqa: E402
from structural_families import get_allowed_stackings  # noqa: E402
from generate_bilayer_combinations import (  # noqa: E402
    load_materials,
    generate_bilayer_combinations,
    format_bilayer_name,
)

BILAYER_COMBINATIONS_FILE = WORKFLOW_ROOT / "relaxation" / "bilayer" / "bilayer_combinations.txt"


def _already_batched(kind):
    """Union of all items across every existing batch of this kind."""
    items = set()
    for batch in iter_batches(kind):
        items.update(batch_items(batch))
    return items


def _next_batch_number(kind):
    existing = iter_batches(kind)
    if not existing:
        return 1
    return max(b["batch_number"] for b in existing) + 1


def _report_excluded(excluded, api_key):
    """Print why each excluded material failed the P6_3/mmc filter."""
    for mat in excluded:
        info = get_material_symmetry(mat, api_key=api_key, use_cache=True)
        sg = info.get("spacegroup_symbol") or "unknown"
        cs = info.get("crystal_system") or "unknown"
        print(f"    ✗ {mat}: excluded (crystal_system={cs}, spacegroup_symbol={sg})")


def find_new_materials(require_p63mmc=True, api_key=None, verbose=False, only_materials=None):
    """
    Returns (eligible_new, excluded_new, already_batched_materials) --
    materials present in materials_list.txt but not yet in any monolayer
    batch, split by the symmetry filter.

    only_materials : optional list, restricts the "new" set to just these
    names (still must be present in materials_list.txt and not yet batched)
    -- use this to batch a specific subset of new materials in one run
    rather than everything currently unbatched.
    """
    all_materials = load_materials()
    already = _already_batched("monolayer")
    candidates = [m for m in all_materials if m not in already]

    if only_materials is not None:
        requested = set(only_materials)
        unknown = requested - set(all_materials)
        if unknown:
            raise ValueError(
                f"--materials named material(s) not in common/materials_list.txt: {sorted(unknown)}"
            )
        already_named = requested & already
        if already_named:
            print(f"  Note: {sorted(already_named)} already batched, ignoring")
        candidates = [m for m in candidates if m in requested]

    if not require_p63mmc:
        return candidates, [], already

    eligible, excluded = filter_materials(candidates, api_key=api_key, verbose=verbose)
    return eligible, excluded, already


def find_new_bilayers(eligible_new, already_batched_materials, tolerance=0.10, verbose=False):
    """
    Returns a list of new bilayer names (new-with-new, new-with-old only --
    never old-with-old, and never a name already present in an existing
    bilayer batch).
    """
    if not eligible_new:
        return []

    already_batched_bilayers = _already_batched("bilayer")
    combined_materials = list(already_batched_materials) + list(eligible_new)

    pairs = generate_bilayer_combinations(
        combined_materials,
        include_homostructures=True,
        include_heterostructures=True,
        only_compatible=True,
        tolerance=tolerance,
        verbose=verbose,
    )

    eligible_new_set = set(eligible_new)
    new_names = []
    for mat1, mat2 in pairs:
        # Skip old-with-old: neither side is a newly eligible material.
        if mat1 not in eligible_new_set and mat2 not in eligible_new_set:
            continue
        for stacking in get_allowed_stackings(mat1, mat2):
            name = format_bilayer_name(mat1, mat2, stacking)
            if name not in already_batched_bilayers and name not in new_names:
                new_names.append(name)

    return new_names


def append_bilayer_combinations_file(new_names, dry_run=False):
    """Append new bilayer names to the authoritative bilayer_combinations.txt."""
    if not new_names:
        return
    if dry_run:
        print(f"  (dry-run) Would append {len(new_names)} name(s) to {BILAYER_COMBINATIONS_FILE}")
        return
    with open(BILAYER_COMBINATIONS_FILE, "a") as f:
        f.write("\n# Newly added combinations (via add_new_batch.py)\n")
        for name in new_names:
            f.write(f"{name}\n")
    print(f"  Appended {len(new_names)} name(s) to {BILAYER_COMBINATIONS_FILE}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Incrementally batch newly added materials/bilayers from materials_list.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add new materials to common/materials_list.txt first, then:
  python3 scripts/batch_management/add_new_batch.py

  # Preview without writing anything
  python3 scripts/batch_management/add_new_batch.py --dry-run

  # Include materials without a cached P6_3/mmc entry too
  python3 scripts/batch_management/add_new_batch.py --no-symmetry-filter

  # Only batch specific new materials this run (others in materials_list.txt
  # that are also unbatched are left for a later run)
  python3 scripts/batch_management/add_new_batch.py --materials silicene germanene
        """,
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Restrict this run to these specific new materials (must be in "
             "materials_list.txt and not yet batched); default: all unbatched materials",
    )
    sym = parser.add_mutually_exclusive_group()
    sym.add_argument(
        "--require-p63mmc",
        dest="require_p63mmc",
        action="store_true",
        default=True,
        help="Only batch hexagonal P6_3/mmc materials (default)",
    )
    sym.add_argument(
        "--no-symmetry-filter",
        dest="require_p63mmc",
        action="store_false",
        help="Include all new materials regardless of symmetry filter",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Max relative lattice-constant mismatch for a compatible bilayer pair (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--bilayer-batch-size",
        type=int,
        default=15,
        help="Max bilayers per new batch file (default: 15, matching existing batches)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing any files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose compatibility/symmetry details",
    )
    args = parser.parse_args()

    api_key = get_api_key()

    print(f"\n{'=' * 60}")
    print("Finding new materials")
    print(f"{'=' * 60}")
    eligible_new, excluded_new, already_batched_materials = find_new_materials(
        require_p63mmc=args.require_p63mmc, api_key=api_key, verbose=args.verbose,
        only_materials=args.materials,
    )

    if not eligible_new and not excluded_new:
        print("No new materials found in common/materials_list.txt -- nothing to do.")
        return

    if eligible_new:
        print(f"  ✓ {len(eligible_new)} new eligible material(s): {', '.join(eligible_new)}")
    if excluded_new:
        print(f"  ✗ {len(excluded_new)} new material(s) excluded by the P6_3/mmc filter:")
        _report_excluded(excluded_new, api_key)

    if not eligible_new:
        print("\nNo newly eligible materials -- nothing to batch.")
        return

    # --- Monolayer batch ---
    mono_batch_number = _next_batch_number("monolayer")
    mono_batch = {
        "kind": "monolayer",
        "batch_number": mono_batch_number,
        "materials": eligible_new,
        "count": len(eligible_new),
    }
    print(f"\n{'=' * 60}")
    print(f"New monolayer batch {mono_batch_number}: {len(eligible_new)} material(s)")
    print(f"{'=' * 60}")
    if args.dry_run:
        print(f"  (dry-run) Would write monolayer_batch_{mono_batch_number}.json: {eligible_new}")
    else:
        path = write_batch("monolayer", mono_batch)
        print(f"  Wrote {path}")

    # --- Bilayer batches ---
    print(f"\n{'=' * 60}")
    print("Finding new bilayer combinations (new-with-new, new-with-old)")
    print(f"{'=' * 60}")
    new_bilayer_names = find_new_bilayers(
        eligible_new, already_batched_materials, tolerance=args.tolerance, verbose=args.verbose
    )

    if not new_bilayer_names:
        print("  No new compatible bilayer combinations found.")
        return

    print(f"  ✓ {len(new_bilayer_names)} new bilayer configuration(s)")

    append_bilayer_combinations_file(new_bilayer_names, dry_run=args.dry_run)

    next_bilayer_num = _next_batch_number("bilayer")
    size = args.bilayer_batch_size
    written = []
    for i in range(0, len(new_bilayer_names), size):
        chunk = new_bilayer_names[i:i + size]
        batch_number = next_bilayer_num + (i // size)
        batch = {
            "kind": "bilayer",
            "batch_number": batch_number,
            "bilayers": chunk,
            "count": len(chunk),
        }
        if args.dry_run:
            print(f"  (dry-run) Would write bilayer_batch_{batch_number}.json ({len(chunk)} items): {chunk}")
        else:
            path = write_batch("bilayer", batch)
            print(f"  Wrote {path} ({len(chunk)} items)")
        written.append(batch_number)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"New monolayer batch: {mono_batch_number} ({len(eligible_new)} materials)")
    print(f"New bilayer batch(es): {written} ({len(new_bilayer_names)} bilayers total)")
    if args.dry_run:
        print("\nDry run -- no files were written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
