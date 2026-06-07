#!/usr/bin/env python3
"""
Report and persist which workflow materials are eligible for batch processing.

Eligible materials must have hexagonal crystal system and space group P6₃/mmc
according to Materials Project metadata (see common/materials_project_api.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "common"))

from materials_project_api import (  # noqa: E402
    TARGET_CRYSTAL_SYSTEM,
    TARGET_SPACEGROUP_SYMBOL,
    filter_materials,
    get_api_key,
    get_material_symmetry,
    is_symmetry_eligible,
)


def load_materials(materials_file: Path) -> list[str]:
    materials = []
    with materials_file.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                materials.append(line)
    return materials


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report hexagonal P6₃/mmc eligibility for workflow materials"
    )
    parser.add_argument(
        "-m",
        "--materials-file",
        type=Path,
        default=WORKFLOW_ROOT / "common" / "materials_list.txt",
        help="Materials list file (default: common/materials_list.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=WORKFLOW_ROOT / "data" / "symmetry_eligible_materials.json",
        help="Output JSON manifest (default: data/symmetry_eligible_materials.json)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-query MP instead of using structure cache only",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-material details")
    args = parser.parse_args()

    materials = load_materials(args.materials_file)
    api_key = get_api_key()

    rows = []
    for name in materials:
        info = get_material_symmetry(
            name,
            api_key=api_key,
            use_cache=not args.refresh_cache,
            refresh_cache=args.refresh_cache,
            verbose=args.verbose,
        )
        eligible = is_symmetry_eligible(
            name,
            api_key=api_key,
            use_cache=not args.refresh_cache,
            refresh_cache=args.refresh_cache,
        )
        rows.append(
            {
                "material": name,
                "crystal_system": info.get("crystal_system"),
                "spacegroup_symbol": info.get("spacegroup_symbol"),
                "spacegroup_number": info.get("spacegroup_number"),
                "material_id": info.get("material_id"),
                "source": info.get("source"),
                "eligible": eligible,
                "reason": info.get("reason"),
            }
        )

    eligible, excluded = filter_materials(
        materials,
        api_key=api_key,
        use_cache=not args.refresh_cache,
        refresh_cache=args.refresh_cache,
        verbose=args.verbose,
    )

    print(f"\n{'Material':<15} {'Crystal':<12} {'Space group':<12} {'Eligible':<8} Source")
    print("-" * 70)
    for row in rows:
        mark = "yes" if row["eligible"] else "no"
        print(
            f"{row['material']:<15} "
            f"{(row['crystal_system'] or '-'):<12} "
            f"{(row['spacegroup_symbol'] or '-'):<12} "
            f"{mark:<8} "
            f"{row['source']}"
        )

    print(f"\nEligible: {len(eligible)}/{len(materials)}")
    if excluded:
        print(f"Excluded: {', '.join(excluded)}")

    manifest = {
        "criteria": {
            "crystal_system": TARGET_CRYSTAL_SYSTEM,
            "spacegroup_symbol": TARGET_SPACEGROUP_SYMBOL,
        },
        "materials_file": str(
            args.materials_file.relative_to(WORKFLOW_ROOT)
            if WORKFLOW_ROOT in args.materials_file.parents
            else args.materials_file
        ),
        "eligible_materials": eligible,
        "excluded_materials": excluded,
        "details": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"\nWrote manifest: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
