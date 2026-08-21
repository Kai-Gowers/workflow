#!/usr/bin/env python3
"""
Convert FINAL_RESULTS phonon data to a Nequix-compatible ASE database.

Each material becomes one entry in the .aselmdb with:
  - atoms: supercell structure (from POSCAR)
  - atoms.info["hessian"]: (3n, 3n) Cartesian force constant matrix
  - energy/forces/stress: zero (equilibrium structures, PFT energy_weight=0)

Usage:
  python3 scripts/convert_to_nequix.py [--output PATH] [--monolayer] [--bilayer]
"""

import argparse
import sys
from pathlib import Path

import ase.db
import numpy as np
import phonopy as ph_module
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from phonopy.harmonic.force_constants import compact_fc_to_full_fc


ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(ROOT / "common"))
from structural_families import STACKING_SUFFIXES  # noqa: E402


def is_bilayer(name: str) -> bool:
    # Longest suffix first (STACKING_SUFFIXES) so "TM_H2" isn't misread as "TM_H".
    return any(name.endswith("_" + suffix) for suffix in STACKING_SUFFIXES)


def convert_material(material_dir: Path, db: ase.db.core.Database) -> str | None:
    """Convert one FINAL_RESULTS subdirectory. Returns None on success, error string on failure."""
    phonopy_yaml = material_dir / "phonopy.yaml"
    if not phonopy_yaml.exists():
        return "missing phonopy.yaml"

    fc_file = material_dir / "FORCE_CONSTANTS"
    if not fc_file.exists():
        return "missing FORCE_CONSTANTS"

    try:
        ph = ph_module.load(
            str(phonopy_yaml),
            force_constants_filename=str(fc_file),
        )
        fc = ph.force_constants  # full (n_sc, n_sc, 3, 3)
    except Exception as e:
        return str(e)

    # Expand compact FC (n_prim, n_sc, 3, 3) → full (n_sc, n_sc, 3, 3) if needed
    if fc.shape[0] != fc.shape[1]:
        fc = compact_fc_to_full_fc(ph.primitive, fc)

    # Build ASE Atoms from phonopy supercell
    sc = ph.supercell
    atoms = Atoms(numbers=sc.numbers, positions=sc.positions, cell=sc.cell, pbc=True)

    n = len(atoms)
    hessian = np.array(fc, dtype=np.float32).swapaxes(1, 2).reshape(3 * n, 3 * n)

    atoms.calc = SinglePointCalculator(
        atoms,
        energy=0.0,
        forces=np.zeros((n, 3), dtype=np.float32),
        stress=np.zeros((3, 3), dtype=np.float32),
    )
    atoms.info["hessian"] = hessian
    atoms.info["material"] = material_dir.name

    db.write(atoms, data=atoms.info)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FINAL_RESULTS to a Nequix PFT training database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each run writes a materials.txt manifest alongside --output, listing every
material included. To keep past dataset versions instead of overwriting them,
give each run its own directory under nequix_datasets/ named vN_<count>materials
(no dates — see nequix_datasets/README.md for version history), e.g.:

Examples:
  python3 scripts/convert_to_nequix.py --source FINAL_RESULTS_HEALTHY \\
      --output nequix_datasets/v1_46materials/dataset.aselmdb
  python3 scripts/convert_to_nequix.py \\
      --output nequix_datasets/v2_54materials/dataset.aselmdb
  python3 scripts/convert_to_nequix.py --monolayer \\
      --output nequix_datasets/v2_monolayers_only/dataset.aselmdb
""",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "FINAL_RESULTS",
        help="Directory of per-material subdirectories to convert (default: workflow/FINAL_RESULTS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "nequix_dataset.aselmdb",
        help="Output .aselmdb path (default: workflow/nequix_dataset.aselmdb; "
        "prefer nequix_datasets/vN_.../dataset.aselmdb to keep past versions)",
    )
    parser.add_argument("--monolayer", action="store_true", help="Include monolayers only")
    parser.add_argument("--bilayer", action="store_true", help="Include bilayers only")
    args = parser.parse_args()

    final_results = args.source
    if not final_results.exists():
        print(f"Error: source directory not found at {final_results}", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(d for d in final_results.iterdir() if d.is_dir())

    if args.monolayer and not args.bilayer:
        dirs = [d for d in dirs if not is_bilayer(d.name)]
    elif args.bilayer and not args.monolayer:
        dirs = [d for d in dirs if is_bilayer(d.name)]

    if not dirs:
        print("No materials found to convert.")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    converted, skipped = 0, []
    included = []

    with ase.db.connect(str(args.output)) as db:
        for d in dirs:
            err = convert_material(d, db)
            if err is None:
                print(f"  ✓ {d.name}")
                converted += 1
                included.append(d.name)
            else:
                print(f"  ✗ {d.name}: {err}")
                skipped.append((d.name, err))

    manifest_path = args.output.parent / "materials.txt"
    with open(manifest_path, "w") as f:
        f.write(f"# Nequix dataset — {len(included)} materials\n")
        f.write(f"# Source: {final_results}\n")
        f.write(f"# Generated: python3 {' '.join(sys.argv)}\n")
        f.write(f"# Count: {len(included)}\n\n")
        for name in sorted(included):
            f.write(f"{name}\n")
    print(f"Wrote material manifest → {manifest_path}")

    print(f"\nConverted {converted}/{len(dirs)} materials → {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} (run postprocess_results.py to regenerate phonopy.yaml):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
