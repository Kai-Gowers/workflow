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
FINAL_RESULTS = ROOT / "FINAL_RESULTS"

# Bilayer names contain an underscore-separated stacking suffix (3R, 2H, AB, BA, TM_H, TM_H2)
_STACKING = {"3R", "2H", "AB", "BA", "TM_H", "TM_H2"}


def is_bilayer(name: str) -> bool:
    parts = name.rsplit("_", 1)
    return len(parts) == 2 and parts[1] in _STACKING


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
Examples:
  python3 scripts/convert_to_nequix.py
  python3 scripts/convert_to_nequix.py --monolayer --output data/monolayers.aselmdb
  python3 scripts/convert_to_nequix.py --bilayer --output data/bilayers.aselmdb
""",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "nequix_dataset.aselmdb",
        help="Output .aselmdb path (default: workflow/nequix_dataset.aselmdb)",
    )
    parser.add_argument("--monolayer", action="store_true", help="Include monolayers only")
    parser.add_argument("--bilayer", action="store_true", help="Include bilayers only")
    args = parser.parse_args()

    if not FINAL_RESULTS.exists():
        print(f"Error: FINAL_RESULTS directory not found at {FINAL_RESULTS}", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(d for d in FINAL_RESULTS.iterdir() if d.is_dir())

    if args.monolayer and not args.bilayer:
        dirs = [d for d in dirs if not is_bilayer(d.name)]
    elif args.bilayer and not args.monolayer:
        dirs = [d for d in dirs if is_bilayer(d.name)]

    if not dirs:
        print("No materials found to convert.")
        sys.exit(1)

    if args.output.exists():
        args.output.unlink()

    converted, skipped = 0, []

    with ase.db.connect(str(args.output)) as db:
        for d in dirs:
            err = convert_material(d, db)
            if err is None:
                print(f"  ✓ {d.name}")
                converted += 1
            else:
                print(f"  ✗ {d.name}: {err}")
                skipped.append((d.name, err))

    print(f"\nConverted {converted}/{len(dirs)} materials → {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} (run postprocess_results.py to regenerate phonopy.yaml):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
