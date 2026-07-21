#!/usr/bin/env python3
"""
Set up graphene_bilayer_BA phonopy jobs at several interlayer spacings.

For each dz value, creates:
  bilayer_examples/graphene_bilayer_BA_dz<NNN>/  (POSCAR, CONTCAR, POTCAR, KPOINTS, INCAR, bat)
  phonopy_bilayer_examples/graphene_bilayer_BA_dz<NNN>_staticpoint/  (disp-XXX directories)

Usage:
  python3 scripts/setup_graphene_dz_scan.py [--no-submit]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent

# Interlayer spacings to scan (Å)
DZ_VALUES = [3.0, 3.2, 3.35, 3.5, 4.0]

# Cell height fixed at 20 Å (same as rest of workflow)
C = 20.0

# Reference POSCAR header / lattice (from graphene_bilayer_BA)
LATTICE = (
    "  2.4672912616030827   0.0000000000000000   0.0000000000000000\n"
    "  1.2336456308015413   2.1367369110836267   0.0000000000000000\n"
    f"  0.0000000000000000   0.0000000000000000  {C:.16f}\n"
)

# BA stacking in-plane fractional coords (fixed)
# Layer 1: A at (0,0), B at (1/3, 1/3)
# Layer 2: A at (2/3, 2/3), B at (0, 0)   — shifted by 2/3, 2/3 relative to layer 1
LAYER1_XY = [(0.0, 0.0), (1 / 3, 1 / 3)]
LAYER2_XY = [(2 / 3, 2 / 3), (0.0, 0.0)]


def make_poscar(dz: float, label: str) -> str:
    z1 = 0.5               # centre layer 1 at half the cell
    z2 = z1 + dz / C      # layer 2 offset upward by dz
    lines = [
        f"Bilayer graphene/graphene BA dz={dz:.2f} A\n",
        " 1.0000000000000000\n",
        LATTICE,
        "   C\n",
        "     4\n",
        "Direct\n",
    ]
    for x, y in LAYER1_XY:
        lines.append(f"  {x:.16f}   {y:.16f}   {z1:.16f}\n")
    for x, y in LAYER2_XY:
        lines.append(f"  {x:.16f}   {y:.16f}   {z2:.16f}\n")
    return "".join(lines)


def setup_example(dz: float, template_dir: Path) -> Path:
    label = f"{int(dz * 100):04d}"          # e.g. 335 → "0335"
    name = f"graphene_bilayer_BA_dz{label}"
    out_dir = WORKFLOW / "bilayer_examples" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    poscar_text = make_poscar(dz, label)

    (out_dir / "POSCAR").write_text(poscar_text)
    (out_dir / "CONTCAR").write_text(poscar_text)   # no relaxation; phonopy reads CONTCAR

    for fname in ("POTCAR", "KPOINTS", "INCAR", "bat"):
        src = template_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
        else:
            print(f"  Warning: {fname} not found in template dir, skipping")

    print(f"  Created {out_dir}")
    return name


def run_phonopy_setup(name: str, submit: bool):
    script = WORKFLOW / "phonopy" / "prepare_and_submit.py"
    cmd = [sys.executable, str(script), "--bilayer", name]
    if not submit:
        cmd.append("--no-submit")
    print(f"  Running: {' '.join(cmd[2:])}")
    result = subprocess.run(cmd, cwd=WORKFLOW)
    if result.returncode != 0:
        print(f"  Warning: prepare_and_submit.py exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-submit", action="store_true",
        help="Create displacement dirs but do not submit SLURM jobs (default: submit)",
    )
    args = parser.parse_args()

    template_dir = WORKFLOW / "bilayer_examples" / "graphene_bilayer_BA"
    if not template_dir.exists():
        sys.exit(f"Template directory not found: {template_dir}")

    print(f"Setting up graphene BA bilayer dz scan: {DZ_VALUES} Å")
    print(f"Submit to SLURM: {not args.no_submit}\n")

    for dz in DZ_VALUES:
        print(f"--- dz = {dz:.2f} Å ---")
        name = setup_example(dz, template_dir)
        run_phonopy_setup(name, submit=not args.no_submit)
        print()

    print("Done.")
    print("\nTo post-process after VASP completes:")
    for dz in DZ_VALUES:
        label = f"{int(dz * 100):04d}"
        name = f"graphene_bilayer_BA_dz{label}"
        print(f"  python3 phonopy/postprocess_results.py --bilayer {name}_staticpoint")


if __name__ == "__main__":
    main()
