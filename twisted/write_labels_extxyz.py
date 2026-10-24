#!/usr/bin/env python3
"""
Write the PBE+D3 energy / force / stress labels of a twisted cell into its tracked results folder:

  FINAL_RESULTS_TWISTED/<name>/displacements.extxyz   every DFT displaced supercell (energy, forces, stress)
  FINAL_RESULTS_TWISTED/<name>/equilibrium.extxyz     the relaxed moire cell (energy, forces, stress of the relaxation run)

FINAL_RESULTS_TWISTED/ only carries phonopy files (FORCE_SETS has forces but no energies / stresses); the labels
live in the gitignored run dirs (phonopy_bilayer_examples_twisted/<name>_staticpoint/disp-*/vasprun.xml and
bilayer_examples_twisted/<name>/OUTCAR). Run this on the host that computed the cell, BEFORE committing
FINAL_RESULTS_TWISTED/<name>/, so git carries the labels too (~1 MB per 42-atom cell, ~10 MB for a 444-displacement
m3 cell). twisted/export_share_bundle.py prefers these files and falls back to the run dirs when they are absent.

Every frame is cross-checked against FORCE_SETS / phonopy.yaml (forces, positions, cell to 1e-5) and the relaxed
cell against the shipped POSCAR, exactly as the share exporter did before. Requires TWIST_VARIANT=twisted only in
the sense that the run dirs must carry the _twisted suffix; the script addresses them directly.

Usage:
  python3 twisted/write_labels_extxyz.py MoS2_twist_m1_near0 [more names]      # default: every cell in FINAL_RESULTS_TWISTED
  python3 twisted/write_labels_extxyz.py --force ...                            # overwrite existing files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import phonopy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from export_shareable_dataset import to_ase  # noqa: E402

SOURCE = ROOT / "FINAL_RESULTS_TWISTED"
STRUCTURES = ROOT / "twisted" / "structures"
RELAX = ROOT / "bilayer_examples_twisted"
STATIC = ROOT / "phonopy_bilayer_examples_twisted"   # overridden by --static-root (e.g. the twisted_gridtight run dirs)
FUNCTIONAL = "PBE+D3(BJ)"


def kmesh(path: Path) -> str:
    return " ".join(path.read_text().splitlines()[3].split()[:3])


def build_frames(name: str) -> tuple[list[Atoms], Atoms]:
    """(displaced supercells with labels, relaxed cell with labels) read from the run dirs, cross-checked."""
    src = SOURCE / name
    tw = json.loads((STRUCTURES / name / "twist.json").read_text())
    ph_fs = phonopy.load(str(src / "phonopy.yaml"), force_sets_filename=str(src / "FORCE_SETS"), log_level=0)
    disps = ph_fs.dataset["first_atoms"]
    vaspruns = sorted((STATIC / f"{name}_staticpoint").glob("disp-*/vasprun.xml"))
    if len(vaspruns) != len(disps):
        sys.exit(f"{name}: {len(vaspruns)} vasprun.xml vs {len(disps)} displacements")
    frames = []
    for i, (disp, sc, vr) in enumerate(zip(disps, ph_fs.supercells_with_displacements, vaspruns)):
        v = read(vr, index=-1)
        atoms = to_ase(sc)
        dfrac = v.get_scaled_positions() - atoms.get_scaled_positions()
        dfrac -= np.round(dfrac)
        if np.abs(v.get_forces() - disp["forces"]).max() > 1e-5 or np.abs(dfrac).max() > 1e-5 \
                or np.abs(v.cell.array - atoms.cell.array).max() > 1e-5:
            sys.exit(f"{name} disp {i}: vasprun does not match FORCE_SETS / phonopy supercell")
        atoms.calc = SinglePointCalculator(atoms, energy=v.get_potential_energy(), forces=v.get_forces(),
                                           stress=v.get_stress())
        atoms.info.update(material=name, displacement_index=i, displaced_atom=int(disp["number"]),
                          displacement=np.array(disp["displacement"]), config_type="phonon_displacement",
                          functional=FUNCTIONAL, theta_deg=tw["theta_deg"])
        frames.append(atoms)

    o = read(RELAX / name / "OUTCAR", index=-1)
    p = read(src / "POSCAR")
    if len(o) != len(p) or np.abs(o.cell.array - p.cell.array).max() > 1e-3 or \
            np.abs(o.get_scaled_positions() - p.get_scaled_positions()).max() > 1e-4:
        sys.exit(f"{name}: relaxation OUTCAR geometry != FINAL_RESULTS_TWISTED POSCAR")
    eq = p.copy()
    eq.calc = SinglePointCalculator(eq, energy=o.get_potential_energy(), forces=o.get_forces(), stress=o.get_stress())
    eq.info.update(material=name, config_type="relaxed_cell", functional=FUNCTIONAL, theta_deg=tw["theta_deg"],
                   kmesh=kmesh(RELAX / name / "KPOINTS"), kmesh_static=kmesh(vaspruns[0].parent / "KPOINTS"),
                   note="relaxation run (ISIF=2, EDIFFG=-1e-4), moire cell fixed")
    return frames, eq


def load_frames(name: str) -> tuple[list[Atoms], Atoms] | None:
    """The tracked label files if both exist, else None."""
    d, e = SOURCE / name / "displacements.extxyz", SOURCE / name / "equilibrium.extxyz"
    if not (d.exists() and e.exists()):
        return None
    frames = read(d, index=":")
    eq = read(e)
    for k in ("kmesh", "kmesh_static"):  # ASE parses "9 9 1" back into an int array; restore the string form
        if not isinstance(eq.info.get(k), str):
            eq.info[k] = " ".join(str(int(x)) for x in np.atleast_1d(eq.info[k]))
    for a in frames + [eq]:
        if a.calc is None:
            sys.exit(f"{name}: {d.name if a is not eq else e.name} has frames without energy/forces/stress")
    return frames, eq


def write_labels(name: str, force: bool = False) -> tuple[int, Path]:
    d, e = SOURCE / name / "displacements.extxyz", SOURCE / name / "equilibrium.extxyz"
    if (d.exists() or e.exists()) and not force:
        sys.exit(f"{name}: {d.name}/{e.name} already exist (use --force to overwrite)")
    frames, eq = build_frames(name)
    write(str(d), frames, format="extxyz")
    write(str(e), [eq], format="extxyz")
    return len(frames), d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="cell names (default: every dir in FINAL_RESULTS_TWISTED)")
    ap.add_argument("--force", action="store_true", help="overwrite existing extxyz files")
    ap.add_argument("--static-root", default=None, help="staticpoint run-dir root to read vasprun.xml from "
                    "(default phonopy_bilayer_examples_twisted; use phonopy_bilayer_examples_twisted_gridtight for grid-tight cells)")
    args = ap.parse_args()
    if args.static_root:
        global STATIC
        STATIC = ROOT / args.static_root
    names = args.names or sorted(p.name for p in SOURCE.iterdir() if (p / "FORCE_SETS").exists())
    for name in names:
        n, d = write_labels(name, args.force)
        print(f"{name}: {n} displaced supercells -> {d.relative_to(ROOT)}, relaxed cell -> equilibrium.extxyz")


if __name__ == "__main__":
    main()
