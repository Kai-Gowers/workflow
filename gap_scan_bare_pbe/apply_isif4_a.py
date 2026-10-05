"""Rebuild bilayer_examples_bare_pbe/<name>/{POSCAR,CONTCAR} at the bare-PBE ISIF=4 lattice constant.

Seed: <name>_isif4/CONTCAR, which already has its intralayer coordinates relaxed at the new `a`. The cell is
rescaled (same in-plane vectors) to `a` from data/bilayer_lattice_overrides_bare_pbe.json, and c is reset to
the production 20 Å. In-plane fractional coordinates and Cartesian z are kept, so only the vacuum changes
(ISIF=4 at fixed volume had shrunk c by up to 2.3%). The gap is NOT trusted; make_scan.py redoes it next.
The previous top-level files are moved to <name>/pre_isif4a/ (INCAR/KPOINTS/POTCAR/bat are copied back).
Used 2026-10-01 for all 11 batch-8 bilayers.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write

WF = Path(__file__).resolve().parent.parent
SRC = WF / "bilayer_examples_bare_pbe"
OVR = json.loads((WF / "data" / "bilayer_lattice_overrides_bare_pbe.json").read_text())["bilayers"]
C = 20.0

for name in sys.argv[1:]:
    ex, i4 = SRC / name, SRC / f"{name}_isif4"
    arch = ex / "pre_isif4a"
    assert not arch.exists(), f"{arch} already exists"
    a = float(OVR[name]["a"])
    seed = read(i4 / "CONTCAR")
    la, lb, lc, al, be, ga = seed.cell.cellpar()
    assert abs(la - lb) < 1e-4 and min(abs(ga - 120), abs(ga - 60)) < 1e-3 and abs(al - 90) < 1e-3 and abs(be - 90) < 1e-3, \
        f"{name}: ISIF=4 cell is not hexagonal {seed.cell.cellpar()}"
    assert abs(la - a) < 1e-3, f"{name}: CONTCAR a {la:.6f} far from override {a:.6f}"

    arch.mkdir()
    for f in ex.iterdir():
        if f.is_file():
            shutil.move(str(f), arch / f.name)
    for f in ("INCAR", "KPOINTS", "POTCAR", "bat"):
        shutil.copy2(arch / f, ex / f)

    frac = seed.get_scaled_positions(wrap=False)
    cell = np.array(seed.cell)  # keep the seed's in-plane convention (60° or 120°), rescaled to exactly `a`
    cell[:2] *= a / la
    cell[:2, 2] = 0.0
    cell[2] = [0.0, 0.0, C]
    pos = frac[:, :2] @ cell[:2, :2]
    atoms = Atoms(seed.get_chemical_symbols(), cell=cell, pbc=True,
                  positions=np.column_stack([pos, seed.positions[:, 2]]))
    for f in ("POSCAR", "CONTCAR"):
        write(ex / f, atoms, format="vasp", direct=True, sort=False)
    z = np.sort(atoms.positions[:, 2])
    print(f"{name}: a {la:.6f} -> {a:.6f}, c {lc:.3f} -> {C:.3f}, gap {z[3]-z[2]:.3f} Å (seed from ISIF=4)")
