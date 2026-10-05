"""Rigid in-plane slide scan for bare-PBE bilayers: E(dx) of one layer sliding over the other.

Phonopy finds a tiny doubly degenerate interlayer shear mode at Gamma (-0.07..-0.09 THz in
MoS2_bilayer_2H / MoS2_MoSe2_3R, 2026-10-03). It is near the finite-displacement force resolution,
so this measures the sliding stiffness directly. For each bilayer, take the fixed-gap
intralayer-relaxed CONTCAR (the phonopy primitive cell), rigidly shift the top layer in-plane, and
run a static SCF.
  x points: top layer along a1 by 0, +-0.05, +-0.1, +-0.2, +-0.3 A  (curvature + cubic term; 3R lacks inversion)
  y points: top layer along the in-plane perpendicular by +-0.1, +-0.2 A (E mode -> should match x)
  b points: BOTTOM layer by -0.2/+0.2 A along a1 (same relative shift as x+0.20/x-0.20 but at a
            different position relative to the FFT grid -> egg-box check)
INCAR = the tight intralayer INCAR with NSW=0 and ISYM=0. ISYM=0 is needed because the shifted cells
have lower symmetry. With symmetry on, the irreducible k-set would change between points and give a
spurious energy offset of the same size as the signal.

Usage: python3 make_slide_scan.py [names...]   (default: all 11 batch-8 bilayers)
"""
import shutil, sys
from pathlib import Path
import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "bilayer_examples_bare_pbe"
PHON = ROOT.parent / "phonopy_bilayer_examples_bare_pbe"
OUT = ROOT / "slide"
ALL = ["MoS2_bilayer_2H", "MoS2_MoSe2_2H", "MoS2_MoSe2_3R", "MoS2_MoTe2_2H", "MoS2_MoTe2_3R",
       "MoS2_WS2_2H", "MoS2_WS2_3R", "MoS2_WSe2_2H", "MoS2_WSe2_3R", "MoS2_WTe2_2H", "MoS2_WTe2_3R"]
NAMES = sys.argv[1:] or ALL

POINTS = [("x", s) for s in (0.0, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3)]
POINTS += [("y", s) for s in (0.1, -0.1, 0.2, -0.2)]
POINTS += [("b", s) for s in (0.2, -0.2)]   # b: bottom moves by -s, i.e. relative shift +s

for name in NAMES:
    atoms = read(SRC / name / "CONTCAR")
    del atoms.constraints   # CONTCAR carries the selective-dynamics flags from the fixed-gap relax
    # Sanity: this must be the same structure phonopy displaced.
    ref = read(PHON / f"{name}_staticpoint" / "POSCAR")
    assert np.allclose(atoms.cell, ref.cell, atol=1e-6)
    assert np.allclose(atoms.get_scaled_positions(), ref.get_scaled_positions(), atol=1e-6), name
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    top, bot = order[3:], order[:3]
    ex = atoms.cell[0] / np.linalg.norm(atoms.cell[0])
    ey = np.cross([0, 0, 1], ex)
    incar = (SRC / name / "INCAR").read_text()
    assert "EDIFF = 1E-8" in incar and "LREAL = .FALSE." in incar
    incar = incar.replace("fixed-gap intralayer relaxation", "rigid slide scan")
    incar = incar.replace("NSW = 400", "NSW = 0").replace("IBRION = 1", "IBRION = -1")
    incar = incar.replace("EDIFFG = -1E-7\n", "").replace("ISIF = 2 # Gemini suggested 4", "ISIF = 2")
    incar = incar.rstrip("\n") + "\n\n# slide scan: identical k-set at every point\nISYM = 0\n"
    for kind, s in POINTS:
        out = OUT / name / f"{kind}{s:+.2f}"
        out.mkdir(parents=True, exist_ok=True)
        a = atoms.copy()
        if kind == "x":
            a.positions[top] += s * ex
        elif kind == "y":
            a.positions[top] += s * ey
        else:
            a.positions[bot] -= s * ex
        write(out / "POSCAR", a, format="vasp", direct=True, sort=False)
        for f in ("POTCAR", "KPOINTS"):
            shutil.copy(SRC / name / f, out / f)
        (out / "INCAR").write_text(incar)
    print(f"{name}: {len(POINTS)} points -> {OUT / name}")
