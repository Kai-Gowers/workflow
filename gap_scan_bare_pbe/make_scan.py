"""Rigid interlayer-gap scan for bare-PBE bilayers whose ISIF=2 relaxation won't settle.

For each bilayer, take the latest CONTCAR, rigidly shift the top layer (the upper 3 atoms by z)
so the chalcogen-chalcogen gap equals d, and set up a static SCF with the production relaxation
INCAR (NSW=0). Energies vs d give the gap that minimizes the energy.
"""
import shutil, sys
from pathlib import Path
import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "bilayer_examples_bare_pbe"
# Default: the 4 that never settled. Pass names on the command line to scan others; --wide adds 3.4/3.5 Å
# for bilayers whose relaxed gap is below 3.6 Å (the 7 accepted batch-8 bilayers, 2026-09-26).
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NAMES = args or ["MoS2_MoTe2_2H", "MoS2_MoSe2_3R", "MoS2_WS2_3R", "MoS2_MoTe2_3R"]
GAPS = np.round(np.arange(3.4 if "--wide" in sys.argv else 3.6, 4.61, 0.1), 2)

for name in NAMES:
    atoms = read(SRC / name / "CONTCAR")
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    top = order[3:]
    gap0 = z[order[3]] - z[order[2]]
    for d in GAPS:
        out = ROOT / name / f"d{d:.2f}"
        out.mkdir(parents=True, exist_ok=True)
        a = atoms.copy()
        a.positions[top, 2] += d - gap0
        write(out / "POSCAR", a, format="vasp", direct=True, sort=False)
        for f in ("POTCAR", "KPOINTS"):
            shutil.copy(SRC / name / f, out / f)
        incar = (SRC / name / "INCAR").read_text()
        incar = incar.replace("relaxation", "gap scan").replace("NSW = 400", "NSW = 0")
        incar = incar.replace("IBRION = 1", "IBRION = -1").replace("EDIFFG = -1E-7\n", "")
        (out / "INCAR").write_text(incar)
    print(f"{name}: gap0={gap0:.4f}, set up {len(GAPS)} points")
