"""Write the gap-scan d_min structure into bilayer_examples_bare_pbe/<name>/{CONTCAR,POSCAR}.

d_min is the mean of a quadratic fit (±0.35 Å around the grid minimum) and a quartic fit (full grid) to E(d).
The previous top-level files are archived to <name>/pre_gapscan/ first (override with --archive=<dirname>, e.g.
--archive=pre_gapscan2 for the 2026-10-01 rescan at the ISIF=4 `a`). Used 2026-09-26 for all 11 batch-8 bilayers.
"""
import glob, os, shutil, sys
from pathlib import Path
import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "bilayer_examples_bare_pbe"

def dmin(name):
    ds, Es = [], []
    for d in sorted(glob.glob(str(ROOT / name / "d*/"))):
        ds.append(float(os.path.basename(d.rstrip("/"))[1:]))
        Es.append(float([l for l in open(d + "/OSZICAR") if "F=" in l][-1].split()[2]))
    ds, E = np.array(ds), np.array(Es)
    i = E.argmin()
    fits = []
    for deg, win in [(2, 0.35), (4, 9)]:
        sel = abs(ds - ds[i]) <= win
        p = np.polyfit(ds[sel], E[sel], deg)
        xs = np.linspace(ds[sel].min(), ds[sel].max(), 2001)
        fits.append(xs[np.polyval(p, xs).argmin()])
    assert ds.min() < np.mean(fits) < ds.max(), f"{name}: d_min at grid edge"
    return round(float(np.mean(fits)), 3)

ARCHIVE = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--archive=")), "pre_gapscan")

for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
    ex = SRC / name
    arch = ex / ARCHIVE
    assert not arch.exists(), f"{arch} already exists"
    arch.mkdir()
    for f in ex.iterdir():
        if f.is_file():
            shutil.copy2(f, arch / f.name)
    atoms = read(ex / "CONTCAR")
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    gap0 = z[order[3]] - z[order[2]]
    d = dmin(name)
    atoms.positions[order[3:], 2] += d - gap0
    for f in ("CONTCAR", "POSCAR"):
        write(ex / f, atoms, format="vasp", direct=True, sort=False)
    zn = np.sort(atoms.positions[:, 2])
    print(f"{name}: gap {gap0:.3f} -> {zn[3]-zn[2]:.3f} (d_min {d})")
