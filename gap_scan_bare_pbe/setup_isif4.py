#!/usr/bin/env python3
"""
Set up bare-PBE ISIF=4 lattice-constant runs in bilayer_examples_bare_pbe/<name>_isif4/.

The batch-8 bare-PBE bilayers were built at the PBE+D3 ISIF=4 'a' (7 pairs, +15-17 kB in-plane
stress) or at the relaxed-monolayer mean (4 pairs, -4..+6 kB). ISIF=4 at fixed volume gives each pair
its own bare-PBE 'a'. Only 'a' is kept from the run (extract it with common/isif4_lattice_extract.py,
and check the c drift is < ~2-3%); the gap is redone by a scan at the new 'a', because bare-PBE ionic
relaxations stall short of d_min.

Seed: the fixed-gap intralayer CONTCAR if that relaxation finished, otherwise the d_min POSCAR.
Selective dynamics is stripped (it fixes fractional z, which would drag the gap as c changes).
INCAR: the dir's current INCAR (EDIFF 1E-8, LREAL .FALSE.) with ISIF = 4; EDIFFG -1E-7 keeps it
NSW-limited, as in the PBE+D3 lattice-correction runs.

Usage: python3 setup_isif4.py <name> [<name> ...]   (then sbatch from each _isif4 dir)
"""

import re
import shutil
import sys
from pathlib import Path

from ase.io import read, write

ROOT = Path(__file__).resolve().parent.parent / "bilayer_examples_bare_pbe"


def setup(name):
    src = ROOT / name
    dst = ROOT / f"{name}_isif4"
    if dst.exists():
        raise SystemExit(f"{dst} already exists")
    contcar = src / "CONTCAR"
    finished = contcar.exists() and "reached required accuracy" in (src / "OUTCAR").read_text()
    seed = contcar if finished else src / "POSCAR"
    atoms = read(seed, format="vasp")
    del atoms.constraints

    dst.mkdir()
    write(dst / "POSCAR", atoms, format="vasp", direct=True, sort=False)
    for f in ("KPOINTS", "POTCAR", "bat"):
        shutil.copy2(src / f, dst / f)
    incar = (src / "INCAR").read_text()
    incar = re.sub(r"^SYSTEM.*$", f"SYSTEM = {name} bare-PBE ISIF=4 lattice correction",
                   incar, count=1, flags=re.M)
    incar, n = re.subn(r"^ISIF\s*=.*$", "ISIF = 4 # fixed-volume shape+ion relax, residual-strain correction",
                       incar, count=1, flags=re.M)
    assert n == 1, f"no ISIF line in {src}/INCAR"
    (dst / "INCAR").write_text(incar)
    print(f"{name}: seed={seed.relative_to(ROOT)}  a={atoms.cell.lengths()[0]:.6f}  c={atoms.cell.lengths()[2]:.4f}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        setup(n)
