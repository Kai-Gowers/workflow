"""Set up a fixed-gap intralayer relaxation of the d_min structure in bilayer_examples_bare_pbe/<name>.

apply_dmin.py only rigidly shifted the top layer, so the atoms inside each layer still carry
0.005-0.036 eV/Å of force. That is enough to make the interlayer phonon modes imaginary. Here the z of
the two facing chalcogens (the atoms bounding the gap) is frozen with selective dynamics and everything
else relaxes. The INCAR is tightened (EDIFF 1E-8, LREAL .FALSE.) so the forces are clean to ~1e-4 eV/Å.
All top-level files are first moved to <name>/pre_intralayer/ (override with --archive=<dirname>). Used 2026-09-30 for all
11 batch-8 bilayers.
"""
import re, shutil, sys
from pathlib import Path
import numpy as np
from ase.io import read, write

SRC = Path(__file__).resolve().parent.parent / "bilayer_examples_bare_pbe"

ARCHIVE = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--archive=")), "pre_intralayer")

for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
    ex = SRC / name
    arch = ex / ARCHIVE
    assert not arch.exists(), f"{arch} already exists"
    pos, con = read(ex / "POSCAR"), read(ex / "CONTCAR")
    assert np.allclose(pos.positions, con.positions, atol=1e-6), f"{name}: POSCAR != CONTCAR (not the d_min structure?)"
    arch.mkdir()
    for f in ex.iterdir():
        if f.is_file():
            shutil.move(str(f), arch / f.name)
    for f in ("INCAR", "KPOINTS", "POTCAR", "bat"):
        shutil.copy2(arch / f, ex / f)

    atoms = pos.copy()
    atoms.set_momenta(None)
    order = np.argsort(atoms.positions[:, 2])
    inner = [int(order[2]), int(order[3])]
    # ASE here writes FixCartesian as all "T T T", so set the selective-dynamics flags by hand
    write(ex / "POSCAR", atoms, format="vasp", direct=True, sort=False)
    lines = (ex / "POSCAR").read_text().splitlines()
    i0 = next(k for k, l in enumerate(lines) if l.strip().lower().startswith("direct")) + 1
    lines.insert(i0 - 1, "Selective dynamics")
    i0 += 1
    for j in range(len(atoms)):
        xyz = lines[i0 + j].split()[:3]
        lines[i0 + j] = "  " + "  ".join(xyz) + ("   T   T   F" if j in inner else "   T   T   T")
    (ex / "POSCAR").write_text("\n".join(lines[: i0 + len(atoms)]) + "\n")

    incar = (ex / "INCAR").read_text()
    incar = re.sub(r"(?m)^SYSTEM\s*=.*$", f"SYSTEM = {name} fixed-gap intralayer relaxation", incar)
    incar = re.sub(r"(?m)^EDIFF\s*=.*$", "EDIFF = 1E-8", incar)
    incar = re.sub(r"(?m)^LREAL\s*=.*$", "LREAL = .FALSE.", incar)
    (ex / "INCAR").write_text(incar)
    z = np.sort(atoms.positions[:, 2])
    print(f"{name}: gap {z[3]-z[2]:.4f} Å, frozen z of atoms {inner} ({atoms.get_chemical_symbols()[inner[0]]}, "
          f"{atoms.get_chemical_symbols()[inner[1]]})")
