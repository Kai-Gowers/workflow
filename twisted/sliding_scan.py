#!/usr/bin/env python3
"""Rigid interlayer-sliding energy scan for a relaxed twisted cell (diagnostic for the Gamma shear pair).

Builds sliding_scan_twisted/<cell>/dx_<d>/ with layer 2 (z > median z) shifted rigidly by d Angstrom along
Cartesian x, static VASP inputs copied from the twisted staticpoint template (EDIFF tightened to 1E-7, ISYM=0 so
the symmetry-broken slid cells are treated identically to d=0). Energies are variational, so E(d) resolves the
shear curvature K = d2E/dd2 per cell independently of finite-difference force noise.

  python3 twisted/sliding_scan.py MoS2_twist_m1_near0 --disp 0 0.1 0.2 0.3 0.5 [--submit]
  python3 twisted/sliding_scan.py MoS2_twist_m1_near0 --analyze     # fit E(d), print K and rigid-shear frequency
"""
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "common" / "staticpoint_templates_twisted"
AMU, EV_A2_TO_N_M = 1.66053907e-27, 16.0218
MASS = {"Mo": 95.95, "W": 183.84, "S": 32.06, "Se": 78.971, "Te": 127.60}


def read_poscar(p):
    L = Path(p).read_text().splitlines()
    scale = float(L[1]); cell = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)]) * scale
    species = L[5].split(); counts = [int(x) for x in L[6].split()]
    i = 7
    if L[i].strip()[0] in "sS": i += 1
    direct = L[i].strip()[0] in "dD"; n = sum(counts)
    pos = np.array([[float(x) for x in L[i + 1 + k].split()[:3]] for k in range(n)])
    if direct: pos = pos @ cell
    return L[0], cell, species, counts, pos


def write_poscar(p, comment, cell, species, counts, pos):
    with open(p, "w") as f:
        f.write(f"{comment}\n1.0\n")
        for v in cell: f.write("  %.12f %.12f %.12f\n" % tuple(v))
        f.write(" ".join(species) + "\n" + " ".join(map(str, counts)) + "\nCartesian\n")
        for r in pos: f.write("  %.12f %.12f %.12f\n" % tuple(r))


def build(cell_name, disps, submit):
    src = ROOT / "bilayer_examples_twisted" / cell_name
    sp = ROOT / "phonopy_bilayer_examples_twisted" / f"{cell_name}_staticpoint" / "disp-001"
    comment, cell, species, counts, pos = read_poscar(src / "CONTCAR")
    assert species == read_poscar(sp / "POSCAR")[2], "species order differs from staticpoint POSCAR"
    layer2 = pos[:, 2] > np.median(pos[:, 2])
    assert layer2.sum() == len(pos) // 2, "layer split by median z failed"
    incar = (TEMPLATE / "INCAR").read_text()
    incar = re.sub(r"^EDIFF\s*=.*$", "EDIFF = 1E-7", incar, flags=re.M)
    incar += "\n# sliding scan: no symmetry so d=0 and d>0 are treated identically\nISYM = 0\n"
    out = ROOT / "sliding_scan_twisted" / cell_name; out.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for d in disps:
        wd = out / f"dx_{d:.2f}"; wd.mkdir(exist_ok=True)
        p = pos.copy(); p[layer2, 0] += d
        write_poscar(wd / "POSCAR", f"{comment.strip()} | layer2 slid {d:.2f} A along x", cell, species, counts, p)
        (wd / "INCAR").write_text(incar)
        for f in ("KPOINTS", "POTCAR"): shutil.copy2(sp / f, wd / f)
        shutil.copy2(TEMPLATE / "bat", wd / "bat")
        if submit:
            r = subprocess.run(["sbatch", "--parsable", f"--job-name={cell_name}-slide-{d:.2f}", "bat"],
                               cwd=wd, capture_output=True, text=True, check=True)
            jobs[f"{d:.2f}"] = r.stdout.strip().split(";")[0]
            (wd / "latest_job_id").write_text(jobs[f"{d:.2f}"] + "\n")
        print(f"  {wd.relative_to(ROOT)}  job {jobs.get(f'{d:.2f}', '-')}")
    (out / "scan.json").write_text(json.dumps({"cell": cell_name, "disp_A": list(disps), "axis": "x",
                                                "layer2_atoms": int(layer2.sum()), "jobs": jobs}, indent=2))


def analyze(cell_name):
    out = ROOT / "sliding_scan_twisted" / cell_name
    _, _, species, counts, pos = read_poscar(out / sorted(out.glob("dx_*"))[0] / "POSCAR")
    layer2 = pos[:, 2] > np.median(pos[:, 2])
    masses = np.repeat([MASS[s] for s in species], counts)
    mu = masses[layer2].sum() * AMU / 2  # reduced mass of two equal layers (per cell)
    d, E = [], []
    for wd in sorted(out.glob("dx_*")):
        osz = wd / "OSZICAR"
        if not osz.exists(): continue
        m = re.findall(r"F=\s*(\S+)\s+E0=\s*(\S+)", osz.read_text())
        if not m: continue
        d.append(float(wd.name[3:])); E.append(float(m[-1][1]))
    d, E = np.array(d), np.array(E)
    if len(d) < 3: print(f"{cell_name}: only {len(d)} finished points"); return
    print(f"{cell_name}: E(d) - E(0) [meV]: " + ", ".join(f"{x:.2f}:{(e - E[d.argmin()]) * 1e3:+.3f}" for x, e in zip(d, E)))
    for name, sel in (("all points", np.ones_like(d, bool)), ("d<=0.3", d <= 0.3)):
        if sel.sum() < 3: continue
        c = np.polyfit(d[sel], E[sel], 4 if sel.sum() >= 5 else 2)
        K = 2 * c[-3]  # d2E/dd2 at d=0 (eV/A^2 per cell)
        w = np.sign(K) * np.sqrt(abs(K) * EV_A2_TO_N_M / mu) / 2 / np.pi / 1e12
        print(f"  fit {name:10s} (deg {len(c) - 1}): K = {K:+.4f} eV/A^2/cell -> rigid shear {w:+.3f} THz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cell"); ap.add_argument("--disp", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--submit", action="store_true"); ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze(a.cell) if a.analyze else build(a.cell, a.disp, a.submit)
