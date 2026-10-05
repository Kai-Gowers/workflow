"""Analyze the rigid slide scan (make_slide_scan.py): sliding stiffness -> Gamma shear frequency.

Two independent readouts per bilayer, both from the x points (top layer along a1):
  energy: fit E(s) = E0 + k/2 s^2 + c3 s^3 + c4 s^4 (eV per primitive cell)
  force : the net force on the top layer along a1, F(s) = -k s - 3 c3 s^2 - ..., gives the slope at s=0
Frequency of a rigid relative slide: f = sqrt(k/mu)/2pi, mu = M_top*M_bot/(M_top+M_bot).
This is the rigid-layer estimate of the Gamma shear mode. Intralayer relaxation can only lower it,
so k < 0 here means the instability is real.
Checks: y vs x curvature (E mode -> equal), b (bottom moved) vs x at the same relative shift (egg-box).

Usage: python3 analyze_slide_scan.py [-v] [names...]
"""
import sys
from pathlib import Path
import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parent / "slide"
EV_A2_AMU_TO_THZ = np.sqrt(16.02176634 / 1.66053907e-27) / (2 * np.pi) / 1e12   # sqrt(eV/A^2/amu) -> THz


def load(d: Path):
    osz = d / "OSZICAR"
    out = d / "OUTCAR"
    if not (out.exists() and "General timing" in out.read_text()):
        return None
    e = float([l for l in osz.read_text().splitlines() if " F=" in l][-1].split("E0=")[1].split()[0])
    atoms = read(out, format="vasp-out")
    return e, atoms


def signed_freq(k, mu):
    return np.sign(k) * np.sqrt(abs(k) / mu) * EV_A2_AMU_TO_THZ


VERBOSE = "-v" in sys.argv
names = [a for a in sys.argv[1:] if not a.startswith("-")] or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
print(f"{'bilayer':16s} {'n':>3s} {'k_E(eV/A2)':>11s} {'k_F(eV/A2)':>11s} {'k_y/k_x':>8s} "
      f"{'f_E(THz)':>9s} {'f_F(THz)':>9s} {'dE(0.3)ueV':>10s} {'eggbox ueV':>10s}")
for name in names:
    pts = {p.name: load(p) for p in sorted((ROOT / name).iterdir())}
    done = {k: v for k, v in pts.items() if v is not None}
    if "x+0.00" not in done:
        print(f"{name:16s} {len(done):3d}/{len(pts)} not ready"); continue
    a0 = done["x+0.00"][1]
    z = a0.positions[:, 2]; order = np.argsort(z); top, bot = order[3:], order[:3]
    m = a0.get_masses(); mu = m[top].sum() * m[bot].sum() / m.sum()
    ex = a0.cell[0] / np.linalg.norm(a0.cell[0]); ey = np.cross([0, 0, 1], ex)
    E0 = done["x+0.00"][0]

    def series(kind, e_dir):
        s, E, F = [], [], []
        for key, (e, at) in done.items():
            if key[0] == kind or key == "x+0.00":
                s.append(0.0 if key == "x+0.00" else float(key[1:]))
                E.append(e - E0)
                F.append(at.get_forces()[top].sum(0) @ e_dir)
        o = np.argsort(s)
        return np.array(s)[o], np.array(E)[o], np.array(F)[o]

    sx, Ex, Fx = series("x", ex)
    deg = 4 if len(sx) >= 7 else 2
    pE = np.polyfit(sx, Ex, deg)
    kE = 2 * pE[-3]
    pF = np.polyfit(sx, Fx, min(3, len(sx) - 1))
    kF = -pF[-2]
    sy, Ey, _ = series("y", ey)
    ky = 2 * np.polyfit(sy, Ey, 2)[-3] if len(sy) >= 3 else np.nan
    kx2 = 2 * np.polyfit(sx[np.abs(sx) <= 0.2 + 1e-9], Ex[np.abs(sx) <= 0.2 + 1e-9], 2)[-3]
    egg = [abs(done[f"b{s:+.2f}"][0] - done[f"x{s:+.2f}"][0]) * 1e6
           for s in (0.2, -0.2) if f"b{s:+.2f}" in done and f"x{s:+.2f}" in done]
    dE3 = np.mean([done[k][0] - E0 for k in ("x+0.30", "x-0.30") if k in done]) * 1e6
    print(f"{name:16s} {len(done):3d} {kE:+11.2e} {kF:+11.2e} {ky / kx2:8.2f} "
          f"{signed_freq(kE, mu):+9.3f} {signed_freq(kF, mu):+9.3f} {dE3:10.1f} "
          f"{(max(egg) if egg else np.nan):10.2f}")
    if VERBOSE:
        for s_, e_, f_ in zip(sx, Ex, Fx):
            print(f"    s={s_:+.2f}  dE={e_ * 1e6:+9.2f} ueV  F_top={f_:+.2e} eV/A")
