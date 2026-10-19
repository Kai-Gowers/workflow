#!/usr/bin/env python3
"""
Write a total phonon DOS (total_dos.dat) into FINAL_RESULTS*/<name>/ from the FORCE_CONSTANTS +
phonopy.yaml that postprocess_results.py already leaves there. Additive: nothing else is touched.

The header records every setting needed to reproduce the identical frequency grid for a model DOS
(nequix/scripts/eval_twisted_phonons.py reads it), so DFT and ML curves can be compared point-wise:

    # natoms = 42
    # mesh = 28 28 1 gamma_center
    # sigma = 0.1 THz
    # freq_min = -1.0 freq_max = 15.0 freq_pitch = 0.02
    # columns: frequency_THz total_dos_states_per_THz_per_cell
    <frequency> <dos>

Mesh: explicit [N, N, 1] with N = max(1, round(mesh_length * |a*|)) (phonopy's length rule, but never
more than one point along the vacuum axis). The DOS integrates to 3 * natoms; divide by natoms for a
per-atom DOS.

Usage (phonon-env, from workflow/):
  python3 phonopy/write_phonon_dos.py FINAL_RESULTS_TWISTED/MoS2_twist_m1_near0 [more dirs...]
  TWIST_VARIANT=twisted python3 phonopy/write_phonon_dos.py --all        # every dir under FINAL_RESULTS_TWISTED/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "common"))
from run_variant import generated_dir  # noqa: E402


def mesh_for(ph, mesh_length: float) -> list[int]:
    """[N, N, 1] with N from phonopy's length rule on the primitive reciprocal lattice (no 2 pi)."""
    rec = np.linalg.inv(np.array(ph.primitive.cell)).T
    n = max(1, int(round(mesh_length * np.linalg.norm(rec[0]))))
    return [n, n, 1]


def compute_total_dos(ph, mesh, sigma: float, freq_min: float, freq_max: float, freq_pitch: float):
    ph.run_mesh(mesh, is_gamma_center=True, with_eigenvectors=False)
    ph.run_total_dos(sigma=sigma, freq_min=freq_min, freq_max=freq_max, freq_pitch=freq_pitch)
    d = ph.get_total_dos_dict()
    return np.asarray(d["frequency_points"]), np.asarray(d["total_dos"])


def write_dos_dat(path: Path, freq, dos, *, natoms: int, mesh, sigma: float, freq_min: float, freq_max: float,
                  freq_pitch: float) -> None:
    with open(path, "w") as f:
        f.write(f"# natoms = {natoms}\n")
        f.write(f"# mesh = {mesh[0]} {mesh[1]} {mesh[2]} gamma_center\n")
        f.write(f"# sigma = {sigma} THz\n")
        f.write(f"# freq_min = {freq_min} freq_max = {freq_max} freq_pitch = {freq_pitch}\n")
        f.write("# columns: frequency_THz total_dos_states_per_THz_per_cell\n")
        for x, y in zip(freq, dos):
            f.write(f"{x:14.7f} {y:18.10f}\n")


def process_dir(d: Path, args) -> bool:
    import phonopy

    if not (d / "FORCE_CONSTANTS").exists() or not (d / "phonopy.yaml").exists():
        print(f"  {d.name}: no FORCE_CONSTANTS/phonopy.yaml - skipped")
        return False
    out = d / "total_dos.dat"
    if out.exists() and not args.force:
        print(f"  {d.name}: total_dos.dat exists (use --force) - skipped")
        return False
    ph = phonopy.load(phonopy_yaml=str(d / "phonopy.yaml"), force_constants_filename=str(d / "FORCE_CONSTANTS"),
                      is_compact_fc=True, log_level=0)
    mesh = args.mesh or mesh_for(ph, args.mesh_length)
    freq, dos = compute_total_dos(ph, mesh, args.sigma, args.freq_min, args.freq_max, args.freq_pitch)
    natoms = len(ph.primitive)
    write_dos_dat(out, freq, dos, natoms=natoms, mesh=mesh, sigma=args.sigma, freq_min=args.freq_min,
                  freq_max=args.freq_max, freq_pitch=args.freq_pitch)
    integral = (getattr(np, "trapezoid", None) or np.trapz)(dos, freq)
    print(f"  {d.name}: mesh {mesh} natoms {natoms} integral {integral:.3f} (expect {3 * natoms}) -> {out}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=Path, help="FINAL_RESULTS*/<name> dirs (phonopy.yaml + FORCE_CONSTANTS)")
    ap.add_argument("--all", action="store_true", help="every subdir of the active variant's FINAL_RESULTS dir")
    ap.add_argument("--mesh-length", type=float, default=200.0, help="A; N = round(length*|a*|) (default 200: 28x28x1 for the 42-atom MoS2 m1 cell, converged to <1e-4 states/THz/atom vs 42x42x1; 100 -> 14x14x1 was NOT converged, 0.003 off in 0-5 THz)")
    ap.add_argument("--mesh", type=int, nargs=3, default=None, help="explicit mesh, overrides --mesh-length")
    ap.add_argument("--sigma", type=float, default=0.1, help="THz smearing (default 0.1)")
    ap.add_argument("--freq-min", type=float, default=-1.0)
    ap.add_argument("--freq-max", type=float, default=15.0)
    ap.add_argument("--freq-pitch", type=float, default=0.02)
    ap.add_argument("--force", action="store_true", help="overwrite an existing total_dos.dat")
    args = ap.parse_args()

    dirs = list(args.dirs)
    if args.all:
        root = generated_dir("FINAL_RESULTS", WORKFLOW_ROOT)
        dirs += sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        ap.error("give result dirs or --all")
    print(f"Writing total_dos.dat for {len(dirs)} dir(s)")
    n_ok = sum(process_dir(d, args) for d in dirs)
    print(f"Done: {n_ok}/{len(dirs)} written")


if __name__ == "__main__":
    main()
