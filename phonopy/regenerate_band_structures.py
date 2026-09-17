#!/usr/bin/env python3
"""Regenerate band.yaml / band.pdf in FINAL_RESULTS*/<material>/ from the stored
phonopy.yaml + FORCE_CONSTANTS, along the corrected Γ-K-M-Γ path.

Background (2026-09-17): the workflow POSCARs are 60° hexagonal cells, where
K = (2/3, 1/3, 0) as band.conf said. But the phonopy CLI defaults to PRIMITIVE_AXES = AUTO
and re-expresses band q-points in a standardized 120° primitive cell (primitive_matrix
[[1,-1,0],[0,1,0],[0,0,1]] in phonopy.yaml). In that basis (2/3, 1/3, 0) lies 2/3 of the way
from Γ to a neighbouring M, so every "K" tick was mislabelled and the true K corner was
never sampled. Correct 120° path: Γ (0,0,0) - K (1/3,1/3,0) -
M (1/2,0,0) - Γ. FORCE_CONSTANTS are untouched; only the sampled q-path changes.

Each file keeps its original number of q-points per segment (51 from the phonopy CLI,
101 from the hiphive scripts) so downstream q-list consumers see the same shape.
Γ and M endpoint frequencies are asserted unchanged against the old band.yaml.

Usage:
    python3 phonopy/regenerate_band_structures.py            # all FINAL_RESULTS*/*
    python3 phonopy/regenerate_band_structures.py --dry-run  # report only, write nothing
    python3 phonopy/regenerate_band_structures.py --only MoS2 WTe2
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")
import phonopy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WAYPOINTS = np.array([[0, 0, 0], [1 / 3, 1 / 3, 0], [0.5, 0, 0], [0, 0, 0]])
LABELS = ["Γ", "K", "M", "Γ"]


def segment(q0, q1, n):
    return np.array([q0 + (q1 - q0) * t / (n - 1) for t in range(n)])


def regenerate(material_dir: Path, dry_run: bool) -> dict:
    old = yaml.safe_load((material_dir / "band.yaml").read_text())
    seg_n = old["segment_nqpoint"]
    if len(seg_n) != 3:
        raise ValueError(f"{material_dir}: expected 3 segments, found {seg_n}")
    f_old = np.array([[b["frequency"] for b in p["band"]] for p in old["phonon"]])
    q_old = np.array([p["q-position"] for p in old["phonon"]])

    ph = phonopy.load(
        str(material_dir / "phonopy.yaml"),
        force_constants_filename=str(material_dir / "FORCE_CONSTANTS"),
        log_level=0,
    )
    cell = ph.primitive.cell
    ang = np.degrees(np.arccos(cell[0] @ cell[1] / np.linalg.norm(cell[0]) / np.linalg.norm(cell[1])))
    if abs(ang - 120) > 0.5:
        raise ValueError(f"{material_dir}: primitive in-plane angle {ang:.1f}°, expected 120°")

    paths = [segment(WAYPOINTS[i], WAYPOINTS[i + 1], seg_n[i]) for i in range(3)]
    ph.run_band_structure(paths, labels=LABELS, path_connections=[True, True, False])
    bs = ph.get_band_structure_dict()
    f_new = np.vstack(bs["frequencies"])

    # Γ (start of seg 1, end of seg 3) and M (end of seg 2 / start of seg 3) must be unchanged.
    i_m = seg_n[0] + seg_n[1] - 1
    checks = {"Γ_start": (0, 0), "M": (i_m, i_m), "Γ_end": (len(f_old) - 1, len(f_new) - 1)}
    for name, (io, inew) in checks.items():
        assert np.allclose(q_old[io], q_new_expected(name)), f"{material_dir}: unexpected old q at {name}: {q_old[io]}"
        d = np.abs(f_old[io] - f_new[inew]).max()
        if d > 1e-4:
            raise AssertionError(f"{material_dir}: {name} frequencies changed by {d:.2e} THz")

    i_k = seg_n[0] - 1
    out = {
        "material": material_dir.name,
        "set": material_dir.parent.name,
        "nq": int(sum(seg_n)),
        "min_old_label_K": float(f_old[i_k].min()),
        "min_true_K": float(f_new[i_k].min()),
        "min_path_old": float(f_old.min()),
        "min_path_new": float(f_new.min()),
    }
    if not dry_run:
        ph.write_yaml_band_structure(filename=str(material_dir / "band.yaml"))
        import matplotlib

        matplotlib.use("Agg")
        fig = ph.plot_band_structure()
        fig.savefig(material_dir / "band.pdf")
        import matplotlib.pyplot as plt

        plt.close("all")
    return out


def q_new_expected(name):
    return {"Γ_start": [0, 0, 0], "M": [0.5, 0, 0], "Γ_end": [0, 0, 0]}[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="material names to process")
    args = ap.parse_args()

    dirs = sorted(d for root in ROOT.glob("FINAL_RESULTS*") for d in root.iterdir()
                  if (d / "phonopy.yaml").exists() and (d / "FORCE_CONSTANTS").exists() and (d / "band.yaml").exists())
    if args.only:
        dirs = [d for d in dirs if d.name in set(args.only)]
    print(f"{len(dirs)} material dirs{' (dry run)' if args.dry_run else ''}")
    print(f"{'set':22s} {'material':30s} {'nq':>4s} {'min@oldK':>9s} {'min@trueK':>10s} {'minpath old':>12s} {'new':>8s}")
    failures = []
    for d in dirs:
        try:
            r = regenerate(d, args.dry_run)
        except Exception as e:  # keep going, report at the end
            failures.append((d, str(e)))
            print(f"{d.parent.name:22s} {d.name:30s} FAILED: {e}")
            continue
        print(f"{r['set']:22s} {r['material']:30s} {r['nq']:4d} {r['min_old_label_K']:9.3f} {r['min_true_K']:10.3f} "
              f"{r['min_path_old']:12.3f} {r['min_path_new']:8.3f}")
    print(f"\ndone: {len(dirs) - len(failures)} ok, {len(failures)} failed")
    for d, e in failures:
        print(f"  {d}: {e}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
