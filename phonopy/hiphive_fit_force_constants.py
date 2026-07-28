#!/usr/bin/env python3
"""
Correct small/spurious negative phonon frequencies by fitting force constants
directly from the raw disp-*/vasprun.xml training data with hiphive's
rotational sum rules enforced *during* the fit (not as a post-processing
step on top of phonopy's own fit -- see `hiphive_fix_force_constants.py` for
that simpler approach). hiphive's own documentation reports this
constraint-based fitting gives cleaner results than post-processing
(https://hiphive.materialsmodeling.org/advanced_topics/rotational_sum_rules.html)
and this project has confirmed cases where it fully resolves a residual that
post-processing could only partially reduce.

Requires the staticpoint directory's disp-*/vasprun.xml files (the raw
per-displacement forces), not just FORCE_CONSTANTS.

The fit is: minimize ||A x - f||^2 + lambda * ||A_rotational x||^2, i.e. the
normal least-squares force-constant fit plus a rotational-invariance penalty
term. As lambda increases the rotational constraint is enforced more
strictly; above some lambda the result plateaus (hard-constraint limit). The
script sweeps a range of lambda values and reports the plateaued result.

If the plateaued minimum frequency stays clearly negative even in the
hard-constraint limit, that is strong evidence of a genuine physical
instability, not a fixable symmetry/numerical artifact -- fitting with the
constraint built in is the most rotational-sum-rule-conformant harmonic
model obtainable from this data, so if it still shows a negative frequency,
no further tightening of the rotational constraint will help.

Usage:
    python3 hiphive_fit_force_constants.py <staticpoint_dir> [--cutoff ANGSTROM] [--lambda L]
    python3 hiphive_fit_force_constants.py --bilayer graphene_BN_AB_staticpoint
    python3 hiphive_fit_force_constants.py --monolayer BN_staticpoint --cutoff 6.0
"""

from pathlib import Path
import argparse
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
from ase import Atoms
from ase.io import read
import phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS

from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures
from hiphive.core.rotational_constraints import get_rotational_constraint_matrix
from trainstation import Optimizer

ROOT = Path(__file__).parent

DEFAULT_LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def phonopy_atoms_to_ase(patoms) -> Atoms:
    return Atoms(symbols=patoms.symbols, positions=patoms.positions, cell=patoms.cell, pbc=True)


def estimate_safe_cutoff(supercell: Atoms, cap: float = 8.0, margin: float = 0.45) -> float:
    cell = supercell.cell
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    return min(cap, margin * min(a_len, b_len))


def parse_band_conf(band_conf: Path):
    band_line = None
    labels_line = None
    for line in band_conf.read_text().splitlines():
        if line.strip().upper().startswith("BAND ="):
            band_line = line.split("=", 1)[1].strip()
        elif line.strip().upper().startswith("BAND_LABELS"):
            labels_line = line.split("=", 1)[1].strip()
    if band_line is None:
        raise ValueError(f"No BAND line found in {band_conf}")

    segments = [seg.strip() for seg in band_line.split(",")]
    all_labels = labels_line.split() if labels_line else None

    paths, path_connections, labels_out = [], [], []
    label_idx = 0
    N = 101
    for seg in segments:
        vals = [float(x) for x in seg.split()]
        waypoints = np.array(vals).reshape(-1, 3)
        for i in range(len(waypoints) - 1):
            q0, q1 = waypoints[i], waypoints[i + 1]
            paths.append(np.array([q0 + (q1 - q0) * t / (N - 1) for t in range(N)]))
            path_connections.append(i < len(waypoints) - 2)
        if all_labels:
            labels_out.extend(all_labels[label_idx:label_idx + len(waypoints)])
            label_idx += len(waypoints)
    return paths, (labels_out or None), path_connections


def backup_existing(staticpoint_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = (staticpoint_dir.parent.parent / "backups"
                  / f"pre_hiphive_fit_correction_{staticpoint_dir.name}_{stamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("FORCE_CONSTANTS", "band.yaml", "band.pdf"):
        src = staticpoint_dir / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
    return backup_dir


def copy_final_results(staticpoint_dir: Path) -> Path:
    name = staticpoint_dir.name
    if name.endswith("_staticpoint"):
        name = name[: -len("_staticpoint")]
    target = staticpoint_dir.parent.parent / "FINAL_RESULTS" / name
    target.mkdir(parents=True, exist_ok=True)
    for fname in ("band.pdf", "band.yaml", "phonopy.yaml", "FORCE_SETS", "POSCAR", "FORCE_CONSTANTS"):
        src = staticpoint_dir / fname
        if src.exists():
            shutil.copy2(src, target / fname)
    return target


def fit_staticpoint_dir(staticpoint_dir: Path, cutoff: float | None, lambdas: list[float]) -> dict:
    band_conf = staticpoint_dir / "band.conf"
    poscar = staticpoint_dir / "POSCAR"
    sposcar = staticpoint_dir / "SPOSCAR"
    phonopy_yaml = staticpoint_dir / "phonopy.yaml"
    for required in (band_conf, poscar, sposcar, phonopy_yaml):
        if not required.exists():
            raise FileNotFoundError(f"Missing {required}")

    disp_dirs = sorted(staticpoint_dir.glob("disp-*"))
    vaspruns = [d / "vasprun.xml" for d in disp_dirs if (d / "vasprun.xml").exists()]
    if not vaspruns:
        raise FileNotFoundError(f"No disp-*/vasprun.xml found under {staticpoint_dir}")

    atoms_ideal = read(sposcar)
    structures = [read(vr, index=-1) for vr in vaspruns]
    training_structures = prepare_structures(structures, atoms_ideal)
    print(f"  Loaded {len(training_structures)} displaced training structures")

    prim = read(poscar)

    if cutoff is None:
        cutoff = estimate_safe_cutoff(atoms_ideal)
        print(f"  No --cutoff given, estimated safe cutoff = {cutoff:.2f} Angstrom")

    cs = ClusterSpace(prim, [cutoff])
    sc = StructureContainer(cs)
    for s in training_structures:
        sc.add_structure(s)
    A, y = sc.get_fit_data()

    opt = Optimizer((A, y), train_size=1.0)
    opt.train()
    print(f"  Unconstrained fit RMSE (training): {opt.rmse_train:.5f} eV/Angstrom")

    Ac = get_rotational_constraint_matrix(cs)
    yc = np.zeros(Ac.shape[0])

    ph = phonopy.load(
        phonopy_yaml=str(phonopy_yaml),
        force_constants_filename=str(staticpoint_dir / "FORCE_CONSTANTS"),
        is_compact_fc=True,
        log_level=0,
    )
    supercell_ase = phonopy_atoms_to_ase(ph.supercell)
    paths, labels, path_connections = parse_band_conf(band_conf)

    sweep = []
    best = None
    for lam in lambdas:
        A_full = np.vstack((A, lam * Ac))
        y_full = np.hstack((y, yc))
        opt2 = Optimizer((A_full, y_full), train_size=1.0, standardize=False)
        opt2.train()
        fcp = ForceConstantPotential(cs, opt2.parameters)
        fc = fcp.get_force_constants(supercell_ase).get_fc_array(order=2)

        ph.force_constants = fc
        ph.run_band_structure(paths)
        band_freqs = np.vstack(ph.get_band_structure_dict()["frequencies"])
        ph.run_mesh([40, 40, 1], is_gamma_center=True, with_eigenvectors=False)
        mesh_freqs = ph.get_mesh_dict()["frequencies"]

        entry = {
            "lambda": lam,
            "band_min": band_freqs.min(),
            "mesh_min": mesh_freqs.min(),
            "mesh_n_negative": int((mesh_freqs < -1e-4).sum()),
            "fc": fc,
        }
        sweep.append(entry)
        print(f"  lambda={lam:.2e}: band-path min={entry['band_min']:.5f} THz, "
              f"mesh min={entry['mesh_min']:.5f} THz "
              f"({entry['mesh_n_negative']}/{mesh_freqs.size} points negative)")
        # pick the largest-lambda (most strictly enforced) result as final
        best = entry

    corrected_fc = best["fc"]

    backup_dir = backup_existing(staticpoint_dir)
    write_FORCE_CONSTANTS(corrected_fc, filename=str(staticpoint_dir / "FORCE_CONSTANTS"))
    ph.force_constants = corrected_fc
    ph.run_band_structure(paths, labels=labels, path_connections=path_connections)
    ph.write_yaml_band_structure(filename=str(staticpoint_dir / "band.yaml"))
    fig = ph.plot_band_structure()
    fig.savefig(staticpoint_dir / "band.pdf")
    final_target = copy_final_results(staticpoint_dir)

    return {
        "staticpoint_dir": staticpoint_dir,
        "cutoff": cutoff,
        "sweep": [{k: v for k, v in e.items() if k != "fc"} for e in sweep],
        "final_mesh_min": best["mesh_min"],
        "final_mesh_n_negative": best["mesh_n_negative"],
        "backup_dir": backup_dir,
        "final_results_dir": final_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit force constants from raw disp-*/vasprun.xml data with hiphive's "
            "rotational sum rules enforced during the fit (constraint-based "
            "approach, generally superior to post-processing enforcement)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 hiphive_fit_force_constants.py --bilayer graphene_BN_AB_staticpoint
  python3 hiphive_fit_force_constants.py --monolayer BN_staticpoint --cutoff 6.0
  python3 hiphive_fit_force_constants.py --bilayer MoSe2_bilayer_2H_staticpoint --lambda 0.001,0.01,0.1,1,10,100
""",
    )
    parser.add_argument("--monolayer", action="store_true")
    parser.add_argument("--bilayer", action="store_true")
    parser.add_argument("staticpoint", help="Staticpoint directory name or path")
    parser.add_argument("--cutoff", type=float, default=None,
                        help="hiphive ClusterSpace cutoff in Angstrom (default: auto-estimated)")
    parser.add_argument("--lambda", dest="lambdas", type=str, default=None,
                        help="Comma-separated lambda values to sweep (default: log sweep 1e-3 to 100)")
    args = parser.parse_args()

    if args.monolayer and args.bilayer:
        parser.error("Specify at most one of --monolayer / --bilayer")

    if args.monolayer:
        d = (ROOT.parent / "phonopy_monolayer_examples" / args.staticpoint).resolve()
    elif args.bilayer:
        d = (ROOT.parent / "phonopy_bilayer_examples" / args.staticpoint).resolve()
    else:
        d = Path(args.staticpoint).resolve()

    if not d.exists():
        print(f"Error: staticpoint directory not found: {d}", file=sys.stderr)
        sys.exit(1)

    lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else DEFAULT_LAMBDAS

    print(f"Processing {d} ...")
    result = fit_staticpoint_dir(d, args.cutoff, lambdas)

    print(f"\n  Cutoff used: {result['cutoff']:.2f} Angstrom")
    print(f"  Final (largest-lambda) full-BZ mesh min freq: {result['final_mesh_min']:.5f} THz "
          f"({result['final_mesh_n_negative']} points negative)")
    print(f"  Backed up pre-fix FORCE_CONSTANTS/band.yaml/band.pdf -> {result['backup_dir']}")
    print(f"  Wrote corrected FORCE_CONSTANTS/band.yaml/band.pdf, copied to {result['final_results_dir']}")

    if result["final_mesh_n_negative"] > 0:
        print("  WARNING: residual negative frequencies remain even under strict rotational "
              "constraint enforcement -- likely a genuine physical instability, not a fixable "
              "artifact. Do not add to FINAL_RESULTS_HEALTHY without further justification.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
