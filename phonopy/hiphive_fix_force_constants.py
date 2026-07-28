#!/usr/bin/env python3
"""
Correct small/spurious negative phonon frequencies near Gamma using hiphive's
rotational sum rule enforcement (https://hiphive.materialsmodeling.org/advanced_topics/rotational_sum_rules.html).

Loads a staticpoint directory's existing phonopy FORCE_CONSTANTS, projects it
onto a hiphive ClusterSpace with a finite real-space cutoff, enforces the
Huang and Born-Huang rotational sum rules, and writes the corrected force
constants back along with a regenerated band.yaml/band.pdf (same BAND/
BAND_LABELS path as the existing band.conf) and an updated FINAL_RESULTS/
copy.

This is NOT a fix for genuine physical instabilities -- it only removes
small residual imaginary frequencies caused by the raw DFT force constants
not exactly satisfying rotational/translational invariance (common for
finite-precision finite-difference force constants on 2D/vdW systems). Only
apply this to materials whose imaginary modes are small in magnitude; check
the reconstruction error and full-BZ mesh scan this script reports before
trusting the result.

Usage:
    python3 hiphive_fix_force_constants.py <staticpoint_dir> [--cutoff ANGSTROM]
    python3 hiphive_fix_force_constants.py --bilayer MoS2_WS2_3R_staticpoint
    python3 hiphive_fix_force_constants.py --monolayer HfSe2_staticpoint --cutoff 6.5

If --cutoff is omitted, a safe default is estimated from the supercell's
in-plane geometry (see `estimate_safe_cutoff`).
"""

from pathlib import Path
import argparse
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
from ase import Atoms
import phonopy
from phonopy.harmonic.force_constants import compact_fc_to_full_fc
from phonopy.file_IO import write_FORCE_CONSTANTS

from hiphive import ForceConstants, ClusterSpace, ForceConstantPotential
from hiphive import enforce_rotational_sum_rules
from hiphive.utilities import extract_parameters

ROOT = Path(__file__).parent


def phonopy_atoms_to_ase(patoms) -> Atoms:
    return Atoms(symbols=patoms.symbols, positions=patoms.positions, cell=patoms.cell, pbc=True)


def parse_band_conf(band_conf: Path):
    """Parse BAND / BAND_LABELS from an existing band.conf into phonopy
    run_band_structure() arguments: (paths, labels, path_connections)."""
    band_line = None
    labels_line = None
    for line in band_conf.read_text().splitlines():
        if line.strip().upper().startswith("BAND ="):
            band_line = line.split("=", 1)[1].strip()
        elif line.strip().upper().startswith("BAND_LABELS"):
            labels_line = line.split("=", 1)[1].strip()

    if band_line is None:
        raise ValueError(f"No BAND line found in {band_conf}")

    # phonopy allows comma-separated disconnected segments; each segment is
    # a whitespace-separated list of q-point triples
    segments = [seg.strip() for seg in band_line.split(",")]
    all_labels = labels_line.split() if labels_line else None

    paths = []
    path_connections = []
    label_idx = 0
    labels_out = []
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


def estimate_safe_cutoff(supercell: Atoms, cap: float = 8.0, margin: float = 0.45) -> float:
    """Estimate a safe hiphive ClusterSpace cutoff from the supercell's
    in-plane lattice vector lengths (assumes vacuum along c, as in this
    project's 2D bilayer/monolayer supercells)."""
    cell = supercell.cell
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    return min(cap, margin * min(a_len, b_len))


def backup_existing(staticpoint_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = staticpoint_dir.parent.parent / "backups" / f"pre_hiphive_correction_{staticpoint_dir.name}_{stamp}"
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
    final_root = staticpoint_dir.parent.parent / "FINAL_RESULTS"
    target = final_root / name
    target.mkdir(parents=True, exist_ok=True)
    for fname in ("band.pdf", "band.yaml", "phonopy.yaml", "FORCE_SETS", "POSCAR", "FORCE_CONSTANTS"):
        src = staticpoint_dir / fname
        if src.exists():
            shutil.copy2(src, target / fname)
    return target


def fix_staticpoint_dir(staticpoint_dir: Path, cutoff: float | None) -> dict:
    band_conf = staticpoint_dir / "band.conf"
    fc_path = staticpoint_dir / "FORCE_CONSTANTS"
    phonopy_yaml = staticpoint_dir / "phonopy.yaml"
    for required in (band_conf, fc_path, phonopy_yaml):
        if not required.exists():
            raise FileNotFoundError(f"Missing {required}")

    ph = phonopy.load(
        phonopy_yaml=str(phonopy_yaml),
        force_constants_filename=str(fc_path),
        is_compact_fc=True,
        log_level=0,
    )

    full_fc = compact_fc_to_full_fc(ph.primitive, ph.force_constants)
    prim_ase = phonopy_atoms_to_ase(ph.primitive)
    supercell_ase = phonopy_atoms_to_ase(ph.supercell)

    if cutoff is None:
        cutoff = estimate_safe_cutoff(supercell_ase)
        print(f"  No --cutoff given, estimated safe cutoff = {cutoff:.2f} Angstrom")

    paths, labels, path_connections = parse_band_conf(band_conf)

    # raw (uncorrected) min frequency, for before/after comparison
    ph.force_constants = full_fc
    ph.run_band_structure(paths)
    raw_freqs = np.vstack(ph.get_band_structure_dict()["frequencies"])
    raw_min = raw_freqs.min()

    fcs_phonopy = ForceConstants.from_arrays(supercell_ase, full_fc)
    cs = ClusterSpace(prim_ase, [cutoff])
    parameters = extract_parameters(fcs_phonopy, cs)

    fcp = ForceConstantPotential(cs, parameters)
    recon_fc = fcp.get_force_constants(supercell_ase).get_fc_array(order=2)
    recon_err_pct = 100 * np.linalg.norm(recon_fc - full_fc) / np.linalg.norm(full_fc)

    enforced_parameters = enforce_rotational_sum_rules(cs, parameters, ["Huang", "Born-Huang"])
    fcp_rot = ForceConstantPotential(cs, enforced_parameters)
    corrected_fc = fcp_rot.get_force_constants(supercell_ase).get_fc_array(order=2)

    ph.force_constants = corrected_fc
    ph.run_band_structure(paths)
    corrected_freqs = np.vstack(ph.get_band_structure_dict()["frequencies"])
    corrected_min = corrected_freqs.min()

    ph.run_mesh([30, 30, 1], is_gamma_center=True, with_eigenvectors=False)
    mesh_freqs = ph.get_mesh_dict()["frequencies"]
    mesh_min = mesh_freqs.min()
    n_negative = int((mesh_freqs < -1e-3).sum())

    backup_dir = backup_existing(staticpoint_dir)

    write_FORCE_CONSTANTS(corrected_fc, filename=str(fc_path))
    ph.run_band_structure(paths, labels=labels, path_connections=path_connections)
    ph.write_yaml_band_structure(filename=str(staticpoint_dir / "band.yaml"))
    fig = ph.plot_band_structure()
    fig.savefig(staticpoint_dir / "band.pdf")

    final_target = copy_final_results(staticpoint_dir)

    return {
        "staticpoint_dir": staticpoint_dir,
        "cutoff": cutoff,
        "reconstruction_error_pct": recon_err_pct,
        "raw_min_freq": raw_min,
        "corrected_min_freq": corrected_min,
        "mesh_min_freq": mesh_min,
        "mesh_n_negative": n_negative,
        "mesh_n_total": mesh_freqs.size,
        "backup_dir": backup_dir,
        "final_results_dir": final_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Correct small residual imaginary phonon frequencies via hiphive's "
            "rotational sum rule enforcement, applied on top of an existing "
            "postprocessed staticpoint directory's FORCE_CONSTANTS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 hiphive_fix_force_constants.py --bilayer MoS2_WS2_3R_staticpoint
  python3 hiphive_fix_force_constants.py --monolayer HfSe2_staticpoint --cutoff 6.5
  python3 hiphive_fix_force_constants.py phonopy_bilayer_examples/MoS2_WS2_3R_staticpoint
""",
    )
    parser.add_argument("--monolayer", action="store_true", help="Resolve NAME under phonopy_monolayer_examples/")
    parser.add_argument("--bilayer", action="store_true", help="Resolve NAME under phonopy_bilayer_examples/")
    parser.add_argument("staticpoint", help="Staticpoint directory name or path")
    parser.add_argument("--cutoff", type=float, default=None,
                        help="hiphive ClusterSpace cutoff in Angstrom (default: auto-estimated from supercell size)")
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

    print(f"Processing {d} ...")
    result = fix_staticpoint_dir(d, args.cutoff)

    print(f"  Cutoff used: {result['cutoff']:.2f} Angstrom")
    print(f"  Force constant reconstruction error: {result['reconstruction_error_pct']:.3f}%")
    print(f"  Band-path min freq: raw={result['raw_min_freq']:.4f} THz -> corrected={result['corrected_min_freq']:.4f} THz")
    print(f"  Full-BZ 30x30x1 mesh min freq: {result['mesh_min_freq']:.5f} THz "
          f"({result['mesh_n_negative']}/{result['mesh_n_total']} points < -0.001 THz)")
    print(f"  Backed up pre-correction FORCE_CONSTANTS/band.yaml/band.pdf -> {result['backup_dir']}")
    print(f"  Wrote corrected FORCE_CONSTANTS/band.yaml/band.pdf, copied to {result['final_results_dir']}")

    if result["mesh_n_negative"] > 0:
        print("  WARNING: residual negative frequencies remain elsewhere in the BZ -- "
              "inspect before adding to FINAL_RESULTS_HEALTHY.", file=sys.stderr)


if __name__ == "__main__":
    main()
