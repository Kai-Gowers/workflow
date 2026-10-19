#!/usr/bin/env python3
"""
Rerun the MoS2 m2 twisted-cell phonopy statics in a 2x2x1 supercell with the grid-tight VASP
settings (TWIST_VARIANT=twisted_gridtight). Additive: nothing in the 1x1x1 run is touched.

Background (logs/2026-10-19.md): the grid-tight 1x1x1 run of MoS2_twist_m2_near0 (114 atoms, the
moire cell itself) removed the egg-box Gamma shear artifact but still shows a -0.057 THz dip in the
flexural (ZA) branch at |q| ~ 0.1 r.l.u. With a 1x1x1 supercell every q between Gamma and the zone
boundary is a Fourier interpolation of the single Gamma-point Hessian, and the ZA q^2 coefficient is
set by pairs at the half-cell distance where the periodic image is ambiguous; the dip is absent in the
old strained cell and in the m1 1x1x1 grid test, i.e. its sign is cell-specific like the shear pair.
hiphive cannot repair it (cutoff capped at half the cell -> rigid-shear curvature 10x the sliding
scan). The other three cells (MoS2 m1, MoS2_WS2 m1, MoTe2 m1) are already 2x2x1 supercells.

What this builds (user decision 2026-10-19):
  phonopy_bilayer_examples_twisted_gridtight/MoS2_twist_m2_near0_dim221_staticpoint/
    POSCAR, POTCAR            copied from the 1x1x1 grid-tight staticpoint (refined PBE+D3 cell)
    INCAR                     from common/staticpoint_templates_twisted_gridtight/INCAR (KPAR=4, NCORE=2)
    KPOINTS                   Gamma 2 2 1  (27.5 A supercell: same k-density as 5x5x1 on the 13.75 A cell)
    phonopy_disp.yaml, SPOSCAR, POSCAR-XXX   from `phonopy-init -d --dim "2 2 1" --amplitude 0.01`
    bat                       grid-tight template bat with 4 nodes x 44 ranks, medium partition, 48 h
    phonopy_dim.txt = "2 2 1", band.conf, provenance.json, jobs.json
  disp-XXX/ folders with POSCAR INCAR KPOINTS POTCAR bat, submitted through job_tracking.submit_bat so
  every job lands in data/job_registry_twisted_gridtight.json as MoS2_twist_m2_near0_dim221-disp-XXX.

Usage (from workflow/):
  TWIST_VARIANT=twisted_gridtight python3 twisted/build_m2_dim221_statics.py --dry-run
  TWIST_VARIANT=twisted_gridtight python3 twisted/build_m2_dim221_statics.py --no-submit
  TWIST_VARIANT=twisted_gridtight python3 twisted/build_m2_dim221_statics.py
  TWIST_VARIANT=twisted_gridtight python3 twisted/build_m2_dim221_statics.py --submit-existing   # after --no-submit
Afterwards:
  TWIST_VARIANT=twisted_gridtight python3 phonopy/postprocess_results.py --bilayer MoS2_twist_m2_near0_dim221_staticpoint --dim "2 2 1"
  -> FINAL_RESULTS_TWISTED_GRIDTIGHT/MoS2_twist_m2_near0_dim221/
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "maintenance"))
from incar_utils import customize_incar  # noqa: E402
from job_tracking import submit_bat  # noqa: E402
from run_variant import generated_dir, templates_dir, variant  # noqa: E402

REQUIRED_VARIANT = "twisted_gridtight"
CELL = "MoS2_twist_m2_near0"
NEW_NAME = f"{CELL}_dim221"
DIM = "2 2 1"
AMPLITUDE = "0.01"
KMESH = "2 2 1"
NODES, RANKS_PER_NODE, PARTITION, TIME = 4, 44, "medium", "48:00:00"


def write_bat(dst: Path) -> None:
    bat = (templates_dir("staticpoint") / "bat").read_text()
    bat = re.sub(r"#SBATCH --nodes=\d+", f"#SBATCH --nodes={NODES}", bat)
    bat = re.sub(r"#SBATCH --ntasks-per-node=\d+", f"#SBATCH --ntasks-per-node={RANKS_PER_NODE}", bat)
    bat = re.sub(r"#SBATCH --partition=\S+", f"#SBATCH --partition={PARTITION}", bat)
    bat = re.sub(r"#SBATCH --time=\S+", f"#SBATCH --time={TIME}", bat)
    (dst / "bat").write_text(bat)


def build(dry_run: bool) -> Path:
    src = generated_dir("phonopy_bilayer_examples") / f"{CELL}_staticpoint"
    dst = generated_dir("phonopy_bilayer_examples") / f"{NEW_NAME}_staticpoint"
    for f in ("POSCAR", "POTCAR"):
        if not (src / f).exists():
            sys.exit(f"missing {src / f}")
    print(f"source cell: {src.relative_to(REPO_ROOT)}  ->  {dst.relative_to(REPO_ROOT)}")
    print(f"  --dim \"{DIM}\", amplitude {AMPLITUDE} A, KPOINTS Gamma {KMESH}, bat {NODES}x{RANKS_PER_NODE} ranks {PARTITION}/{TIME}")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {dst}")
    if dry_run:
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "POSCAR", dst / "POSCAR")
    shutil.copy2(src / "POTCAR", dst / "POTCAR")
    customize_incar(templates_dir("staticpoint") / "INCAR", dst / "INCAR", NEW_NAME, suffix="phonon")
    (dst / "KPOINTS").write_text(f"Automatic mesh\n0\nGamma\n{KMESH}\n0 0 0\n")
    subprocess.run(["phonopy-init", "-d", "--dim", DIM, "--amplitude", AMPLITUDE, "-c", "POSCAR"],
                   cwd=dst, check=True, capture_output=True, text=True)
    poscars = sorted(p for p in dst.iterdir() if re.fullmatch(r"POSCAR-\d+", p.name))
    n_super = int(sum(int(x) for x in (dst / "SPOSCAR").read_text().splitlines()[6].split()))
    print(f"  phonopy-init: {len(poscars)} displacements, supercell {n_super} atoms")
    (dst / "phonopy_dim.txt").write_text(DIM + "\n")
    (dst / "band.conf").write_text(
        "ATOM_NAME = Mo S\n" f"DIM = {DIM}\n" "BAND = 0 0 0  0.3333 0.3333 0  0.5 0 0  0 0 0\n"
        "BAND_LABELS = Γ K M Γ\n" "FC_SYMMETRY = .TRUE.\n")
    write_bat(dst)
    (dst / "provenance.json").write_text(json.dumps({
        "source": str(src.relative_to(REPO_ROOT)),
        "copied": ["POSCAR", "POTCAR"],
        "incar_template": str((templates_dir("staticpoint") / "INCAR").relative_to(REPO_ROOT)),
        "kpoints": f"Gamma {KMESH}",
        "phonopy": f"phonopy-init -d --dim '{DIM}' --amplitude {AMPLITUDE}",
        "sbatch": f"{NODES} nodes x {RANKS_PER_NODE} ranks, {PARTITION}, {TIME}",
        "reason": "2x2x1 supercell rerun of the 1x1x1 grid-tight statics: ZA interpolation dip (-0.057 THz), 2026-10-19",
    }, indent=1))
    for p in poscars:
        d = dst / f"disp-{int(p.name.split('-')[1]):03d}"
        d.mkdir(exist_ok=True)
        shutil.copy2(p, d / "POSCAR")
        for f in ("INCAR", "KPOINTS", "POTCAR", "bat"):
            shutil.copy2(dst / f, d / f)
    print(f"  built {len(poscars)} disp-XXX folders")
    return dst


def submit(dst: Path) -> dict:
    jobs = {}
    for d in sorted(dst.glob("disp-*")):
        ok, jid, msg = submit_bat(d, job_name=f"{NEW_NAME}-{d.name}", job_type="phonopy-displacement")
        if not ok or not jid:
            print(f"    ✗ {d.name}: {msg}")
            continue
        jobs[d.name] = jid
    (dst / "jobs.json").write_text(json.dumps(jobs, indent=1))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-submit", action="store_true", help="build the directories but do not sbatch")
    ap.add_argument("--submit-existing", action="store_true",
                    help="skip the build and sbatch the disp-XXX folders already built with --no-submit")
    args = ap.parse_args()
    if variant() != REQUIRED_VARIANT:
        sys.exit(f"Refusing to run: export TWIST_VARIANT={REQUIRED_VARIANT} first (got {variant()!r})")
    if args.submit_existing:
        dst = generated_dir("phonopy_bilayer_examples") / f"{NEW_NAME}_staticpoint"
        if (dst / "jobs.json").exists():
            sys.exit(f"{dst / 'jobs.json'} exists: already submitted")
    else:
        dst = build(args.dry_run)
        if args.dry_run or args.no_submit:
            return
    jobs = submit(dst)
    ids = list(jobs.values())
    print(f"\nSUMMARY: {len(jobs)} jobs submitted" + (f" {ids[0]}..{ids[-1]}" if ids else ""))


if __name__ == "__main__":
    main()
