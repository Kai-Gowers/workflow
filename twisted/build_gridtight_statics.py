#!/usr/bin/env python3
"""
Rerun the phonopy statics of finished twisted cells with grid-tight VASP settings
(TWIST_VARIANT=twisted_gridtight).

Background (logs/2026-10-16.md): the Gamma layer-shear pair of every twisted cell shipped in
FINAL_RESULTS_TWISTED is a finite-difference artifact of the real-space-grid (egg-box) error on the
displaced atom's own force (random sign, +-0.2..0.4 THz). A 42-displacement test on MoS2 m1 with
LREAL=.FALSE., ADDGRID=.TRUE., EDIFF=1E-8 brought the rigid-shear curvature from 0.168 to 0.036
eV/A^2/cell, in agreement with the independent sliding scan (0.030-0.041), and the Gamma pair from
0.230 to +0.026 THz. This script repeats the PRODUCTION statics with those settings only.

Additive glue: nothing in the production tree is touched. For each cell it
  * clones phonopy_bilayer_examples_twisted/<cell>_staticpoint/ -> phonopy_bilayer_examples_twisted_gridtight/<cell>_staticpoint/
    copying phonopy_disp.yaml, SPOSCAR, POSCAR, POSCAR-XXX, POTCAR, KPOINTS, band.conf BYTE-FOR-BYTE
    (same displacements, same supercell, same k-mesh as the shipped run);
  * writes INCAR from common/staticpoint_templates_twisted_gridtight/INCAR (the only physics change);
  * writes phonopy_dim.txt (from the supercell matrix) for postprocess_results.py;
  * hands the directory to the standard submitter (common/setup_displacements_impl.py), which builds the
    disp-XXX folders, copies the variant `bat` (short partition, 12 h) and registers every job in
    data/job_registry_twisted_gridtight.json under the name <cell>-disp-XXX.

Usage (from workflow/):
  TWIST_VARIANT=twisted_gridtight python3 twisted/build_gridtight_statics.py --dry-run
  TWIST_VARIANT=twisted_gridtight python3 twisted/build_gridtight_statics.py            # the 4 shipped cells
  TWIST_VARIANT=twisted_gridtight python3 twisted/build_gridtight_statics.py --names MoS2_twist_m1_near0 --no-submit
Afterwards:
  TWIST_VARIANT=twisted_gridtight python3 phonopy/postprocess_results.py --bilayer <cell>_staticpoint --dim "$(cat .../phonopy_dim.txt)"
  -> FINAL_RESULTS_TWISTED_GRIDTIGHT/<cell>/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "common"))
from incar_utils import customize_incar  # noqa: E402
from run_variant import generated_dir, templates_dir, variant  # noqa: E402
from setup_displacements_impl import setup_and_submit_displacements  # noqa: E402

REQUIRED_VARIANT = "twisted_gridtight"
SOURCE_VARIANT_DIR = REPO_ROOT / "phonopy_bilayer_examples_twisted"   # the shipped production statics
DEFAULT_NAMES = ["MoS2_twist_m1_near0", "MoS2_twist_m2_near0", "MoS2_WS2_twist_m1_near0", "MoTe2_twist_m1_near0"]
COPY_FILES = ["phonopy_disp.yaml", "SPOSCAR", "POSCAR", "POTCAR", "KPOINTS", "band.conf"]


def clone_staticpoint(name: str, dry_run: bool) -> Path:
    src = SOURCE_VARIANT_DIR / f"{name}_staticpoint"
    dst = generated_dir("phonopy_bilayer_examples") / f"{name}_staticpoint"
    if not src.is_dir():
        sys.exit(f"source staticpoint not found: {src}")
    poscars = sorted(p for p in src.iterdir() if p.name.startswith("POSCAR-") and p.suffix == "")
    disp = yaml.safe_load((src / "phonopy_disp.yaml").read_text())
    dim = " ".join(str(disp["supercell_matrix"][i][i]) for i in range(3))
    n_atoms = len(disp["supercell"]["points"])
    print(f"{name}: {len(poscars)} displacements, supercell {n_atoms} atoms, --dim \"{dim}\", KPOINTS "
          f"{(src / 'KPOINTS').read_text().splitlines()[3].strip()}")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {dst}")
    if dry_run:
        print(f"  would create {dst}")
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    for p in poscars:
        shutil.copy2(p, dst / p.name)
    customize_incar(templates_dir("staticpoint") / "INCAR", dst / "INCAR", name, suffix="phonon")
    (dst / "phonopy_dim.txt").write_text(dim + "\n")
    (dst / "provenance.json").write_text(json.dumps({
        "source": str(src.relative_to(REPO_ROOT)),
        "incar_template": str(templates_dir("staticpoint").relative_to(REPO_ROOT) / "INCAR"),
        "reason": "grid-tight rerun of the shipped statics (egg-box Gamma shear artifact), 2026-10-16",
        "copied_byte_for_byte": COPY_FILES + [p.name for p in poscars],
    }, indent=1))
    print(f"  wrote {dst.relative_to(REPO_ROOT)} (INCAR from {templates_dir('staticpoint').name})")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-submit", action="store_true", help="build the directories but do not sbatch")
    args = ap.parse_args()
    if variant() != REQUIRED_VARIANT:
        sys.exit(f"Refusing to run: export TWIST_VARIANT={REQUIRED_VARIANT} first (got {variant()!r})")
    summary = {}
    for name in args.names:
        dst = clone_staticpoint(name, args.dry_run)
        if args.dry_run:
            continue
        res = setup_and_submit_displacements(dst, "phonopy_bilayer_examples", submit=not args.no_submit)
        summary[name] = {"disp": res["displacements_setup"], "jobs": res["job_ids"]}
        if res["job_ids"]:
            (dst / "jobs.json").write_text(json.dumps(res["job_ids"], indent=1))
    if summary:
        print("\nSUMMARY")
        for n, s in summary.items():
            ids = s["jobs"]
            print(f"  {n}: {s['disp']} disp folders, {len(ids)} jobs" + (f" {ids[0]}..{ids[-1]}" if ids else ""))


if __name__ == "__main__":
    main()
