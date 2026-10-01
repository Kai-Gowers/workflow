#!/usr/bin/env python3
"""
Build VASP relaxation input directories for twisted bilayers so they can go through the
standard DFT phonon pipeline (relax -> phonopy/prepare_and_submit.py -> postprocess_results.py).

Additive glue only: nothing in the production pipeline is modified. Mirrors the strain-pilot
recipe (scripts/generate_strain_variants.py) -- POSCAR + POTCAR + INCAR + KPOINTS + bat -- but
reads the structures from twisted/structures/<name>/ (built by build_twisted_bilayer.py) and
sizes the k-mesh per structure, because the production templates (21x21x1 relax, 7x7x1
static) are sized for a 3.19 A primitive cell, not an 8-30 A moire cell.

Requires TWIST_VARIANT=twisted so that every generated path carries the _twisted suffix
(bilayer_examples_twisted/, phonopy_bilayer_examples_twisted/, FINAL_RESULTS_TWISTED/,
data/job_registry_twisted.json) and the templates come from common/*_templates_twisted/.
The script refuses to run under any other variant to keep twisted cells out of the
production tree.

Per structure it writes bilayer_examples_twisted/<name>/:
  POSCAR           byte-for-byte copy of twisted/structures/<name>/POSCAR
  POTCAR           relaxation/monolayer/generate_potcar.generate_potcar
  INCAR            common/incar_utils.customize_incar from the twisted relaxation template
  KPOINTS          Gamma n n 1 with n = ceil(RELAX_KLENGTH_A / a_moire)
                   (RELAX_KLENGTH_A = 67 A = the production 21 x 3.19 A relaxation density)
  bat              twisted relaxation template; --nodes=4 when the cell has > BIG_CELL_ATOMS atoms
  phonopy_dim.txt  the --dim to pass to prepare_and_submit.py / postprocess_results.py:
                   k = ceil(DFT_MIN_SUPERCELL_A / a_moire), i.e. the smallest phonopy supercell
                   at least as large as the production 4x4x1 (12.8 A) one.

Then (all with TWIST_VARIANT=twisted exported):
  python3 relaxation/bilayer/submit_bilayer_job.py bilayer_examples_twisted/<name>
  python3 phonopy/prepare_and_submit.py --bilayer bilayer_examples_twisted/<name> --dim "$(cat .../phonopy_dim.txt)" --no-submit
  python3 phonopy/bilayer/setup_displacements.py <name>_staticpoint
  python3 phonopy/postprocess_results.py --bilayer <name>_staticpoint --dim "..."

Usage (from workflow/):
  TWIST_VARIANT=twisted python3 twisted/prepare_dft_inputs.py                 # the default 3 structures
  TWIST_VARIANT=twisted python3 twisted/prepare_dft_inputs.py --names MoS2_twist_m1_near60 --dry-run
"""

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "relaxation" / "monolayer"))
sys.path.insert(0, str(REPO_ROOT / "common"))
from run_variant import generated_dir, templates_dir, variant  # noqa: E402

from generate_potcar import generate_potcar  # noqa: E402
from incar_utils import customize_incar  # noqa: E402

REQUIRED_VARIANT = "twisted"
STRUCTURES_DIR = Path(__file__).resolve().parent / "structures"
DEFAULT_NAMES = ["MoS2_twist_m1_near0", "MoS2_twist_m2_near0", "MoS2_twist_m3_near0"]

RELAX_KLENGTH_A = 21 * 3.19224  # production relaxation KPOINTS 21x21x1 on the a = 3.19 A MoS2 cell
DFT_MIN_SUPERCELL_A = 4 * 3.19224  # production phonopy supercell 4x4x1
BIG_CELL_ATOMS = 200  # above this, relax on 4 nodes (precedent: MoS2_WS2_3R 216-atom statics)


def kmesh_n(a_moire: float) -> int:
    return max(1, math.ceil(RELAX_KLENGTH_A / a_moire))


def phonopy_k(a_moire: float) -> int:
    return max(1, math.ceil(DFT_MIN_SUPERCELL_A / a_moire))


def write_kpoints(path: Path, n: int) -> None:
    path.write_text(f"Automatic mesh\n0\nGamma\n{n} {n} 1\n0 0 0\n")


def write_bat(template: Path, out: Path, n_atoms: int) -> int:
    text = template.read_text()
    nodes = 4 if n_atoms > BIG_CELL_ATOMS else None
    if nodes is not None:
        text, count = re.subn(r"^#SBATCH --nodes=\d+$", f"#SBATCH --nodes={nodes}", text, flags=re.M)
        if count != 1:
            raise RuntimeError(f"expected exactly one '#SBATCH --nodes=' line in {template}, found {count}")
    out.write_text(text)
    return nodes or int(re.search(r"^#SBATCH --nodes=(\d+)$", text, flags=re.M).group(1))


def prepare_one(name: str, out_root: Path, template_dir: Path, dry_run: bool) -> None:
    src = STRUCTURES_DIR / name
    poscar = src / "POSCAR"
    meta = json.loads((src / "twist.json").read_text())
    a_moire = float(meta["a_moire_A"])
    n_atoms = int(meta["n_atoms"])
    n_k = kmesh_n(a_moire)
    k = phonopy_k(a_moire)
    outdir = out_root / name

    print(f"{name}: {n_atoms} atoms, a_moire {a_moire:.3f} A, theta {meta['theta_deg']:.2f} deg")
    print(f"  relax KPOINTS {n_k}x{n_k}x1, phonopy --dim \"{k} {k} 1\" "
          f"({n_atoms * k * k}-atom supercell, {a_moire * k:.1f} A), "
          f"nodes {4 if n_atoms > BIG_CELL_ATOMS else 2}")
    print(f"  -> {outdir}")
    if dry_run:
        return
    if outdir.exists():
        print(f"  WARNING: {outdir} exists; overwriting inputs (POSCAR/POTCAR/INCAR/KPOINTS/bat)")
    outdir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(poscar, outdir / "POSCAR")
    generate_potcar(outdir / "POSCAR", outdir / "POTCAR")
    customize_incar(template_dir / "INCAR", outdir / "INCAR", name, suffix="relaxation")
    write_kpoints(outdir / "KPOINTS", n_k)
    write_bat(template_dir / "bat", outdir / "bat", n_atoms)
    (outdir / "phonopy_dim.txt").write_text(f"{k} {k} 1\n")
    shutil.copyfile(src / "twist.json", outdir / "twist.json")
    assert (outdir / "POSCAR").read_bytes() == poscar.read_bytes()
    print("  wrote POSCAR POTCAR INCAR KPOINTS bat phonopy_dim.txt twist.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES,
                        help=f"Structure dirs under twisted/structures/ (default: {' '.join(DEFAULT_NAMES)})")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing")
    args = parser.parse_args()

    if variant() != REQUIRED_VARIANT:
        sys.exit(f"Refusing to run: export TWIST_VARIANT={REQUIRED_VARIANT} first "
                 f"(active variant: {variant() or 'none'}). This keeps twisted cells out of the production tree.")

    template_dir = templates_dir("relaxation")
    for fname in ("INCAR", "bat"):
        if not (template_dir / fname).exists():
            sys.exit(f"Missing template {template_dir / fname} (host-local, gitignored; see workflow/CLAUDE.md)")

    out_root = generated_dir("bilayer_examples")
    missing = [n for n in args.names if not (STRUCTURES_DIR / n / "twist.json").exists()]
    if missing:
        sys.exit(f"No twisted/structures/<name>/twist.json for: {missing} (run twisted/build_twisted_bilayer.py)")

    print(f"templates: {template_dir}\noutput root: {out_root}\n")
    for name in args.names:
        prepare_one(name, out_root, template_dir, args.dry_run)
    print("\nNext: TWIST_VARIANT=twisted python3 relaxation/bilayer/submit_bilayer_job.py "
          f"{out_root.name}/<name>  (then prepare_and_submit.py --bilayer {out_root.name}/<name> "
          "--dim \"$(cat <name>/phonopy_dim.txt)\")")


if __name__ == "__main__":
    main()
