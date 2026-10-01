#!/usr/bin/env python3
"""
Export the 48-material Mo/W TMD phonon dataset as a self-contained, model-agnostic bundle
that someone outside this repo can use to benchmark any ASE-compatible interatomic potential.

Selection: every FINAL_RESULTS_HEALTHY entry whose name is a pure Mo/W dichalcogenide
monolayer, homobilayer or heterobilayer (2H/3R). No train/val/test split is imposed —
recipients decide that themselves.

Bundle layout (share/<name>/):
  README.md                 provenance, units, conventions, how to evaluate
  materials.csv             one row per material (formula, type, atoms, supercell, min DFT freq, ...)
  displacements.extxyz      every DFT displaced supercell with its forces (standard MLIP format)
  evaluate_phonons.py       reference evaluation script (any ASE calculator -> phonon RMSE vs DFT)
  requirements.txt
  SHA256SUMS
  materials/<name>/         POSCAR, phonopy.yaml, FORCE_SETS, FORCE_CONSTANTS, band.yaml, band.pdf
and a zip of the same directory next to it.

Usage:
  python3 scripts/export_shareable_dataset.py [--source FINAL_RESULTS_HEALTHY] [--name tmd48_phonon_dataset_v1]
"""

import argparse
import csv
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import phonopy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

ROOT = Path(__file__).resolve().parent.parent
TMD_RE = re.compile(r"^(Mo|W)(S|Se|Te)2(_(Mo|W)(S|Se|Te)2_(2H|3R)|_bilayer_(2H|3R))?$")
PER_MATERIAL_FILES = ["POSCAR", "phonopy.yaml", "FORCE_SETS", "FORCE_CONSTANTS", "band.yaml", "band.pdf"]
BUNDLE_SCRIPT_DIR = ROOT / "scripts" / "share_bundle"


def classify(name: str) -> tuple[str, str, str]:
    """Return (kind, stacking, layers) for a material name."""
    if "_bilayer_" in name:
        mono, stacking = name.split("_bilayer_")
        return "homobilayer", stacking, f"{mono}/{mono}"
    parts = name.split("_")
    if len(parts) == 3:
        return "heterobilayer", parts[2], f"{parts[0]}/{parts[1]}"
    return "monolayer", "", name


def to_ase(patoms) -> Atoms:
    return Atoms(numbers=patoms.numbers, positions=patoms.positions, cell=patoms.cell, pbc=True)


def fc_source(ph_fs, ph_fc, q_list) -> str:
    """'phonopy' if the shipped FORCE_CONSTANTS reproduce a plain symmetrized phonopy fit of FORCE_SETS
    to within 0.01 THz on the band path, else 'hiphive-corrected' (sum rules enforced with hiphive)."""
    ph_fs.produce_force_constants()
    ph_fs.symmetrize_force_constants()
    ph_fs.run_qpoints(q_list)
    ph_fc.run_qpoints(q_list)
    df = np.abs(ph_fs.get_qpoints_dict()["frequencies"] - ph_fc.get_qpoints_dict()["frequencies"]).max()
    return "phonopy" if df < 0.01 else "hiphive-corrected"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="FINAL_RESULTS_HEALTHY")
    parser.add_argument("--name", default="tmd48_phonon_dataset_v1")
    parser.add_argument("--out-root", default="share")
    args = parser.parse_args()

    source = ROOT / args.source
    out = ROOT / args.out_root / args.name
    materials = sorted(p.name for p in source.iterdir() if p.is_dir() and TMD_RE.match(p.name))
    if len(materials) != 48:
        sys.exit(f"expected 48 TMD materials in {source}, found {len(materials)}: {materials}")

    if out.exists():
        shutil.rmtree(out)
    (out / "materials").mkdir(parents=True)

    rows = []
    all_disp = []
    for name in materials:
        src = source / name
        dst = out / "materials" / name
        dst.mkdir()
        for f in PER_MATERIAL_FILES:
            if not (src / f).exists():
                sys.exit(f"{name}: missing {f}")
            shutil.copy2(src / f, dst / f)

        ph_fs = phonopy.load(str(src / "phonopy.yaml"), force_sets_filename=str(src / "FORCE_SETS"), log_level=0)
        ph_fc = phonopy.load(
            str(src / "phonopy.yaml"), force_constants_filename=str(src / "FORCE_CONSTANTS"), log_level=0
        )
        for i, (disp, sc) in enumerate(zip(ph_fs.dataset["first_atoms"], ph_fs.supercells_with_displacements)):
            atoms = to_ase(sc)
            atoms.calc = SinglePointCalculator(atoms, forces=np.array(disp["forces"]))
            atoms.info.update(
                material=name,
                displacement_index=i,
                displaced_atom=int(disp["number"]),
                displacement=np.array(disp["displacement"]),
                config_type="phonon_displacement",
            )
            all_disp.append(atoms)

        import yaml

        band = yaml.safe_load((src / "band.yaml").read_text())
        q_list = np.array([q["q-position"] for q in band["phonon"]])
        freqs = np.array([[b["frequency"] for b in q["band"]] for q in band["phonon"]])
        kind, stacking, layers = classify(name)
        dim = ph_fs.supercell_matrix.diagonal()
        rows.append(
            dict(
                material=name,
                kind=kind,
                stacking=stacking,
                layers=layers,
                formula_unitcell=to_ase(ph_fs.unitcell).get_chemical_formula(),
                n_atoms_unitcell=len(ph_fs.unitcell),
                supercell=f"{dim[0]}x{dim[1]}x{dim[2]}",
                n_atoms_supercell=len(ph_fs.supercell),
                n_displacements=len(ph_fs.dataset["first_atoms"]),
                n_qpoints=freqs.shape[0],
                n_bands=freqs.shape[1],
                min_freq_THz=f"{freqs.min():.4f}",
                max_freq_THz=f"{freqs.max():.4f}",
                force_constants_source=fc_source(ph_fs, ph_fc, q_list),
            )
        )
        print(f"{name:<22} {kind:<14} {rows[-1]['supercell']} {rows[-1]['n_displacements']:>2} disp  "
              f"fc={rows[-1]['force_constants_source']}")

    with (out / "materials.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ase_write(str(out / "displacements.extxyz"), all_disp, format="extxyz")

    for f in ["README.md", "evaluate_phonons.py", "requirements.txt"]:
        shutil.copy2(BUNDLE_SCRIPT_DIR / f, out / f)

    with (out / "SHA256SUMS").open("w") as fh:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n")

    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zf.write(p, Path(out.name) / p.relative_to(out))
    n_hiphive = sum(r["force_constants_source"] == "hiphive-corrected" for r in rows)
    print(f"\n{len(rows)} materials, {len(all_disp)} displaced supercells, {n_hiphive} hiphive-corrected")
    print(f"bundle: {out}\nzip:    {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
