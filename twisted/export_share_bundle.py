#!/usr/bin/env python3
"""
Export the twisted-MoS2 DFT phonon references (FINAL_RESULTS_TWISTED/) as a self-contained,
model-agnostic bundle, same layout and evaluation script as share/tmd48_phonon_dataset_v1 (PBE+D3).

Bundle layout (share/<name>/):
  README.md                what this is, how it was made, units, caveats, how to evaluate
  materials.csv / .txt     one row / one name per twisted cell (angle, atoms, supercell, min DFT freq, ...)
  displacements.extxyz     every DFT displaced supercell with PBE+D3 energy, forces, stress (ASE extended XYZ)
  equilibrium.extxyz       the relaxed moire cells with their PBE+D3 energy, forces, stress (relaxation run)
  evaluate_phonons.py      reference benchmark script (any ASE calculator -> phonon RMSE vs DFT), unchanged
  requirements.txt, SHA256SUMS
  materials/<name>/        POSCAR (DFT-relaxed), POSCAR_asbuilt (rigid template the relaxation started from),
                           twist.json, phonopy.yaml, FORCE_SETS, FORCE_CONSTANTS, band.yaml, band.pdf
and a zip next to it. Additive: reuses scripts/export_shareable_dataset.py helpers, changes nothing there.

Usage (workflow conda env; no dftd3 needed):
  python3 twisted/export_share_bundle.py [--materials MoS2_twist_m1_near0 ...] [--name twisted_mos2_phonon_dataset_v2]

v1 (2026-10-10, m1+m2 at the MP a = 3.1922 Å, +2.1 GPa strained) is superseded by v2 (refined cells at the PBE+D3
equilibrium a; campaign restarted 2026-10-10). Pass --materials to add cells as their FINAL_RESULTS_TWISTED land.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import phonopy
import yaml
from ase.io import read
from ase.io import write as ase_write

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from export_shareable_dataset import BUNDLE_SCRIPT_DIR, fc_source  # noqa: E402
sys.path.insert(0, str(ROOT / "twisted"))
from write_labels_extxyz import build_frames, load_frames  # noqa: E402

SOURCE = ROOT / "FINAL_RESULTS_TWISTED"
STRUCTURES = ROOT / "twisted" / "structures"
PER_MATERIAL_FILES = ["POSCAR", "phonopy.yaml", "FORCE_SETS", "FORCE_CONSTANTS", "band.yaml", "band.pdf"]
OPTIONAL_FILES = ["total_dos.dat"]  # written by phonopy/write_phonon_dos.py; shipped when present
EV_A3_TO_GPA = 160.21766
DEFAULT_MATERIALS = ["MoS2_twist_m1_near0"]


def interlayer_gap(atoms) -> tuple[float, float]:
    """(S–S gap across the interlayer region, Mo-plane separation) in Å for a MoS2 homobilayer cell."""
    z = atoms.positions[:, 2]
    mo = z[atoms.numbers == 42]
    mid = mo.mean()
    s = z[atoms.numbers == 16]
    gap = s[s > mid].min() - s[s < mid].max()
    sep = mo[mo > mid].mean() - mo[mo < mid].mean()
    return float(gap), float(sep)


def readme(rows: list[dict], n_frames: int) -> str:
    a_layer = float(rows[0]["a_layer_A"])
    dmx = float(json.loads((STRUCTURES / rows[0]["material"] / "twist.json").read_text())["dMX_A"])
    gap0 = float(rows[0]["gap_asbuilt_A"])
    sxx = float(rows[0]["relax_stress_xx_GPa"])
    ex = rows[0]["material"]
    tab = "\n".join(
        f"| `{r['material']}` | {float(r['theta_deg']):.2f}° | ({r['m']}, {r['n']}) | {r['n_atoms_unitcell']} | "
        f"{float(r['a_moire_A']):.3f} | {r['supercell']} ({r['n_atoms_supercell']} atoms) | {r['n_displacements']} | "
        f"{float(r['gap_relaxed_A']):.2f} (as-built {float(r['gap_asbuilt_A']):.2f}) | {r['force_constants_source']} | "
        f"{float(r['min_freq_THz']):+.3f} |" for r in rows)
    return f"""# Twisted MoS2 bilayer phonon references (DFT, PBE+D3(BJ)) — v2

Harmonic phonon reference data for commensurate twisted MoS2 homobilayers (moiré cells), computed with the
same code, functional and phonon protocol as `tmd48_phonon_dataset_v1` (untwisted mono/bilayers), so a model
benchmarked there can be tested here unchanged. Built {date.today().isoformat()}. {len(rows)} cell(s) so far; the campaign
is MoS2, MoTe2 and the MoS2/WS2 heterobilayer at m = 1, 2, 3 (θ = 21.8 / 13.2 / 9.4°, 42 / 114 / 222 atoms) and cells are
added to this bundle as they finish. **v2 supersedes v1** (2026-10-10): v1's cells sat at the Materials-Project a = 3.1922 Å
with +2.1 GPa residual in-plane PBE+D3 stress; every v2 cell is built at the PBE+D3 equilibrium lattice constant of the
refined untwisted references (the same cells as `tmd48_bare_pbe_subtract_v3` / the refreshed PBE+D3 set), so there is no
residual strain and cell-relaxing models are compared on equal footing.

| cell | θ | (m, n) | atoms | a_moiré (Å) | phonon supercell | displacements | S–S gap (Å) | force constants | min DFT freq (THz) |
|---|---|---|---|---|---|---|---|---|---|
{tab}

**Geometry.** Commensurate (m, n = m+1) twisted homobilayers of the "near-0°" family: the untwisted limit is the
3R (AB, same-orientation) homobilayer; the top layer is rotated by +θ about a Mo atom of the bottom layer. The
as-built rigid cell (`POSCAR_asbuilt`, `twist.json`: one in-plane MoS2 lattice constant **a = {a_layer:.4f} Å** for both
layers, dMX = {dmx:.3f} Å, S–S gap {gap0:.2f} Å from the relaxed untwisted 3R bilayer, c = 20 Å) was relaxed with DFT at fixed
cell (positions only), which opens the gap to the values in the table; `POSCAR` is the relaxed cell the phonons were
computed on. Both layers share one lattice constant by construction (commensurability), so there is no lattice
mismatch and the moiré is a pure rotation.

> **Strain.** a = {a_layer:.4f} Å is the PBE+D3 equilibrium of the untwisted MoS2 bilayer (ISIF = 4 refinement of the
> reference set, 2026-10-10), so the moiré cell carries no residual in-plane stress by construction (DFT σxx of the
> relaxed cell = {sxx:+.2f} GPa, see `materials.csv` / `equilibrium.extxyz`). Evaluate models at this fixed cell
> (`--relax none` or `--relax positions`); the moiré a is fixed by commensurability, and a model's own cell relaxation
> should leave it essentially unchanged — a sizeable drift is a model error, not a reference artefact.

## Contents

```
materials.csv / materials.txt   one row / one name per cell (columns described below)
displacements.extxyz            {n_frames} DFT displaced supercells (ASE extended XYZ): PBE+D3 energy, forces, stress;
                                atoms.info has material, displacement_index, displaced_atom, displacement
equilibrium.extxyz              the relaxed moiré cells with the PBE+D3 energy / forces / stress of the relaxation run
evaluate_phonons.py             benchmark any ASE calculator against band.yaml (unchanged from tmd48_phonon_dataset_v1)
requirements.txt, SHA256SUMS
materials/<name>/
    POSCAR                      DFT-relaxed moiré cell (VASP 5 format, 60° hexagonal cell, 20 Å c)
    POSCAR_asbuilt              rigid template the relaxation started from (same cell, unrelaxed positions)
    twist.json                  θ, (m, n), atom count, a, dMX, gap, moiré a, rotation convention
    phonopy.yaml                unit cell, supercell + primitive matrix, symmetry (P321)
    FORCE_SETS                  raw DFT (PBE+D3) forces for every symmetry-inequivalent 0.01 Å displacement
    FORCE_CONSTANTS             harmonic force constants used for the reference dispersion (eV/Å²)
    band.yaml / band.pdf        reference dispersion (THz) on Γ–K–M–Γ of the moiré Brillouin zone
    total_dos.dat               total phonon DOS (states/THz/cell) from FORCE_CONSTANTS on a Γ-centred mesh
                                (grid in the header; 200 Å mesh length, converged to 1e-5 states/THz/atom)
```

`materials.csv` columns: `theta_deg`, `m`, `n`, `family`, `seed_stacking`, `n_atoms_unitcell`, `a_layer_A`,
`a_moire_A`, `gap_asbuilt_A`, `gap_relaxed_A` (S–S across the interlayer region), `mo_plane_sep_relaxed_A`,
`supercell`, `n_atoms_supercell`, `n_displacements`, `n_qpoints`, `n_bands`, `min_freq_THz`, `max_freq_THz`,
`force_constants_source`, `relax_energy_pbe_d3_eV`, `relax_stress_xx_GPa` (in-plane, PBE+D3, ASE sign: positive =
tensile), `relax_max_force_eV_per_A`, `kmesh_relax`, `kmesh_static`.

## How the data were generated

- **Code / functional / numerics**: VASP 6.5.1, PAW PBE (`Mo_sv`, `S`), PBE + Grimme D3(BJ) (`IVDW = 12`: s6 = 1.0,
  s8 = 0.7875, a1 = 0.4289, a2 = 4.4407, two-body cutoff 50.2022 Å, CN cutoff 21.1671 Å, no three-body term),
  spin-unpolarised, `ENCUT = 520`, `PREC = Accurate`, `EDIFF = 1e-6`, `ISMEAR = 0`, `SIGMA = 0.05`, no U, no SOC —
  identical to the untwisted set.
- **Relaxation**: positions only (`ISIF = 2`), fixed moiré cell, `EDIFFG = -1e-4` eV/Å (the untwisted set used
  −1e-7, unreachable for 100–200-atom cells in the walltime; residual forces ≤ 1e-4 eV/Å, see
  `relax_max_force_eV_per_A`), Γ-centred k-mesh in `kmesh_relax` (chosen to match the untwisted 21×21×1 density).
- **Phonons**: 0.01 Å finite displacements, supercell in the table (chosen so the supercell is ≥ 12.8 Å, as the
  untwisted 4×4×1 cells), Γ-centred 5×5×1 k-mesh, phonopy 4.1.0, `fc_symmetry = .TRUE.`. P321 symmetry was kept by
  VASP, so the number of displacements equals the number of atoms.
- **Force-constant provenance**: `force_constants_source` is `phonopy` when `FORCE_CONSTANTS` is the plain
  symmetrised phonopy fit of `FORCE_SETS` (reproduced to < 0.01 THz), `hiphive-corrected` when the plain fit left a
  small spurious negative interlayer-shear pair at Γ (numerical violation of the rotational sum rules in a 2D slab,
  not an instability) and the shipped `FORCE_CONSTANTS` / `band.yaml` were refitted with hiphive enforcing the
  translational and rotational sum rules. `FORCE_SETS` is always the untouched DFT data, so you can refit yourself.
- **q-path**: Γ (0,0,0) → K (1/3,1/3,0) → M (1/2,0,0) → Γ in reduced coordinates of **phonopy's standardised 120°
  primitive cell** of the moiré lattice (`primitive_matrix` in `phonopy.yaml`), 51 points per segment. This is the
  moiré Brillouin zone: the low-frequency region holds the folded acoustic branches, the interlayer shear (moiré
  phason) and breathing modes.

## Loading the data

```python
import phonopy, yaml
from ase.io import read
ph = phonopy.load("materials/{ex}/phonopy.yaml",
                  force_constants_filename="materials/{ex}/FORCE_CONSTANTS")
band = yaml.safe_load(open("materials/{ex}/band.yaml"))
q = [p["q-position"] for p in band["phonon"]]; f = [[b["frequency"] for b in p["band"]] for p in band["phonon"]]
frames = read("displacements.extxyz", index=":")      # PBE+D3 energy / forces / stress per displaced supercell
cells = read("equilibrium.extxyz", index=":")         # relaxed moiré cells with PBE+D3 energy / forces / stress
```

Energies are VASP `e_0_energy` (eV, σ→0 extrapolated). Stresses in eV/Å³, ASE sign convention, Voigt order
xx yy zz yz xz xy, for the full 20 Å-vacuum cell. All quantities are **PBE+D3**; a model trained on the bare-PBE
bundles (`tmd48_bare_pbe_subtract_v*`) must add the D3(BJ) term (energy, forces, stress **and Hessian**) with the
parameters above before comparing here.

## Benchmarking a model

Same script and modes as the untwisted bundle; the only practical difference is size (m = 2 / 3 cells have 114 / 222
atoms and their phonon supercell is the cell itself):

```bash
pip install -r requirements.txt   # + your model's package
python evaluate_phonons.py --calc "my_pkg.my_module:make_calculator" --relax positions --plots
```

`--relax none` evaluates at the DFT geometry; `--relax positions` relaxes the atoms with your model at the fixed
moiré cell first (recommended; the DFT relaxation is only converged to 1e-4 eV/Å, so a model's own minimum is the
fairer comparison). Please state which one you report, and look at the Γ shear pair and the lowest breathing mode in
the plots as well as the whole-band RMSE: those interlayer modes are what twisting changes.

## Citation / contact

Dataset produced by Kai Gowers (Boston College). Please get in touch before publishing results derived from it.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materials", nargs="*", default=DEFAULT_MATERIALS)
    ap.add_argument("--name", default="twisted_mos2_phonon_dataset_v2")
    ap.add_argument("--out-root", default="share")
    args = ap.parse_args()

    out = ROOT / args.out_root / args.name
    if out.exists():
        shutil.rmtree(out)
    (out / "materials").mkdir(parents=True)

    rows, frames, eq_rows = [], [], []
    for name in args.materials:
        src, dst = SOURCE / name, out / "materials" / name
        dst.mkdir()
        for f in PER_MATERIAL_FILES:
            if not (src / f).exists():
                sys.exit(f"{name}: missing {f}")
            shutil.copy2(src / f, dst / f)
        for f in OPTIONAL_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        shutil.copy2(STRUCTURES / name / "POSCAR", dst / "POSCAR_asbuilt")
        shutil.copy2(STRUCTURES / name / "twist.json", dst / "twist.json")
        tw = json.loads((STRUCTURES / name / "twist.json").read_text())

        ph_fs = phonopy.load(str(src / "phonopy.yaml"), force_sets_filename=str(src / "FORCE_SETS"), log_level=0)
        ph_fc = phonopy.load(str(src / "phonopy.yaml"), force_constants_filename=str(src / "FORCE_CONSTANTS"), log_level=0)
        disps = ph_fs.dataset["first_atoms"]
        # labels: prefer the tracked FINAL_RESULTS_TWISTED/<name>/*.extxyz (written by twisted/write_labels_extxyz.py
        # on the computing host), fall back to rebuilding them from the local run dirs
        loaded = load_frames(name)
        label_src = "tracked extxyz" if loaded else "run dirs"
        cell_frames, eq = loaded if loaded else build_frames(name)
        if len(cell_frames) != len(disps):
            sys.exit(f"{name}: {len(cell_frames)} labelled frames vs {len(disps)} displacements")
        frames.extend(cell_frames)
        p = read(src / "POSCAR")
        eq_rows.append(eq)

        band = yaml.safe_load((src / "band.yaml").read_text())
        q_list = np.array([q["q-position"] for q in band["phonon"]])
        freqs = np.array([[b["frequency"] for b in q["band"]] for q in band["phonon"]])
        dim = ph_fs.supercell_matrix.diagonal()
        gap_rel, sep_rel = interlayer_gap(p)
        gap_ab, _ = interlayer_gap(read(STRUCTURES / name / "POSCAR"))
        s = eq.get_stress() * EV_A3_TO_GPA
        rows.append(dict(
            material=name, theta_deg=f"{tw['theta_deg']:.6f}", m=tw["m"], n=tw["n"], family=tw["family"],
            seed_stacking=tw["seed_stacking"], n_atoms_unitcell=len(p), a_layer_A=f"{tw['a_layer_A']:.6f}",
            a_moire_A=f"{tw['a_moire_A']:.6f}", gap_asbuilt_A=f"{gap_ab:.4f}", gap_relaxed_A=f"{gap_rel:.4f}",
            mo_plane_sep_relaxed_A=f"{sep_rel:.4f}", supercell=f"{dim[0]}x{dim[1]}x{dim[2]}",
            n_atoms_supercell=len(ph_fs.supercell), n_displacements=len(disps), n_qpoints=freqs.shape[0],
            n_bands=freqs.shape[1], min_freq_THz=f"{freqs.min():.4f}", max_freq_THz=f"{freqs.max():.4f}",
            force_constants_source=fc_source(ph_fs, ph_fc, q_list),
            relax_energy_pbe_d3_eV=f"{eq.get_potential_energy():.6f}", relax_stress_xx_GPa=f"{s[0]:.4f}",
            relax_max_force_eV_per_A=f"{np.abs(eq.get_forces()).max():.2e}",
            kmesh_relax=eq.info["kmesh"], kmesh_static=eq.info["kmesh_static"],
        ))
        r = rows[-1]
        print(f"{name:<22} θ={float(r['theta_deg']):5.2f}°  {r['n_atoms_unitcell']:>3} atoms  {r['supercell']} ({r['n_atoms_supercell']})  "
              f"{r['n_displacements']:>3} disp  gap {r['gap_asbuilt_A']}->{r['gap_relaxed_A']} Å  fc={r['force_constants_source']}  "
              f"min {r['min_freq_THz']} THz  σxx {r['relax_stress_xx_GPa']} GPa  labels: {label_src}", flush=True)

    with (out / "materials.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "materials.txt").write_text("\n".join(r["material"] for r in rows) + "\n")
    ase_write(str(out / "displacements.extxyz"), frames, format="extxyz")
    ase_write(str(out / "equilibrium.extxyz"), eq_rows, format="extxyz")
    for f in ("evaluate_phonons.py", "requirements.txt"):
        shutil.copy2(BUNDLE_SCRIPT_DIR / f, out / f)
    (out / "README.md").write_text(readme(rows, len(frames)))
    with (out / "SHA256SUMS").open("w") as fh:
        for pth in sorted(out.rglob("*")):
            if pth.is_file() and pth.name != "SHA256SUMS":
                fh.write(f"{hashlib.sha256(pth.read_bytes()).hexdigest()}  {pth.relative_to(out)}\n")
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pth in sorted(out.rglob("*")):
            if pth.is_file():
                zf.write(pth, Path(out.name) / pth.relative_to(out))
    print(f"\n{len(rows)} cells, {len(frames)} displaced supercells\nbundle: {out}\nzip:    {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
