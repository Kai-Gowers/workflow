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
  python3 twisted/export_share_bundle.py [--materials MoS2_twist_m1_near0 MoS2_twist_m2_near0] [--name twisted_mos2_phonon_dataset_v1]
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
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.io import write as ase_write

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from export_shareable_dataset import BUNDLE_SCRIPT_DIR, fc_source, to_ase  # noqa: E402

SOURCE = ROOT / "FINAL_RESULTS_TWISTED"
STRUCTURES = ROOT / "twisted" / "structures"
RELAX = ROOT / "bilayer_examples_twisted"
STATIC = ROOT / "phonopy_bilayer_examples_twisted"
PER_MATERIAL_FILES = ["POSCAR", "phonopy.yaml", "FORCE_SETS", "FORCE_CONSTANTS", "band.yaml", "band.pdf"]
EV_A3_TO_GPA = 160.21766
DEFAULT_MATERIALS = ["MoS2_twist_m1_near0", "MoS2_twist_m2_near0"]


def interlayer_gap(atoms) -> tuple[float, float]:
    """(S–S gap across the interlayer region, Mo-plane separation) in Å for a MoS2 homobilayer cell."""
    z = atoms.positions[:, 2]
    mo = z[atoms.numbers == 42]
    mid = mo.mean()
    s = z[atoms.numbers == 16]
    gap = s[s > mid].min() - s[s < mid].max()
    sep = mo[mo > mid].mean() - mo[mo < mid].mean()
    return float(gap), float(sep)


def kmesh(path: Path) -> str:
    return " ".join(path.read_text().splitlines()[3].split()[:3])


def readme(rows: list[dict], n_frames: int) -> str:
    tab = "\n".join(
        f"| `{r['material']}` | {float(r['theta_deg']):.2f}° | ({r['m']}, {r['n']}) | {r['n_atoms_unitcell']} | "
        f"{float(r['a_moire_A']):.3f} | {r['supercell']} ({r['n_atoms_supercell']} atoms) | {r['n_displacements']} | "
        f"{float(r['gap_relaxed_A']):.2f} (as-built {float(r['gap_asbuilt_A']):.2f}) | {r['force_constants_source']} | "
        f"{float(r['min_freq_THz']):+.3f} |" for r in rows)
    return f"""# Twisted MoS2 bilayer phonon references (DFT, PBE+D3(BJ)) — v1

Harmonic phonon reference data for commensurate twisted MoS2 homobilayers (moiré cells), computed with the
same code, functional and phonon protocol as `tmd48_phonon_dataset_v1` (untwisted mono/bilayers), so a model
benchmarked there can be tested here unchanged. Built {date.today().isoformat()}. {len(rows)} cells; a third angle
(m = 3, θ = 9.43°, 222 atoms) is being computed and will be added in a later version.

| cell | θ | (m, n) | atoms | a_moiré (Å) | phonon supercell | displacements | S–S gap (Å) | force constants | min DFT freq (THz) |
|---|---|---|---|---|---|---|---|---|---|
{tab}

**Geometry.** Commensurate (m, n = m+1) twisted homobilayers of the "near-0°" family: the untwisted limit is the
3R (AB, same-orientation) homobilayer; the top layer is rotated by +θ about a Mo atom of the bottom layer. The
as-built rigid cell (`POSCAR_asbuilt`, `twist.json`: one in-plane MoS2 lattice constant **a = 3.1922 Å** for both
layers, dMX = 1.564 Å, S–S gap 2.90 Å from the relaxed untwisted 3R bilayer, c = 20 Å) was relaxed with DFT at fixed
cell (positions only), which opens the gap to the values in the table; `POSCAR` is the relaxed cell the phonons were
computed on. Both layers share one lattice constant by construction (commensurability), so there is no lattice
mismatch and the moiré is a pure rotation.

> **Caveat — residual in-plane strain.** a = 3.1922 Å is the Materials-Project value, at which PBE+D3 MoS2 carries
> **+2.1 GPa residual in-plane stress** (the PBE+D3 equilibrium is a = 3.155 Å; the untwisted MoS2 references in
> `tmd48_bare_pbe_subtract_v3` / the refreshed PBE+D3 set were re-relaxed to it on 2026-10-10, these twisted cells
> were not). The relaxed cells here are therefore slightly tensile-strained (+1.2 %). Evaluate models at this fixed
> cell (`--relax none` or `--relax positions`); a model that also relaxes the cell will contract it and its phonons
> will drift away from these references for a physical, not a model, reason. Stress is reported in `materials.csv`
> and `equilibrium.extxyz` so you can check your model's stress against DFT at this cell.

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
ph = phonopy.load("materials/MoS2_twist_m2_near0/phonopy.yaml",
                  force_constants_filename="materials/MoS2_twist_m2_near0/FORCE_CONSTANTS")
band = yaml.safe_load(open("materials/MoS2_twist_m2_near0/band.yaml"))
q = [p["q-position"] for p in band["phonon"]]; f = [[b["frequency"] for b in p["band"]] for p in band["phonon"]]
frames = read("displacements.extxyz", index=":")      # PBE+D3 energy / forces / stress per displaced supercell
cells = read("equilibrium.extxyz", index=":")         # relaxed moiré cells with PBE+D3 energy / forces / stress
```

Energies are VASP `e_0_energy` (eV, σ→0 extrapolated). Stresses in eV/Å³, ASE sign convention, Voigt order
xx yy zz yz xz xy, for the full 20 Å-vacuum cell. All quantities are **PBE+D3**; a model trained on the bare-PBE
bundles (`tmd48_bare_pbe_subtract_v*`) must add the D3(BJ) term (energy, forces, stress **and Hessian**) with the
parameters above before comparing here.

## Benchmarking a model

Same script and modes as the untwisted bundle; the only practical difference is size (the m = 2 cell has 114 atoms
and its phonon supercell is the cell itself):

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
    ap.add_argument("--name", default="twisted_mos2_phonon_dataset_v1")
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
        shutil.copy2(STRUCTURES / name / "POSCAR", dst / "POSCAR_asbuilt")
        shutil.copy2(STRUCTURES / name / "twist.json", dst / "twist.json")
        tw = json.loads((STRUCTURES / name / "twist.json").read_text())

        ph_fs = phonopy.load(str(src / "phonopy.yaml"), force_sets_filename=str(src / "FORCE_SETS"), log_level=0)
        ph_fc = phonopy.load(str(src / "phonopy.yaml"), force_constants_filename=str(src / "FORCE_CONSTANTS"), log_level=0)
        disps = ph_fs.dataset["first_atoms"]
        vaspruns = sorted((STATIC / f"{name}_staticpoint").glob("disp-*/vasprun.xml"))
        if len(vaspruns) != len(disps):
            sys.exit(f"{name}: {len(vaspruns)} vasprun.xml vs {len(disps)} displacements")
        for i, (disp, sc, vr) in enumerate(zip(disps, ph_fs.supercells_with_displacements, vaspruns)):
            v = read(vr, index=-1)
            atoms = to_ase(sc)
            dfrac = v.get_scaled_positions() - atoms.get_scaled_positions()
            dfrac -= np.round(dfrac)
            if np.abs(v.get_forces() - disp["forces"]).max() > 1e-5 or np.abs(dfrac).max() > 1e-5 \
                    or np.abs(v.cell.array - atoms.cell.array).max() > 1e-5:
                sys.exit(f"{name} disp {i}: vasprun does not match FORCE_SETS / phonopy supercell")
            atoms.calc = SinglePointCalculator(atoms, energy=v.get_potential_energy(), forces=v.get_forces(), stress=v.get_stress())
            atoms.info.update(material=name, displacement_index=i, displaced_atom=int(disp["number"]),
                              displacement=np.array(disp["displacement"]), config_type="phonon_displacement",
                              functional="PBE+D3(BJ)", theta_deg=tw["theta_deg"])
            frames.append(atoms)

        # relaxed cell + its energetics (must be the shipped POSCAR)
        o = read(RELAX / name / "OUTCAR", index=-1)
        p = read(src / "POSCAR")
        if len(o) != len(p) or np.abs(o.cell.array - p.cell.array).max() > 1e-3 or \
                np.abs(o.get_scaled_positions() - p.get_scaled_positions()).max() > 1e-4:
            sys.exit(f"{name}: relaxation OUTCAR geometry != FINAL_RESULTS_TWISTED POSCAR")
        eq = p.copy()
        eq.calc = SinglePointCalculator(eq, energy=o.get_potential_energy(), forces=o.get_forces(), stress=o.get_stress())
        eq.info.update(material=name, config_type="relaxed_cell", functional="PBE+D3(BJ)", theta_deg=tw["theta_deg"],
                       kmesh=kmesh(RELAX / name / "KPOINTS"), note="relaxation run (ISIF=2, EDIFFG=-1e-4), moire cell fixed")
        eq_rows.append(eq)

        band = yaml.safe_load((src / "band.yaml").read_text())
        q_list = np.array([q["q-position"] for q in band["phonon"]])
        freqs = np.array([[b["frequency"] for b in q["band"]] for q in band["phonon"]])
        dim = ph_fs.supercell_matrix.diagonal()
        gap_rel, sep_rel = interlayer_gap(p)
        gap_ab, _ = interlayer_gap(read(STRUCTURES / name / "POSCAR"))
        s = o.get_stress() * EV_A3_TO_GPA
        rows.append(dict(
            material=name, theta_deg=f"{tw['theta_deg']:.6f}", m=tw["m"], n=tw["n"], family=tw["family"],
            seed_stacking=tw["seed_stacking"], n_atoms_unitcell=len(p), a_layer_A=f"{tw['a_layer_A']:.6f}",
            a_moire_A=f"{tw['a_moire_A']:.6f}", gap_asbuilt_A=f"{gap_ab:.4f}", gap_relaxed_A=f"{gap_rel:.4f}",
            mo_plane_sep_relaxed_A=f"{sep_rel:.4f}", supercell=f"{dim[0]}x{dim[1]}x{dim[2]}",
            n_atoms_supercell=len(ph_fs.supercell), n_displacements=len(disps), n_qpoints=freqs.shape[0],
            n_bands=freqs.shape[1], min_freq_THz=f"{freqs.min():.4f}", max_freq_THz=f"{freqs.max():.4f}",
            force_constants_source=fc_source(ph_fs, ph_fc, q_list),
            relax_energy_pbe_d3_eV=f"{o.get_potential_energy():.6f}", relax_stress_xx_GPa=f"{s[0]:.4f}",
            relax_max_force_eV_per_A=f"{np.abs(o.get_forces()).max():.2e}",
            kmesh_relax=kmesh(RELAX / name / "KPOINTS"), kmesh_static=kmesh(vaspruns[0].parent / "KPOINTS"),
        ))
        r = rows[-1]
        print(f"{name:<22} θ={float(r['theta_deg']):5.2f}°  {r['n_atoms_unitcell']:>3} atoms  {r['supercell']} ({r['n_atoms_supercell']})  "
              f"{r['n_displacements']:>3} disp  gap {r['gap_asbuilt_A']}->{r['gap_relaxed_A']} Å  fc={r['force_constants_source']}  "
              f"min {r['min_freq_THz']} THz  σxx {r['relax_stress_xx_GPa']} GPa", flush=True)

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
