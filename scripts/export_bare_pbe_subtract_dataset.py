#!/usr/bin/env python3
"""
Export FINAL_RESULTS_BARE_PBE_SUBTRACT (bare-PBE phonons of the 48 Mo/W TMDs obtained by explicit
D3 subtraction) as a self-contained, model-agnostic bundle, same layout as
scripts/export_shareable_dataset.py produced for the PBE+D3 data (share/tmd48_phonon_dataset_v1),
but with energies and stresses on every displaced supercell.

Bundle layout (share/<name>/):
  README.md                 what this is, how it was made, units, caveats
  materials.csv / .txt      one row / one name per material
  displacements.extxyz      every DFT displaced supercell. Standard keys = bare PBE
                            (energy, forces, stress); PBE+D3 and D3 kept alongside.
  evaluate_phonons.py       reference evaluation script (any ASE calculator -> phonon RMSE)
  requirements.txt, SHA256SUMS
  materials/<name>/         POSCAR, phonopy.yaml, FORCE_SETS (bare, fz-corrected), FORCE_CONSTANTS,
                            band.yaml, band.pdf, d3_subtraction.{json,npz}, bare_vs_pbed3.json,
                            [hiphive_fit.json], plain_phonopy/, reference_pbe_d3/{band.yaml,band.pdf}
and a zip next to it.

Needs simple-dftd3 (nequix uv env):
    cd ../nequix && uv run python ../workflow/scripts/export_bare_pbe_subtract_dataset.py
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
from ase.stress import full_3x3_to_voigt_6_stress

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "nequix" / "explicit_dispersion"))
from export_shareable_dataset import BUNDLE_SCRIPT_DIR, TMD_RE, classify, to_ase  # noqa: E402
import d3  # noqa: E402

SOURCE = ROOT / "FINAL_RESULTS_BARE_PBE_SUBTRACT"
REFERENCE = ROOT / "FINAL_RESULTS_HEALTHY"
PER_MATERIAL_FILES = ["POSCAR", "phonopy.yaml", "FORCE_SETS", "FORCE_CONSTANTS", "band.yaml", "band.pdf",
                      "d3_subtraction.json", "d3_subtraction.npz", "bare_vs_pbed3.json"]
OPTIONAL_FILES = ["hiphive_fit.json"]
EV_A3_TO_GPA = 160.21766


def staticpoint_dir(name: str) -> Path:
    for d in (ROOT / "phonopy_monolayer_examples" / f"{name}_staticpoint", ROOT / "phonopy_bilayer_examples" / f"{name}_staticpoint"):
        if d.is_dir():
            return d
    sys.exit(f"{name}: no staticpoint directory with vasprun.xml files on this host")


def relaxation_dir(name: str) -> Path | None:
    for d in (ROOT / "monolayer_examples" / name, ROOT / "bilayer_examples" / name):
        if (d / "OUTCAR").exists():
            return d
    return None


def unitcell_energetics(name: str, poscar: Path) -> dict:
    """Relaxed unit cell energy/stress from the relaxation OUTCAR, only if its final geometry is the
    shipped POSCAR (some cells were ISIF=4 refined afterwards; then the OUTCAR is stale -> blank)."""
    blank = dict(unitcell_energy_pbe_d3_eV="", unitcell_energy_d3_eV="", unitcell_energy_bare_eV="",
                 unitcell_stress_pbe_d3_GPa_voigt="", unitcell_stress_d3_GPa_voigt="", unitcell_energetics_note="")
    rd = relaxation_dir(name)
    if rd is None:
        return {**blank, "unitcell_energetics_note": "no relaxation OUTCAR on this host"}
    try:
        out = read(rd / "OUTCAR", index=-1)
    except Exception as e:  # noqa: BLE001
        return {**blank, "unitcell_energetics_note": f"OUTCAR unreadable: {e}"}
    ref = read(poscar)
    if len(out) != len(ref) or np.abs(out.cell.array - ref.cell.array).max() > 1e-3 or \
            np.abs(out.get_scaled_positions() - ref.get_scaled_positions()).max() > 1e-4:
        return {**blank, "unitcell_energetics_note": "relaxation OUTCAR geometry != shipped POSCAR (cell refined afterwards)"}
    e = out.get_potential_energy()
    s = out.get_stress()
    d = d3.d3_efs(ref)
    sd = full_3x3_to_voigt_6_stress(d["stress"])
    fmt = lambda v: " ".join(f"{x:.5f}" for x in v)  # noqa: E731
    return dict(unitcell_energy_pbe_d3_eV=f"{e:.6f}", unitcell_energy_d3_eV=f"{d['energy']:.6f}",
                unitcell_energy_bare_eV=f"{e - d['energy']:.6f}",
                unitcell_stress_pbe_d3_GPa_voigt=fmt(s * EV_A3_TO_GPA), unitcell_stress_d3_GPa_voigt=fmt(sd * EV_A3_TO_GPA),
                unitcell_energetics_note="relaxation OUTCAR, 21x21x1 k-mesh (supercell statics use 7x7x1)")


def readme(n_mat: int, n_frames: int, n_hiphive: int, forced: list[str]) -> str:
    return f"""# 48-material Mo/W TMD phonon dataset — bare PBE by explicit D3 subtraction

Harmonic phonon reference data for the same 48 dynamically stable 2D transition-metal dichalcogenides
as `tmd48_phonon_dataset_v1` (6 monolayers, 12 homobilayers, 30 heterobilayers built from MoS2, MoSe2,
MoTe2, WS2, WSe2, WTe2), but with the Grimme D3(BJ) dispersion term **removed analytically** so the
targets are **bare PBE**. Built {date.today().isoformat()} from `FINAL_RESULTS_BARE_PBE_SUBTRACT/` in the
twist workflow.

> **Read every number here as bare-PBE curvature / energetics at the PBE+D3-relaxed geometry.**
> No structure was re-relaxed without D3. The structures are therefore *not* stationary points of
> bare PBE: each carries a residual bare-PBE force of 0.10–0.17 eV/Å (= −F_D3 at equilibrium, mostly
> on the chalcogen / interlayer z coordinate) and a nonzero stress. A genuine bare-PBE (IVDW off)
> re-relaxation + phonon campaign exists separately and is the right reference for bare-PBE minima.

## What was subtracted

The DFT reference is VASP PBE + DFT-D3(BJ) (`IVDW = 12`, VASP defaults: s6 = 1.0, s8 = 0.7875,
a1 = 0.4289, a2 = 4.4407, two-body cutoff 50.2022 Å, CN cutoff 21.1671 Å, no three-body term). The
identical D3 term was re-evaluated with simple-dftd3 on every structure (checked against VASP's own
`Edisp` on 390 supercells: max 0.85 meV, mean 0.49 meV per supercell) and subtracted:

| quantity | stored as | definition |
|---|---|---|
| `energy`, `forces`, `stress` (standard ASE keys in `displacements.extxyz`) | bare PBE | `X_pbe_d3 − X_d3` on the displaced supercell |
| `energy_pbe_d3`, `forces_pbe_d3`, `stress_pbe_d3` | PBE+D3 | straight from VASP `vasprun.xml` |
| `energy_d3`, `forces_d3`, `stress_d3` | D3 only | simple-dftd3 on the same geometry |
| `materials/<name>/FORCE_SETS` | bare PBE, **residual-corrected** | `F_pbe_d3 − [F_d3(disp) − F_d3(eq)]` |

The residual-corrected FORCE_SETS is what the phonons were built from: phonopy's finite differences
assume zero force on the undisplaced cell, so the constant bare-PBE equilibrium force is removed
(phonopy's `--fz` treatment). `F_d3(eq)` is in `materials/<name>/d3_subtraction.npz` (`f_d3_eq`),
so `forces` in the extxyz and FORCE_SETS differ by exactly that constant per atom.

Energies are VASP `e_0_energy` (σ→0 extrapolated, ISMEAR=0 / SIGMA=0.05 so free-energy differences
are < 1 meV). Stresses are in eV/Å³ in the ASE sign convention (positive = tensile), 6-component
Voigt order xx yy zz yz xz xy, for the full 20 Å-vacuum cell (not thickness-renormalised).

## Post-processing (same two steps the PBE+D3 reference went through)

1. plain phonopy (`FC_SYMMETRY = .TRUE.`) on the bare FORCE_SETS for all {n_mat} materials; that result
   is kept in `materials/<name>/plain_phonopy/`.
2. hiphive rotational-sum-rule constrained fit (cutoff ≈ 6 Å, λ sweep 1e-3…1e2, largest λ kept) on
   {n_hiphive} materials: the 20 whose PBE+D3 reference had itself been hiphive-corrected, plus
   {', '.join(forced)} where the plain build kept a small (−0.002…−0.03 THz) flexural dip near Γ. These
   {n_hiphive} are flagged `hiphive_applied = True` in `materials.csv`, with `hiphive_fit.json` recording the fit.
   `FORCE_SETS` is always the untouched (subtracted) force data, so you can always refit yourself.

After step 2 every material is ≥ 0 THz on the Γ–K–M–Γ path and on a 40×40×1 mesh.

`materials/<name>/reference_pbe_d3/band.yaml` is the PBE+D3 dispersion the bare one was derived from.
`bare_vs_pbed3.json` / `materials.csv` give the RMS difference two ways: `rmse_plain_bare_vs_plain_pbed3_THz`
(plain phonopy on both sides = the pure D3 effect at fixed geometry, 0.007–0.018 THz, always a
softening) and `rmse_vs_pbed3_THz` (final vs shipped reference; for hiphive materials this also contains
the mismatch between this refit and the reference's own hiphive step, so quote the former for "what D3
does").

## Contents

```
materials.csv / materials.txt   per-material table / plain list of the {n_mat} names
displacements.extxyz            {n_frames} displaced supercells (ASE extended XYZ), keys as above;
                                atoms.info also has material, displacement_index, displaced_atom, displacement
evaluate_phonons.py             benchmark any ASE calculator against band.yaml (unchanged from v1 bundle)
requirements.txt, SHA256SUMS
materials/<name>/
    POSCAR                      PBE+D3-relaxed unit cell (60° hexagonal, 20 Å c)
    phonopy.yaml                unit cell, supercell + primitive matrix, symmetry
    FORCE_SETS                  bare-PBE displaced forces, residual-corrected (see above)
    FORCE_CONSTANTS             final bare-PBE force constants (eV/Å²), plain or hiphive-refit
    band.yaml / band.pdf        bare-PBE dispersion on Γ–K–M–Γ (THz; 153 or 303 q-points)
    plain_phonopy/              step-1 result (band.yaml, band.pdf, FORCE_CONSTANTS)
    reference_pbe_d3/           PBE+D3 band.yaml / band.pdf from the parent dataset
    d3_subtraction.json / .npz  every D3 quantity used (f_ref, f_d3_disp, f_d3_eq, f_bare, E_d3, stress_d3)
    bare_vs_pbed3.json          minima, mesh check, RMS vs PBE+D3 (both definitions)
    hiphive_fit.json            (hiphive materials only) cutoff, λ sweep, chosen λ
```

`materials.csv` additionally carries, where the relaxation OUTCAR is on the exporting host and its final
geometry equals the shipped POSCAR, the relaxed **unit-cell** energy (PBE+D3, D3, bare; 21×21×1 k-mesh) and
stress (GPa, Voigt). Blank cells mean not available; see `unitcell_energetics_note`.

## DFT settings of the parent data

VASP 6.5.1, PAW PBE (`Mo_sv`, `W_sv`, `S`, `Se`, `Te`), spin-unpolarised, `ENCUT = 520`, `PREC = Accurate`,
`EDIFF = 1e-6`, `ISMEAR = 0`, `SIGMA = 0.05`, no U, no SOC. Relaxation `ISIF = 2`, `EDIFFG = -1e-7`,
21×21×1 Γ-centred (a few cells ISIF=4-refined in-plane). Phonons: 0.01 Å displacements in 4×4×1
(5×5×1 / 6×6×1 where noted in `supercell`), 7×7×1 Γ-centred, phonopy 4.1.0. q-path in phonopy's
standardised 120° primitive cell: Γ (0,0,0) → K (1/3,1/3,0) → M (1/2,0,0) → Γ.

## Loading

```python
from ase.io import read
frames = read("displacements.extxyz", index=":")
a = frames[0]
a.get_potential_energy(), a.get_forces(), a.get_stress()     # bare PBE
a.info["energy_pbe_d3"], a.arrays["forces_pbe_d3"], a.info["stress_pbe_d3"]
a.info["energy_d3"],     a.arrays["forces_d3"],     a.info["stress_d3"]
```

## Benchmarking a model

`python evaluate_phonons.py --calc "pkg.module:make_calculator" [--relax positions] [--plots]` compares
on exactly the stored q-points; state whether you report `--relax none` (DFT geometry) or `--relax positions`.
A model that has *no* dispersion term should be compared against this bundle; a model that includes D3
belongs with `tmd48_phonon_dataset_v1`.

## Contact

Dataset produced by Kai Gowers (Boston College). Please get in touch before publishing results derived from it.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="tmd48_bare_pbe_subtract_v1")
    ap.add_argument("--out-root", default="share")
    args = ap.parse_args()

    materials = sorted(p.name for p in SOURCE.iterdir() if p.is_dir() and TMD_RE.match(p.name))
    if len(materials) != 48:
        sys.exit(f"expected 48 materials in {SOURCE}, found {len(materials)}")
    summary = {r["material"]: r for r in csv.DictReader((SOURCE / "summary.csv").open())}

    out = ROOT / args.out_root / args.name
    if out.exists():
        shutil.rmtree(out)
    (out / "materials").mkdir(parents=True)

    rows, frames = [], []
    for name in materials:
        src, ref, dst = SOURCE / name, REFERENCE / name, out / "materials" / name
        dst.mkdir()
        for f in PER_MATERIAL_FILES:
            if not (src / f).exists():
                sys.exit(f"{name}: missing {f}")
            shutil.copy2(src / f, dst / f)
        for f in OPTIONAL_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        shutil.copytree(src / "plain_phonopy", dst / "plain_phonopy")
        (dst / "reference_pbe_d3").mkdir()
        for f in ("band.yaml", "band.pdf"):
            shutil.copy2(ref / f, dst / "reference_pbe_d3" / f)

        # displaced supercells: VASP (PBE+D3) energy/forces/stress + D3 -> bare
        ph = phonopy.load(str(ref / "phonopy.yaml"), force_sets_filename=str(ref / "FORCE_SETS"), log_level=0)
        disps = ph.dataset["first_atoms"]
        vaspruns = sorted(staticpoint_dir(name).glob("disp-*/vasprun.xml"))
        if len(vaspruns) != len(disps):
            sys.exit(f"{name}: {len(vaspruns)} vasprun.xml vs {len(disps)} displacements")
        for i, (disp, sc, vr) in enumerate(zip(disps, ph.supercells_with_displacements, vaspruns)):
            v = read(vr, index=-1)
            atoms = to_ase(sc)
            dfrac = v.get_scaled_positions() - atoms.get_scaled_positions()
            dfrac -= np.round(dfrac)  # minimum image: VASP wraps atoms displaced across a cell boundary
            if np.abs(v.get_forces() - disp["forces"]).max() > 1e-5 or np.abs(dfrac).max() > 1e-5 \
                    or np.abs(v.cell.array - atoms.cell.array).max() > 1e-5:
                sys.exit(f"{name} disp {i}: vasprun does not match FORCE_SETS / phonopy supercell")
            e_ref, f_ref, s_ref = v.get_potential_energy(), v.get_forces(), v.get_stress()
            dd = d3.d3_efs(atoms)
            s_d3 = full_3x3_to_voigt_6_stress(dd["stress"])
            atoms.calc = SinglePointCalculator(atoms, energy=e_ref - dd["energy"], forces=f_ref - dd["forces"], stress=s_ref - s_d3)
            atoms.arrays["forces_pbe_d3"] = f_ref
            atoms.arrays["forces_d3"] = dd["forces"]
            atoms.info.update(
                material=name, displacement_index=i, displaced_atom=int(disp["number"]),
                displacement=np.array(disp["displacement"]), config_type="phonon_displacement",
                energy_pbe_d3=e_ref, energy_d3=dd["energy"], stress_pbe_d3=s_ref, stress_d3=s_d3,
                functional="PBE (D3(BJ) subtracted)", geometry="PBE+D3 relaxed",
            )
            frames.append(atoms)

        band = yaml.safe_load((src / "band.yaml").read_text())
        freqs = np.array([[b["frequency"] for b in q["band"]] for q in band["phonon"]])
        kind, stacking, layers = classify(name)
        dim = ph.supercell_matrix.diagonal()
        s = summary[name]
        info = json.loads((src / "d3_subtraction.json").read_text())
        rows.append(dict(
            material=name, kind=kind, stacking=stacking, layers=layers,
            formula_unitcell=to_ase(ph.unitcell).get_chemical_formula(), n_atoms_unitcell=len(ph.unitcell),
            supercell=f"{dim[0]}x{dim[1]}x{dim[2]}", n_atoms_supercell=len(ph.supercell), n_displacements=len(disps),
            n_qpoints=freqs.shape[0], n_bands=freqs.shape[1],
            min_freq_THz=f"{freqs.min():.4f}", max_freq_THz=f"{freqs.max():.4f}",
            force_constants_source="hiphive-refit" if s["hiphive_applied"] == "True" else "phonopy",
            hiphive_applied=s["hiphive_applied"], hiphive_reason=s.get("hiphive_reason", ""),
            reference_pbe_d3_was_hiphive=s["hiphive_required"],
            min_freq_plain_THz=f"{float(s['min_freq_plain_THz']):.4f}",
            mesh40_min_THz=f"{float(s['mesh40_min_THz']):.4f}",
            rmse_plain_bare_vs_plain_pbed3_THz=f"{float(s['rmse_plain_bare_vs_plain_pbed3_THz']):.4f}",
            rmse_vs_pbed3_THz=f"{float(s['rmse_vs_pbed3_THz']):.4f}",
            max_abs_F_d3_eq_eV_per_A=f"{info['max_abs_F_d3_eq_eV_per_A']:.4f}",
            E_d3_eq_supercell_eV=f"{info['E_d3_eq_eV']:.4f}",
            **unitcell_energetics(name, src / "POSCAR"),
        ))
        print(f"{name:<20} {kind:<14} {rows[-1]['supercell']} {len(disps):>2} disp  fc={rows[-1]['force_constants_source']:<14} "
              f"unitcell E: {'yes' if rows[-1]['unitcell_energy_bare_eV'] else 'no'}", flush=True)

    with (out / "materials.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "materials.txt").write_text("\n".join(materials) + "\n")
    ase_write(str(out / "displacements.extxyz"), frames, format="extxyz")
    for f in ("evaluate_phonons.py", "requirements.txt"):
        shutil.copy2(BUNDLE_SCRIPT_DIR / f, out / f)
    forced = [r["material"] for r in rows if r["hiphive_applied"] == "True" and r["reference_pbe_d3_was_hiphive"] != "True"]
    n_h = sum(r["hiphive_applied"] == "True" for r in rows)
    (out / "README.md").write_text(readme(len(rows), len(frames), n_h, forced))
    with (out / "SHA256SUMS").open("w") as fh:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n")
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zf.write(p, Path(out.name) / p.relative_to(out))
    n_uc = sum(bool(r["unitcell_energy_bare_eV"]) for r in rows)
    print(f"\n{len(rows)} materials, {len(frames)} displaced supercells, {n_h} hiphive-refit, unit-cell energetics for {n_uc}/48")
    print(f"bundle: {out}\nzip:    {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
