#!/usr/bin/env python3
"""
Stage 1 of the bare-PBE-by-subtraction pipeline: build bare-PBE FORCE_SETS for the 48 Mo/W TMDs
from the PBE+D3 (VASP IVDW = 12) reference in FINAL_RESULTS_HEALTHY/, writing one directory per
material into FINAL_RESULTS_BARE_PBE_SUBTRACT/<material>/.

For each displaced supercell that VASP actually ran (taken from the reference FORCE_SETS):

    F_bare(disp) = F_ref(disp) - [ F_D3(disp) - F_D3(eq) ]

* F_D3(disp): explicit DFT-D3(BJ) force on the displaced supercell with VASP's own parameters
  (nequix/explicit_dispersion/d3.py; checked against VASP's Edisp to < 1 meV).
* F_D3(eq): D3 force on the undisplaced supercell. The structure is a PBE+D3 minimum, so the
  bare-PBE equilibrium force is -F_D3(eq) != 0 (0.10-0.17 eV/A). Phonopy's finite differences
  assume zero residual force, so it is subtracted explicitly here (phonopy's "--fz" treatment)
  instead of relying on +/- displacement pairs to cancel it.

Also written per material: POSCAR (unchanged geometry), phonopy_disp.yaml (= reference phonopy.yaml), band.conf (same DIM / BAND path as the
reference run; ATOM_NAME from POSCAR), d3_subtraction.npz (every D3 quantity used) and
d3_subtraction.json (scalars + a flag saying whether the reference FORCE_CONSTANTS was a plain
phonopy rebuild of FORCE_SETS or had been hiphive-corrected — decided by the RMS band-frequency
change, see reference_is_plain_rebuild). Stage 2
(postprocess_bare_pbe_subtract.py) turns these into band.yaml / band.pdf / FORCE_CONSTANTS.

Environment: needs simple-dftd3, which lives in the nequix uv env, not the workflow conda env:

    cd ../nequix && uv run python ../workflow/phonopy/subtract_d3_force_sets.py [--materials M ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import phonopy
import yaml
from ase import Atoms
from phonopy.file_IO import write_FORCE_SETS
from phonopy.harmonic.force_constants import compact_fc_to_full_fc

ROOT = Path(__file__).resolve().parent
WORKFLOW_ROOT = ROOT.parent
TWIST_ROOT = WORKFLOW_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TWIST_ROOT / "nequix" / "explicit_dispersion"))
from postprocess_results import parse_atom_names_from_poscar, write_band_conf  # noqa: E402
import d3  # noqa: E402  (nequix/explicit_dispersion/d3.py: VASP IVDW=12 parametrisation)

FINAL_RESULTS_HEALTHY = WORKFLOW_ROOT / "FINAL_RESULTS_HEALTHY"
OUT_ROOT = WORKFLOW_ROOT / "FINAL_RESULTS_BARE_PBE_SUBTRACT"
DATASET = WORKFLOW_ROOT / "nequix_datasets" / "v4_tmd_only"
FC_REPRODUCE_TOL = 5e-3  # eV/A^2, informational (nequix/explicit_dispersion/strip_d3_dataset.py used this alone)
FREQ_REPRODUCE_TOL_THZ = 2e-3  # decides the hiphive flag: RMS(plain rebuild - shipped FC) on the band path


def tmd48() -> list[str]:
    mats: list[str] = []
    for split in ("train", "val", "holdout"):
        for line in (DATASET / split / "materials.txt").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                mats.append(line)
    if len(mats) != 48:
        raise RuntimeError(f"expected 48 materials in {DATASET}, found {len(mats)}")
    return mats


def to_ase(patoms) -> Atoms:
    return Atoms(numbers=patoms.numbers, positions=patoms.positions, cell=patoms.cell, pbc=True)


def reference_is_plain_rebuild(ref_dir: Path) -> tuple[bool, float, float]:
    """Does phonopy on FORCE_SETS reproduce the shipped FORCE_CONSTANTS (i.e. no hiphive step)?

    Two numbers: max |FC diff| and the RMS frequency difference on the reference band.yaml
    q-points. The frequency criterion decides — a FC max-diff just above tolerance with a
    negligible frequency change (MoTe2_WSe2_3R: 1e-2 eV/A^2 but 0.0004 THz) is numerical noise,
    not a hiphive correction, and must not trigger a refit (the fit's finite cutoff changes the
    bands by ~0.05 THz RMS, far more than the D3 effect itself).
    """
    ph = phonopy.load(str(ref_dir / "phonopy.yaml"), force_constants_filename=str(ref_dir / "FORCE_CONSTANTS"),
                      is_compact_fc=True, log_level=0)
    fc_ref = ph.force_constants
    if fc_ref.shape[0] != fc_ref.shape[1]:
        fc_ref = compact_fc_to_full_fc(ph.primitive, fc_ref)
    ph2 = phonopy.load(str(ref_dir / "phonopy.yaml"), force_sets_filename=str(ref_dir / "FORCE_SETS"), log_level=0)
    ph2.produce_force_constants()
    ph2.symmetrize_force_constants()
    fc2 = ph2.force_constants
    if fc2.shape[0] != fc2.shape[1]:
        fc2 = compact_fc_to_full_fc(ph2.primitive, fc2)
    err = float(np.abs(fc2 - fc_ref).max())
    by = yaml.safe_load((ref_dir / "band.yaml").read_text())
    q = np.array([p["q-position"] for p in by["phonon"]])
    f_ref = np.array([[b["frequency"] for b in p["band"]] for p in by["phonon"]])
    ph2.run_qpoints(q)
    f_plain = ph2.get_qpoints_dict()["frequencies"]
    freq_rms = float(np.sqrt(((f_plain - f_ref) ** 2).mean()))
    return freq_rms <= FREQ_REPRODUCE_TOL_THZ, err, freq_rms


def process(material: str, overwrite: bool) -> dict:
    ref_dir = FINAL_RESULTS_HEALTHY / material
    out_dir = OUT_ROOT / material
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} exists (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    ph = phonopy.load(str(ref_dir / "phonopy.yaml"), force_sets_filename=str(ref_dir / "FORCE_SETS"), log_level=0)
    dataset = ph.dataset
    sc_eq = to_ase(ph.supercell)
    eq = d3.d3_efs(sc_eq)
    f_d3_eq = eq["forces"]

    f_ref, f_d3, f_bare = [], [], []
    new_dataset = {"natom": dataset["natom"], "first_atoms": []}
    for disp, supercell in zip(dataset["first_atoms"], ph.supercells_with_displacements):
        fd = d3.d3_efs(to_ase(supercell))["forces"]
        fb = disp["forces"] - (fd - f_d3_eq)
        f_ref.append(disp["forces"]); f_d3.append(fd); f_bare.append(fb)
        new_dataset["first_atoms"].append({"number": disp["number"], "displacement": disp["displacement"], "forces": fb})

    write_FORCE_SETS(new_dataset, filename=str(out_dir / "FORCE_SETS"))
    shutil.copy2(ref_dir / "POSCAR", out_dir / "POSCAR")
    # phonopy >= 4 CLI needs phonopy_disp.yaml (unit cell, supercell + primitive matrices); the
    # reference phonopy.yaml carries exactly that, so stage 2 reproduces the reference's cell setup.
    shutil.copy2(ref_dir / "phonopy.yaml", out_dir / "phonopy_disp.yaml")
    sm = np.array(ph.supercell_matrix)
    dim = " ".join(str(int(sm[i, i])) for i in range(3))
    write_band_conf(out_dir, parse_atom_names_from_poscar(out_dir / "POSCAR"), dim=dim)

    plain, reproduce_err, reproduce_freq_rms = reference_is_plain_rebuild(ref_dir)
    f_ref, f_d3, f_bare = map(np.array, (f_ref, f_d3, f_bare))
    np.savez(out_dir / "d3_subtraction.npz", f_ref=f_ref, f_d3_disp=f_d3, f_d3_eq=f_d3_eq, f_bare=f_bare,
             e_d3_eq=eq["energy"], stress_d3_eq=eq["stress"])
    info = {
        "material": material,
        "source": str(ref_dir.relative_to(WORKFLOW_ROOT)),
        "n_displacements": int(len(f_bare)),
        "n_atoms_supercell": int(dataset["natom"]),
        "dim": dim,
        "d3_params": {**d3.VASP_PBE_D3BJ, "cutoff_disp2_A": d3.VASP_CUTOFF_DISP2_A, "cutoff_cn_A": d3.VASP_CUTOFF_CN_A},
        "E_d3_eq_eV": float(eq["energy"]),
        "max_abs_F_d3_eq_eV_per_A": float(np.abs(f_d3_eq).max()),
        "max_abs_F_d3_disp_eV_per_A": float(np.abs(f_d3).max()),
        "max_abs_dF_d3_eV_per_A": float(np.abs(f_d3 - f_d3_eq).max()),
        "max_abs_F_ref_disp_eV_per_A": float(np.abs(f_ref).max()),
        "stress_d3_eq_eV_per_A3": np.asarray(eq["stress"]).tolist(),
        "reference_fc_is_plain_phonopy_rebuild": bool(plain),
        "reference_fc_reproduce_maxdiff_eV_per_A2": reproduce_err,
        "reference_fc_reproduce_freq_rms_THz": reproduce_freq_rms,
        "hiphive_rotational_fit_required": bool(not plain),
        "note": "F_bare(disp) = F_ref(disp) - (F_D3(disp) - F_D3(eq)); geometry is the PBE+D3 minimum, not a bare-PBE one",
    }
    (out_dir / "d3_subtraction.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materials", nargs="*", default=None, help="subset (default: all 48 of v4_tmd_only)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    mats = args.materials or tmd48()
    OUT_ROOT.mkdir(exist_ok=True)
    rows = []
    for m in mats:
        info = process(m, args.overwrite)
        rows.append(info)
        print(f"{m:22s} ndisp={info['n_displacements']:3d} |F_D3(eq)|max={info['max_abs_F_d3_eq_eV_per_A']:.3f} "
              f"|dF_D3|max={info['max_abs_dF_d3_eV_per_A']:.4f} eV/A  ref_plain={info['reference_fc_is_plain_phonopy_rebuild']}",
              flush=True)
    (OUT_ROOT / "stage1_subtraction_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} materials to {OUT_ROOT}")


if __name__ == "__main__":
    main()
