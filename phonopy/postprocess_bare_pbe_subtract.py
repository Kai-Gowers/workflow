#!/usr/bin/env python3
"""
Stage 2 of the bare-PBE-by-subtraction pipeline: turn the bare-PBE FORCE_SETS written by
subtract_d3_force_sets.py into band.yaml / band.pdf / FORCE_CONSTANTS in
FINAL_RESULTS_BARE_PBE_SUBTRACT/<material>/, following the same two-step post-processing the
PBE+D3 reference went through:

1. plain phonopy (`phonopy -p -s --writefc --config band.conf`, FC_SYMMETRY = .TRUE.) for all
   48 materials -> a copy of band.yaml / band.pdf / FORCE_CONSTANTS is kept in plain_phonopy/.
2. for the materials whose PBE+D3 reference FORCE_CONSTANTS was hiphive-corrected
   (`hiphive_rotational_fit_required` in d3_subtraction.json), the constraint-based hiphive fit
   from phonopy/hiphive_fit_force_constants.py (rotational sum rules enforced during the force
   constant fit, lambda sweep, largest lambda kept) is re-applied to the *bare* displaced forces.
   The displaced training structures are rebuilt from FORCE_SETS instead of disp-*/vasprun.xml,
   because the bare forces exist only in FORCE_SETS (and 8/48 staticpoint dirs are not on this
   host). Otherwise the fit (cutoff estimate, lambda sweep, band path, mesh check) is identical.
   The result overwrites the top-level band.yaml / band.pdf / FORCE_CONSTANTS; the sweep is
   recorded in hiphive_fit.json.

Every material also gets bare_vs_pbed3.json: band-path minimum frequency (plain / final), full
40x40x1 mesh minimum, and the RMS frequency difference to the PBE+D3 reference band.yaml computed
on the reference's own q-points. summary.csv and README.md at the top level collect these.

Environment: workflow conda env (phonopy >= 4, hiphive, trainstation; no dftd3 needed here).

    python3 phonopy/postprocess_bare_pbe_subtract.py [--materials M ...] [--skip-hiphive] [--cutoff A]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import phonopy
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
from hiphive.core.rotational_constraints import get_rotational_constraint_matrix
from hiphive.utilities import prepare_structures
from phonopy.file_IO import write_FORCE_CONSTANTS
from trainstation import Optimizer

ROOT = Path(__file__).resolve().parent
WORKFLOW_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from hiphive_fit_force_constants import DEFAULT_LAMBDAS, estimate_safe_cutoff, parse_band_conf, phonopy_atoms_to_ase  # noqa: E402
from postprocess_results import run_phonopy_band  # noqa: E402

OUT_ROOT = WORKFLOW_ROOT / "FINAL_RESULTS_BARE_PBE_SUBTRACT"
FINAL_RESULTS_HEALTHY = WORKFLOW_ROOT / "FINAL_RESULTS_HEALTHY"
PLAIN_FILES = ("band.yaml", "band.pdf", "FORCE_CONSTANTS")


def band_min(mat_dir: Path) -> float:
    by = yaml.safe_load((mat_dir / "band.yaml").read_text())
    return float(min(b["frequency"] for q in by["phonon"] for b in q["band"]))


def hiphive_fit(mat_dir: Path, cutoff: float | None, lambdas: list[float]) -> dict:
    ph = phonopy.load(str(mat_dir / "phonopy.yaml"), force_sets_filename=str(mat_dir / "FORCE_SETS"), log_level=0)
    ideal = phonopy_atoms_to_ase(ph.supercell)
    structures = []
    for disp in ph.dataset["first_atoms"]:
        a = ideal.copy()
        a.positions[disp["number"]] += disp["displacement"]
        a.calc = SinglePointCalculator(a, forces=disp["forces"])
        structures.append(a)
    training = prepare_structures(structures, ideal)
    prim = read(mat_dir / "POSCAR")
    if cutoff is None:
        cutoff = estimate_safe_cutoff(ideal)
    cs = ClusterSpace(prim, [cutoff])
    sc = StructureContainer(cs)
    for s in training:
        sc.add_structure(s)
    A, y = sc.get_fit_data()
    opt = Optimizer((A, y), train_size=1.0)
    opt.train()
    Ac = get_rotational_constraint_matrix(cs)
    yc = np.zeros(Ac.shape[0])
    paths, labels, conns = parse_band_conf(mat_dir / "band.conf")

    sweep, best = [], None
    for lam in lambdas:
        opt2 = Optimizer((np.vstack((A, lam * Ac)), np.hstack((y, yc))), train_size=1.0, standardize=False)
        opt2.train()
        fc = ForceConstantPotential(cs, opt2.parameters).get_force_constants(ideal).get_fc_array(order=2)
        ph.force_constants = fc
        ph.run_band_structure(paths)
        bmin = float(np.vstack(ph.get_band_structure_dict()["frequencies"]).min())
        ph.run_mesh([40, 40, 1], is_gamma_center=True, with_eigenvectors=False)
        mesh = ph.get_mesh_dict()["frequencies"]
        entry = {"lambda": lam, "band_min": bmin, "mesh_min": float(mesh.min()), "mesh_n_negative": int((mesh < -1e-4).sum())}
        sweep.append(entry); best = (entry, fc)
        print(f"    lambda={lam:.0e}: band min {bmin:+.5f} THz, mesh min {entry['mesh_min']:+.5f} THz", flush=True)

    entry, fc = best
    write_FORCE_CONSTANTS(fc, filename=str(mat_dir / "FORCE_CONSTANTS"))
    ph.force_constants = fc
    ph.run_band_structure(paths, labels=labels, path_connections=conns)
    ph.write_yaml_band_structure(filename=str(mat_dir / "band.yaml"))
    ph.plot_band_structure().savefig(mat_dir / "band.pdf")
    out = {"method": "hiphive constraint-based fit from bare FORCE_SETS (phonopy/hiphive_fit_force_constants.py recipe)",
           "cutoff_A": cutoff, "unconstrained_rmse_train_eV_per_A": float(opt.rmse_train), "sweep": sweep,
           "chosen_lambda": entry["lambda"], "n_training_structures": len(training)}
    (mat_dir / "hiphive_fit.json").write_text(json.dumps(out, indent=2))
    return out


def compare_to_reference(mat_dir: Path, material: str) -> dict:
    """Bare (final) vs PBE+D3 reference on the reference's q-points, plus a decomposition: the pure
    D3 effect is RMS(plain bare, plain rebuild of the reference FORCE_SETS); the rest of the final
    difference for hiphive materials is method mismatch between this refit and whatever hiphive
    step the reference got (cutoff / lambda / fix-vs-fit), not dispersion."""
    ref_dir = FINAL_RESULTS_HEALTHY / material
    by = yaml.safe_load((ref_dir / "band.yaml").read_text())
    q = np.array([p["q-position"] for p in by["phonon"]])
    f_ref = np.array([[b["frequency"] for b in p["band"]] for p in by["phonon"]])

    def freqs(ph):
        ph.run_qpoints(q)
        return ph.get_qpoints_dict()["frequencies"]

    ph = phonopy.load(str(mat_dir / "phonopy.yaml"), force_constants_filename=str(mat_dir / "FORCE_CONSTANTS"), log_level=0)
    f_bare = freqs(ph)
    ph.run_mesh([40, 40, 1], is_gamma_center=True, with_eigenvectors=False)
    mesh = ph.get_mesh_dict()["frequencies"]
    ph_bp = phonopy.load(str(mat_dir / "phonopy.yaml"), force_constants_filename=str(mat_dir / "plain_phonopy" / "FORCE_CONSTANTS"), log_level=0)
    f_bare_plain = freqs(ph_bp)
    ph_rp = phonopy.load(str(ref_dir / "phonopy.yaml"), force_sets_filename=str(ref_dir / "FORCE_SETS"), log_level=0)
    ph_rp.produce_force_constants()
    ph_rp.symmetrize_force_constants()
    f_ref_plain = freqs(ph_rp)
    rms = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))  # noqa: E731
    d = f_bare - f_ref
    return {"rmse_vs_pbed3_THz": rms(f_bare, f_ref), "max_abs_diff_vs_pbed3_THz": float(np.abs(d).max()),
            "mean_shift_vs_pbed3_THz": float(d.mean()), "min_freq_final_THz": float(f_bare.min()),
            "min_freq_pbed3_ref_THz": float(f_ref.min()),
            "rmse_plain_bare_vs_plain_pbed3_THz": rms(f_bare_plain, f_ref_plain),
            "rmse_pbed3_ref_vs_plain_pbed3_THz": rms(f_ref, f_ref_plain),
            "mesh40_min_THz": float(mesh.min()), "mesh40_n_negative": int((mesh < -1e-4).sum()), "n_q": int(len(q))}


def process(material: str, cutoff: float | None, lambdas: list[float], skip_hiphive: bool) -> dict:
    mat_dir = OUT_ROOT / material
    info = json.loads((mat_dir / "d3_subtraction.json").read_text())
    print(f"{material}", flush=True)
    for f in PLAIN_FILES + ("phonopy.yaml",):
        (mat_dir / f).unlink(missing_ok=True)
    if not run_phonopy_band(mat_dir, mat_dir / "band.conf"):
        raise RuntimeError(f"phonopy failed for {material}")
    plain_dir = mat_dir / "plain_phonopy"
    plain_dir.mkdir(exist_ok=True)
    for f in PLAIN_FILES:
        shutil.copy2(mat_dir / f, plain_dir / f)
    row = {"material": material, "hiphive_required": info["hiphive_rotational_fit_required"],
           "min_freq_plain_THz": band_min(mat_dir), "hiphive_applied": False, "hiphive_cutoff_A": None, "hiphive_lambda": None}
    if info["hiphive_rotational_fit_required"] and not skip_hiphive:
        print("  hiphive rotational-sum-rule fit (reference was hiphive-corrected)", flush=True)
        fit = hiphive_fit(mat_dir, cutoff, lambdas)
        row.update(hiphive_applied=True, hiphive_cutoff_A=fit["cutoff_A"], hiphive_lambda=fit["chosen_lambda"])
    (mat_dir / "hiphive_fit.json").unlink(missing_ok=True) if not row["hiphive_applied"] else None
    row.update(compare_to_reference(mat_dir, material))
    row["max_abs_F_d3_eq_eV_per_A"] = info["max_abs_F_d3_eq_eV_per_A"]
    (mat_dir / "bare_vs_pbed3.json").write_text(json.dumps(row, indent=2))
    print(f"  min plain {row['min_freq_plain_THz']:+.4f} -> final {row['min_freq_final_THz']:+.4f} THz, "
          f"mesh min {row['mesh40_min_THz']:+.4f}, RMS vs PBE+D3 {row['rmse_vs_pbed3_THz']:.4f} THz", flush=True)
    return row


def summarize(material: str) -> dict:
    mat_dir = OUT_ROOT / material
    row = json.loads((mat_dir / "bare_vs_pbed3.json").read_text())
    row.update(compare_to_reference(mat_dir, material))
    (mat_dir / "bare_vs_pbed3.json").write_text(json.dumps(row, indent=2))
    return row


def write_readme(rows: list[dict]) -> None:
    n_h = sum(r["hiphive_applied"] for r in rows)
    worst = min(rows, key=lambda r: r["min_freq_final_THz"])
    rms = np.array([r["rmse_vs_pbed3_THz"] for r in rows])
    txt = f"""# FINAL_RESULTS_BARE_PBE_SUBTRACT — bare-PBE phonons by explicit D3 subtraction

Generated {date.today().isoformat()} by `phonopy/subtract_d3_force_sets.py` (stage 1, nequix uv env) and
`phonopy/postprocess_bare_pbe_subtract.py` (stage 2, workflow conda env). {len(rows)} Mo/W TMDs, same set and
geometries as `FINAL_RESULTS_HEALTHY` restricted to `nequix_datasets/v4_tmd_only`.

**What these are.** Bare-PBE harmonic phonons *at the PBE+D3-relaxed geometry*. The DFT-D3(BJ) term (VASP
`IVDW = 12` parameters, evaluated with simple-dftd3; see `nequix/explicit_dispersion/README.md`) is subtracted
from the VASP forces on every displaced supercell, together with the equilibrium D3 force, so the result is what
phonopy would give for PBE-only forces on a structure that *is* a PBE minimum in the harmonic sense:

    F_bare(disp) = F_ref(disp) - [ F_D3(disp) - F_D3(eq) ]

The structures were **not** re-relaxed with bare PBE (that is the separate `TWIST_VARIANT=bare_pbe` DFT campaign,
`FINAL_RESULTS_BARE_PBE/`). The residual bare-PBE force -F_D3(eq) (0.10-0.17 eV/A, mostly interlayer / chalcogen z)
is recorded per material and the stress placeholder is unchanged from the reference.

**Pipeline per material** (`<material>/`):

| file | content |
|---|---|
| `POSCAR`, `phonopy_disp.yaml`, `band.conf` | unchanged geometry / cell setup / band path from the reference run |
| `FORCE_SETS` | bare-PBE displaced forces (above) |
| `d3_subtraction.npz/.json` | every D3 quantity used (F_D3 per displacement, F_D3(eq), E_D3, stress) and sanity numbers |
| `plain_phonopy/` | step 1: plain `phonopy -p -s --writefc` result (FC_SYMMETRY on), kept for all {len(rows)} |
| `FORCE_CONSTANTS`, `band.yaml`, `band.pdf`, `phonopy.yaml` | final result: = plain for {len(rows) - n_h} materials, hiphive-fit for {n_h} |
| `hiphive_fit.json` | ({n_h} materials) cutoff, lambda sweep, chosen lambda — only where the PBE+D3 reference had been hiphive-corrected |
| `bare_vs_pbed3.json` | band-path / 40x40x1-mesh minima and RMS difference to the PBE+D3 reference on its own q-points |

**Headline.** RMS(bare - PBE+D3) over all bands: mean {rms.mean():.4f} THz, range {rms.min():.4f}-{rms.max():.4f} THz.
The pure D3 effect at fixed geometry is `rmse_plain_bare_vs_plain_pbed3_THz` (plain phonopy on both sides); for the
{n_h} hiphive materials `rmse_vs_pbed3_THz` additionally contains the mismatch between this refit and the reference's own
hiphive step (`rmse_pbed3_ref_vs_plain_pbed3_THz` shows how large that step was), so use the plain-vs-plain column for
"what does D3 do to the bands".
Lowest band-path frequency after post-processing: {worst['min_freq_final_THz']:+.4f} THz ({worst['material']}).
See `summary.csv` for every material.
"""
    (OUT_ROOT / "README.md").write_text(txt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materials", nargs="*", default=None)
    ap.add_argument("--cutoff", type=float, default=None, help="hiphive cutoff (default: estimate_safe_cutoff)")
    ap.add_argument("--lambdas", type=str, default=None, help="comma-separated lambda sweep (default as hiphive_fit_force_constants.py)")
    ap.add_argument("--skip-hiphive", action="store_true")
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute bare_vs_pbed3.json / summary.csv / README.md from the existing FORCE_CONSTANTS, no phonopy/hiphive rerun")
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else DEFAULT_LAMBDAS
    mats = args.materials or sorted(p.name for p in OUT_ROOT.iterdir() if (p / "d3_subtraction.json").exists())
    rows = [summarize(m) if args.summary_only else process(m, args.cutoff, lambdas, args.skip_hiphive) for m in mats]
    if args.materials is None:
        with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        write_readme(rows)
        print(f"\nwrote {OUT_ROOT / 'summary.csv'} and README.md")


if __name__ == "__main__":
    main()
