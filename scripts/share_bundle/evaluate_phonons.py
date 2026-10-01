#!/usr/bin/env python3
"""
Benchmark any ASE calculator against the DFT phonon dispersions in this bundle.

For each material the script
  1. loads the DFT-relaxed unit cell + supercell matrix from materials/<name>/phonopy.yaml,
  2. generates symmetry-reduced finite displacements with phonopy (same protocol as the DFT),
  3. evaluates forces with YOUR calculator and builds force constants,
  4. computes frequencies on exactly the DFT q-points stored in band.yaml (Γ–K–M–Γ),
  5. reports RMSE / MAE (THz) against DFT, acoustic/optical breakdown and the minimum frequency
     (negative = imaginary = model predicts a dynamical instability; DFT has none here).

The DFT geometry is used as-is by default ("fixed-geometry" eval). With --relax positions the
unit cell's atomic positions are first relaxed with your model (cell fixed), which is the
fairer "deployment" test — report which one you used.

Plug in a model with --calc "module:function" where function() returns an ASE calculator, e.g.

  # MACE-MP-0
  python evaluate_phonons.py --calc "mace.calculators:mace_mp"
  # MACE with args
  python evaluate_phonons.py --calc "mace.calculators:mace_mp" --calc-kwargs '{"model":"medium","default_dtype":"float64"}'
  # Nequix
  python evaluate_phonons.py --calc "nequix.calculator:NequixCalculator" --calc-kwargs '{"model_name":"nequix-omat-1"}'
  # your own: any importable callable returning an ase.calculators.calculator.Calculator

Outputs results.csv (one row per material) and a summary to stdout; --plots adds per-material
dispersion overlays (model vs DFT) as PNGs.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import phonopy
import yaml
from ase import Atoms

HERE = Path(__file__).resolve().parent
MATERIALS_DIR = HERE / "materials"


def load_calculator(spec: str, kwargs: dict):
    module, _, func = spec.partition(":")
    if not func:
        raise SystemExit("--calc must look like 'package.module:callable'")
    return getattr(importlib.import_module(module), func)(**kwargs)


def phonopy_to_ase(patoms) -> Atoms:
    return Atoms(numbers=patoms.numbers, positions=patoms.positions, cell=patoms.cell, pbc=True)


def load_dft(material: str):
    mdir = MATERIALS_DIR / material
    ph = phonopy.load(str(mdir / "phonopy.yaml"), force_constants_filename=str(mdir / "FORCE_CONSTANTS"), log_level=0)
    band = yaml.safe_load((mdir / "band.yaml").read_text())
    q_list = np.array([q["q-position"] for q in band["phonon"]])
    distances = np.array([q["distance"] for q in band["phonon"]])
    dft_freqs = np.array([[b["frequency"] for b in q["band"]] for q in band["phonon"]])
    return ph, q_list, distances, dft_freqs


def relax_positions(unitcell: Atoms, calc, fmax: float, steps: int) -> Atoms:
    from ase.optimize import BFGS

    atoms = unitcell.copy()
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms


def model_force_constants(ph_ref, unitcell: Atoms, calc, displacement: float):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    pa = PhonopyAtoms(symbols=unitcell.get_chemical_symbols(), cell=unitcell.cell[:],
                      scaled_positions=unitcell.get_scaled_positions())
    ph = Phonopy(pa, supercell_matrix=ph_ref.supercell_matrix, primitive_matrix=ph_ref.primitive_matrix, log_level=0)
    ph.generate_displacements(distance=displacement)
    forces = []
    for sc in ph.supercells_with_displacements:
        atoms = phonopy_to_ase(sc)
        atoms.calc = calc
        forces.append(atoms.get_forces())
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    return ph


def metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    diff = pred - ref
    ac, op = diff[:, :3], diff[:, 3:]
    return dict(
        rmse_THz=float(np.sqrt(np.mean(diff**2))),
        mae_THz=float(np.mean(np.abs(diff))),
        rmse_acoustic_THz=float(np.sqrt(np.mean(ac**2))),
        rmse_optical_THz=float(np.sqrt(np.mean(op**2))) if op.size else float("nan"),
        min_freq_model_THz=float(pred.min()),
        min_freq_dft_THz=float(ref.min()),
    )


def plot(material: str, distances, dft, model, out: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(distances, dft, color="k", lw=1.2, label="DFT")
    ax.plot(distances, model, color="tab:red", lw=1.0, ls="--", label=label)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:1] + h[dft.shape[1]:dft.shape[1] + 1], l[:1] + l[dft.shape[1]:dft.shape[1] + 1])
    ax.axhline(0, color="gray", lw=0.5)
    n = len(distances) // 3  # three equal segments: Γ-K, K-M, M-Γ
    ticks = [distances[0], distances[n - 1], distances[2 * n - 1], distances[-1]]
    for t in ticks[1:-1]:
        ax.axvline(t, color="gray", lw=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Γ", "K", "M", "Γ"])
    ax.set_xlim(distances[0], distances[-1])
    ax.set_title(f"{material}  (path Γ–K–M–Γ)")
    ax.set_ylabel("Frequency (THz)")
    fig.tight_layout()
    fig.savefig(out / f"{material}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calc", required=True, help="'module:callable' returning an ASE calculator")
    p.add_argument("--calc-kwargs", default="{}", help="JSON dict of keyword args for the callable")
    p.add_argument("--label", default=None, help="Model label used in outputs (default: --calc)")
    p.add_argument("--materials", nargs="*", default=None, help="Subset of material names (default: all 48)")
    p.add_argument("--relax", choices=["none", "positions"], default="none",
                   help="none = DFT geometry as-is (default); positions = relax atoms with the model, cell fixed")
    p.add_argument("--fmax", type=float, default=1e-3, help="Relaxation force threshold, eV/Å")
    p.add_argument("--relax-steps", type=int, default=500)
    p.add_argument("--displacement", type=float, default=0.01, help="Finite-displacement amplitude, Å (DFT used 0.01)")
    p.add_argument("--output-dir", default=None, help="Default: results/<label>[_relax-positions]/")
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    label = args.label or args.calc.split(":")[-1]
    out = Path(args.output_dir) if args.output_dir else HERE / "results" / (label + ("_relax-positions" if args.relax == "positions" else ""))
    out.mkdir(parents=True, exist_ok=True)
    plot_dir = out / "plots" if args.plots else None
    if plot_dir:
        plot_dir.mkdir(exist_ok=True)

    calc = load_calculator(args.calc, json.loads(args.calc_kwargs))
    materials = args.materials or sorted(d.name for d in MATERIALS_DIR.iterdir() if d.is_dir())

    rows = []
    for material in materials:
        ph_ref, q_list, distances, dft_freqs = load_dft(material)
        unitcell = phonopy_to_ase(ph_ref.unitcell)
        if args.relax == "positions":
            unitcell = relax_positions(unitcell, calc, args.fmax, args.relax_steps)
        ph_model = model_force_constants(ph_ref, unitcell, calc, args.displacement)
        ph_model.run_qpoints(q_list)
        model_freqs = ph_model.get_qpoints_dict()["frequencies"]
        row = dict(material=material, model=label, geometry=("dft-fixed" if args.relax == "none" else "model-relaxed-positions"),
                   **metrics(model_freqs, dft_freqs))
        rows.append(row)
        print(f"{material:<22} RMSE {row['rmse_THz']:.4f} THz  MAE {row['mae_THz']:.4f}  "
              f"min(model) {row['min_freq_model_THz']:+.3f}  min(DFT) {row['min_freq_dft_THz']:+.3f}", flush=True)
        if plot_dir:
            plot(material, distances, dft_freqs, model_freqs, plot_dir, label)

    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rm = np.array([r["rmse_THz"] for r in rows])
    n_unstable = sum(r["min_freq_model_THz"] < -0.1 for r in rows)
    print(f"\n{label} ({rows[0]['geometry']}): {len(rows)} materials  mean RMSE {rm.mean():.4f} THz  "
          f"median {np.median(rm):.4f}  worst {rm.max():.4f} ({rows[int(rm.argmax())]['material']})  "
          f"materials with modes below -0.1 THz: {n_unstable}")
    print(f"wrote {out / 'results.csv'}")


if __name__ == "__main__":
    main()
