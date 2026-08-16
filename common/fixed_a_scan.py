#!/usr/bin/env python3
"""
Build a fixed-in-plane-lattice-constant energy scan from a converged bilayer
CONTCAR.

Used when ISIF=4 diverges instead of converging (see
project_bilayer_batch7_lattice_mismatch memory: the fixed-volume constraint
on a vacuum-padded slab cell can pay for a large in-plane correction by
collapsing the vacuum gap instead of finding a real equilibrium, when the
needed correction is large). This sidesteps the issue entirely: rescale only
the in-plane lattice vectors (fractional/Direct coordinates are unaffected by
a uniform in-plane rescale for a hexagonal cell), leave `c` untouched, and
relax ions only (ISIF=2) at each fixed trial `a`. Fit E(a) afterward to find
the true minimum.
"""

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phonopy"))
from poscar_utils import read_poscar, write_poscar  # noqa: E402


def build_scan_poscars(contcar_path, trial_a_values, output_dir_template):
    """
    Write one rescaled POSCAR per trial `a`, at `output_dir_template.format(i=i)`.

    Parameters
    ----------
    contcar_path : str or Path
        Converged CONTCAR to rescale (must be Direct/fractional coordinates).
    trial_a_values : list of float
        In-plane lattice constants (Angstroms) to scan.
    output_dir_template : str
        Format string with `{i}` for the trial index, e.g.
        "bilayer_examples/MoTe2_InSe_TM_H_ascan_{i}".

    Returns
    -------
    list of (a_value, output_dir) tuples, in the same order as trial_a_values.
    """
    data = read_poscar(contcar_path)
    if data.coord_type != "Direct":
        raise ValueError(
            f"{contcar_path}: coord_type={data.coord_type!r}, expected Direct "
            "(fractional coordinates are required for a pure in-plane rescale)"
        )

    a0 = (data.lattice[0][0] ** 2 + data.lattice[0][1] ** 2 + data.lattice[0][2] ** 2) ** 0.5

    results = []
    for i, a_trial in enumerate(trial_a_values):
        factor = a_trial / a0
        new_lattice = [
            [c * factor for c in data.lattice[0]],
            [c * factor for c in data.lattice[1]],
            list(data.lattice[2]),
        ]
        out_dir = Path(output_dir_template.format(i=i))
        out_dir.mkdir(parents=True, exist_ok=True)

        scan_data = type(data)(
            comment=f"{data.comment} (fixed-a scan, a={a_trial:.6f})",
            scale=data.scale,
            lattice=new_lattice,
            symbols=data.symbols,
            positions=data.positions,
            coord_type=data.coord_type,
            selective_dynamics=data.selective_dynamics,
        )
        write_poscar(scan_data, out_dir / "POSCAR")
        results.append((a_trial, out_dir))

    return results


def populate_scan_dir(source_example_dir, out_dir):
    """Copy POTCAR/KPOINTS/bat/INCAR (ISIF=2, unchanged) from an existing
    bilayer example directory into a scan directory (POSCAR already written).
    """
    source_example_dir = Path(source_example_dir)
    out_dir = Path(out_dir)
    for fname in ("POTCAR", "KPOINTS", "bat", "INCAR"):
        src = source_example_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a fixed-a energy scan from a converged bilayer CONTCAR",
        epilog="""
Example:
  python3 fixed_a_scan.py bilayer_examples/MoTe2_InSe_TM_H/CONTCAR \\
      --source-dir bilayer_examples/MoTe2_InSe_TM_H \\
      --a-values 3.85 3.90 3.95 4.00 4.05 \\
      --out-template bilayer_examples/MoTe2_InSe_TM_H_ascan_{i}
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("contcar", help="Converged CONTCAR to rescale")
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Existing bilayer example dir to copy POTCAR/KPOINTS/bat/INCAR from",
    )
    parser.add_argument(
        "--a-values", type=float, nargs="+", required=True, help="Trial a values (Angstrom)"
    )
    parser.add_argument(
        "--out-template",
        required=True,
        help="Output dir template with {i}, e.g. bilayer_examples/NAME_ascan_{i}",
    )
    args = parser.parse_args()

    results = build_scan_poscars(args.contcar, args.a_values, args.out_template)
    for a_trial, out_dir in results:
        populate_scan_dir(args.source_dir, out_dir)
        print(f"a={a_trial:.6f} -> {out_dir}")


if __name__ == "__main__":
    main()
