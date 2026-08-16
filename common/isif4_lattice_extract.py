#!/usr/bin/env python3
"""
Extract the plateaued in-plane lattice constant from an ISIF=4 OUTCAR.

An ISIF=4 relaxation (fixed volume, ion + cell-shape relax) prints a fresh
"direct lattice vectors" block each ionic step. The in-plane hexagonal
lattice constant `a` (norm of the first lattice vector) should converge to a
stable plateau over the last several steps -- this is the workflow's
established way to find the true equilibrium `a` for a bilayer, since the
initial anchor/average-based POSCAR is only a guess (see
data/bilayer_lattice_overrides.json for prior corrections built this way).

No existing script did this before this project decided to reuse it broadly
(previously done by hand, reading OUTCAR "direct lattice vectors" blocks and
averaging in a scratch shell/python session per material).
"""

from pathlib import Path
import re
import statistics
import sys

_LATTICE_BLOCK_RE = re.compile(
    r"direct lattice vectors\s+reciprocal lattice vectors\n"
    r"\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
)


def extract_in_plane_a_per_step(outcar_path):
    """Return a list of in-plane lattice constant `a` (Angstroms), one per
    ionic step that printed a "direct lattice vectors" block, in step order.

    `a` is the norm of the first lattice vector, matching
    relaxed_monolayer.hexagonal_lattice_a's convention.
    """
    text = Path(outcar_path).read_text()
    a_values = []
    for match in _LATTICE_BLOCK_RE.finditer(text):
        x, y, z = (float(v) for v in match.groups())
        a_values.append((x**2 + y**2 + z**2) ** 0.5)
    return a_values


def plateau_average(a_values, window=50):
    """Average (and std) of the last `window` values, or all of them if
    fewer than `window` steps were recorded.

    Returns (mean, std, n_used, n_total).
    """
    if not a_values:
        raise ValueError("No 'direct lattice vectors' blocks found in OUTCAR")
    tail = a_values[-window:]
    mean = statistics.fmean(tail)
    std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
    return mean, std, len(tail), len(a_values)


def report(outcar_path, window=50):
    a_values = extract_in_plane_a_per_step(outcar_path)
    mean, std, n_used, n_total = plateau_average(a_values, window=window)
    return {
        "outcar": str(outcar_path),
        "n_ionic_steps": n_total,
        "n_averaged": n_used,
        "a_plateau": mean,
        "a_std": std,
        "a_first": a_values[0],
        "a_last": a_values[-1],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract plateaued in-plane lattice constant from an ISIF=4 OUTCAR",
        epilog="""
Examples:
  python3 isif4_lattice_extract.py bilayer_examples/MoS2_MoTe2_3R_isif4/OUTCAR
  python3 isif4_lattice_extract.py bilayer_examples/MoS2_MoTe2_3R_isif4 --window 30
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path", help="Path to OUTCAR, or a directory containing OUTCAR"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Number of trailing ionic steps to average over (default: 50)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    outcar_path = path / "OUTCAR" if path.is_dir() else path

    if not outcar_path.exists():
        print(f"Error: {outcar_path} not found", file=sys.stderr)
        sys.exit(1)

    result = report(outcar_path, window=args.window)
    print(f"OUTCAR: {result['outcar']}")
    print(f"Ionic steps with a lattice-vector block: {result['n_ionic_steps']}")
    print(f"a (first step):  {result['a_first']:.6f} A")
    print(f"a (last step):   {result['a_last']:.6f} A")
    print(
        f"Plateau average over last {result['n_averaged']} steps: "
        f"{result['a_plateau']:.6f} A (std = {result['a_std']:.6f} A)"
    )


if __name__ == "__main__":
    main()
