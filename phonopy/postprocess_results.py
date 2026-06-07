#!/usr/bin/env python3
"""
Post-process phonopy static-point calculations:

For each staticpoint directory (monolayer or bilayer):
1. Collect disp-*/vasprun.xml files
2. Run: phonopy --vasp -f disp-001/vasprun.xml disp-002/vasprun.xml ...
   to build FORCE_SETS
3. Create band.conf with material-specific ATOM_NAME, fixed DIM and BAND path
4. Run: phonopy -p band.conf --save
   to generate band.pdf, band.yaml, FORCE_CONSTANTS
5. Copy band.pdf, band.yaml, and FORCE_SETS into FINAL_RESULTS/<material_or_bilayer_name>/
"""

from pathlib import Path
import argparse
import sys
import subprocess
from typing import List, Tuple


ROOT = Path(__file__).parent


def find_staticpoint_dirs(kind: str) -> List[Path]:
    """Return list of staticpoint directories for given kind."""
    if kind == "monolayer":
        base = ROOT.parent / "phonopy_monolayer_examples"
    elif kind == "bilayer":
        base = ROOT.parent / "phonopy_bilayer_examples"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    if not base.exists():
        return []

    return sorted(
        d for d in base.iterdir()
        if d.is_dir() and d.name.endswith("_staticpoint")
    )


def collect_vaspruns(staticpoint_dir: Path) -> List[Path]:
    """Return list of disp-*/vasprun.xml paths that exist."""
    vasps: List[Path] = []
    for sub in sorted(staticpoint_dir.iterdir()):
        if sub.is_dir() and sub.name.startswith("disp-"):
            vr = sub / "vasprun.xml"
            if vr.exists():
                vasps.append(vr)
    return vasps


def parse_atom_names_from_poscar(poscar_path: Path) -> List[str]:
    """
    Heuristically parse ATOM_NAME from POSCAR/CONTCAR.

    Assumes VASP 5+ style where the element symbols appear on one line,
    followed by a line of counts.
    """
    with poscar_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Skip first 8 lines (comment + scale + 3 lattice + 2 more) conservatively,
    # then look for:
    #   line of element symbols (no digits),
    #   next line with counts (digits).
    for i in range(5, min(len(lines) - 1, 20)):
        symbols = lines[i].split()
        counts = lines[i + 1].split()
        if (symbols
                and all(token.isalpha() for token in symbols)
                and counts
                and all(any(ch.isdigit() for ch in token) for token in counts)):
            return symbols

    # Fallback: if we can't detect, just return empty and let caller handle it
    return []


def write_band_conf(staticpoint_dir: Path,
                    atom_names: List[str],
                    dim: str = "3 3 1") -> Path:
    """Write band.conf into staticpoint_dir and return its path."""
    if not atom_names:
        atom_line = "# ATOM_NAME could not be detected automatically\n"
    else:
        atom_line = f"ATOM_NAME = {' '.join(atom_names)}\n"

    contents = (
        f"{atom_line}"
        f"DIM = {dim}\n"
        "BAND = 0 0 0  0.6667 0.3333 0  0.5 0 0  0 0 0\n"
        "BAND_LABELS = Γ K M Γ\n"
    )

    band_conf = staticpoint_dir / "band.conf"
    band_conf.write_text(contents)
    return band_conf


def run_phonopy_force_sets(staticpoint_dir: Path,
                           vaspruns: List[Path]) -> bool:
    """Run phonopy --vasp -f ... to build FORCE_SETS."""
    if not vaspruns:
        print(f"  ✗ No vasprun.xml files found in disp-* directories")
        return False

    cmd = ["phonopy", "--vasp", "-f"] + [str(v) for v in vaspruns]
    print(f"  Running: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            cwd=str(staticpoint_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        print("  ✓ Generated FORCE_SETS")
        return True
    except FileNotFoundError:
        print("  ✗ phonopy command not found (is it in your PATH?)",
              file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:  # noqa: BLE001
        print("  ✗ Error running phonopy --vasp -f",
              file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def run_phonopy_band(staticpoint_dir: Path,
                     band_conf: Path) -> bool:
    """Run phonopy -p band.conf --save in staticpoint_dir."""
    cmd = ["phonopy", "-p", str(band_conf.name), "--save"]
    print(f"  Running: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            cwd=str(staticpoint_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        print("  ✓ Generated band.pdf, band.yaml, FORCE_CONSTANTS (using FORCE_SETS)")
        return True
    except FileNotFoundError:
        print("  ✗ phonopy command not found (is it in your PATH?)",
              file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:  # noqa: BLE001
        print("  ✗ Error running phonopy -p band.conf --save",
              file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def copy_final_results(staticpoint_dir: Path,
                       kind: str) -> Tuple[bool, Path]:
    """
    Copy band.pdf, band.yaml, FORCE_SETS into FINAL_RESULTS/<name>/.

    Returns (success, target_dir).
    """
    workflow_root = ROOT.parent
    final_root = workflow_root / "FINAL_RESULTS"
    final_root.mkdir(exist_ok=True)

    # Name = staticpoint_dir name without trailing "_staticpoint"
    name = staticpoint_dir.name
    if name.endswith("_staticpoint"):
        name = name[:-len("_staticpoint")]

    target = final_root / name
    target.mkdir(exist_ok=True)

    files = ["band.pdf", "band.yaml", "FORCE_SETS"]
    missing = []
    for fname in files:
        src = staticpoint_dir / fname
        if src.exists():
            dst = target / fname
            dst.write_bytes(src.read_bytes())
        else:
            missing.append(fname)

    if missing:
        print(f"  ✗ Missing files in {staticpoint_dir}: {', '.join(missing)}")
        return False, target

    print(f"  ✓ Copied band.pdf, band.yaml, FORCE_SETS → {target}")
    return True, target


def process_staticpoint_dir(staticpoint_dir: Path,
                            kind: str,
                            dim: str = "3 3 1") -> dict:
    """Process a single staticpoint directory end-to-end."""
    print(f"\n{'=' * 60}")
    print(f"Processing {kind} staticpoint: {staticpoint_dir.name}")
    print(f"{'=' * 60}")

    # 1. Collect vasprun.xml files
    vaspruns = collect_vaspruns(staticpoint_dir)
    print(f"  Found {len(vaspruns)} vasprun.xml file(s)")

    # 2. Build FORCE_SETS
    ok_force_sets = run_phonopy_force_sets(staticpoint_dir, vaspruns)
    if not ok_force_sets:
        return {"success": False, "staticpoint_dir": staticpoint_dir}

    # 3. band.conf
    poscar = staticpoint_dir / "POSCAR"
    atom_names = parse_atom_names_from_poscar(poscar) if poscar.exists() else []
    band_conf = write_band_conf(staticpoint_dir, atom_names, dim=dim)
    print(f"  ✓ Wrote band.conf (ATOM_NAME = {' '.join(atom_names) if atom_names else 'auto-detect failed'})")

    # 4. Run phonopy -p band.conf --save
    ok_band = run_phonopy_band(staticpoint_dir, band_conf)
    if not ok_band:
        return {"success": False, "staticpoint_dir": staticpoint_dir}

    # 5. Copy results
    ok_copy, target = copy_final_results(staticpoint_dir, kind)

    return {
        "success": ok_copy,
        "staticpoint_dir": staticpoint_dir,
        "final_dir": target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process phonopy staticpoint calculations into FINAL_RESULTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single monolayer staticpoint directory
  python3 postprocess_results.py --monolayer HfSe2_staticpoint

  # Process a single bilayer staticpoint directory
  python3 postprocess_results.py --bilayer MoS2_TaTe2_3R_staticpoint

  # Process all monolayer staticpoints
  python3 postprocess_results.py --monolayer --all

  # Process all bilayer staticpoints
  python3 postprocess_results.py --bilayer --all
""",
    )

    parser.add_argument(
        "--monolayer",
        action="store_true",
        help="Process monolayer staticpoint directory(ies)",
    )
    parser.add_argument(
        "--bilayer",
        action="store_true",
        help="Process bilayer staticpoint directory(ies)",
    )
    parser.add_argument(
        "staticpoint",
        nargs="?",
        help=(
            "Staticpoint directory name (e.g. 'HfSe2_staticpoint', "
            "'MoS2_TaTe2_3R_staticpoint')"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all staticpoint directories of the selected kind(s)",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default="3 3 1",
        help='DIM value for band.conf (default: "3 3 1")',
    )

    args = parser.parse_args()

    if not args.monolayer and not args.bilayer:
        parser.error("Must specify --monolayer and/or --bilayer")
    if args.all and args.staticpoint:
        parser.error("Cannot use --all with a specific staticpoint")
    if not args.all and not args.staticpoint:
        parser.error("Must provide a staticpoint name or use --all")

    results = []

    if args.monolayer:
        if args.all:
            dirs = find_staticpoint_dirs("monolayer")
            if not dirs:
                print("No monolayer staticpoint directories found.")
            for d in dirs:
                results.append(process_staticpoint_dir(d, "monolayer", dim=args.dim))
        else:
            base = ROOT.parent / "phonopy_monolayer_examples"
            d = (base / args.staticpoint).resolve()
            if not d.exists():
                print(f"Error: staticpoint directory not found: {d}", file=sys.stderr)
            else:
                results.append(process_staticpoint_dir(d, "monolayer", dim=args.dim))

    if args.bilayer:
        if args.all:
            dirs = find_staticpoint_dirs("bilayer")
            if not dirs:
                print("No bilayer staticpoint directories found.")
            for d in dirs:
                results.append(process_staticpoint_dir(d, "bilayer", dim=args.dim))
        else:
            base = ROOT.parent / "phonopy_bilayer_examples"
            d = (base / args.staticpoint).resolve()
            if not d.exists():
                print(f"Error: staticpoint directory not found: {d}", file=sys.stderr)
            else:
                results.append(process_staticpoint_dir(d, "bilayer", dim=args.dim))

    # Summary
    ok = sum(1 for r in results if r.get("success"))
    print(f"\nProcessed {len(results)} staticpoint directory(ies), {ok} successful.")
    sys.exit(0 if ok == len(results) and results else 1)


if __name__ == "__main__":
    main()


