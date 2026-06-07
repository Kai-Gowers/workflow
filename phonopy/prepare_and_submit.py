#!/usr/bin/env python3
"""
Master script to prepare static-point phonopy calculations, set up
displacements, and submit VASP jobs for both monolayers and bilayers.

Pipeline per example:
1. Take relaxed example (monolayer or bilayer)
2. Prepare static-point directory (CONTCAR → POSCAR, copy POTCAR/KPOINTS,
   create INCAR and bat from static templates)
3. Run phonopy to generate displacements (POSCAR-001, POSCAR-002, ...)
4. Create disp-XXX folders and copy POSCAR/INCAR/KPOINTS/POTCAR/bat
5. Optionally submit each displacement with `sbatch bat`

This script can operate:
- On a single example
- On all examples of a given type
- On a specific relaxation batch (using data/batches.json)
"""

from pathlib import Path
import argparse
import sys
import importlib.util
import json


#
# Import helpers (avoid module name collisions)
#

ROOT = Path(__file__).parent


def _get_workflow_root() -> Path:
    """Return the workflow root directory (parent of phonopy/)."""
    return ROOT.parent


def _load_module(name: str, path: Path):
    """Load a module from a specific file path with a unique name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------
# Path resolution helpers
# --------

def _resolve_relaxed_example(example_path, kind: str) -> Path:
    """
    Resolve a relaxed example directory.

    If `example_path` exists as given (absolute or relative), use it.
    Otherwise, treat it as an example *name* and resolve to:
      - workflow/monolayer_examples/<name>  (kind='monolayer')
      - workflow/bilayer_examples/<name>    (kind='bilayer')
    """
    p = Path(example_path)

    # If user gave an absolute path, or a relative path that exists, use it
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()

    workflow_root = _get_workflow_root()  # .../workflow
    if kind == "monolayer":
        return (workflow_root / "monolayer_examples" / p.name).resolve()
    if kind == "bilayer":
        return (workflow_root / "bilayer_examples" / p.name).resolve()
    raise ValueError(f"Unknown kind: {kind}")


# Monolayer modules
mono_prep = _load_module(
    "mono_prepare_staticpoint",
    ROOT / "monolayer" / "prepare_staticpoint.py",
)
mono_setup = _load_module(
    "mono_setup_displacements",
    ROOT / "monolayer" / "setup_displacements.py",
)

# Bilayer modules
bi_prep = _load_module(
    "bi_prepare_staticpoint",
    ROOT / "bilayer" / "prepare_staticpoint.py",
)
bi_setup = _load_module(
    "bi_setup_displacements",
    ROOT / "bilayer" / "setup_displacements.py",
)

# Extract functions
prepare_monolayer_staticpoint = mono_prep.prepare_staticpoint
setup_monolayer_displacements = mono_setup.setup_and_submit_displacements
prepare_bilayer_staticpoint = bi_prep.prepare_staticpoint
setup_bilayer_displacements = bi_setup.setup_and_submit_displacements


#
# Per-example pipelines
#

def process_monolayer(example_path, supercell_dim="3 3 1",
                      submit=True, dry_run=False):
    """
    Process a single monolayer example: prepare staticpoint and set up displacements.

    Parameters
    ----------
    example_path : str or Path
        Path or name of relaxed monolayer example (e.g. 'MoS2' or
        'monolayer_examples/MoS2')
    supercell_dim : str
        Supercell dimensions for phonopy (default: "3 3 1")
    submit : bool
        Whether to submit jobs for displacements
    dry_run : bool
        If True, only print actions without executing
    """
    print(f"\n{'=' * 60}")
    print(f"Processing monolayer: {example_path}")
    print(f"{'=' * 60}\n")

    # Step 1: prepare staticpoint (let module use its own default base_dir)
    print("Step 1: Preparing static-point calculation...")
    try:
        resolved_example = _resolve_relaxed_example(example_path, kind="monolayer")
        prep_result = prepare_monolayer_staticpoint(
            resolved_example,
            base_dir=None,              # use default phonopy_monolayer_examples
            supercell_dim=supercell_dim,
            generate_displacements=True,
        )
        staticpoint_dir = prep_result["output_dir"]
        print(f"  ✓ Created static-point directory: {staticpoint_dir}\n")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ Error preparing staticpoint: {e}\n", file=sys.stderr)
        return {"success": False, "error": str(e)}

    # Step 2: set up displacement folders and optionally submit
    print("Step 2: Setting up displacement calculations...")
    try:
        disp_result = setup_monolayer_displacements(
            staticpoint_dir,
            submit=submit,
            dry_run=dry_run,
        )
        print(f"  ✓ Set up {disp_result['displacements_setup']} displacement(s)")
        if disp_result["jobs_submitted"] > 0:
            print(f"  ✓ Submitted {disp_result['jobs_submitted']} job(s)\n")

        return {
            "success": True,
            "type": "monolayer",
            "staticpoint_dir": staticpoint_dir,
            "displacements_setup": disp_result["displacements_setup"],
            "jobs_submitted": disp_result["jobs_submitted"],
            "job_ids": disp_result["job_ids"],
        }
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ Error setting up displacements: {e}\n", file=sys.stderr)
        return {"success": False, "error": str(e)}


def process_bilayer(example_path, supercell_dim="3 3 1",
                    submit=True, dry_run=False):
    """
    Process a single bilayer example: prepare staticpoint and set up displacements.

    Parameters
    ----------
    example_path : str or Path
        Path or name of relaxed bilayer example (e.g. 'MoS2_TaTe2_3R' or
        'bilayer_examples/MoS2_TaTe2_3R')
    supercell_dim : str
        Supercell dimensions for phonopy (default: "3 3 1")
    submit : bool
        Whether to submit jobs for displacements
    dry_run : bool
        If True, only print actions without executing
    """
    print(f"\n{'=' * 60}")
    print(f"Processing bilayer: {example_path}")
    print(f"{'=' * 60}\n")

    # Step 1: prepare staticpoint (let module use its own default base_dir)
    print("Step 1: Preparing static-point calculation...")
    try:
        resolved_example = _resolve_relaxed_example(example_path, kind="bilayer")
        prep_result = prepare_bilayer_staticpoint(
            resolved_example,
            base_dir=None,              # use default phonopy_bilayer_examples
            supercell_dim=supercell_dim,
            generate_displacements=True,
        )
        staticpoint_dir = prep_result["output_dir"]
        print(f"  ✓ Created static-point directory: {staticpoint_dir}\n")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ Error preparing staticpoint: {e}\n", file=sys.stderr)
        return {"success": False, "error": str(e)}

    # Step 2: set up displacement folders and optionally submit
    print("Step 2: Setting up displacement calculations...")
    try:
        disp_result = setup_bilayer_displacements(
            staticpoint_dir,
            submit=submit,
            dry_run=dry_run,
        )
        print(f"  ✓ Set up {disp_result['displacements_setup']} displacement(s)")
        if disp_result["jobs_submitted"] > 0:
            print(f"  ✓ Submitted {disp_result['jobs_submitted']} job(s)\n")

        return {
            "success": True,
            "type": "bilayer",
            "staticpoint_dir": staticpoint_dir,
            "displacements_setup": disp_result["displacements_setup"],
            "jobs_submitted": disp_result["jobs_submitted"],
            "job_ids": disp_result["job_ids"],
        }
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ Error setting up displacements: {e}\n", file=sys.stderr)
        return {"success": False, "error": str(e)}


#
# Batch helpers
#

def process_all_monolayers(supercell_dim="3 3 1",
                           submit=True, dry_run=False):
    """Process all monolayer examples in ../monolayer_examples."""
    base_examples_dir = ROOT.parent / "monolayer_examples"
    if not base_examples_dir.exists():
        print(
            f"Error: Monolayer examples directory not found: {base_examples_dir}",
            file=sys.stderr,
        )
        return []

    examples = [d for d in base_examples_dir.iterdir() if d.is_dir()]
    if not examples:
        print(f"No monolayer examples found in {base_examples_dir}")
        return []

    print(f"Found {len(examples)} monolayer example(s)\n")

    results = []
    for example_dir in sorted(examples):
        result = process_monolayer(
            example_dir,
            supercell_dim=supercell_dim,
            submit=submit,
            dry_run=dry_run,
        )
        results.append(result)

    return results


def process_all_bilayers(supercell_dim="3 3 1",
                         submit=True, dry_run=False):
    """Process all bilayer examples in ../bilayer_examples."""
    base_examples_dir = ROOT.parent / "bilayer_examples"
    if not base_examples_dir.exists():
        print(
            f"Error: Bilayer examples directory not found: {base_examples_dir}",
            file=sys.stderr,
        )
        return []

    examples = [d for d in base_examples_dir.iterdir() if d.is_dir()]
    if not examples:
        print(f"No bilayer examples found in {base_examples_dir}")
        return []

    print(f"Found {len(examples)} bilayer example(s)\n")

    results = []
    for example_dir in sorted(examples):
        result = process_bilayer(
            example_dir,
            supercell_dim=supercell_dim,
            submit=submit,
            dry_run=dry_run,
        )
        results.append(result)

    return results


#
# CLI
#

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare staticpoint directories, set up displacements, and "
            "submit jobs for monolayer/bilayer phonopy calculations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single monolayer
  python3 prepare_and_submit.py --monolayer MoS2

  # Single bilayer
  python3 prepare_and_submit.py --bilayer MoS2_TaTe2_3R

  # All monolayers
  python3 prepare_and_submit.py --monolayer --all

  # All bilayers
  python3 prepare_and_submit.py --bilayer --all

  # Both monolayers and bilayers (all)
  python3 prepare_and_submit.py --monolayer --bilayer --all

  # Set up only (no submission)
  python3 prepare_and_submit.py --monolayer MoS2 --no-submit

  # Custom supercell dimensions
  python3 prepare_and_submit.py --monolayer MoS2 --dim "4 4 1"

  # Dry run
  python3 prepare_and_submit.py --monolayer MoS2 --dry-run
""",
    )

    parser.add_argument(
        "--monolayer",
        action="store_true",
        help="Process monolayer example(s)",
    )
    parser.add_argument(
        "--bilayer",
        action="store_true",
        help="Process bilayer example(s)",
    )
    parser.add_argument(
        "example",
        nargs="?",
        help=(
            "Example name or path "
            "(e.g. 'MoS2', 'monolayer_examples/MoS2', "
            "'MoS2_TaTe2_3R', 'bilayer_examples/MoS2_TaTe2_3R')"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all examples of selected type(s)",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default="3 3 1",
        help='Supercell dimensions for phonopy (default: "3 3 1")',
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Set up displacement folders but do not submit jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )

    args = parser.parse_args()

    if not args.monolayer and not args.bilayer:
        parser.error("Must specify --monolayer and/or --bilayer")

    if args.all and args.example:
        parser.error("Cannot use --all with a specific example name/path")

    if not args.all and not args.example:
        parser.error("Must provide an example name or use --all")

    submit = not args.no_submit

    # Monolayers
    monolayer_results = []
    if args.monolayer:
        if args.all:
            monolayer_results = process_all_monolayers(
                supercell_dim=args.dim,
                submit=submit,
                dry_run=args.dry_run,
            )
        else:
            monolayer_results = [
                process_monolayer(
                    args.example,
                    supercell_dim=args.dim,
                    submit=submit,
                    dry_run=args.dry_run,
                ),
            ]

    # Bilayers
    bilayer_results = []
    if args.bilayer:
        if args.all:
            bilayer_results = process_all_bilayers(
                supercell_dim=args.dim,
                submit=submit,
                dry_run=args.dry_run,
            )
        else:
            bilayer_results = [
                process_bilayer(
                    args.example,
                    supercell_dim=args.dim,
                    submit=submit,
                    dry_run=args.dry_run,
                ),
            ]

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if monolayer_results:
        ok = sum(1 for r in monolayer_results if r.get("success"))
        total_disps = sum(r.get("displacements_setup", 0) for r in monolayer_results)
        total_jobs = sum(r.get("jobs_submitted", 0) for r in monolayer_results)
        print("\nMonolayers:")
        print(f"  Processed: {len(monolayer_results)}")
        print(f"  Successful: {ok}")
        print(f"  Displacements set up: {total_disps}")
        print(f"  Jobs submitted: {total_jobs}")

    if bilayer_results:
        ok = sum(1 for r in bilayer_results if r.get("success"))
        total_disps = sum(r.get("displacements_setup", 0) for r in bilayer_results)
        total_jobs = sum(r.get("jobs_submitted", 0) for r in bilayer_results)
        print("\nBilayers:")
        print(f"  Processed: {len(bilayer_results)}")
        print(f"  Successful: {ok}")
        print(f"  Displacements set up: {total_disps}")
        print(f"  Jobs submitted: {total_jobs}")

    all_results = monolayer_results + bilayer_results
    failed = [r for r in all_results if not r.get("success")]
    if failed:
        print(f"\n✗ {len(failed)} example(s) failed")
        sys.exit(1)

    print("\n✓ All examples processed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()


