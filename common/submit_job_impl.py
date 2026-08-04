#!/usr/bin/env python3
"""
Shared implementation for submitting a single relaxation job.

Used identically by relaxation/monolayer/submit_monolayer_job.py and
relaxation/bilayer/submit_bilayer_job.py -- the only difference between
monolayer and bilayer is which top-level "*_examples" directory holds the
example, passed in as `examples_root_name`.
"""

from pathlib import Path
import argparse
import sys

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts" / "maintenance"))
from job_tracking import submit_bat  # noqa: E402


def submit_example_job(example_path, examples_root_name, dry_run=False):
    """
    Submit a VASP job for a relaxation example.

    Parameters:
    -----------
    example_path : str or Path
        Path to the example directory (e.g., "monolayer_examples/MoS2") or just
        the example name (e.g., "MoS2"), resolved against `examples_root_name`.
    examples_root_name : str
        Top-level examples directory to resolve a bare name against
        ("monolayer_examples" or "bilayer_examples").
    dry_run : bool, optional
        If True, print what would be done without actually submitting (default: False)

    Returns:
    --------
    tuple : (success: bool, job_id: str or None, message: str)
    """
    example_path = Path(example_path)

    # If it's already an absolute path or exists as-is, use it
    if example_path.is_absolute() and example_path.exists():
        example_dir = example_path
    elif example_path.exists():
        # Relative path that exists
        example_dir = example_path.resolve()
    else:
        # Assume it's a bare name in the given examples root
        base_dir = WORKFLOW_ROOT / examples_root_name
        example_dir = base_dir / example_path.name

    # Resolve to absolute path
    example_dir = example_dir.resolve()

    # Check if directory exists
    if not example_dir.exists():
        return False, None, f"Example directory not found: {example_dir}"

    if not example_dir.is_dir():
        return False, None, f"Path is not a directory: {example_dir}"

    # Check if bat file exists
    bat_file = example_dir / "bat"
    if not bat_file.exists():
        return False, None, f"Batch script not found: {bat_file}"

    # Check for required VASP files
    required_files = ["POSCAR", "POTCAR", "INCAR", "KPOINTS"]
    missing_files = [f for f in required_files if not (example_dir / f).exists()]
    if missing_files:
        return False, None, f"Missing required files: {', '.join(missing_files)}"

    if dry_run:
        print(f"Would submit job from: {example_dir}")
        print(f"  Batch script: {bat_file}")
        print(f"  Command: sbatch bat")
        return True, None, "Dry run - no job submitted"

    return submit_bat(
        example_dir,
        job_name=example_dir.name,
        job_type="relaxation",
        label=example_dir.name,
        dry_run=dry_run,
    )


def main(submit_fn, script_name, description, example_id, examples_root_name):
    """CLI entry point shared by the monolayer/bilayer wrapper scripts.

    Parameters:
    -----------
    submit_fn : callable
        The wrapper's own submit_job-style function, called as
        submit_fn(example_arg, dry_run=...).
    script_name : str
        e.g. "submit_monolayer_job.py", used only in --help text.
    description : str
        argparse description.
    example_id : str
        A representative example name for --help text (e.g. "MoS2").
    examples_root_name : str
        "monolayer_examples" or "bilayer_examples", used only in --help text.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Submit job for {example_id}
  python3 {script_name} {example_id}

  # Submit job using full path
  python3 {script_name} {examples_root_name}/{example_id}

  # Dry run (see what would be done)
  python3 {script_name} {example_id} --dry-run

  # Submit multiple jobs
  python3 {script_name} {example_id} another_example
        """
    )
    parser.add_argument(
        "examples",
        nargs="+",
        help=f"Example ID(s) or path(s) to example directory (e.g., '{example_id}' or '{examples_root_name}/{example_id}')"
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Print what would be done without actually submitting"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    # Submit jobs for each example
    all_success = True
    for example_arg in args.examples:
        success, job_id, message = submit_fn(example_arg, dry_run=args.dry_run)

        if success:
            if job_id:
                print(f"✓ Submitted job {job_id} for {example_arg}")
                if args.verbose:
                    print(f"  Directory: {Path(example_arg).resolve()}")
            else:
                print(f"✓ {example_arg}: {message}")
        else:
            print(f"✗ {example_arg}: {message}", file=sys.stderr)
            all_success = False

    sys.exit(0 if all_success else 1)
