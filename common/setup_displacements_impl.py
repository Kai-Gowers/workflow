#!/usr/bin/env python3
"""
Shared implementation for setting up + submitting phonopy displacement jobs.

Used identically by phonopy/monolayer/setup_displacements.py and
phonopy/bilayer/setup_displacements.py -- the only difference between
monolayer and bilayer is which top-level "phonopy_*_examples" directory
holds the staticpoint folders, passed in as `examples_root_name`.

This script:
1. Finds all POSCAR-XXX files in a staticpoint directory
2. Creates disp-XXX subdirectories
3. Copies POSCAR-XXX -> disp-XXX/POSCAR
4. Copies INCAR, KPOINTS, POTCAR from the staticpoint dir; always copies bat from common/staticpoint_templates
5. Optionally submits jobs using sbatch bat
"""

from pathlib import Path
import argparse
import re
import shutil
import sys

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts" / "maintenance"))
from job_tracking import submit_bat  # noqa: E402


def find_displacement_poscars(staticpoint_dir):
    """
    Find all POSCAR-XXX files in the staticpoint directory.

    Parameters:
    -----------
    staticpoint_dir : Path
        Path to the staticpoint directory

    Returns:
    --------
    list : List of (POSCAR file path, displacement number) tuples
    """
    staticpoint_dir = Path(staticpoint_dir)
    poscars = []

    # Pattern to match POSCAR-001, POSCAR-002, etc.
    pattern = re.compile(r'^POSCAR-(\d+)$')

    for file in staticpoint_dir.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                disp_num = int(match.group(1))
                poscars.append((file, disp_num))

    # Sort by displacement number
    poscars.sort(key=lambda x: x[1])
    return poscars


def _staticpoint_bat_template() -> Path:
    """Always use the shared SLURM template for displacement jobs."""
    return WORKFLOW_ROOT / "common" / "staticpoint_templates" / "bat"


def setup_displacement_folder(staticpoint_dir, poscar_file, disp_num):
    """
    Set up a displacement folder for a single POSCAR-XXX file.

    Parameters:
    -----------
    staticpoint_dir : Path
        Path to the staticpoint directory
    poscar_file : Path
        Path to the POSCAR-XXX file
    disp_num : int
        Displacement number (e.g., 1 for POSCAR-001)

    Returns:
    --------
    Path : Path to the created displacement folder
    """
    staticpoint_dir = Path(staticpoint_dir)
    poscar_file = Path(poscar_file)

    # Create disp-XXX folder
    disp_folder = staticpoint_dir / f"disp-{disp_num:03d}"
    disp_folder.mkdir(exist_ok=True)

    # Copy POSCAR-XXX → disp-XXX/POSCAR
    shutil.copy2(poscar_file, disp_folder / "POSCAR")

    # Copy INCAR, KPOINTS, POTCAR from parent staticpoint directory
    files_to_copy = ["INCAR", "KPOINTS", "POTCAR"]
    for filename in files_to_copy:
        src = staticpoint_dir / filename
        if src.exists():
            shutil.copy2(src, disp_folder / filename)
        else:
            raise FileNotFoundError(f"Required file not found: {src}")

    bat_template = _staticpoint_bat_template()
    if not bat_template.exists():
        raise FileNotFoundError(f"bat template not found: {bat_template}")
    shutil.copy2(bat_template, disp_folder / "bat")

    return disp_folder


def submit_displacement_job(disp_folder, dry_run=False):
    """
    Submit a VASP job for a displacement folder.

    Parameters:
    -----------
    disp_folder : Path
        Path to the displacement folder
    dry_run : bool
        If True, print what would be done without submitting

    Returns:
    --------
    tuple : (success: bool, job_id: str or None, message: str)
    """
    disp_folder = Path(disp_folder)
    bat_file = disp_folder / "bat"

    if not bat_file.exists():
        return False, None, f"Batch script not found: {bat_file}"

    if dry_run:
        return True, None, f"Would submit job from: {disp_folder}"

    staticpoint_name = disp_folder.parent.name.replace("_staticpoint", "")
    job_name = f"{staticpoint_name}-{disp_folder.name}"

    return submit_bat(
        disp_folder,
        job_name=job_name,
        job_type="phonopy-displacement",
        label=job_name,
        dry_run=False,
    )


def setup_and_submit_displacements(staticpoint_path, examples_root_name, submit=True, dry_run=False):
    """
    Set up all displacement folders and optionally submit jobs.

    Parameters:
    -----------
    staticpoint_path : str or Path
        Path to the staticpoint directory (e.g., "phonopy_monolayer_examples/MoS2_staticpoint"),
        or just a bare name (e.g. "MoS2_staticpoint"), resolved against `examples_root_name`.
    examples_root_name : str
        Top-level examples directory to resolve a bare name against
        ("phonopy_monolayer_examples" or "phonopy_bilayer_examples").
    submit : bool
        If True, submit jobs after setting up folders
    dry_run : bool
        If True, print what would be done without actually doing it

    Returns:
    --------
    dict : Summary of results
    """
    staticpoint_path = Path(staticpoint_path)

    # Resolve path
    if staticpoint_path.is_absolute() and staticpoint_path.exists():
        staticpoint_dir = staticpoint_path
    elif staticpoint_path.exists():
        staticpoint_dir = staticpoint_path.resolve()
    else:
        # Assume it's a bare name in the given examples root
        base_dir = WORKFLOW_ROOT / examples_root_name
        staticpoint_dir = base_dir / f"{staticpoint_path.name}_staticpoint"

    staticpoint_dir = staticpoint_dir.resolve()

    if not staticpoint_dir.exists():
        raise FileNotFoundError(f"Staticpoint directory not found: {staticpoint_dir}")

    if not staticpoint_dir.is_dir():
        raise ValueError(f"Path is not a directory: {staticpoint_dir}")

    # Find all POSCAR-XXX files
    displacement_poscars = find_displacement_poscars(staticpoint_dir)

    if not displacement_poscars:
        print(f"  No POSCAR-XXX files found in {staticpoint_dir}")
        return {
            'staticpoint_dir': staticpoint_dir,
            'displacements_found': 0,
            'displacements_setup': 0,
            'jobs_submitted': 0,
            'job_ids': []
        }

    print(f"  Found {len(displacement_poscars)} displacement(s)")

    # Set up each displacement folder
    setup_folders = []
    for poscar_file, disp_num in displacement_poscars:
        disp_folder_name = f"disp-{disp_num:03d}"
        disp_folder = staticpoint_dir / disp_folder_name

        if not dry_run:
            try:
                setup_displacement_folder(staticpoint_dir, poscar_file, disp_num)
                setup_folders.append((disp_folder, disp_num))
                print(f"    ✓ Set up {disp_folder_name}")
            except Exception as e:
                print(f"    ✗ Failed to set up {disp_folder_name}: {e}")
        else:
            setup_folders.append((disp_folder, disp_num))
            print(f"    Would set up {disp_folder_name}")

    # Submit jobs if requested
    job_ids = []
    if submit and not dry_run:
        print(f"  Submitting {len(setup_folders)} job(s)...")
        for disp_folder, disp_num in setup_folders:
            success, job_id, message = submit_displacement_job(disp_folder, dry_run=False)
            if success:
                if job_id:
                    print(f"    ✓ Submitted job {job_id} for disp-{disp_num:03d}")
                    job_ids.append(job_id)
                else:
                    print(f"    ✓ disp-{disp_num:03d}: {message}")
            else:
                print(f"    ✗ disp-{disp_num:03d}: {message}")
    elif submit and dry_run:
        print(f"  Would submit {len(setup_folders)} job(s)...")
        for disp_folder, disp_num in setup_folders:
            success, job_id, message = submit_displacement_job(disp_folder, dry_run=True)
            print(f"    Would submit: {disp_folder}")

    return {
        'staticpoint_dir': staticpoint_dir,
        'displacements_found': len(displacement_poscars),
        'displacements_setup': len(setup_folders),
        'jobs_submitted': len(job_ids),
        'job_ids': job_ids
    }


def main(examples_root_name, example_name):
    """CLI entry point shared by the monolayer/bilayer wrapper scripts.

    Parameters:
    -----------
    examples_root_name : str
        "phonopy_monolayer_examples" or "phonopy_bilayer_examples"
    example_name : str
        A representative staticpoint name for --help text (e.g. "MoS2_staticpoint")
    """
    parser = argparse.ArgumentParser(
        description="Set up and submit VASP jobs for phonopy displacement calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Set up and submit jobs for a specific staticpoint directory
  python3 setup_displacements.py {example_name}

  # Set up only (don't submit)
  python3 setup_displacements.py {example_name} --no-submit

  # Dry run
  python3 setup_displacements.py {example_name} --dry-run

  # Process all staticpoint directories
  python3 setup_displacements.py --all
        """
    )
    parser.add_argument(
        "staticpoint",
        nargs="?",
        help=f"Staticpoint directory name or path (e.g., '{example_name}' or '{examples_root_name}/{example_name}')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Process all staticpoint directories in {examples_root_name}"
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Set up folders but don't submit jobs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    if not args.staticpoint and not args.all:
        parser.error("Either provide a staticpoint directory or use --all")

    if args.all:
        # Find all staticpoint directories
        base_dir = WORKFLOW_ROOT / examples_root_name
        if not base_dir.exists():
            print(f"Error: Directory not found: {base_dir}", file=sys.stderr)
            sys.exit(1)

        staticpoint_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.endswith("_staticpoint")]

        if not staticpoint_dirs:
            print(f"No staticpoint directories found in {base_dir}")
            sys.exit(0)

        print(f"Processing {len(staticpoint_dirs)} staticpoint directory(ies)...\n")

        all_results = []
        for staticpoint_dir in sorted(staticpoint_dirs):
            print(f"Processing: {staticpoint_dir.name}")
            try:
                result = setup_and_submit_displacements(
                    staticpoint_dir,
                    examples_root_name,
                    submit=not args.no_submit,
                    dry_run=args.dry_run
                )
                all_results.append(result)
                print()
            except Exception as e:
                print(f"  ✗ Error: {e}\n", file=sys.stderr)

        # Summary
        total_disps = sum(r['displacements_setup'] for r in all_results)
        total_jobs = sum(r['jobs_submitted'] for r in all_results)
        print(f"Summary: {len(all_results)} staticpoint directory(ies), {total_disps} displacement(s) set up, {total_jobs} job(s) submitted")

    else:
        # Process single staticpoint directory
        try:
            result = setup_and_submit_displacements(
                args.staticpoint,
                examples_root_name,
                submit=not args.no_submit,
                dry_run=args.dry_run
            )

            print(f"\n✓ Successfully processed: {result['staticpoint_dir'].name}")
            print(f"  Displacements found: {result['displacements_found']}")
            print(f"  Displacements set up: {result['displacements_setup']}")
            if result['jobs_submitted'] > 0:
                print(f"  Jobs submitted: {result['jobs_submitted']}")
                if args.verbose:
                    print(f"  Job IDs: {', '.join(result['job_ids'])}")

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
