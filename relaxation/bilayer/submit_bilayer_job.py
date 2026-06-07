#!/usr/bin/env python3
"""
Submit VASP relaxation jobs for bilayer examples.

Similar to submit_job.py but adapted for bilayer example naming convention.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "maintenance"))
from job_tracking import submit_bat  # noqa: E402


def submit_bilayer_job(example_path, dry_run=False):
    """
    Submit a VASP job for a bilayer example.
    
    Parameters:
    -----------
    example_path : str or Path
        Path to bilayer example directory (e.g., "bilayer_examples/MoS2_bilayer_3R")
        or just the example name (e.g., "MoS2_bilayer_3R")
    dry_run : bool, optional
        If True, print what would be done without actually submitting
    
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
        # Assume it's a bilayer name in bilayer_examples directory
        base_dir = Path(__file__).parent.parent.parent / "bilayer_examples"
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


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Submit VASP relaxation job for a bilayer example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit job for bilayer_5_3R
  python3 submit_bilayer_job.py 5
  
  # Submit using full path
  python3 submit_bilayer_job.py bilayer_examples/bilayer_10_2H
  
  # Dry run
  python3 submit_bilayer_job.py 5 --dry-run
        """
    )
    parser.add_argument(
        "examples",
        nargs="+",
        help="Example ID(s) or path(s) to bilayer example directory"
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
        success, job_id, message = submit_bilayer_job(example_arg, dry_run=args.dry_run)
        
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


if __name__ == "__main__":
    main()

