#!/usr/bin/env python3
"""
Cleanup script to cancel all jobs and delete all examples.

This script will:
1. Cancel all SLURM jobs for the current user
2. Delete monolayer_examples and bilayer_examples directories
3. Optionally delete phonopy examples and FINAL_RESULTS
"""

import subprocess
import sys
from pathlib import Path
import argparse


def cancel_all_jobs(dry_run=False):
    """Cancel all SLURM jobs for the current user."""
    try:
        import os
        username = os.environ.get("USER", os.environ.get("USERNAME", ""))
        if not username:
            print("Warning: Could not determine username. Skipping job cancellation.")
            return 0
        
        # Get all job IDs for current user
        result = subprocess.run(
            ["squeue", "-u", username, "-h", "-o", "%i"],
            capture_output=True,
            text=True,
            check=False
        )
        
        job_ids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        
        if not job_ids:
            print("No jobs found to cancel.")
            return 0
        
        print(f"Found {len(job_ids)} job(s) to cancel:")
        for job_id in job_ids[:10]:  # Show first 10
            print(f"  Job {job_id}")
        if len(job_ids) > 10:
            print(f"  ... and {len(job_ids) - 10} more")
        
        if dry_run:
            print("(Dry run: would cancel these jobs)")
            return len(job_ids)
        
        # Cancel all jobs
        cancelled = 0
        for job_id in job_ids:
            try:
                subprocess.run(
                    ["scancel", job_id],
                    capture_output=True,
                    check=True
                )
                cancelled += 1
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Failed to cancel job {job_id}: {e}", file=sys.stderr)
        
        print(f"Cancelled {cancelled}/{len(job_ids)} job(s)")
        return cancelled
        
    except FileNotFoundError:
        print("Warning: squeue/scancel commands not found. Skipping job cancellation.")
        return 0
    except Exception as e:
        print(f"Error canceling jobs: {e}", file=sys.stderr)
        return 0


def delete_directory(path, name, dry_run=False):
    """Delete a directory if it exists."""
    path = Path(path)
    if not path.exists():
        print(f"{name}: Does not exist, skipping")
        return False
    
    if not path.is_dir():
        print(f"{name}: Not a directory, skipping")
        return False
    
    if dry_run:
        # Count subdirectories
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        print(f"{name}: Would delete ({len(subdirs)} subdirectories)")
        return True
    
    try:
        import shutil
        shutil.rmtree(path)
        print(f"{name}: Deleted")
        return True
    except Exception as e:
        print(f"{name}: Error deleting - {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Cancel all jobs and delete all examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would be deleted)
  python3 cleanup_all.py --dry-run
  
  # Cancel jobs and delete examples (with confirmation)
  python3 cleanup_all.py
  
  # Force delete without confirmation
  python3 cleanup_all.py --force
  
  # Also delete phonopy examples and FINAL_RESULTS
  python3 cleanup_all.py --include-phonopy
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--include-phonopy",
        action="store_true",
        help="Also delete phonopy examples and FINAL_RESULTS"
    )
    parser.add_argument(
        "--jobs-only",
        action="store_true",
        help="Only cancel jobs, don't delete directories"
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Only delete examples, don't cancel jobs"
    )
    
    args = parser.parse_args()
    
    workflow_root = Path(__file__).parent.parent.parent
    
    print("="*60)
    print("Cleanup All Jobs and Examples")
    print("="*60)
    
    if args.dry_run:
        print("DRY RUN MODE - Nothing will be deleted")
        print("="*60)
    
    cancelled = 0
    # Cancel jobs
    if not args.examples_only:
        print("\n1. Canceling jobs...")
        cancelled = cancel_all_jobs(dry_run=args.dry_run)
    
    if args.jobs_only:
        print("\nDone (jobs only mode)")
        return
    
    # Delete directories
    print("\n2. Deleting example directories...")
    
    dirs_to_delete = [
        (workflow_root / "monolayer_examples", "monolayer_examples"),
        (workflow_root / "bilayer_examples", "bilayer_examples"),
    ]
    
    if args.include_phonopy:
        dirs_to_delete.extend([
            (workflow_root / "phonopy_monolayer_examples", "phonopy_monolayer_examples"),
            (workflow_root / "phonopy_bilayer_examples", "phonopy_bilayer_examples"),
            (workflow_root / "FINAL_RESULTS", "FINAL_RESULTS"),
        ])
    
    # Show what will be deleted
    total_dirs = 0
    for path, name in dirs_to_delete:
        if path.exists():
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            total_dirs += len(subdirs)
            print(f"  {name}: {len(subdirs)} subdirectories")
    
    if total_dirs == 0:
        print("  No example directories found to delete")
        return
    
    # Confirmation
    if not args.force and not args.dry_run:
        print(f"\n⚠️  WARNING: This will delete {total_dirs} example directories!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    # Delete directories
    deleted = 0
    for path, name in dirs_to_delete:
        if delete_directory(path, name, dry_run=args.dry_run):
            deleted += 1
    
    print(f"\n{'='*60}")
    if args.dry_run:
        print("Dry run complete - nothing was actually deleted")
    else:
        print(f"Cleanup complete!")
        if not args.examples_only:
            print(f"  Cancelled jobs: {cancelled}")
        print(f"  Deleted directories: {deleted}/{len(dirs_to_delete)}")
    print("="*60)


if __name__ == "__main__":
    main()

