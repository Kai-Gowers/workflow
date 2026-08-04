#!/usr/bin/env python3
"""
Cancel SLURM jobs and delete directories for a specific batch.

This script:
1. Loads batch information from data/batches/
2. Cancels any running SLURM jobs for the batch
3. Deletes the example directories for the batch
"""

import sys
import subprocess
import shutil
from pathlib import Path

workflow_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workflow_root / "common"))
sys.path.insert(0, str(workflow_root / "scripts" / "maintenance"))

from batches import load_batch, items_key
import job_tracking


def get_batch_bilayers(batch_number, batch_type='bilayer'):
    """
    Get list of bilayer/material names from a specific batch.

    Parameters:
    -----------
    batch_number : int
        Batch number to retrieve
    batch_type : str
        'bilayer' or 'monolayer'

    Returns:
    --------
    list : List of bilayer/material names
    """
    try:
        batch = load_batch(batch_type, batch_number)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return []
    return batch.get(items_key(batch_type), [])


def cancel_jobs_for_directories(directories, dry_run=False):
    """
    Cancel SLURM jobs for given directories.
    
    Parameters:
    -----------
    directories : list of Path
        List of directory paths
    dry_run : bool
        If True, only print what would be done
    
    Returns:
    --------
    int : Number of jobs cancelled
    """
    cancelled = 0

    # job_id -> directory, built once from slurm-<jobid>.out files (see
    # scripts/maintenance/job_tracking.py) rather than the old vasp_<id>.out
    # filename pattern, which never matched actual SLURM output filenames.
    slurm_index = job_tracking.build_slurm_output_index()
    job_ids_by_dir = {}
    for job_id, path in slurm_index.items():
        job_ids_by_dir.setdefault(path.resolve(), []).append(job_id)

    for directory in directories:
        if not directory.exists():
            continue

        job_ids = job_ids_by_dir.get(directory.resolve(), [])

        # Cancel jobs
        for job_id in set(job_ids):  # Remove duplicates
            if dry_run:
                print(f"  Would cancel job {job_id} for {directory.name}")
            else:
                try:
                    result = subprocess.run(
                        ["scancel", job_id],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        print(f"  ✓ Cancelled job {job_id} for {directory.name}")
                        cancelled += 1
                    else:
                        print(f"  ✗ Failed to cancel job {job_id}: {result.stderr.strip()}")
                except Exception as e:
                    print(f"  ✗ Error cancelling job {job_id}: {e}")
    
    return cancelled


def delete_directories(directories, dry_run=False):
    """
    Delete directories.
    
    Parameters:
    -----------
    directories : list of Path
        List of directory paths to delete
    dry_run : bool
        If True, only print what would be done
    
    Returns:
    --------
    int : Number of directories deleted
    """
    deleted = 0
    
    for directory in directories:
        if not directory.exists():
            continue
        
        if dry_run:
            print(f"  Would delete: {directory}")
        else:
            try:
                shutil.rmtree(directory)
                print(f"  ✓ Deleted: {directory.name}")
                deleted += 1
            except Exception as e:
                print(f"  ✗ Error deleting {directory.name}: {e}", file=sys.stderr)
    
    return deleted


def cancel_and_delete_batch(batch_number, batch_type='bilayer', base_dir=None, dry_run=False):
    """
    Cancel jobs and delete directories for a specific batch.
    
    Parameters:
    -----------
    batch_number : int
        Batch number to process
    batch_type : str
        'bilayer' or 'monolayer'
    base_dir : Path, optional
        Base directory (default: script's parent)
    dry_run : bool
        If True, only print what would be done
    
    Returns:
    --------
    dict : Statistics about the operation
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent
    else:
        base_dir = Path(base_dir)
    
    # Get batch items
    items = get_batch_bilayers(batch_number, batch_type)
    
    if not items:
        print(f"Error: No items found for {batch_type} batch {batch_number}", file=sys.stderr)
        return {'cancelled': 0, 'deleted': 0, 'total': 0}
    
    # Determine base directory for examples
    if batch_type == 'bilayer':
        examples_dir = base_dir / "bilayer_examples"
    else:
        examples_dir = base_dir / "monolayer_examples"
    
    # Find directories
    directories = []
    for item in items:
        item_dir = examples_dir / item
        if item_dir.exists():
            directories.append(item_dir)
        elif dry_run:
            # In dry run, show what would be checked
            directories.append(item_dir)
    
    stats = {
        'total': len(items),
        'found': len([d for d in directories if d.exists()]),
        'cancelled': 0,
        'deleted': 0
    }
    
    print(f"\n{'='*60}")
    print(f"Processing {batch_type} batch {batch_number}")
    print(f"{'='*60}")
    print(f"Total items in batch: {stats['total']}")
    print(f"Directories found: {stats['found']}")
    
    if dry_run:
        print("\nDRY RUN MODE - No changes will be made\n")
    
    # Cancel jobs
    print(f"\nCancelling SLURM jobs...")
    stats['cancelled'] = cancel_jobs_for_directories(directories, dry_run=dry_run)
    
    # Delete directories
    print(f"\nDeleting directories...")
    stats['deleted'] = delete_directories(directories, dry_run=dry_run)
    
    return stats


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cancel jobs and delete directories for a specific batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be done
  python3 cancel_and_delete_batch.py --bilayer 2 --dry-run
  
  # Cancel and delete bilayer batch 2
  python3 cancel_and_delete_batch.py --bilayer 2
  
  # Cancel and delete monolayer batch 1
  python3 cancel_and_delete_batch.py --monolayer 1
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bilayer",
        type=int,
        metavar='N',
        help="Bilayer batch number to cancel and delete"
    )
    group.add_argument(
        "--monolayer",
        type=int,
        metavar='N',
        help="Monolayer batch number to cancel and delete"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    if args.bilayer:
        batch_type = 'bilayer'
        batch_number = args.bilayer
    else:
        batch_type = 'monolayer'
        batch_number = args.monolayer
    
    stats = cancel_and_delete_batch(
        batch_number=batch_number,
        batch_type=batch_type,
        dry_run=args.dry_run
    )
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total items in batch: {stats['total']}")
    print(f"Directories found: {stats['found']}")
    print(f"Jobs cancelled: {stats['cancelled']}")
    print(f"Directories deleted: {stats['deleted']}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("This was a dry run. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()


