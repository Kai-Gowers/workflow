#!/usr/bin/env python3
"""
Create hard-coded batches of monolayers and bilayers for manual sequential submission.

This script divides all materials and bilayer combinations into batches and saves
each batch as its own file under data/batches/. You can then use submit_batch.py to
submit batches one at a time.
"""

import sys
from pathlib import Path

# Add workflow directories to path
workflow_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workflow_root / "relaxation" / "monolayer"))
sys.path.insert(0, str(workflow_root / "relaxation" / "bilayer"))
sys.path.insert(0, str(workflow_root / "common"))

from generate_bilayer_combinations import load_bilayer_combinations, load_materials
from materials_project_api import filter_materials, get_api_key
from batches import write_batch


def create_batches(
    monolayer_batch_size=50,
    bilayer_batch_size=50,
    require_p63mmc=True,
    api_key=None,
    verbose=False,
):
    """
    Create batches of monolayers and bilayers, writing one file per batch to
    data/batches/.

    Parameters:
    -----------
    monolayer_batch_size : int
        Number of monolayers per batch
    bilayer_batch_size : int
        Number of bilayers per batch

    Returns:
    --------
    dict : Batch information (same summary shape as before, for print_batch_summary)
    """
    workflow_root = Path(__file__).parent.parent.parent

    # Load all materials
    materials_file = workflow_root / "common" / "materials_list.txt"
    materials = load_materials(materials_file)

    if require_p63mmc:
        if api_key is None:
            api_key = get_api_key()
        eligible, excluded = filter_materials(
            materials, api_key=api_key, verbose=verbose
        )
        if excluded:
            print(
                f"Symmetry filter (hexagonal P6₃/mmc): "
                f"{len(eligible)} eligible, {len(excluded)} excluded"
            )
        materials = eligible

    # Load all bilayer combinations
    bilayer_file = workflow_root / "relaxation" / "bilayer" / "bilayer_combinations.txt"
    bilayers = load_bilayer_combinations(bilayer_file)
    
    # Create monolayer batches
    monolayer_batches = []
    for i in range(0, len(materials), monolayer_batch_size):
        batch = materials[i:i + monolayer_batch_size]
        monolayer_batches.append({
            'kind': 'monolayer',
            'batch_number': len(monolayer_batches) + 1,
            'materials': batch,
            'count': len(batch)
        })

    # Create bilayer batches
    bilayer_batches = []
    for i in range(0, len(bilayers), bilayer_batch_size):
        batch = bilayers[i:i + bilayer_batch_size]
        bilayer_batches.append({
            'kind': 'bilayer',
            'batch_number': len(bilayer_batches) + 1,
            'bilayers': batch,
            'count': len(batch)
        })

    # Write one file per batch to data/batches/
    written_paths = []
    for batch in monolayer_batches:
        written_paths.append(write_batch("monolayer", batch))
    for batch in bilayer_batches:
        written_paths.append(write_batch("bilayer", batch))

    # Batch information structure, kept for print_batch_summary() and as a
    # return value -- not written to disk (each batch already wrote itself).
    batch_info = {
        'created': str(Path.cwd()),
        'monolayer_batch_size': monolayer_batch_size,
        'bilayer_batch_size': bilayer_batch_size,
        'symmetry_filter': 'hexagonal_P63mmc' if require_p63mmc else None,
        'total_monolayers': len(materials),
        'total_bilayers': len(bilayers),
        'total_monolayer_batches': len(monolayer_batches),
        'total_bilayer_batches': len(bilayer_batches),
        'monolayer_batches': monolayer_batches,
        'bilayer_batches': bilayer_batches
    }

    print(f"\n{'='*60}")
    print("Batch Information Created")
    print(f"{'='*60}")
    print(f"Monolayers: {len(materials)} total")
    print(f"  Batches: {len(monolayer_batches)} batches of ~{monolayer_batch_size}")
    print(f"\nBilayers: {len(bilayers)} total")
    print(f"  Batches: {len(bilayer_batches)} batches of ~{bilayer_batch_size}")
    print(f"\nWrote {len(written_paths)} batch file(s) to: {workflow_root / 'data' / 'batches'}")
    print(f"{'='*60}\n")

    return batch_info


def print_batch_summary(batch_info):
    """Print a summary of all batches"""
    print(f"\n{'='*60}")
    print("Batch Summary")
    print(f"{'='*60}")
    
    print(f"\nMonolayer Batches ({batch_info['total_monolayer_batches']} total):")
    for batch in batch_info['monolayer_batches']:
        print(f"  Batch {batch['batch_number']}: {batch['count']} materials")
        if batch['batch_number'] <= 3:  # Show first 3
            print(f"    Examples: {', '.join(batch['materials'][:5])}{'...' if len(batch['materials']) > 5 else ''}")
    
    print(f"\nBilayer Batches ({batch_info['total_bilayer_batches']} total):")
    for batch in batch_info['bilayer_batches']:
        print(f"  Batch {batch['batch_number']}: {batch['count']} bilayers")
        if batch['batch_number'] <= 3:  # Show first 3
            print(f"    Examples: {', '.join(batch['bilayers'][:5])}{'...' if len(batch['bilayers']) > 5 else ''}")
    
    print(f"{'='*60}\n")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create hard-coded batches of monolayers and bilayers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create batches with default sizes (50 monolayers, 50 bilayers per batch)
  python3 create_batches.py
  
  # Create batches with custom sizes
  python3 create_batches.py --monolayer-size 30 --bilayer-size 40
        """
    )
    parser.add_argument(
        "--monolayer-size",
        type=int,
        default=50,
        help="Number of monolayers per batch (default: 50)"
    )
    parser.add_argument(
        "--bilayer-size",
        type=int,
        default=50,
        help="Number of bilayers per batch (default: 50)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print detailed summary of batches"
    )
    sym = parser.add_mutually_exclusive_group()
    sym.add_argument(
        "--require-p63mmc",
        dest="require_p63mmc",
        action="store_true",
        default=True,
        help="Only batch hexagonal P6₃/mmc materials (default)",
    )
    sym.add_argument(
        "--no-symmetry-filter",
        dest="require_p63mmc",
        action="store_false",
        help="Include all materials from materials_list.txt",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print symmetry filter details",
    )

    args = parser.parse_args()

    batch_info = create_batches(
        monolayer_batch_size=args.monolayer_size,
        bilayer_batch_size=args.bilayer_size,
        require_p63mmc=args.require_p63mmc,
        verbose=args.verbose,
    )
    
    if args.summary:
        print_batch_summary(batch_info)


if __name__ == "__main__":
    main()

