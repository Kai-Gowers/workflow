#!/usr/bin/env python3
"""
Submit a specific batch of monolayers or bilayers.

This script reads batch information from data/batches/ and submits a specific batch.
You can submit batches sequentially as previous ones complete.
"""

import sys
from pathlib import Path

# Add workflow directories to path
workflow_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workflow_root / "relaxation" / "monolayer"))
sys.path.insert(0, str(workflow_root / "relaxation" / "bilayer"))
sys.path.insert(0, str(workflow_root / "common"))

from create_monolayer_example import create_training_example
from submit_monolayer_job import submit_job as submit_monolayer_job
from create_bilayer_example import create_bilayer_example
from submit_bilayer_job import submit_bilayer_job
from materials_project_api import load_symmetry_eligible_set, parse_bilayer_components
from batches import load_batch, iter_batches
from cli_helpers import add_mp_args


def _skip_symmetry(name: str, eligible_set, is_bilayer: bool = False) -> bool:
    """Return True if name should be skipped (not symmetry-eligible)."""
    if eligible_set is None:
        return False
    if is_bilayer:
        components = parse_bilayer_components(name)
        if not components:
            return False
        return not all(c in eligible_set for c in components)
    return name not in eligible_set


def _print_job_id_summary(results, name_key):
    """Print material/bilayer -> job_id table for submitted jobs."""
    submitted = [
        r for r in results
        if r.get("submitted") and r.get("job_id")
    ]
    if not submitted:
        return

    print("Submitted jobs:")
    for r in submitted:
        name = r.get(name_key, "?")
        print(f"  {name:<30} job {r['job_id']}")
    print()


def submit_monolayer_batch(
    batch_number,
    create=True,
    generate_potcar=True,
    dry_run=False,
    verbose=False,
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
    strict_validation=False,
):
    """
    Submit a specific monolayer batch.
    
    Parameters:
    -----------
    batch_number : int
        Batch number to submit
    create : bool
        Create examples if they don't exist
    generate_potcar : bool
        Generate POTCAR files
    dry_run : bool
        Don't actually submit, just show what would be done
    verbose : bool
        Print verbose output

    Returns:
    --------
    dict : Submission results
    """
    batch = load_batch("monolayer", batch_number)

    print(f"\n{'='*60}")
    print(f"Submitting Monolayer Batch {batch_number}")
    print(f"{'='*60}")
    print(f"Materials in batch: {batch['count']}")
    if verbose:
        print(f"Materials: {', '.join(batch['materials'])}")
    print(f"{'='*60}\n")
    
    results = []
    created = 0
    submitted = 0
    skipped = 0
    
    workflow_root = Path(__file__).parent.parent.parent
    monolayer_dir = workflow_root / "monolayer_examples"
    eligible_set = load_symmetry_eligible_set()

    for i, material in enumerate(batch['materials'], 1):
        if _skip_symmetry(material, eligible_set):
            print(f"  Skipping {material}: not in symmetry_eligible_materials.json")
            skipped += 1
            continue
        if verbose:
            print(f"[{i}/{batch['count']}] Processing: {material}")
        
        # Check if example already exists
        example_dir = monolayer_dir / material
        if example_dir.exists() and (example_dir / "bat").exists():
            if verbose:
                print(f"  Example already exists: {example_dir}")
            skipped += 1
        elif create:
            # Create the example
            try:
                result = create_training_example(
                    example_name=material,
                    base_dir=monolayer_dir,
                    generate_potcar_file=generate_potcar,
                    use_mp=use_mp,
                    mp_api_key=mp_api_key,
                    mp_refresh=mp_refresh,
                    mp_verbose=mp_verbose,
                    strict_validation=strict_validation,
                )
                example_dir = result['directory']
                created += 1
                if verbose:
                    print(f"  ✓ Created: {example_dir}")
            except Exception as e:
                print(f"  ✗ Failed to create {material}: {e}", file=sys.stderr)
                results.append({
                    'material': material,
                    'created': False,
                    'submitted': False,
                    'error': str(e)
                })
                continue
        else:
            print(f"  ⊘ Skipping {material} (not created, use --create to create)")
            skipped += 1
            continue
        
        # Submit the job
        if not dry_run:
            try:
                success, job_id, message = submit_monolayer_job(example_dir, dry_run=False)
                if success:
                    submitted += 1
                    if job_id:
                        print(f"  ✓ Submitted job {job_id} for {material}")
                    else:
                        print(f"  ✓ {message} for {material}")
                    results.append({
                        'material': material,
                        'created': True,
                        'submitted': True,
                        'job_id': job_id
                    })
                else:
                    print(f"  ✗ Failed to submit {material}: {message}", file=sys.stderr)
                    results.append({
                        'material': material,
                        'created': True,
                        'submitted': False,
                        'error': message
                    })
            except Exception as e:
                print(f"  ✗ Error submitting {material}: {e}", file=sys.stderr)
                results.append({
                    'material': material,
                    'created': True,
                    'submitted': False,
                    'error': str(e)
                })
        else:
            print(f"  (Dry run: would submit {material})")
            results.append({
                'material': material,
                'created': True,
                'submitted': False,
                'dry_run': True
            })
    
    print(f"\n{'='*60}")
    print(f"Batch {batch_number} Summary")
    print(f"{'='*60}")
    print(f"Total materials: {batch['count']}")
    print(f"Created: {created}")
    print(f"Already existed: {skipped}")
    print(f"Submitted: {submitted}")
    _print_job_id_summary(results, "material")
    print(f"{'='*60}\n")

    return {
        'batch_number': batch_number,
        'type': 'monolayer',
        'total': batch['count'],
        'created': created,
        'skipped': skipped,
        'submitted': submitted,
        'results': results
    }


def submit_bilayer_batch(
    batch_number,
    create=True,
    generate_potcar=True,
    dry_run=False,
    verbose=False,
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
    strict_validation=False,
):
    """
    Submit a specific bilayer batch.
    
    Parameters:
    -----------
    batch_number : int
        Batch number to submit
    create : bool
        Create examples if they don't exist
    generate_potcar : bool
        Generate POTCAR files
    dry_run : bool
        Don't actually submit, just show what would be done
    verbose : bool
        Print verbose output

    Returns:
    --------
    dict : Submission results
    """
    batch = load_batch("bilayer", batch_number)

    print(f"\n{'='*60}")
    print(f"Submitting Bilayer Batch {batch_number}")
    print(f"{'='*60}")
    print(f"Bilayers in batch: {batch['count']}")
    if verbose:
        print(f"Bilayers: {', '.join(batch['bilayers'][:10])}{'...' if len(batch['bilayers']) > 10 else ''}")
    print(f"{'='*60}\n")
    
    results = []
    created = 0
    submitted = 0
    skipped = 0
    
    workflow_root = Path(__file__).parent.parent.parent
    bilayer_dir = workflow_root / "bilayer_examples"
    eligible_set = load_symmetry_eligible_set()

    for i, bilayer_name in enumerate(batch['bilayers'], 1):
        if _skip_symmetry(bilayer_name, eligible_set, is_bilayer=True):
            print(f"  Skipping {bilayer_name}: not symmetry-eligible")
            skipped += 1
            continue
        if verbose:
            print(f"[{i}/{batch['count']}] Processing: {bilayer_name}")
        
        # Check if example already exists
        example_dir = bilayer_dir / bilayer_name
        if example_dir.exists() and (example_dir / "bat").exists():
            if verbose:
                print(f"  Example already exists: {example_dir}")
            skipped += 1
        elif create:
            # Create the example
            try:
                result = create_bilayer_example(
                    bilayer_name=bilayer_name,
                    example_name=bilayer_name,
                    base_dir=bilayer_dir,
                    generate_potcar_file=generate_potcar,
                    use_mp=use_mp,
                    mp_api_key=mp_api_key,
                    mp_refresh=mp_refresh,
                    mp_verbose=mp_verbose,
                    strict_validation=strict_validation,
                )
                example_dir = result['directory']
                created += 1
                if verbose:
                    print(f"  ✓ Created: {example_dir}")
            except Exception as e:
                print(f"  ✗ Failed to create {bilayer_name}: {e}", file=sys.stderr)
                results.append({
                    'bilayer': bilayer_name,
                    'created': False,
                    'submitted': False,
                    'error': str(e)
                })
                continue
        else:
            print(f"  ⊘ Skipping {bilayer_name} (not created, use --create to create)")
            skipped += 1
            continue
        
        # Submit the job
        if not dry_run:
            try:
                success, job_id, message = submit_bilayer_job(example_dir, dry_run=False)
                if success:
                    submitted += 1
                    if job_id:
                        print(f"  ✓ Submitted job {job_id} for {bilayer_name}")
                    else:
                        print(f"  ✓ {message} for {bilayer_name}")
                    results.append({
                        'bilayer': bilayer_name,
                        'created': True,
                        'submitted': True,
                        'job_id': job_id
                    })
                else:
                    print(f"  ✗ Failed to submit {bilayer_name}: {message}", file=sys.stderr)
                    results.append({
                        'bilayer': bilayer_name,
                        'created': True,
                        'submitted': False,
                        'error': message
                    })
            except Exception as e:
                print(f"  ✗ Error submitting {bilayer_name}: {e}", file=sys.stderr)
                results.append({
                    'bilayer': bilayer_name,
                    'created': True,
                    'submitted': False,
                    'error': str(e)
                })
        else:
            print(f"  (Dry run: would submit {bilayer_name})")
            results.append({
                'bilayer': bilayer_name,
                'created': True,
                'submitted': False,
                'dry_run': True
            })
    
    print(f"\n{'='*60}")
    print(f"Batch {batch_number} Summary")
    print(f"{'='*60}")
    print(f"Total bilayers: {batch['count']}")
    print(f"Created: {created}")
    print(f"Already existed: {skipped}")
    print(f"Submitted: {submitted}")
    _print_job_id_summary(results, "bilayer")
    print(f"{'='*60}\n")
    
    return {
        'batch_number': batch_number,
        'type': 'bilayer',
        'total': batch['count'],
        'created': created,
        'skipped': skipped,
        'submitted': submitted,
        'results': results
    }


def list_batches():
    """List all available batches"""
    mono_batches = iter_batches("monolayer")
    bi_batches = iter_batches("bilayer")

    print(f"\n{'='*60}")
    print("Available Batches")
    print(f"{'='*60}")
    print(f"\nMonolayer Batches ({len(mono_batches)} total):")
    for batch in mono_batches:
        print(f"  Batch {batch['batch_number']}: {batch['count']} materials")

    print(f"\nBilayer Batches ({len(bi_batches)} total):")
    for batch in bi_batches:
        print(f"  Batch {batch['batch_number']}: {batch['count']} bilayers")

    print(f"{'='*60}\n")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Submit a specific batch of monolayers or bilayers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available batches
  python3 submit_batch.py --list
  
  # Submit monolayer batch 1
  python3 submit_batch.py --monolayer 1
  
  # Submit bilayer batch 2
  python3 submit_batch.py --bilayer 2
  
  # Submit batch without creating new examples (only submit existing ones)
  python3 submit_batch.py --monolayer 1 --no-create
  
  # Dry run (see what would be done)
  python3 submit_batch.py --monolayer 1 --dry-run
        """
    )
    parser.add_argument(
        "-m", "--monolayer",
        type=int,
        default=None,
        help="Monolayer batch number to submit"
    )
    parser.add_argument(
        "-b", "--bilayer",
        type=int,
        default=None,
        help="Bilayer batch number to submit"
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Don't create new examples, only submit existing ones"
    )
    parser.add_argument(
        "--no-potcar",
        action="store_true",
        help="Skip POTCAR generation"
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Dry run: show what would be done without actually doing it"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available batches"
    )
    add_mp_args(parser)
    
    args = parser.parse_args()
    
    if args.list:
        list_batches()
        return

    if args.monolayer is None and args.bilayer is None:
        parser.print_help()
        print("\nError: Must specify either --monolayer or --bilayer, or use --list to see available batches", file=sys.stderr)
        sys.exit(1)

    if args.monolayer is not None and args.bilayer is not None:
        print("Error: Cannot specify both --monolayer and --bilayer", file=sys.stderr)
        sys.exit(1)

    if args.monolayer is not None:
        submit_monolayer_batch(
            args.monolayer,
            create=not args.no_create,
            generate_potcar=not args.no_potcar,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_mp=not args.no_mp,
            mp_api_key=args.mp_api_key,
            mp_refresh=args.mp_refresh,
            mp_verbose=args.mp_verbose,
            strict_validation=args.strict_validation,
        )
    elif args.bilayer is not None:
        submit_bilayer_batch(
            args.bilayer,
            create=not args.no_create,
            generate_potcar=not args.no_potcar,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_mp=not args.no_mp,
            mp_api_key=args.mp_api_key,
            mp_refresh=args.mp_refresh,
            mp_verbose=args.mp_verbose,
            strict_validation=args.strict_validation,
        )


if __name__ == "__main__":
    main()

