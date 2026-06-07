#!/usr/bin/env python3
"""
Create a new training example and submit it to the cluster.

This script combines create_monolayer_example.py and submit_monolayer_job.py to:
1. Create a new training example with random material
2. Automatically submit the job to the cluster

This is a convenience script for quickly generating and submitting new relaxation jobs.
"""

import sys
from pathlib import Path
from create_monolayer_example import create_training_example
from submit_monolayer_job import submit_job


def create_and_submit(base_dir=None, generate_potcar_file=True, 
                      dry_run=False, verbose=False):
    """
    Create a new training example and submit it to the cluster.
    
    Parameters:
    -----------
    base_dir : str
        Base directory where training examples are stored
    generate_potcar_file : bool
        Whether to generate POTCAR file (default: True)
    dry_run : bool
        If True, don't actually submit the job (default: False)
    verbose : bool
        Print verbose output (default: False)
    
    Returns:
    --------
    dict : Information about the created and submitted example
    """
    # Default base_dir is monolayer_examples in parent directory (workflow level)
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "monolayer_examples"
    
    # Create the training example
    if verbose:
        print("Creating new training example...")
    
    try:
        result = create_training_example(
            example_name=None,  # Auto-generate from material name
            base_dir=base_dir,
            generate_potcar_file=generate_potcar_file
        )
    except Exception as e:
        print(f"Error creating training example: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    example_name = result['name']
    example_dir = result['directory']
    
    if verbose:
        print(f"\nTraining example '{example_name}' created successfully!")
        print(f"  Material: {result['material']}")
        print(f"  Directory: {example_dir}")
    
    # Submit the job using the example directory
    if dry_run:
        print(f"\nDry run: Would submit job for {example_name}")
        success, job_id, message = submit_job(example_dir, dry_run=True)
        if verbose:
            print(f"  {message}")
    else:
        if verbose:
            print(f"\nSubmitting job for {example_name}...")
        
        success, job_id, message = submit_job(example_dir, dry_run=False)
        
        if success:
            result['job_id'] = job_id
            result['submitted'] = True
            if job_id:
                print(f"✓ Job {job_id} submitted successfully!")
            else:
                print(f"✓ Job submitted: {message}")
        else:
            result['job_id'] = None
            result['submitted'] = False
            print(f"✗ Failed to submit job: {message}", file=sys.stderr)
            print(f"  Example created at: {example_dir}")
            print(f"  You can submit manually with: python3 submit_monolayer_job.py {example_name}")
    
    return result


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create a new training example and submit it to the cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create and submit a new training example
  python3 new_monolayer_relax.py
  
  # Create multiple examples and submit them
  python3 new_monolayer_relax.py --count 3
  
  # Create example without submitting
  python3 new_monolayer_relax.py --no-submit
  
  # Dry run (create example but don't submit)
  python3 new_monolayer_relax.py --dry-run
  
  # Create example in specific directory
  python3 new_monolayer_relax.py --base-dir my_examples
  
  # Create example without POTCAR
  python3 new_monolayer_relax.py --no-potcar
        """
    )
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=1,
        help="Number of training examples to create and submit (default: 1)"
    )
    parser.add_argument(
        "-b", "--base-dir",
        type=str,
        default=None,
        help="Base directory for monolayer examples (default: ../monolayer_examples)"
    )
    parser.add_argument(
        "--no-potcar",
        action="store_true",
        help="Skip POTCAR generation"
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Create example but don't submit to cluster"
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Dry run: create example but don't actually submit (shows what would be done)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Create and submit examples
    all_success = True
    results = []
    
    for i in range(args.count):
        if args.verbose and args.count > 1:
            print(f"\n{'='*60}")
            print(f"Creating example {i+1} of {args.count}")
            print(f"{'='*60}")
        
        try:
            result = create_and_submit(
                base_dir=args.base_dir,
                generate_potcar_file=not args.no_potcar,
                dry_run=args.dry_run if not args.no_submit else True,
                verbose=args.verbose
            )
            results.append(result)
            
            if not args.no_submit and not args.dry_run:
                if not result.get('submitted', False):
                    all_success = False
            
            if args.count > 1 and i < args.count - 1:
                print()  # Blank line between examples
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            all_success = False
            break
    
    # Summary
    if args.count > 1:
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        for result in results:
            status = "✓" if result.get('submitted', False) else "○"
            job_info = f" (job {result.get('job_id', 'N/A')})" if result.get('job_id') else ""
            print(f"{status} {result['name']}: {result['material']}{job_info}")
    
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()

