#!/usr/bin/env python3
"""
Systematically create all bilayer training examples from bilayer_combinations.txt.

This script reads the bilayer combinations list and creates training examples
for each bilayer with both 3R and 2H stacking.
"""

from pathlib import Path
from create_bilayer_example import create_bilayer_example


def load_bilayer_combinations(bilayer_file=None):
    """Load bilayer combinations from file"""
    if bilayer_file is None:
        bilayer_file = Path(__file__).parent / "bilayer_combinations.txt"
    else:
        bilayer_file = Path(bilayer_file)
    
    if not bilayer_file.exists():
        raise FileNotFoundError(f"Bilayer combinations file not found: {bilayer_file}")
    
    bilayers = []
    with open(bilayer_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                bilayers.append(line)
    
    return bilayers


def create_all_bilayer_examples(bilayer_file=None, base_dir=None, generate_potcar_file=True, 
                                max_examples=None):
    """
    Create training examples for all bilayer combinations.
    
    Parameters:
    -----------
    bilayer_file : str or Path, optional
        Path to bilayer combinations file
    base_dir : Path, optional
        Base directory for examples
    generate_potcar_file : bool
        Whether to generate POTCAR files
    max_examples : int, optional
        Maximum number of examples to create (for testing)
    
    Returns:
    --------
    list : List of created example information
    """
    bilayers = load_bilayer_combinations(bilayer_file)
    
    if max_examples:
        bilayers = bilayers[:max_examples]
    
    print(f"Creating {len(bilayers)} bilayer examples...")
    
    results = []
    
    for i, bilayer_name in enumerate(bilayers, 1):
        try:
            print(f"\n[{i}/{len(bilayers)}] Processing: {bilayer_name}")
            result = create_bilayer_example(
                bilayer_name=bilayer_name,
                example_name=None,  # Auto-generate from bilayer_name
                base_dir=base_dir,
                generate_potcar_file=generate_potcar_file
            )
            results.append(result)
        except Exception as e:
            print(f"  Error creating {bilayer_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Summary: Created {len(results)}/{len(bilayers)} bilayer examples")
    print(f"{'='*60}")
    
    return results


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create all bilayer training examples from combinations list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all bilayer examples
  python3 create_all_bilayers.py
  
  # Create first 10 examples (for testing)
  python3 create_all_bilayers.py --max 10
        """
    )
    parser.add_argument(
        "-f", "--bilayer-file",
        type=str,
        default=None,
        help="Path to bilayer combinations file (default: bilayer_combinations.txt)"
    )
    parser.add_argument(
        "-b", "--base-dir",
        type=str,
        default=None,
        help="Base directory for examples (default: ../bilayer_examples)"
    )
    parser.add_argument(
        "--no-potcar",
        action="store_true",
        help="Skip POTCAR generation"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of examples to create (for testing)"
    )
    
    args = parser.parse_args()
    
    try:
        results = create_all_bilayer_examples(
            bilayer_file=args.bilayer_file,
            base_dir=args.base_dir,
            generate_potcar_file=not args.no_potcar,
            max_examples=args.max
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()

