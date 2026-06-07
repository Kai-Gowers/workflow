#!/usr/bin/env python3
"""
Create bilayer training examples for VASP relaxation calculations.

This script generates complete bilayer training examples by:
1. Selecting a bilayer combination (from bilayer_combinations.txt or randomly)
2. Generating POSCAR file for that bilayer with specified stacking (3R or 2H)
3. Generating POTCAR file from POSCAR elements
4. Copying and customizing INCAR, KPOINTS, and batch script templates

Each bilayer example directory contains all files needed for VASP relaxation.
"""

from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "monolayer"))
from generate_potcar import generate_potcar
from generate_bilayer_poscar import generate_bilayer_poscar


def customize_incar(template_path, output_path, material_name):
    """
    Copy INCAR template and customize it for the specific material.
    
    Parameters:
    -----------
    template_path : Path
        Path to INCAR template file
    output_path : Path
        Path where customized INCAR will be written
    material_name : str
        Material name to use in SYSTEM line
    """
    with open(template_path, 'r') as f:
        lines = f.readlines()
    
    # Customize the INCAR
    customized_lines = []
    system_found = False
    
    for line in lines:
        # Update SYSTEM line with material name (replace first occurrence, skip duplicates)
        stripped = line.strip()
        if stripped.startswith('SYSTEM'):
            if not system_found:
                # First SYSTEM line: use material name
                customized_lines.append(f"SYSTEM = {material_name} relaxation\n")
                system_found = True
            # Skip duplicate SYSTEM lines
        else:
            customized_lines.append(line)
    
    # If no SYSTEM line found, add one at the beginning
    if not system_found:
        customized_lines.insert(0, f"SYSTEM = {material_name} relaxation\n")
    
    # Write customized INCAR
    with open(output_path, 'w') as f:
        f.writelines(customized_lines)


def find_unique_bilayer_name(bilayer_name, base_dir=None):
    """
    Find a unique directory name for a bilayer, handling duplicates.
    
    Parameters:
    -----------
    bilayer_name : str
        Bilayer name (e.g., "MoS2_bilayer_3R" or "MoS2_WS2_2H")
    base_dir : Path, optional
        Base directory where bilayer examples are stored
    
    Returns:
    --------
    str : Unique directory name
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "bilayer_examples"
    else:
        base_dir = Path(base_dir)
    
    if not base_dir.exists():
        return bilayer_name
    
    # Check if base name exists
    base_path = base_dir / bilayer_name
    if not base_path.exists():
        return bilayer_name
    
    # If exists, try with suffix
    counter = 1
    while True:
        candidate = f"{bilayer_name}_{counter}"
        candidate_path = base_dir / candidate
        if not candidate_path.exists():
            return candidate
        counter += 1


def create_bilayer_example(
    bilayer_name,
    example_name=None,
    base_dir=None,
    generate_potcar_file=True,
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
    strict_validation=False,
    anchor=None,
    vacuum=None,
    monolayer_examples_dir=None,
    use_relaxed_monolayers=True,
):
    """
    Create a new bilayer training example.
    
    Parameters:
    -----------
    bilayer_name : str
        Bilayer name (e.g., "MoS2_bilayer_3R" or "MoS2_WS2_2H")
    example_name : str, optional
        Desired directory name. If None, uses bilayer_name (with auto-increment for duplicates).
    base_dir : Path, optional
        Base directory for examples (default: ../bilayer_examples)
    generate_potcar_file : bool
        Whether to generate POTCAR file (default: True)
    
    Returns:
    --------
    dict : Information about the created example
    """
    # Default base_dir
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "bilayer_examples"
    else:
        base_dir = Path(base_dir)
    
    # Determine example directory name
    if example_name is None:
        example_name = find_unique_bilayer_name(bilayer_name, base_dir)
    else:
        # Check if specified name already exists
        if (base_dir / example_name).exists():
            raise ValueError(f"Example directory already exists: {base_dir / example_name}")
    
    # Extract stacking from bilayer name
    if '_3R' in bilayer_name:
        stacking = '3R'
    elif '_2H' in bilayer_name:
        stacking = '2H'
    else:
        stacking = '3R'  # default
    
    # Create directory with bilayer name
    example_dir = base_dir / example_name
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate POSCAR
    poscar_metadata = {}
    poscar_path = generate_bilayer_poscar(
        bilayer_name=bilayer_name,
        output_dir=example_dir,
        filename="POSCAR",
        use_mp=use_mp,
        mp_api_key=mp_api_key,
        mp_refresh=mp_refresh,
        mp_verbose=mp_verbose,
        strict_validation=strict_validation,
        anchor=anchor,
        vacuum=vacuum,
        metadata_out=poscar_metadata,
        monolayer_examples_dir=monolayer_examples_dir,
        use_relaxed_monolayers=use_relaxed_monolayers,
    )
    
    # Generate POTCAR
    potcar_path = None
    if generate_potcar_file:
        try:
            potcar_path, selected_variants = generate_potcar(
                poscar_path=poscar_path,
                potcar_path=example_dir / "POTCAR"
            )
        except Exception as e:
            print(f"  Warning: Could not generate POTCAR: {e}")
    
    # Copy and customize template files
    template_dir = Path(__file__).parent.parent.parent / "common" / "relaxation_templates"
    copied_files = []
    
    try:
        # Copy KPOINTS
        if (template_dir / "KPOINTS").exists():
            shutil.copy2(template_dir / "KPOINTS", example_dir / "KPOINTS")
            copied_files.append("KPOINTS")
        
        # Copy bat script
        if (template_dir / "bat").exists():
            shutil.copy2(template_dir / "bat", example_dir / "bat")
            copied_files.append("bat")
        
        # Copy and customize INCAR
        if (template_dir / "INCAR").exists():
            customize_incar(
                template_path=template_dir / "INCAR",
                output_path=example_dir / "INCAR",
                material_name=bilayer_name
            )
            copied_files.append("INCAR")
    except Exception as e:
        print(f"  Warning: Could not copy template files: {e}")
    
    poscar_method = poscar_metadata.get("poscar_method", "template")
    print(f"Created bilayer example: {example_name}")
    print(f"  Bilayer: {bilayer_name}")
    print(f"  Stacking: {stacking}")
    print(f"  POSCAR method: {poscar_method}")
    print(f"  POSCAR: {poscar_path}")
    if potcar_path:
        print(f"  POTCAR: {potcar_path}")
    if copied_files:
        print(f"  Template files: {', '.join(copied_files)}")
    print(f"  Directory: {example_dir}")
    
    result = {
        'name': example_name,
        'bilayer_name': bilayer_name,
        'stacking': stacking,
        'poscar_path': poscar_path,
        'potcar_path': potcar_path,
        'directory': example_dir,
        'poscar_method': poscar_method,
    }
    return result


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create a bilayer training example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create bilayer example
  python3 create_bilayer_example.py MoS2_bilayer_3R
  
  # Create with specific name
  python3 create_bilayer_example.py MoS2_WS2_2H --name MoS2_WS2_2H_custom
  
  # Create without POTCAR
  python3 create_bilayer_example.py MoS2_bilayer_3R --no-potcar
        """
    )
    parser.add_argument(
        "bilayer_name",
        type=str,
        help="Bilayer name (e.g., 'MoS2_bilayer_3R' or 'MoS2_WS2_2H')"
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="Example directory name (default: uses bilayer_name with auto-increment for duplicates)"
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
        "--no-mp",
        action="store_true",
        help="Disable Materials Project structure lookup for bilayer POSCAR generation",
    )
    parser.add_argument(
        "--mp-api-key",
        type=str,
        default=None,
        help="Materials Project API key (default: MP_API_KEY environment variable)",
    )
    parser.add_argument(
        "--mp-refresh",
        action="store_true",
        help="Refresh MP cache entries for queried materials",
    )
    parser.add_argument(
        "--mp-verbose",
        action="store_true",
        help="Print MP selection/fallback details",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail bilayer creation if MP/geometry validation fails",
    )
    parser.add_argument(
        "--anchor",
        type=int,
        choices=[1, 2],
        default=None,
        help="Lock bilayer in-plane lattice a to mat1 (1) or mat2 (2); default: average",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=None,
        help="c-axis cell height for bilayers (default: 20 Å)",
    )
    parser.add_argument(
        "--monolayer-examples-dir",
        type=str,
        default=None,
        help="Root directory of relaxed monolayer examples (default: monolayer_examples)",
    )
    parser.add_argument(
        "--use-templates",
        action="store_true",
        help="Use ideal template monolayer geometry instead of relaxed CONTCAR files",
    )

    args = parser.parse_args()

    try:
        result = create_bilayer_example(
            bilayer_name=args.bilayer_name,
            example_name=args.name,
            base_dir=args.base_dir,
            generate_potcar_file=not args.no_potcar,
            use_mp=not args.no_mp,
            mp_api_key=args.mp_api_key,
            mp_refresh=args.mp_refresh,
            mp_verbose=args.mp_verbose,
            strict_validation=args.strict_validation,
            anchor=args.anchor,
            vacuum=args.vacuum,
            monolayer_examples_dir=args.monolayer_examples_dir,
            use_relaxed_monolayers=not args.use_templates,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

