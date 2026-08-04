#!/usr/bin/env python3
"""
Create training examples for VASP relaxation calculations.

This script generates complete training examples by:
1. Randomly selecting a material from common/materials_list.txt
2. Generating POSCAR file for that material
3. Generating POTCAR file from POSCAR elements
4. Copying and customizing INCAR, KPOINTS, and batch script templates

Each training example directory contains all files needed for VASP relaxation.
"""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
from incar_utils import customize_incar  # noqa: E402
from cli_helpers import add_mp_args  # noqa: E402
from generate_monolayer_poscar import generate_random_poscar, generate_poscar, load_materials_list
from generate_potcar import generate_potcar


def find_unique_example_name(material_name, base_dir="monolayer_examples"):
    """
    Find a unique directory name for a material, handling duplicates.
    
    Parameters:
    -----------
    material_name : str
        Material name (e.g., "MoS2", "graphene")
    base_dir : str or Path
        Base directory where training examples are stored
    
    Returns:
    --------
    str : Unique directory name (e.g., "MoS2", "MoS2_1", "MoS2_2", etc.)
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        return material_name
    
    # Check if base name exists
    base_path = base_dir / material_name
    if not base_path.exists():
        return material_name
    
    # If exists, try with suffix
    counter = 1
    while True:
        candidate = f"{material_name}_{counter}"
        candidate_path = base_dir / candidate
        if not candidate_path.exists():
            return candidate
        counter += 1


def create_training_example(
    example_name=None,
    base_dir=None,
    generate_potcar_file=True,
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
    strict_validation=False,
):
    """
    Create a new training example with a random material, including POSCAR and POTCAR.
    
    Parameters:
    -----------
    example_name : str, optional
        Desired directory name. If None, uses material name (with auto-increment for duplicates).
    base_dir : str or Path, optional
        Base directory where monolayer examples are stored. If None, uses ../monolayer_examples
    generate_potcar_file : bool
        Whether to generate POTCAR file (default: True)
    
    Returns:
    --------
    dict : Information about the created example
    """
    # Default base_dir is monolayer_examples in parent directory (workflow level)
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "monolayer_examples"
    else:
        base_dir = Path(base_dir)
    
    # Determine material name
    # If example_name is provided and matches a material name, use it
    # Otherwise, generate a random material
    materials_file = Path(__file__).parent.parent.parent / "common" / "materials_list.txt"
    available_materials = load_materials_list(materials_file)
    
    temp_dir = base_dir / ".temp_poscar"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    if example_name is not None and example_name in available_materials:
        # Use the provided material name
        material = example_name
        # Check if directory already exists
        if (base_dir / example_name).exists():
            raise ValueError(f"Example directory already exists: {base_dir / example_name}")
        # Generate POSCAR for the specific material
        generate_poscar(
            material,
            temp_dir,
            filename="POSCAR",
            use_mp=use_mp,
            mp_api_key=mp_api_key,
            mp_refresh=mp_refresh,
            mp_verbose=mp_verbose,
            strict_validation=strict_validation,
        )
        temp_poscar = temp_dir / "POSCAR"
    elif example_name is not None:
        # example_name provided but not a material - use it as directory name and generate random material
        # Check if directory already exists
        if (base_dir / example_name).exists():
            raise ValueError(f"Example directory already exists: {base_dir / example_name}")
        # Generate random POSCAR to get material name
        material, temp_poscar = generate_random_poscar(
            output_dir=temp_dir,
            use_mp=use_mp,
            mp_api_key=mp_api_key,
            mp_refresh=mp_refresh,
            mp_verbose=mp_verbose,
            strict_validation=strict_validation,
        )
    else:
        # Generate random POSCAR to get material name
        material, temp_poscar = generate_random_poscar(
            output_dir=temp_dir,
            use_mp=use_mp,
            mp_api_key=mp_api_key,
            mp_refresh=mp_refresh,
            mp_verbose=mp_verbose,
            strict_validation=strict_validation,
        )
        # Determine example directory name based on material
        example_name = find_unique_example_name(material, base_dir)
    
    # Create directory for this example
    example_dir = base_dir / example_name
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Move POSCAR from temp to final location
    poscar_path = example_dir / "POSCAR"
    import shutil
    shutil.move(str(temp_poscar), str(poscar_path))
    # Clean up temp directory
    try:
        temp_dir.rmdir()
    except:
        pass
    
    potcar_path = None
    if generate_potcar_file:
        try:
            # Generate POTCAR from POSCAR
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
        # Copy KPOINTS (no customization needed)
        if (template_dir / "KPOINTS").exists():
            shutil.copy2(template_dir / "KPOINTS", example_dir / "KPOINTS")
            copied_files.append("KPOINTS")
        
        # Copy bat script (no customization needed)
        if (template_dir / "bat").exists():
            shutil.copy2(template_dir / "bat", example_dir / "bat")
            copied_files.append("bat")
        
        # Copy and customize INCAR (update SYSTEM line)
        if (template_dir / "INCAR").exists():
            customize_incar(
                template_path=template_dir / "INCAR",
                output_path=example_dir / "INCAR",
                name=material
            )
            copied_files.append("INCAR")
        
    except Exception as e:
        print(f"  Warning: Could not copy template files: {e}")
    
    print(f"Created training example: {example_name}")
    print(f"  Material: {material}")
    print(f"  POSCAR: {poscar_path}")
    if potcar_path:
        print(f"  POTCAR: {potcar_path}")
    if copied_files:
        print(f"  Template files: {', '.join(copied_files)}")
    print(f"  Directory: {example_dir}")
    
    return {
        'name': example_name,
        'material': material,
        'poscar_path': poscar_path,
        'potcar_path': potcar_path,
        'directory': example_dir
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create a new training example with random material",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new training example (auto-increments ID)
  python3 create_monolayer_example.py
  
  # Create multiple examples
  python3 create_monolayer_example.py --count 5
  
  # Create example in specific directory
  python3 create_monolayer_example.py --base-dir my_examples
  
  # Create example without POTCAR
  python3 create_monolayer_example.py --no-potcar
        """
    )
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=1,
        help="Number of training examples to create (default: 1)"
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
        "-n", "--name",
        type=str,
        default=None,
        help="Specific example name to use (default: uses material name)"
    )
    add_mp_args(parser)
    
    args = parser.parse_args()
    
    # Create training examples
    for i in range(args.count):
        try:
            # Use specified name for first example, then auto-generate for subsequent ones
            if args.name is not None and i == 0:
                example_name = args.name
            else:
                example_name = None  # Auto-generate from material name
            
            result = create_training_example(
                example_name=example_name,
                base_dir=args.base_dir,
                generate_potcar_file=not args.no_potcar,
                use_mp=not args.no_mp,
                mp_api_key=args.mp_api_key,
                mp_refresh=args.mp_refresh,
                mp_verbose=args.mp_verbose,
                strict_validation=args.strict_validation,
            )
            print()
        except Exception as e:
            print(f"Error creating training example: {e}")
            import traceback
            traceback.print_exc()
            break

