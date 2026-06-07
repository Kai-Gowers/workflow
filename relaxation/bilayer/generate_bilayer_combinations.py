#!/usr/bin/env python3
"""
Generate systematic bilayer combinations from materials list.

This script creates all possible bilayer combinations (homostructures and heterostructures)
and generates a list with both 3R (0-degree) and 2H (180-degree) stacking versions.

Compatibility checking is now based on lattice constant matching from The Materials Project API.
Materials are considered compatible if their lattice constants are within 20% of each other.
"""

import sys
import os
from pathlib import Path
from itertools import combinations_with_replacement, combinations

# Import Materials Project API functions
def _get_mp_api_path():
    """Get path to Materials Project API module"""
    workflow_root = Path(__file__).parent.parent.parent
    return workflow_root / "common" / "materials_project_api.py"

def _import_mp_api_functions():
    """Import Materials Project API compatibility functions"""
    mp_api_path = _get_mp_api_path()
    if not mp_api_path.exists():
        return None, None, None, None
    
    # Add parent directory to path
    sys.path.insert(0, str(mp_api_path.parent))
    try:
        from materials_project_api import (
            are_materials_compatible_by_lattice,
            filter_materials,
            get_all_lattice_constants,
            get_api_key,
        )
        return are_materials_compatible_by_lattice, get_all_lattice_constants, get_api_key, filter_materials
    except ImportError as e:
        print(f"Warning: Could not import Materials Project API functions: {e}", file=sys.stderr)
        return None, None, None, None


# Try to import Materials Project API functions
_are_compatible_mp, _get_all_lattice_constants, _get_api_key, _filter_materials = _import_mp_api_functions()

# Single-element 2D materials: only 3R stacking is defined (no 2H)
SINGLE_ELEMENT_MATERIALS = frozenset([
    'graphene', 'phosphorene', 'silicene', 'germanene', 'stanene'
])


def is_single_element_material(material_name):
    """Return True if the material is single-element (only 3R stacking, no 2H)."""
    return material_name in SINGLE_ELEMENT_MATERIALS


def are_materials_compatible(mat1, mat2, api_key=None, tolerance=0.20, verbose=False, lattice_constants=None):
    """
    Check if two materials are compatible for bilayer stacking based on lattice constants.
    
    Uses The Materials Project API to retrieve lattice constants and checks if they
    are within the specified tolerance (default 20%).
    
    Parameters:
    -----------
    mat1 : str
        First material name
    mat2 : str
        Second material name
    api_key : str, optional
        Materials Project API key (if None, tries to get from environment)
    tolerance : float
        Maximum relative difference in lattice constants (default: 0.20 = 20%)
    verbose : bool
        Print detailed information
    
    Returns:
    --------
    bool : True if materials are compatible
    """
    # Homostructures are always compatible
    if mat1 == mat2:
        return True

    # Use pre-fetched lattice constants when provided (same tolerance semantics, no extra API calls)
    if lattice_constants is not None:
        a1 = lattice_constants.get(mat1)
        a2 = lattice_constants.get(mat2)
        if a1 is not None and a2 is not None:
            sys.path.insert(0, str(_get_mp_api_path().parent))
            from materials_project_api import are_lattice_constants_compatible
            compatible = are_lattice_constants_compatible(a1, a2, tolerance)
            if verbose:
                relative_diff = abs(a1 - a2) / min(a1, a2) * 100
                status = "compatible" if compatible else "incompatible"
                print(
                    f"  {mat1}: a = {a1:.4f} Å, {mat2}: a = {a2:.4f} Å, "
                    f"diff = {relative_diff:.2f}% ({status})"
                )
            return compatible
    
    # Check if API functions are available
    if _are_compatible_mp is None:
        print(f"Warning: Materials Project API not available. Cannot check compatibility for {mat1}-{mat2}", file=sys.stderr)
        print("  Install mp-api with: pip install mp-api", file=sys.stderr)
        print("  Set MP_API_KEY environment variable", file=sys.stderr)
        return False
    
    if api_key is None:
        if _get_api_key:
            api_key = _get_api_key()
    
    if api_key is None:
        print(f"Warning: No API key found. Cannot check compatibility for {mat1}-{mat2}", file=sys.stderr)
        print("  Set MP_API_KEY environment variable", file=sys.stderr)
        return False
    
    return _are_compatible_mp(
        mat1, mat2, api_key, tolerance, verbose, lattice_constants=lattice_constants
    )


def load_materials(materials_file=None):
    """Load monolayer materials from materials list"""
    if materials_file is None:
        materials_file = Path(__file__).parent.parent.parent / "common" / "materials_list.txt"
    else:
        materials_file = Path(materials_file)
    
    if not materials_file.exists():
        raise FileNotFoundError(f"Materials list not found: {materials_file}")
    
    materials = []
    with open(materials_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                materials.append(line)
    
    return materials


def generate_bilayer_combinations(materials, include_homostructures=True, include_heterostructures=True, 
                                   only_compatible=True, api_key=None, tolerance=0.20, verbose=False):
    """
    Generate all bilayer combinations, optionally filtering for compatible lattice constants.
    
    Parameters:
    -----------
    materials : list
        List of material names
    include_homostructures : bool
        Include same-material bilayers (default: True)
    include_heterostructures : bool
        Include different-material bilayers (default: True)
    only_compatible : bool
        Only include heterostructures with compatible lattice constants (default: True)
        Compatible means lattice constants are within tolerance (default: 20%)
    api_key : str, optional
        Materials Project API key (if None, tries to get from environment)
    tolerance : float
        Maximum relative difference in lattice constants (default: 0.20 = 20%)
    verbose : bool
        Print detailed information about compatibility checks
    
    Returns:
    --------
    list : List of tuples (mat1, mat2) for bilayer combinations
    """
    combinations_list = []
    
    # Pre-fetch lattice constants for all materials if checking compatibility
    lattice_constants = {}
    if only_compatible and include_heterostructures and _get_all_lattice_constants:
        if api_key is None and _get_api_key:
            api_key = _get_api_key()
        
        if api_key:
            if verbose:
                print(f"Fetching lattice constants for {len(materials)} materials from Materials Project API...")
            lattice_constants = _get_all_lattice_constants(materials, api_key, verbose=verbose)
            
            # Report materials that couldn't be found
            missing = [mat for mat, a in lattice_constants.items() if a is None]
            if missing:
                print(f"Warning: Could not retrieve lattice constants for {len(missing)} materials: {', '.join(missing)}", file=sys.stderr)
        else:
            print("Warning: No API key found. Cannot check compatibility.", file=sys.stderr)
            print("  Set MP_API_KEY environment variable to enable compatibility checking.", file=sys.stderr)
    
    if include_homostructures:
        # Homostructures: same material (always compatible)
        for mat1, mat2 in combinations_with_replacement(materials, 2):
            if mat1 == mat2:
                combinations_list.append((mat1, mat2))
    
    if include_heterostructures:
        # Heterostructures: different materials
        compatible_count = 0
        incompatible_count = 0
        
        for mat1, mat2 in combinations(materials, 2):
            if only_compatible:
                # Check compatibility based on lattice constants
                if are_materials_compatible(
                    mat1, mat2, api_key, tolerance, verbose, lattice_constants=lattice_constants
                ):
                    combinations_list.append((mat1, mat2))
                    compatible_count += 1
                else:
                    incompatible_count += 1
            else:
                # Include all combinations (old behavior)
                combinations_list.append((mat1, mat2))
        
        if only_compatible and verbose:
            print(f"\nCompatibility summary:")
            print(f"  Compatible pairs: {compatible_count}")
            print(f"  Incompatible pairs: {incompatible_count}")
    
    return combinations_list


def format_bilayer_name(mat1, mat2, stacking):
    """
    Format bilayer name for use in POSCAR generation.
    
    Parameters:
    -----------
    mat1 : str
        First material name
    mat2 : str
        Second material name
    stacking : str
        Stacking type: '3R' or '2H'
    
    Returns:
    --------
    str : Formatted bilayer name
    """
    if mat1 == mat2:
        # Homostructure
        return f"{mat1}_bilayer_{stacking}"
    else:
        # Heterostructure
        return f"{mat1}_{mat2}_{stacking}"


def load_bilayer_combinations(bilayer_file=None):
    """
    Load bilayer combinations from existing file.
    
    Parameters:
    -----------
    bilayer_file : str or Path, optional
        Path to bilayer combinations file (default: bilayer_combinations.txt)
    
    Returns:
    --------
    list : List of bilayer names
    """
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


def generate_bilayer_list(
    materials_file=None,
    output_file=None,
    only_compatible=True,
    api_key=None,
    tolerance=0.20,
    verbose=False,
    require_p63mmc=True,
):
    """
    Generate complete list of bilayer combinations with both stacking types.
    
    Parameters:
    -----------
    materials_file : str or Path, optional
        Path to materials list file
    output_file : str or Path, optional
        Path where bilayer list will be written (default: bilayer_combinations.txt)
    only_compatible : bool
        Only include heterostructures with compatible lattice constants (default: True)
        Compatible means lattice constants are within tolerance (default: 20%)
    api_key : str, optional
        Materials Project API key (if None, tries to get from environment)
    tolerance : float
        Maximum relative difference in lattice constants (default: 0.20 = 20%)
    verbose : bool
        Print detailed information about compatibility checks
    
    Returns:
    --------
    list : List of bilayer names
    """
    # Load materials
    materials = load_materials(materials_file)

    if require_p63mmc:
        if _filter_materials is None:
            print(
                "Warning: symmetry filter requested but materials_project_api unavailable.",
                file=sys.stderr,
            )
        else:
            eligible, excluded = _filter_materials(
                materials, api_key=api_key, verbose=verbose
            )
            if excluded:
                print(
                    f"Symmetry filter (hexagonal {('P6₃/mmc')}): "
                    f"{len(eligible)} eligible, {len(excluded)} excluded"
                )
                if verbose:
                    print(f"  Excluded: {', '.join(excluded)}")
            materials = eligible

    print(f"Loaded {len(materials)} materials for bilayer generation")
    
    # Get API key if not provided
    if api_key is None and _get_api_key:
        api_key = _get_api_key()
    
    if only_compatible and api_key is None:
        print("Warning: No API key found. Cannot check compatibility.", file=sys.stderr)
        print("  Set MP_API_KEY environment variable to enable compatibility checking.", file=sys.stderr)
        print("  Proceeding without compatibility filtering...", file=sys.stderr)
    
    # Generate combinations
    bilayer_combos = generate_bilayer_combinations(
        materials,
        include_homostructures=True,
        include_heterostructures=True,
        only_compatible=only_compatible,
        api_key=api_key,
        tolerance=tolerance,
        verbose=verbose
    )
    
    if only_compatible:
        print(f"Generated {len(bilayer_combos)} bilayer combinations (compatible lattice constants, tolerance = {tolerance:.0%})")
    else:
        print(f"Generated {len(bilayer_combos)} bilayer combinations (all combinations)")
    
    # Create list with stacking types: single-element only 3R; multi-element 3R and 2H
    bilayer_list = []
    for mat1, mat2 in bilayer_combos:
        bilayer_list.append(format_bilayer_name(mat1, mat2, '3R'))
        # 2H only for multi-element materials (single-element has no 2H)
        if not (is_single_element_material(mat1) and is_single_element_material(mat2)):
            bilayer_list.append(format_bilayer_name(mat1, mat2, '2H'))
    
    n_2h = sum(1 for n in bilayer_list if n.endswith('_2H'))
    print(f"Total bilayer configurations: {len(bilayer_list)} ({len(bilayer_combos)} combinations; 3R for all, 2H for multi-element only)")
    
    # Write to file
    if output_file is None:
        output_file = Path(__file__).parent / "bilayer_combinations.txt"
    else:
        output_file = Path(output_file)
    
    with open(output_file, 'w') as f:
        f.write("# Bilayer Combinations\n")
        f.write("# Format: material1_material2_stacking or material_bilayer_stacking\n")
        f.write("# Stacking: 3R (0-degree twist) or 2H (180-degree twist)\n")
        f.write("# Generated systematically from materials list\n\n")
        
        # Group by combination type (single-element: 3R only; multi-element: 3R and 2H)
        f.write("# Homostructures (same material)\n")
        for mat1, mat2 in bilayer_combos:
            if mat1 == mat2:
                f.write(f"{format_bilayer_name(mat1, mat2, '3R')}\n")
                if not (is_single_element_material(mat1) and is_single_element_material(mat2)):
                    f.write(f"{format_bilayer_name(mat1, mat2, '2H')}\n")
        
        f.write("\n# Heterostructures (different materials)\n")
        for mat1, mat2 in bilayer_combos:
            if mat1 != mat2:
                f.write(f"{format_bilayer_name(mat1, mat2, '3R')}\n")
                if not (is_single_element_material(mat1) and is_single_element_material(mat2)):
                    f.write(f"{format_bilayer_name(mat1, mat2, '2H')}\n")
    
    print(f"Bilayer list written to: {output_file}")
    
    return bilayer_list


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate systematic bilayer combinations from materials list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate bilayer combinations from default materials list
  python3 generate_bilayer_combinations.py
  
  # Specify custom materials file
  python3 generate_bilayer_combinations.py -m ../common/materials_list.txt
  
  # Specify output file
  python3 generate_bilayer_combinations.py -o my_bilayers.txt
        """
    )
    parser.add_argument(
        "-m", "--materials-file",
        type=str,
        default=None,
        help="Path to materials list file (default: ../common/materials_list.txt)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file for bilayer list (default: bilayer_combinations.txt)"
    )
    parser.add_argument(
        "--no-homostructures",
        action="store_true",
        help="Exclude homostructures (same-material bilayers)"
    )
    parser.add_argument(
        "--no-heterostructures",
        action="store_true",
        help="Exclude heterostructures (different-material bilayers)"
    )
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="Include all combinations, even incompatible structures (default: only compatible)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.20,
        help="Maximum relative difference in lattice constants for compatibility (default: 0.20 = 20%%)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Materials Project API key (default: from MP_API_KEY environment variable)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about compatibility checks"
    )
    sym = parser.add_mutually_exclusive_group()
    sym.add_argument(
        "--require-p63mmc",
        dest="require_p63mmc",
        action="store_true",
        default=True,
        help="Only use hexagonal P6₃/mmc materials (default)",
    )
    sym.add_argument(
        "--no-symmetry-filter",
        dest="require_p63mmc",
        action="store_false",
        help="Disable hexagonal P6₃/mmc material filter",
    )

    args = parser.parse_args()

    try:
        bilayer_list = generate_bilayer_list(
            materials_file=args.materials_file,
            output_file=args.output,
            only_compatible=not args.all_combinations,
            api_key=args.api_key,
            tolerance=args.tolerance,
            verbose=args.verbose,
            require_p63mmc=args.require_p63mmc,
        )
        
        print(f"\nSuccessfully generated {len(bilayer_list)} bilayer configurations")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()

