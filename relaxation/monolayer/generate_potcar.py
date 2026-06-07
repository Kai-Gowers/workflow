#!/usr/bin/env python3
"""
Script to generate POTCAR file from POSCAR file.
Reads element names from POSCAR and concatenates appropriate POTCAR files.
"""

import os
import sys
from pathlib import Path

# Default POTCAR path (can be overridden)
DEFAULT_POTCAR_PATH = "/projects/twist2d/modules/vasp/potpaw_PBE.64"

# Priority order for pseudopotential variants (higher priority first)
# This determines which variant to use when multiple are available
POTCAR_VARIANT_PRIORITY = [
    "_sv",      # Semi-core valence (preferred for transition metals)
    "_pv",      # p-valence
    "_d",       # d-valence
    "_h",       # Hard pseudopotential
    "",         # Basic (no suffix)
    "_GW",      # GW variant (usually combined with others)
    "_sv_GW",   # Combined variants
    "_pv_GW",
]

# Elements that typically benefit from _sv (semi-core valence) treatment
# Transition metals and heavy elements
SV_ELEMENTS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
}


def read_elements_from_poscar(poscar_path):
    """
    Read element names from a POSCAR file.
    
    Parameters:
    -----------
    poscar_path : str or Path
        Path to POSCAR file
    
    Returns:
    --------
    list : List of element symbols
    """
    poscar_path = Path(poscar_path)
    
    if not poscar_path.exists():
        raise FileNotFoundError(f"POSCAR file not found: {poscar_path}")
    
    with open(poscar_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comment line (line 1)
    # Skip scale factor (line 2)
    # Skip lattice vectors (lines 3-5)
    # Element symbols are on line 6 (index 5)
    if len(lines) < 6:
        raise ValueError(f"POSCAR file {poscar_path} appears to be incomplete or malformed")
    
    # Get element line (line 6, index 5)
    element_line = lines[5].strip()
    
    if not element_line:
        raise ValueError(f"Could not find element symbols in POSCAR file {poscar_path}")
    
    # Split by whitespace to get element names
    elements = element_line.split()
    
    # Filter out empty strings
    elements = [elem for elem in elements if elem]
    
    if not elements:
        raise ValueError(f"No elements found in POSCAR file {poscar_path}")
    
    return elements


def find_available_potcar_variants(element, potcar_base_path):
    """
    Find all available POTCAR variants for an element.
    
    Parameters:
    -----------
    element : str
        Element symbol (e.g., 'Mo', 'W', 'S')
    potcar_base_path : Path
        Base path to POTCAR directory
    
    Returns:
    --------
    list : List of available variant names (e.g., ['Mo', 'Mo_pv', 'Mo_sv', 'Mo_sv_GW'])
    """
    potcar_base_path = Path(potcar_base_path)
    available = []
    
    # List all directories in potcar_base_path
    if not potcar_base_path.exists():
        return available
    
    # Check for exact match and variants
    element_upper = element.capitalize()  # Standardize to first letter capital
    
    for item in potcar_base_path.iterdir():
        if item.is_dir():
            name = item.name
            # Check if it starts with the element name
            if name.startswith(element_upper) or name.startswith(element):
                # Check if it's the exact element or a variant
                if name == element_upper or name == element:
                    available.append(name)
                elif name.startswith(element_upper + '_') or name.startswith(element + '_'):
                    available.append(name)
    
    return sorted(available)


def select_potcar_variant(element, potcar_base_path, preferred_variant=None):
    """
    Select the best POTCAR variant for an element.
    
    Parameters:
    -----------
    element : str
        Element symbol
    potcar_base_path : Path
        Base path to POTCAR directory
    preferred_variant : str, optional
        Preferred variant suffix (e.g., '_sv', '_pv'). If None, uses default priority.
    
    Returns:
    --------
    str : Selected variant name (e.g., 'Mo_sv')
    Path : Path to POTCAR file
    """
    potcar_base_path = Path(potcar_base_path)
    
    # Find all available variants
    available = find_available_potcar_variants(element, potcar_base_path)
    
    if not available:
        # Try exact element name as fallback
        element_potcar = potcar_base_path / element / "POTCAR"
        if element_potcar.exists():
            return element, element_potcar
        raise FileNotFoundError(
            f"No POTCAR variants found for element {element} in {potcar_base_path}\n"
            f"Available elements: {', '.join(sorted([d.name for d in potcar_base_path.iterdir() if d.is_dir()]))[:100]}"
        )
    
    # If preferred variant is specified, try to use it
    if preferred_variant:
        for variant in available:
            if variant.endswith(preferred_variant) or variant == preferred_variant:
                potcar_path = potcar_base_path / variant / "POTCAR"
                if potcar_path.exists():
                    return variant, potcar_path
    
    # Use priority order to select best variant
    # For transition metals, prefer _sv; for others, prefer basic
    element_upper = element.capitalize()
    
    if element_upper in SV_ELEMENTS:
        # For transition metals, prefer _sv variants
        priority_order = ["_sv", "_pv", "", "_sv_GW", "_pv_GW", "_GW"]
    else:
        # For other elements, prefer basic variant
        priority_order = ["", "_pv", "_sv", "_h", "_d", "_GW"]
    
    # Try each priority in order
    for priority in priority_order:
        for variant in available:
            if priority == "":
                # Exact match (no suffix)
                if variant == element_upper or variant == element:
                    potcar_path = potcar_base_path / variant / "POTCAR"
                    if potcar_path.exists():
                        return variant, potcar_path
            elif variant.endswith(priority):
                potcar_path = potcar_base_path / variant / "POTCAR"
                if potcar_path.exists():
                    return variant, potcar_path
    
    # Fallback: use first available variant
    variant = available[0]
    potcar_path = potcar_base_path / variant / "POTCAR"
    if potcar_path.exists():
        return variant, potcar_path
    
    raise FileNotFoundError(
        f"POTCAR file not found for element {element} (tried variants: {', '.join(available)})"
    )


def generate_potcar(poscar_path, potcar_path=None, potcar_base_path=None, output_path=None, 
                    preferred_variants=None, verbose=False):
    """
    Generate POTCAR file from POSCAR file.
    
    Parameters:
    -----------
    poscar_path : str or Path
        Path to input POSCAR file
    potcar_path : str or Path, optional
        Path where POTCAR will be written (default: same directory as POSCAR, named "POTCAR")
    potcar_base_path : str or Path, optional
        Base path to POTCAR pseudopotential files (default: DEFAULT_POTCAR_PATH)
    output_path : str or Path, optional
        Alias for potcar_path (for backward compatibility)
    preferred_variants : dict, optional
        Dictionary mapping element symbols to preferred variant suffixes
        (e.g., {'Mo': '_sv', 'S': ''})
    verbose : bool, optional
        Print detailed information about variant selection
    
    Returns:
    --------
    Path : Path to generated POTCAR file
    dict : Dictionary mapping elements to selected variants
    """
    poscar_path = Path(poscar_path)
    
    # Determine output path
    if output_path is not None:
        potcar_path = output_path
    if potcar_path is None:
        potcar_path = poscar_path.parent / "POTCAR"
    else:
        potcar_path = Path(potcar_path)
    
    # Use default base path if not specified
    if potcar_base_path is None:
        potcar_base_path = DEFAULT_POTCAR_PATH
    else:
        potcar_base_path = Path(potcar_base_path)
    
    if preferred_variants is None:
        preferred_variants = {}
    
    # Read elements from POSCAR
    elements = read_elements_from_poscar(poscar_path)
    
    if verbose:
        print(f"Reading POSCAR: {poscar_path}")
        print(f"Found elements: {', '.join(elements)}")
    
    # Select POTCAR variants for each element
    potcar_files = []
    selected_variants = {}
    
    for element in elements:
        preferred = preferred_variants.get(element)
        variant, potcar_file = select_potcar_variant(element, potcar_base_path, preferred)
        potcar_files.append(potcar_file)
        selected_variants[element] = variant
        
        if verbose:
            available = find_available_potcar_variants(element, potcar_base_path)
            print(f"  {element}: selected '{variant}' (available: {', '.join(available)})")
    
    # Concatenate POTCAR files
    try:
        with open(potcar_path, 'wb') as outfile:
            for potcar_file in potcar_files:
                with open(potcar_file, 'rb') as infile:
                    outfile.write(infile.read())
        
        if verbose:
            print(f"POTCAR written to: {potcar_path}")
            print(f"Selected variants: {selected_variants}")
        else:
            print(f"POTCAR written to: {potcar_path} (variants: {', '.join([f'{k}={v}' for k, v in selected_variants.items()])})")
        
        return potcar_path, selected_variants
        
    except Exception as e:
        raise RuntimeError(f"Error writing POTCAR file: {e}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate POTCAR file from POSCAR file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate POTCAR from POSCAR in current directory
  python3 generate_potcar.py POSCAR
  
  # Specify input and output paths
  python3 generate_potcar.py POSCAR -o POTCAR
  
  # Use custom POTCAR base path
  python3 generate_potcar.py POSCAR --potcar-base /path/to/potpaw_PBE.64
  
  # Generate POTCAR in a different directory
  python3 generate_potcar.py ./input/POSCAR -o ./input/POTCAR
  
  # Specify preferred variants for elements
  python3 generate_potcar.py POSCAR --variant Mo:_sv --variant W:_sv
  
  # Verbose output showing variant selection
  python3 generate_potcar.py POSCAR -v
        """
    )
    parser.add_argument(
        "poscar",
        type=str,
        help="Path to POSCAR file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path for output POTCAR file (default: POTCAR in same directory as POSCAR)"
    )
    parser.add_argument(
        "--potcar-base",
        type=str,
        default=None,
        help=f"Base path to POTCAR pseudopotential files (default: {DEFAULT_POTCAR_PATH})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output including variant selection details"
    )
    parser.add_argument(
        "--variant",
        type=str,
        action="append",
        metavar="ELEMENT:VARIANT",
        help="Specify preferred variant for an element (e.g., --variant Mo:_sv --variant S:). Can be used multiple times."
    )
    
    args = parser.parse_args()
    
    # Parse variant preferences
    preferred_variants = {}
    if args.variant:
        for variant_spec in args.variant:
            if ':' not in variant_spec:
                print(f"Warning: Invalid variant specification '{variant_spec}'. Expected format: ELEMENT:VARIANT", file=sys.stderr)
                continue
            element, variant = variant_spec.split(':', 1)
            preferred_variants[element] = variant
    
    try:
        potcar_path, selected_variants = generate_potcar(
            poscar_path=args.poscar,
            potcar_path=args.output,
            potcar_base_path=args.potcar_base,
            preferred_variants=preferred_variants if preferred_variants else None,
            verbose=args.verbose
        )
        
        if args.verbose:
            print(f"Successfully generated POTCAR at: {potcar_path}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

