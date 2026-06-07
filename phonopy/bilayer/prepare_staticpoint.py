#!/usr/bin/env python3
"""
Prepare static-point calculation directories from relaxed bilayer examples.

This script:
1. Takes CONTCAR from a relaxed bilayer example
2. Creates a new clean directory for static-point calculation
3. Copies CONTCAR → POSCAR, grouping atoms by species for phonopy/VASP compatibility
4. Copies POTCAR from the original example; KPOINTS from staticpoint_templates if present, else from the relaxed example
5. Uses INCAR and bat from staticpoint_templates (with customized SYSTEM line)
6. Generates phonopy displacements using phonopy --dim="3 3 1" -d -c POSCAR
"""

from pathlib import Path
import shutil
import argparse
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poscar_utils import reorder_poscar_for_phonopy


def customize_incar(template_path, output_path, bilayer_name):
    """
    Copy INCAR template and customize it for the specific bilayer.
    
    Parameters:
    -----------
    template_path : Path
        Path to INCAR template file
    output_path : Path
        Path where customized INCAR will be written
    bilayer_name : str
        Bilayer name to use in SYSTEM line (e.g., "MoS2_bilayer_3R")
    """
    with open(template_path, 'r') as f:
        lines = f.readlines()
    
    # Customize the INCAR
    customized_lines = []
    system_found = False
    
    for line in lines:
        # Update SYSTEM line with bilayer name (replace first occurrence, skip duplicates)
        stripped = line.strip()
        if stripped.startswith('SYSTEM'):
            if not system_found:
                # First SYSTEM line: use bilayer name
                customized_lines.append(f"SYSTEM = {bilayer_name} phonon\n")
                system_found = True
            # Skip duplicate SYSTEM lines
        else:
            customized_lines.append(line)
    
    # If no SYSTEM line found, add one at the beginning
    if not system_found:
        customized_lines.insert(0, f"SYSTEM = {bilayer_name} phonon\n")
    
    # Write customized INCAR
    with open(output_path, 'w') as f:
        f.writelines(customized_lines)


def generate_phonopy_displacements(work_dir, supercell_dim="3 3 1"):
    """
    Generate phonopy displacements in the given directory.
    
    Parameters:
    -----------
    work_dir : Path
        Working directory containing POSCAR
    supercell_dim : str
        Supercell dimensions (default: "3 3 1")
    
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    work_dir = Path(work_dir)
    poscar = work_dir / "POSCAR"
    
    if not poscar.exists():
        raise FileNotFoundError(f"POSCAR not found in {work_dir}")
    
    dim_parts = supercell_dim.split()
    import shutil
    if shutil.which("phonopy-init"):
        cmd = ["phonopy-init", "--dim", *dim_parts, "-d", "-c", str(poscar)]
    else:
        cmd = ["phonopy", "--dim", supercell_dim, "-d", "-c", str(poscar)]
    
    try:
        subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Warning: phonopy command failed: {e}")
        print(f"  Command: {' '.join(cmd)}")
        if e.stderr:
            print(f"  Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  Warning: phonopy command not found. Make sure phonopy is installed and in PATH")
        return False


def prepare_staticpoint(relaxed_example_path, output_dir=None, base_dir=None, supercell_dim="3 3 1", generate_displacements=True):
    """
    Prepare a static-point calculation directory from a relaxed bilayer example.
    
    Parameters:
    -----------
    relaxed_example_path : str or Path
        Path to the relaxed bilayer example directory (e.g., "bilayer_examples/MoS2_bilayer_3R")
    output_dir : str or Path, optional
        Output directory for static-point calculation. If None, uses base_dir/<bilayer_name>_staticpoint
    base_dir : Path, optional
        Base directory for static-point examples. Default: ../phonopy_bilayer_examples
    
    Returns:
    --------
    dict : Information about the prepared static-point example
    """
    relaxed_example_path = Path(relaxed_example_path).resolve()
    
    # Get bilayer name from directory name
    bilayer_name = relaxed_example_path.name
    
    # Default base_dir
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "phonopy_bilayer_examples"
    else:
        base_dir = Path(base_dir)
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output directory
    if output_dir is None:
        output_dir = base_dir / f"{bilayer_name}_staticpoint"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check that CONTCAR exists
    contcar = relaxed_example_path / "CONTCAR"
    if not contcar.exists():
        raise FileNotFoundError(f"CONTCAR not found in {relaxed_example_path}")
    
    # Check that POTCAR exists (KPOINTS may come from static template)
    potcar = relaxed_example_path / "POTCAR"
    kpoints_relaxed = relaxed_example_path / "KPOINTS"

    if not potcar.exists():
        raise FileNotFoundError(f"POTCAR not found in {relaxed_example_path}")

    # Get template paths
    template_dir = Path(__file__).parent.parent.parent / "common" / "staticpoint_templates"
    incar_template = template_dir / "INCAR"
    bat_template = template_dir / "bat"
    kpoints_template = template_dir / "KPOINTS"

    if not incar_template.exists():
        raise FileNotFoundError(f"INCAR template not found at {incar_template}")
    if not bat_template.exists():
        raise FileNotFoundError(f"bat template not found at {bat_template}")
    if not kpoints_template.exists() and not kpoints_relaxed.exists():
        raise FileNotFoundError(
            f"KPOINTS not found: need {kpoints_template} or {kpoints_relaxed}"
        )
    
    # Copy CONTCAR → POSCAR, grouping atoms by species so phonopy and VASP
    # use the same atom ordering in phonopy_disp.yaml and vasprun.xml.
    poscar_path = output_dir / "POSCAR"
    if reorder_poscar_for_phonopy(contcar, poscar_path):
        print(f"  Copied CONTCAR → POSCAR (grouped atoms by species for phonopy)")
    else:
        print(f"  Copied CONTCAR → POSCAR")
    
    # Copy POTCAR
    shutil.copy2(potcar, output_dir / "POTCAR")
    print(f"  Copied POTCAR")
    
    # Copy KPOINTS: prefer static-point template (separate mesh from relaxation)
    if kpoints_template.exists():
        shutil.copy2(kpoints_template, output_dir / "KPOINTS")
        print(f"  Copied KPOINTS (from staticpoint_templates)")
    else:
        shutil.copy2(kpoints_relaxed, output_dir / "KPOINTS")
        print(f"  Copied KPOINTS (from relaxed example)")
    
    # Copy and customize INCAR
    customize_incar(incar_template, output_dir / "INCAR", bilayer_name)
    print(f"  Created customized INCAR")
    
    # Copy bat
    shutil.copy2(bat_template, output_dir / "bat")
    print(f"  Copied bat script")
    
    # Generate phonopy displacements
    if generate_displacements:
        print(f"  Generating phonopy displacements (supercell: {supercell_dim})...")
        if generate_phonopy_displacements(output_dir, supercell_dim):
            print(f"  ✓ Generated phonopy displacements")
        else:
            print(f"  ✗ Failed to generate phonopy displacements")
    
    return {
        'bilayer': bilayer_name,
        'output_dir': output_dir,
        'relaxed_example': relaxed_example_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare static-point calculation directory from relaxed bilayer example"
    )
    parser.add_argument(
        'example_path',
        nargs='?',
        type=str,
        default=None,
        help="Path to relaxed bilayer example directory (e.g., 'bilayer_examples/MoS2_bilayer_3R' or 'MoS2_bilayer_3R')"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Output directory for static-point calculation (default: <base_dir>/<bilayer>_staticpoint)"
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help="Base directory for static-point examples (default: ../phonopy_bilayer_examples)"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Process all examples in bilayer_examples directory"
    )
    parser.add_argument(
        '--dim',
        type=str,
        default="3 3 1",
        help='Supercell dimensions for phonopy (default: "3 3 1")'
    )
    parser.add_argument(
        '--no-displacements',
        action='store_true',
        help="Skip phonopy displacement generation"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Process all examples
        if args.base_dir is None:
            base_dir = Path(__file__).parent.parent.parent / "bilayer_examples"
        else:
            base_dir = Path(args.base_dir).parent / "bilayer_examples"
        
        if not base_dir.exists():
            print(f"Error: {base_dir} does not exist")
            sys.exit(1)
        
        examples = [d for d in base_dir.iterdir() if d.is_dir()]
        print(f"Processing {len(examples)} bilayer examples...\n")
        
        results = []
        for example_dir in examples:
            try:
                print(f"Processing: {example_dir.name}")
                result = prepare_staticpoint(
                    example_dir,
                    base_dir=args.base_dir,
                    supercell_dim=args.dim,
                    generate_displacements=not args.no_displacements
                )
                results.append(result)
                print(f"  ✓ Created static-point directory: {result['output_dir']}\n")
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                continue
        
        print(f"\n{'='*60}")
        print(f"Summary: Prepared {len(results)}/{len(examples)} static-point examples")
        print(f"{'='*60}")
    else:
        if args.example_path is None:
            print("Error: Must provide example_path or use --all flag")
            parser.print_help()
            sys.exit(1)
        
        # Process single example
        example_path = Path(args.example_path)
        
        # If just a name, assume it's in bilayer_examples
        if not example_path.is_absolute() and not example_path.parent.name:
            base_dir = Path(__file__).parent.parent.parent / "bilayer_examples"
            example_path = base_dir / example_path
        
        try:
            print(f"Preparing static-point calculation from: {example_path}")
            result = prepare_staticpoint(
                example_path,
                args.output_dir,
                args.base_dir,
                supercell_dim=args.dim,
                generate_displacements=not args.no_displacements
            )
            print(f"\n✓ Successfully created static-point directory:")
            print(f"  {result['output_dir']}")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

