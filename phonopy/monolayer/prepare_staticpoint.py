#!/usr/bin/env python3
"""
Prepare static-point calculation directories from relaxed monolayer examples.

This script:
1. Takes CONTCAR from a relaxed monolayer example
2. Creates a new clean directory for static-point calculation
3. Copies CONTCAR → POSCAR
4. Copies POTCAR from the original example; KPOINTS from staticpoint_templates if present, else from the relaxed example
5. Uses INCAR and bat from staticpoint_templates (with customized SYSTEM line)
6. Generates phonopy displacements using phonopy --dim="3 3 1" -d -c POSCAR
"""

from pathlib import Path
import shutil
import argparse
import sys
import subprocess


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
                customized_lines.append(f"SYSTEM = {material_name} phonon\n")
                system_found = True
            # Skip duplicate SYSTEM lines
        else:
            customized_lines.append(line)
    
    # If no SYSTEM line found, add one at the beginning
    if not system_found:
        customized_lines.insert(0, f"SYSTEM = {material_name} phonon\n")
    
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
    # phonopy v4: displacement setup moved to phonopy-init
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
    Prepare a static-point calculation directory from a relaxed monolayer example.
    
    Parameters:
    -----------
    relaxed_example_path : str or Path
        Path to the relaxed monolayer example directory (e.g., "monolayer_examples/MoS2")
    output_dir : str or Path, optional
        Output directory for static-point calculation. If None, uses base_dir/<material_name>_staticpoint
    base_dir : Path, optional
        Base directory for static-point examples. Default: ../phonopy_monolayer_examples
    
    Returns:
    --------
    dict : Information about the prepared static-point example
    """
    relaxed_example_path = Path(relaxed_example_path).resolve()
    
    # Get material name from directory name
    material_name = relaxed_example_path.name
    
    # Default base_dir
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "phonopy_monolayer_examples"
    else:
        base_dir = Path(base_dir)
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output directory
    if output_dir is None:
        output_dir = base_dir / f"{material_name}_staticpoint"
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
    
    # Copy CONTCAR → POSCAR
    shutil.copy2(contcar, output_dir / "POSCAR")
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
    customize_incar(incar_template, output_dir / "INCAR", material_name)
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
        'material': material_name,
        'output_dir': output_dir,
        'relaxed_example': relaxed_example_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare static-point calculation directory from relaxed monolayer example"
    )
    parser.add_argument(
        'example_path',
        type=str,
        help="Path to relaxed monolayer example directory (e.g., 'monolayer_examples/MoS2' or 'MoS2')"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Output directory for static-point calculation (default: <base_dir>/<material>_staticpoint)"
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help="Base directory for static-point examples (default: ../phonopy_monolayer_examples)"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Process all examples in monolayer_examples directory"
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
            base_dir = Path(__file__).parent.parent.parent / "monolayer_examples"
        else:
            base_dir = Path(args.base_dir).parent / "monolayer_examples"
        
        if not base_dir.exists():
            print(f"Error: {base_dir} does not exist")
            sys.exit(1)
        
        examples = [d for d in base_dir.iterdir() if d.is_dir()]
        print(f"Processing {len(examples)} monolayer examples...\n")
        
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
        # Process single example
        example_path = Path(args.example_path)
        
        # If just a name, assume it's in monolayer_examples
        if not example_path.is_absolute() and not example_path.parent.name:
            base_dir = Path(__file__).parent.parent.parent / "monolayer_examples"
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

