#!/usr/bin/env python3
"""
Shared implementation for preparing phonopy staticpoint directories from a
relaxed monolayer or bilayer example.

Used identically by phonopy/monolayer/prepare_staticpoint.py and
phonopy/bilayer/prepare_staticpoint.py. The only real (non-cosmetic)
difference between monolayer and bilayer is `reorder_species`: bilayer
relaxation can produce a CONTCAR with interleaved species blocks, which
breaks phonopy/VASP atom-ordering assumptions, so bilayer callers pass
reorder_species=True to re-group atoms by species when writing POSCAR;
monolayer callers pass False (never interleaved, plain copy).
"""

from pathlib import Path
import argparse
import shutil
import subprocess
import sys

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "phonopy"))
from poscar_utils import reorder_poscar_for_phonopy  # noqa: E402

from incar_utils import customize_incar as _customize_incar
from interlayer_check import check_bilayer_interlayer_gap


def customize_incar(template_path, output_path, name):
    return _customize_incar(template_path, output_path, name, suffix="phonon")


def generate_phonopy_displacements(work_dir, supercell_dim="4 4 1"):
    """
    Generate phonopy displacements in the given directory.

    Parameters:
    -----------
    work_dir : Path
        Working directory containing POSCAR
    supercell_dim : str
        Supercell dimensions (default: "4 4 1")

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


def prepare_staticpoint(
    relaxed_example_path,
    examples_root_name,
    name_key,
    reorder_species,
    output_dir=None,
    base_dir=None,
    supercell_dim="4 4 1",
    generate_displacements=True,
):
    """
    Prepare a static-point calculation directory from a relaxed example.

    Parameters:
    -----------
    relaxed_example_path : str or Path
        Path to the relaxed example directory (e.g., "monolayer_examples/MoS2")
    examples_root_name : str
        Default staticpoint base dir name ("phonopy_monolayer_examples" or
        "phonopy_bilayer_examples").
    name_key : str
        Key used for the example name in the returned dict ("material" or "bilayer").
    reorder_species : bool
        If True, re-group CONTCAR atoms by species when writing POSCAR (bilayer);
        if False, plain copy (monolayer).
    output_dir : str or Path, optional
        Output directory for static-point calculation. If None, uses base_dir/<name>_staticpoint
    base_dir : Path, optional
        Base directory for static-point examples. Default: ../<examples_root_name>

    Returns:
    --------
    dict : Information about the prepared static-point example
    """
    relaxed_example_path = Path(relaxed_example_path).resolve()

    # Get example name from directory name
    name = relaxed_example_path.name

    # Default base_dir
    if base_dir is None:
        base_dir = WORKFLOW_ROOT / examples_root_name
    else:
        base_dir = Path(base_dir)

    base_dir.mkdir(parents=True, exist_ok=True)

    # Determine output directory
    if output_dir is None:
        output_dir = base_dir / f"{name}_staticpoint"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check that CONTCAR exists
    contcar = relaxed_example_path / "CONTCAR"
    if not contcar.exists():
        raise FileNotFoundError(f"CONTCAR not found in {relaxed_example_path}")

    # Bilayers only: guard against the trapped-relaxation bug (see
    # feedback_trapped_bilayer_relaxation_bug memory) -- VASP's own force
    # convergence can be satisfied for some material families while the
    # layers are still ~2-3x too far apart, which later trivially passes the
    # phonon-stability check as two decoupled monolayers. Fail loudly here,
    # before the expensive displacement + static-VASP phase, instead of
    # silently promoting a decoupled pair to FINAL_RESULTS_HEALTHY.
    if reorder_species:
        gap = check_bilayer_interlayer_gap(contcar, name)
        print(f"  Interlayer gap check passed ({gap:.3f} Å)")

    # Check that POTCAR exists (KPOINTS may come from static template)
    potcar = relaxed_example_path / "POTCAR"
    kpoints_relaxed = relaxed_example_path / "KPOINTS"

    if not potcar.exists():
        raise FileNotFoundError(f"POTCAR not found in {relaxed_example_path}")

    # Get template paths
    template_dir = WORKFLOW_ROOT / "common" / "staticpoint_templates"
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

    # Copy CONTCAR -> POSCAR, optionally grouping atoms by species so phonopy
    # and VASP use the same atom ordering in phonopy_disp.yaml and vasprun.xml.
    poscar_path = output_dir / "POSCAR"
    if reorder_species:
        if reorder_poscar_for_phonopy(contcar, poscar_path):
            print(f"  Copied CONTCAR → POSCAR (grouped atoms by species for phonopy)")
        else:
            print(f"  Copied CONTCAR → POSCAR")
    else:
        shutil.copy2(contcar, poscar_path)
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
    customize_incar(incar_template, output_dir / "INCAR", name)
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
        name_key: name,
        'output_dir': output_dir,
        'relaxed_example': relaxed_example_path
    }


def main(prepare_fn, examples_dirname, description, example_hint):
    """CLI entry point shared by the monolayer/bilayer wrapper scripts.

    Parameters:
    -----------
    prepare_fn : callable
        The wrapper's own prepare_staticpoint-style function, called as
        prepare_fn(example_path, output_dir=, base_dir=, supercell_dim=, generate_displacements=).
    examples_dirname : str
        "monolayer_examples" or "bilayer_examples"
    description : str
        argparse description.
    example_hint : str
        A representative example name for --help text (e.g. "MoS2" or "MoS2_bilayer_3R").
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'example_path',
        nargs='?',
        type=str,
        default=None,
        help=f"Path to relaxed example directory (e.g., '{examples_dirname}/{example_hint}' or '{example_hint}')"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Output directory for static-point calculation (default: <base_dir>/<name>_staticpoint)"
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help=f"Base directory for static-point examples (default: ../phonopy_{examples_dirname})"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help=f"Process all examples in {examples_dirname} directory"
    )
    parser.add_argument(
        '--dim',
        type=str,
        default="4 4 1",
        help='Supercell dimensions for phonopy (default: "4 4 1")'
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
            base_dir = WORKFLOW_ROOT / examples_dirname
        else:
            base_dir = Path(args.base_dir).parent / examples_dirname

        if not base_dir.exists():
            print(f"Error: {base_dir} does not exist")
            sys.exit(1)

        examples = [d for d in base_dir.iterdir() if d.is_dir()]
        print(f"Processing {len(examples)} examples...\n")

        results = []
        for example_dir in examples:
            try:
                print(f"Processing: {example_dir.name}")
                result = prepare_fn(
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

        # If just a name, assume it's in the default examples directory
        if not example_path.is_absolute() and not example_path.parent.name:
            base_dir = WORKFLOW_ROOT / examples_dirname
            example_path = base_dir / example_path

        try:
            print(f"Preparing static-point calculation from: {example_path}")
            result = prepare_fn(
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
