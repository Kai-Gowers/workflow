#!/usr/bin/env python3
"""
Prepare static-point calculation directories from relaxed monolayer examples.

This script:
1. Takes CONTCAR from a relaxed monolayer example
2. Creates a new clean directory for static-point calculation
3. Copies CONTCAR → POSCAR
4. Copies POTCAR from the original example; KPOINTS from staticpoint_templates if present, else from the relaxed example
5. Uses INCAR and bat from staticpoint_templates (with customized SYSTEM line)
6. Generates phonopy displacements using phonopy --dim="4 4 1" -d -c POSCAR

Thin wrapper around common/prepare_staticpoint_impl.py -- identical logic to
phonopy/bilayer/prepare_staticpoint.py, except monolayer CONTCARs never need
species reordering (reorder_species=False).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
from prepare_staticpoint_impl import (  # noqa: E402
    prepare_staticpoint as _prepare_staticpoint,
    generate_phonopy_displacements,
    customize_incar,
    main as _main,
)

EXAMPLES_ROOT_NAME = "phonopy_monolayer_examples"
EXAMPLES_DIRNAME = "monolayer_examples"


def prepare_staticpoint(relaxed_example_path, output_dir=None, base_dir=None,
                         supercell_dim="4 4 1", generate_displacements=True):
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
    return _prepare_staticpoint(
        relaxed_example_path,
        EXAMPLES_ROOT_NAME,
        name_key="material",
        reorder_species=False,
        output_dir=output_dir,
        base_dir=base_dir,
        supercell_dim=supercell_dim,
        generate_displacements=generate_displacements,
    )


def main():
    _main(
        prepare_staticpoint,
        EXAMPLES_DIRNAME,
        "Prepare static-point calculation directory from relaxed monolayer example",
        "MoS2",
    )


if __name__ == "__main__":
    main()
