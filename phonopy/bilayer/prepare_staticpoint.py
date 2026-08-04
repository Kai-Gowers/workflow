#!/usr/bin/env python3
"""
Prepare static-point calculation directories from relaxed bilayer examples.

This script:
1. Takes CONTCAR from a relaxed bilayer example
2. Creates a new clean directory for static-point calculation
3. Copies CONTCAR → POSCAR, grouping atoms by species for phonopy/VASP compatibility
4. Copies POTCAR from the original example; KPOINTS from staticpoint_templates if present, else from the relaxed example
5. Uses INCAR and bat from staticpoint_templates (with customized SYSTEM line)
6. Generates phonopy displacements using phonopy --dim="4 4 1" -d -c POSCAR

Thin wrapper around common/prepare_staticpoint_impl.py -- identical logic to
phonopy/monolayer/prepare_staticpoint.py, except bilayer CONTCARs need
species reordering (reorder_species=True) since bilayer relaxation can
produce interleaved species blocks.
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

EXAMPLES_ROOT_NAME = "phonopy_bilayer_examples"
EXAMPLES_DIRNAME = "bilayer_examples"


def prepare_staticpoint(relaxed_example_path, output_dir=None, base_dir=None,
                         supercell_dim="4 4 1", generate_displacements=True):
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
    return _prepare_staticpoint(
        relaxed_example_path,
        EXAMPLES_ROOT_NAME,
        name_key="bilayer",
        reorder_species=True,
        output_dir=output_dir,
        base_dir=base_dir,
        supercell_dim=supercell_dim,
        generate_displacements=generate_displacements,
    )


def main():
    _main(
        prepare_staticpoint,
        EXAMPLES_DIRNAME,
        "Prepare static-point calculation directory from relaxed bilayer example",
        "MoS2_bilayer_3R",
    )


if __name__ == "__main__":
    main()
