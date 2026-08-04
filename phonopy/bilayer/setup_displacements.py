#!/usr/bin/env python3
"""
Set up and submit VASP jobs for phonopy displacement calculations (bilayer).

Thin wrapper around common/setup_displacements_impl.py -- identical logic to
phonopy/monolayer/setup_displacements.py, parameterized only by which
phonopy_*_examples directory holds the staticpoint folders.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
from setup_displacements_impl import (  # noqa: E402
    setup_and_submit_displacements as _setup_and_submit_displacements,
    find_displacement_poscars,
    setup_displacement_folder,
    submit_displacement_job,
    main as _main,
)

EXAMPLES_ROOT_NAME = "phonopy_bilayer_examples"
EXAMPLE_NAME = "MoS2_TaTe2_3R_staticpoint"


def setup_and_submit_displacements(staticpoint_path, submit=True, dry_run=False):
    return _setup_and_submit_displacements(
        staticpoint_path, EXAMPLES_ROOT_NAME, submit=submit, dry_run=dry_run
    )


def main():
    _main(EXAMPLES_ROOT_NAME, EXAMPLE_NAME)


if __name__ == "__main__":
    main()
