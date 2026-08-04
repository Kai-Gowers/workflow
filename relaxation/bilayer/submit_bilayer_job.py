#!/usr/bin/env python3
"""
Submit VASP relaxation jobs for bilayer examples.

Similar to submit_monolayer_job.py but adapted for bilayer example naming
convention. Thin wrapper around common/submit_job_impl.py -- identical logic
to relaxation/monolayer/submit_monolayer_job.py, parameterized only by which
*_examples directory holds the example.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
from submit_job_impl import submit_example_job, main as _main  # noqa: E402


def submit_bilayer_job(example_path, dry_run=False):
    """
    Submit a VASP job for a bilayer example.

    Parameters:
    -----------
    example_path : str or Path
        Path to bilayer example directory (e.g., "bilayer_examples/MoS2_bilayer_3R")
        or just the example name (e.g., "MoS2_bilayer_3R")
    dry_run : bool
        If True, print what would be done without actually submitting

    Returns:
    --------
    tuple : (success: bool, job_id: str or None, message: str)
    """
    return submit_example_job(example_path, "bilayer_examples", dry_run=dry_run)


def main():
    _main(
        submit_bilayer_job,
        "submit_bilayer_job.py",
        "Submit VASP relaxation job for a bilayer example",
        "MoS2_bilayer_3R",
        "bilayer_examples",
    )


if __name__ == "__main__":
    main()
