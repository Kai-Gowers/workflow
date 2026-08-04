#!/usr/bin/env python3
"""
Submit VASP relaxation jobs to the cluster.

This script submits a training example job to the cluster using sbatch.
It changes to the example directory and runs 'sbatch bat' to submit the job.

Thin wrapper around common/submit_job_impl.py -- identical logic to
relaxation/bilayer/submit_bilayer_job.py, parameterized only by which
*_examples directory holds the example.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
from submit_job_impl import submit_example_job, main as _main  # noqa: E402


def submit_job(example_path, dry_run=False):
    """
    Submit a VASP job for a training example.

    Parameters:
    -----------
    example_path : str or Path
        Path to the monolayer example directory (e.g., "monolayer_examples/MoS2")
        or just the example name (e.g., "MoS2")
    dry_run : bool, optional
        If True, print what would be done without actually submitting (default: False)

    Returns:
    --------
    tuple : (success: bool, job_id: str or None, message: str)
    """
    return submit_example_job(example_path, "monolayer_examples", dry_run=dry_run)


def main():
    _main(
        submit_job,
        "submit_monolayer_job.py",
        "Submit VASP relaxation job for a training example",
        "0",
        "monolayer_examples",
    )


if __name__ == "__main__":
    main()
