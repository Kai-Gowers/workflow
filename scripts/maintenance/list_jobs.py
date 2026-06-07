#!/usr/bin/env python3
"""
List SLURM jobs with their corresponding workflow material/directory.

Usage:
  python3 list_jobs.py
  python3 list_jobs.py --job-id 2504059
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from job_tracking import WORKFLOW_ROOT, get_queued_jobs, resolve_job  # noqa: E402


def _print_lookup(job_id: str) -> int:
    info = resolve_job(job_id)
    if not info:
        print(f"No workflow mapping found for job {job_id}")
        print("Tip: older jobs may only appear after slurm-<id>.out is created.")
        return 1

    path = info.get("path", "")
    label = info.get("label", "")
    job_type = info.get("job_type", "")
    source = info.get("source", "")
    submitted = info.get("submitted_at", "")

    print(f"Job ID:   {job_id}")
    print(f"Label:    {label}")
    if job_type:
        print(f"Type:     {job_type}")
    print(f"Path:     {path}")
    if path:
        print(f"Full:     {WORKFLOW_ROOT / path}")
    if submitted:
        print(f"Submitted:{submitted}")
    print(f"Source:   {source}")
    return 0


def _print_queue(jobs: list[dict]) -> int:
    if not jobs:
        print("No jobs in queue for your user.")
        return 0

    rows = []
    for job in jobs:
        rows.append(
            (
                job["job_id"],
                job["state"],
                job["label"],
                job.get("path") or job.get("work_dir", ""),
                job["time"],
                job["nodes"],
                job.get("reason", ""),
            )
        )

    headers = ("JOBID", "ST", "MATERIAL", "PATH", "TIME", "NODES", "REASON")
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print(fmt_row(headers))
    print(fmt_row("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))

    unnamed = sum(1 for j in jobs if j["name"] in ("vasp", "vasp_phonon"))
    if unnamed:
        print(
            f"\nNote: {unnamed} job(s) still use generic SLURM names. "
            "Resubmit via submit_monolayer_job.py to get material names in squeue."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map SLURM job IDs to workflow materials/directories.",
    )
    parser.add_argument(
        "--job-id",
        "--lookup",
        dest="job_id",
        metavar="ID",
        help="Look up a specific job ID",
    )
    args = parser.parse_args()

    if args.job_id:
        sys.exit(_print_lookup(args.job_id))

    jobs = get_queued_jobs()
    sys.exit(_print_queue(jobs))


if __name__ == "__main__":
    main()
