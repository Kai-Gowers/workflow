#!/usr/bin/env python3
"""Shared helpers for mapping SLURM job IDs to workflow directories."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


WORKFLOW_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = WORKFLOW_ROOT / "data" / "job_registry.json"

SEARCH_ROOTS = (
    "monolayer_examples",
    "bilayer_examples",
    "phonopy_monolayer_examples",
    "phonopy_bilayer_examples",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_job_name(label: str) -> str:
    """Return a SLURM-safe job name (max 64 chars)."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip())
    return (name or "vasp")[:64]


def load_registry() -> Dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with open(REGISTRY_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_registry(registry: Dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
        f.write("\n")


def record_job(
    job_id: str,
    work_dir: Path,
    *,
    job_type: str = "relaxation",
    label: Optional[str] = None,
) -> None:
    """Persist job_id -> directory mapping for later lookup."""
    work_dir = work_dir.resolve()
    try:
        rel_path = work_dir.relative_to(WORKFLOW_ROOT)
        path_str = str(rel_path)
    except ValueError:
        path_str = str(work_dir)

    registry = load_registry()
    registry[str(job_id)] = {
        "path": path_str,
        "label": label or work_dir.name,
        "job_type": job_type,
        "submitted_at": _now_iso(),
    }
    save_registry(registry)

    latest = work_dir / "latest_job_id"
    latest.write_text(f"{job_id}\n")


def submit_bat(
    work_dir: Path,
    *,
    job_name: str,
    job_type: str = "relaxation",
    label: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[bool, Optional[str], str]:
    """
    Submit sbatch with a descriptive job name and record the mapping.

    Returns (success, job_id, message).
    """
    work_dir = Path(work_dir).resolve()
    bat_file = work_dir / "bat"
    if not bat_file.exists():
        return False, None, f"Batch script not found: {bat_file}"

    safe_name = sanitize_job_name(job_name)
    cmd = ["sbatch", f"--job-name={safe_name}", "bat"]

    if dry_run:
        return True, None, f"Would run in {work_dir}: {' '.join(cmd)}"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        msg = e.stderr or e.stdout or str(e)
        return False, None, f"Error submitting job: {msg}"
    except FileNotFoundError:
        return False, None, "sbatch command not found. Are you on a cluster node with SLURM?"

    output = result.stdout.strip()
    job_id = None
    if "Submitted batch job" in output:
        job_id = output.split()[-1]
        record_job(job_id, work_dir, job_type=job_type, label=label or job_name)

    return True, job_id, output


def build_slurm_output_index() -> Dict[str, Path]:
    """Map job IDs to directories via slurm-<jobid>.out files."""
    index: Dict[str, Path] = {}
    for root_name in SEARCH_ROOTS:
        root = WORKFLOW_ROOT / root_name
        if not root.exists():
            continue
        for out_file in root.rglob("slurm-*.out"):
            match = re.fullmatch(r"slurm-(\d+)\.out", out_file.name)
            if match:
                index[match.group(1)] = out_file.parent
    return index


def resolve_job(job_id: str) -> Optional[dict]:
    """Resolve a job ID to path/label using registry and slurm output files."""
    job_id = str(job_id)
    registry = load_registry()
    if job_id in registry:
        entry = registry[job_id].copy()
        entry["source"] = "registry"
        return entry

    slurm_index = build_slurm_output_index()
    if job_id in slurm_index:
        path = slurm_index[job_id]
        try:
            rel_path = path.relative_to(WORKFLOW_ROOT)
            path_str = str(rel_path)
        except ValueError:
            path_str = str(path)
        return {
            "path": path_str,
            "label": path.name,
            "job_type": "unknown",
            "source": "slurm-output",
        }

    return None


def _path_from_workdir(work_dir: str) -> str:
    if not work_dir:
        return ""
    path = Path(work_dir)
    try:
        return str(path.relative_to(WORKFLOW_ROOT))
    except ValueError:
        if "workflow" in work_dir:
            idx = work_dir.find("workflow")
            suffix = work_dir[idx + len("workflow") :].lstrip("/")
            return suffix
        return work_dir


def get_queued_jobs(username: Optional[str] = None) -> List[dict]:
    """Return current queue entries for the user, enriched with workflow paths."""
    import os

    username = username or os.environ.get("USER") or os.environ.get("USERNAME")
    if not username:
        return []

    try:
        result = subprocess.run(
            [
                "squeue",
                "-u",
                username,
                "-h",
                "-o",
                "%i|%j|%t|%M|%D|%R|%Z",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    jobs = []
    for line in result.stdout.splitlines():
        parts = line.split("|", 6)
        if len(parts) < 7:
            continue
        job_id, name, state, time_str, nodes, reason, work_dir = parts
        info = resolve_job(job_id) or {}
        jobs.append(
            {
                "job_id": job_id,
                "name": name,
                "state": state,
                "time": time_str,
                "nodes": nodes,
                "reason": reason,
                "work_dir": work_dir,
                "path": info.get("path") or _path_from_workdir(work_dir),
                "label": info.get("label") or name,
                "job_type": info.get("job_type", ""),
                "source": info.get("source", "squeue"),
            }
        )
    return jobs
