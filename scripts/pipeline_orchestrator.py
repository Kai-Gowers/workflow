#!/usr/bin/env python3
"""
Pipeline orchestrator: runs each batch through the full workflow
(relax → phonopy displacements → postprocess band structures) one at a time.

Usage:
    python3 scripts/pipeline_orchestrator.py [--dry-run] [--reset]

Run every 2 minutes via /loop to advance the pipeline and restart stuck jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE    = WORKFLOW_ROOT / "data" / "pipeline_state.json"
REGISTRY_FILE = WORKFLOW_ROOT / "data" / "job_registry.json"
BATCHES_FILE  = WORKFLOW_ROOT / "data" / "batches.json"

PHONON_PYTHON = "/home/gowers/miniconda3/envs/phonon-env/bin/python3"

# ── Thresholds ─────────────────────────────────────────────────────────────────

WAVECAR_MARKER      = "WAVECAR not read"
STALE_SECONDS       = 5 * 60    # OUTCAR not updated → stuck
MIN_RUNTIME_SECONDS = 10 * 60   # don't flag stuck before this runtime

# ── SLURM helpers ──────────────────────────────────────────────────────────────

def get_queue() -> set[str]:
    """Return set of currently queued/running SLURM job IDs for this user."""
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "gowers"), "-h", "-o", "%i"],
            capture_output=True, text=True, timeout=30,
        )
        return {line.strip() for line in r.stdout.splitlines() if line.strip()}
    except Exception:
        return set()


def get_queue_details() -> dict[str, dict]:
    """Return {job_id: {state, time, reason}} for all queued jobs."""
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "gowers"), "-h", "-o", "%i|%t|%M|%R"],
            capture_output=True, text=True, timeout=30,
        )
        out: dict[str, dict] = {}
        for line in r.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 4:
                jid, state, elapsed, reason = parts
                out[jid.strip()] = {
                    "state": state.strip(),
                    "time": elapsed.strip(),
                    "reason": reason.strip(),
                }
        return out
    except Exception:
        return {}


def parse_slurm_minutes(t: str) -> int:
    """
    Convert SLURM elapsed time string to minutes.

    Formats:
      MM:SS          → 2 colon-parts  → minutes = int(parts[0])
      HH:MM:SS       → 3 colon-parts  → minutes = int(parts[0])*60 + int(parts[1])
      D-HH:MM:SS     → dash prefix    → days*1440 + HH*60 + MM
    """
    if not t or t in ("0:00", "INVALID"):
        return 0
    try:
        if "-" in t:
            # D-HH:MM:SS
            day_part, rest = t.split("-", 1)
            colon_parts = rest.split(":")
            days = int(day_part)
            hours = int(colon_parts[0]) if len(colon_parts) >= 1 else 0
            mins  = int(colon_parts[1]) if len(colon_parts) >= 2 else 0
            return days * 1440 + hours * 60 + mins
        colon_parts = t.split(":")
        if len(colon_parts) == 2:
            # MM:SS — first part is minutes
            return int(colon_parts[0])
        # HH:MM:SS
        return int(colon_parts[0]) * 60 + int(colon_parts[1])
    except (ValueError, IndexError):
        return 0

# ── Registry helpers ───────────────────────────────────────────────────────────

def load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {}
    with REGISTRY_FILE.open() as f:
        return json.load(f)


def save_registry(reg: dict) -> None:
    with REGISTRY_FILE.open("w") as f:
        json.dump(reg, f, indent=2)


def new_registry_jobs(before_ids: set[str]) -> list[str]:
    """Return job IDs added to registry since before_ids snapshot."""
    reg = load_registry()
    return [jid for jid in reg if jid not in before_ids]


def job_work_dir(job_id: str) -> Optional[Path]:
    reg = load_registry()
    entry = reg.get(job_id)
    if not entry:
        return None
    return WORKFLOW_ROOT / entry["path"]

# ── Stuck detection & restart ──────────────────────────────────────────────────

def is_stuck(job_id: str, details: dict[str, dict]) -> Optional[str]:
    """
    Return a reason string if the job appears stuck, else None.

    Checks:
    1. Last non-empty line of slurm-JOBID.out contains "WAVECAR not read"
    2. Runtime >= MIN_RUNTIME_SECONDS AND OUTCAR stale > STALE_SECONDS AND no vasprun.xml
    """
    work_dir = job_work_dir(job_id)
    if work_dir is None:
        return None

    # Check 1: WAVECAR hang
    slurm_out = work_dir / f"slurm-{job_id}.out"
    if slurm_out.exists():
        try:
            lines = [l.strip() for l in slurm_out.read_text().splitlines() if l.strip()]
            if lines and WAVECAR_MARKER in lines[-1]:
                return "WAVECAR not read"
        except OSError:
            pass

    # Check 2: stale OUTCAR (also catches incomplete vasprun.xml stubs < 10 KB)
    job_info = details.get(job_id)
    if job_info:
        runtime_min = parse_slurm_minutes(job_info.get("time", "0:00"))
        if runtime_min * 60 >= MIN_RUNTIME_SECONDS:
            outcar = work_dir / "OUTCAR"
            vasprun = work_dir / "vasprun.xml"
            vasprun_incomplete = (
                not vasprun.exists()
                or vasprun.stat().st_size < 10_000
            )
            if outcar.exists() and vasprun_incomplete:
                age = time.time() - outcar.stat().st_mtime
                if age >= STALE_SECONDS:
                    return f"stale OUTCAR ({int(age/60)}min)"

    return None


def restart_job(job_id: str, dry_run: bool) -> Optional[str]:
    """
    Cancel job_id, resubmit via `sbatch bat` in its work directory.
    Returns new job ID on success, None on failure.
    """
    work_dir = job_work_dir(job_id)
    if work_dir is None:
        print(f"    ✗ Cannot find work dir for job {job_id}")
        return None

    print(f"    → Restarting job {job_id} in {work_dir}")
    if dry_run:
        print(f"      [dry-run] would: scancel {job_id} && sbatch bat")
        return None

    subprocess.run(["scancel", job_id], capture_output=True)

    r = subprocess.run(
        ["sbatch", "bat"],
        cwd=str(work_dir),
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"    ✗ sbatch failed: {r.stderr.strip()}")
        return None

    # Extract new job ID from "Submitted batch job NNNNNN"
    new_jid = None
    for token in r.stdout.split():
        if token.isdigit():
            new_jid = token
    if not new_jid:
        print(f"    ✗ Could not parse job ID from: {r.stdout.strip()}")
        return None

    # Update registry
    reg = load_registry()
    old_entry = reg.get(job_id, {})
    reg[new_jid] = {
        "job_type": old_entry.get("job_type", "phonopy-displacement"),
        "label": old_entry.get("label", work_dir.name),
        "path": str(work_dir.relative_to(WORKFLOW_ROOT)),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    save_registry(reg)

    print(f"    ✓ Restarted as job {new_jid}")
    return new_jid

# ── State persistence ──────────────────────────────────────────────────────────

def build_initial_state() -> list[dict]:
    batches = json.loads(BATCHES_FILE.read_text())
    seq = []
    # Monolayer batches first
    for b in batches["monolayer_batches"]:
        seq.append({
            "kind": "monolayer",
            "batch": b["batch_number"],
            "stage": "not_started",
            "job_ids": [],
            "submitted_at": None,
        })
    # Then bilayer batches
    for b in batches["bilayer_batches"]:
        seq.append({
            "kind": "bilayer",
            "batch": b["batch_number"],
            "stage": "not_started",
            "job_ids": [],
            "submitted_at": None,
        })
    return seq


def load_state() -> list[dict]:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        if data:
            return data
    return build_initial_state()


def save_state(seq: list[dict]) -> None:
    with STATE_FILE.open("w") as f:
        json.dump(seq, f, indent=2)

# ── Command runner ─────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], dry_run: bool) -> tuple[int, str, str]:
    """Run a command, print indented output, return (returncode, stdout, stderr)."""
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        print("  [dry-run] skipping")
        return 0, "", ""
    r = subprocess.run(
        cmd,
        cwd=str(WORKFLOW_ROOT),
        capture_output=True, text=True,
    )
    for line in r.stdout.splitlines():
        print(f"    {line}")
    if r.returncode != 0 and r.stderr:
        for line in r.stderr.splitlines():
            print(f"    [err] {line}")
    return r.returncode, r.stdout, r.stderr

# ── Pipeline stage logic ───────────────────────────────────────────────────────

def check_and_restart_stuck(
    task: dict,
    queued: set[str],
    details: dict[str, dict],
    dry_run: bool,
) -> None:
    """Check each running job for stuck conditions and restart if needed."""
    still_running = [j for j in task["job_ids"] if j in queued]
    new_ids = list(task["job_ids"])  # copy

    for jid in still_running:
        reason = is_stuck(jid, details)
        if reason:
            print(f"  ⚠ Job {jid} stuck ({reason}) — restarting")
            new_jid = restart_job(jid, dry_run)
            if new_jid:
                try:
                    new_ids.remove(jid)
                except ValueError:
                    pass
                new_ids.append(new_jid)

    task["job_ids"] = new_ids


def process_active_task(
    task: dict,
    queued: set[str],
    details: dict[str, dict],
    dry_run: bool,
) -> None:
    """Advance a single task through its pipeline stages."""
    kind  = task["kind"]
    batch = task["batch"]
    stage = task["stage"]
    flag  = f"--{kind}"

    print(f"\n[{kind} batch {batch}] stage={stage}")

    # ── Stage: not_started → submit relaxation ─────────────────────────────────
    if stage == "not_started":
        before = set(load_registry())
        rc, _, _ = run_cmd(
            [PHONON_PYTHON, "scripts/batch_management/submit_batch.py", flag, str(batch)],
            dry_run,
        )
        if rc != 0:
            print(f"  ✗ Relaxation submission failed (rc={rc})")
            task["stage"] = "error"
            return

        new_jobs = new_registry_jobs(before)
        if new_jobs:
            print(f"  ✓ Submitted {len(new_jobs)} relaxation job(s)")
            task["stage"] = "relax_running"
            task["job_ids"] = new_jobs
            task["submitted_at"] = datetime.now(timezone.utc).isoformat()
        else:
            # Already relaxed — skip straight to phonopy
            print(f"  → No new relaxation jobs (already done), proceeding to phonopy")
            task["stage"] = "relax_done"
            task["job_ids"] = []

    # ── Stage: relax_running → wait ────────────────────────────────────────────
    elif stage == "relax_running":
        check_and_restart_stuck(task, queued, details, dry_run)
        still = [j for j in task["job_ids"] if j in queued]
        if still:
            print(f"  ▶ {len(still)}/{len(task['job_ids'])} relaxation jobs still running")
        else:
            print(f"  ✓ All relaxation jobs complete")
            task["stage"] = "relax_done"

    # ── Stage: relax_done → submit phonopy displacements ──────────────────────
    elif stage == "relax_done":
        before = set(load_registry())
        rc, _, _ = run_cmd(
            [PHONON_PYTHON, "phonopy/submit_batch.py", flag, "--batch", str(batch)],
            dry_run,
        )
        if rc != 0:
            print(f"  ✗ Phonopy submission failed (rc={rc})")
            task["stage"] = "error"
            return

        new_jobs = new_registry_jobs(before)
        print(f"  ✓ Submitted {len(new_jobs)} displacement job(s)")
        task["stage"] = "phonopy_running"
        task["job_ids"] = new_jobs
        task["submitted_at"] = datetime.now(timezone.utc).isoformat()

    # ── Stage: phonopy_running → wait → postprocess ────────────────────────────
    elif stage == "phonopy_running":
        check_and_restart_stuck(task, queued, details, dry_run)
        still = [j for j in task["job_ids"] if j in queued]
        if still:
            print(f"  ▶ {len(still)}/{len(task['job_ids'])} displacement jobs still running")
        else:
            print(f"  ✓ All displacement jobs complete — post-processing")
            # Remove stale FORCE_CONSTANTS so phonopy rebuilds from FORCE_SETS cleanly
            sub = "phonopy_monolayer_examples" if kind == "monolayer" else "phonopy_bilayer_examples"
            for fc in (WORKFLOW_ROOT / sub).rglob("FORCE_CONSTANTS"):
                fc.unlink()
            rc, _, _ = run_cmd(
                [PHONON_PYTHON, "phonopy/postprocess_batch.py", flag, "--batch", str(batch)],
                dry_run,
            )
            if rc != 0:
                print(f"  ✗ Post-processing failed (rc={rc})")
                task["stage"] = "error"
                return
            task["stage"] = "done"
            print(f"  ✓ [{kind} batch {batch}] DONE")

# ── Status display ─────────────────────────────────────────────────────────────

STAGE_ICON = {
    "not_started":    "·",
    "relax_running":  "▶",
    "relax_done":     "·",
    "phonopy_running": "▶",
    "done":           "✓",
    "error":          "✗",
}


def print_status(seq: list[dict]) -> None:
    done   = sum(1 for t in seq if t["stage"] == "done")
    errors = sum(1 for t in seq if t["stage"] == "error")
    print(f"\n=== Pipeline Status ({done}/{len(seq)} done, {errors} errors) ===")
    for i, t in enumerate(seq, 1):
        icon  = STAGE_ICON.get(t["stage"], "?")
        njobs = len(t["job_ids"])
        print(f"  {i:2}. [{icon}] {t['kind']:9} batch {t['batch']:2}  {t['stage']:<18}  jobs={njobs}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline orchestrator")
    parser.add_argument("--dry-run",       action="store_true", help="Print commands without running them")
    parser.add_argument("--reset",         action="store_true", help="Delete state file and start fresh")
    parser.add_argument("--monolayer-only", action="store_true", help="Only process monolayer batches")
    args = parser.parse_args()

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            print("State file deleted — starting fresh.")
        else:
            print("No state file to delete.")

    seq     = load_state()
    queued  = get_queue()
    details = get_queue_details()

    # Optionally restrict to monolayer tasks only
    working_seq = [t for t in seq if t["kind"] == "monolayer"] if args.monolayer_only else seq

    # Find the first non-finished task
    active = next((t for t in working_seq if t["stage"] not in ("done", "error")), None)

    if active is None:
        print("All batches complete!")
        print_status(working_seq)
        return

    process_active_task(active, queued, details, args.dry_run)
    if not args.dry_run:
        save_state(seq)
    print_status(working_seq)


if __name__ == "__main__":
    main()
