#!/usr/bin/env python3
"""
Watchdog for long VASP ionic relaxations that silently hang on this cluster.

Polls a set of relaxation directories and, for each one whose latest job is
  * RUNNING but has produced no output for --stale-minutes   (silent mid-run hang), or
  * finished (TIMEOUT / FAILED / CANCELLED / COMPLETED ...) without reaching EDIFFG,
applies the usual recovery playbook automatically:
  scancel (if still running) -> archive all output to stale_hang_<jobid>/ (nothing is
  deleted) -> seed POSCAR from the last complete CONTCAR -> resubmit through the normal
  submit helper (registry + latest_job_id are updated as for a manual submission).

A directory is finished once OUTCAR contains "reached required accuracy". The number of
automatic restarts per directory is capped by --max-restarts (counted as stale_hang_* dirs).

Additive: nothing in the production pipeline is touched. Variant-aware through --variant,
which is applied before the workflow modules are imported.

Usage (login node, detached so it survives the shell):
  setsid nohup python3 scripts/relaxation_watchdog.py --variant twisted \\
      bilayer_examples_twisted/MoS2_twist_m3_near0 ... > bilayer_examples_twisted/watchdog.log 2>&1 &
  python3 scripts/relaxation_watchdog.py --variant twisted <dirs> --once --dry-run   # single inspection pass
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILES = {"INCAR", "KPOINTS", "POTCAR", "POSCAR", "bat", "twist.json",
               "phonopy_dim.txt", "latest_job_id"}
ENDED_STATES = {"COMPLETED", "TIMEOUT", "FAILED", "CANCELLED", "NODE_FAIL",
                "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
CONVERGED_MARK = "reached required accuracy"


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def sacct_state(job_id):
    """Return (state, start_datetime_or_None) for a job id, ('UNKNOWN', None) if sacct has nothing."""
    try:
        out = subprocess.run(
            ["sacct", "-j", str(job_id), "-X", "-n", "-P", "-o", "State,Start"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        log(f"  sacct failed for {job_id}: {e}")
        return "UNKNOWN", None
    if not out:
        return "UNKNOWN", None
    state, start = out.splitlines()[0].split("|")[:2]
    state = state.split()[0]  # "CANCELLED by 12345" -> CANCELLED
    try:
        start_dt = datetime.fromisoformat(start)
    except ValueError:
        start_dt = None
    return state, start_dt


def latest_job_id(d):
    f = d / "latest_job_id"
    return f.read_text().strip() if f.exists() else None


def is_converged(d):
    outcar = d / "OUTCAR"
    if not outcar.exists():
        return False
    with open(outcar, "rb") as fh:
        fh.seek(max(0, outcar.stat().st_size - 200_000))
        return CONVERGED_MARK.encode() in fh.read()


def last_output_time(d, job_id):
    """Newest mtime among the files VASP writes continuously (None if none exist yet)."""
    times = []
    for name in ("OSZICAR", "OUTCAR", "CONTCAR", "XDATCAR", f"slurm-{job_id}.out"):
        p = d / name
        if p.exists():
            times.append(datetime.fromtimestamp(p.stat().st_mtime))
    return max(times) if times else None


def ionic_steps(d):
    osz = d / "OSZICAR"
    if not osz.exists():
        return 0
    return sum(1 for line in open(osz, errors="ignore") if " F= " in line)


def contcar_complete(d):
    poscar, contcar = d / "POSCAR", d / "CONTCAR"
    if not contcar.exists() or contcar.stat().st_size == 0:
        return False
    n_pos = sum(1 for _ in open(poscar))
    n_con = sum(1 for _ in open(contcar))
    return n_con >= n_pos


def restart_count(d):
    return len([p for p in d.iterdir() if p.is_dir() and p.name.startswith("stale_hang_")])


def wait_until_gone(job_id, timeout_s=180):
    """After scancel, wait until sacct no longer reports RUNNING, then a grace period for file flushes."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        state, _ = sacct_state(job_id)
        if state not in ("RUNNING", "COMPLETING", "PENDING"):
            break
        time.sleep(10)
    time.sleep(20)


def archive_and_reseed(d, job_id, dry_run):
    """Move all output into stale_hang_<jobid>/, seed POSCAR from CONTCAR if it is complete."""
    stale = d / f"stale_hang_{job_id}"
    reseed = contcar_complete(d)
    steps = ionic_steps(d)
    log(f"  archiving -> {stale.name}  (ionic steps banked: {steps}, reseed from CONTCAR: {reseed})")
    if dry_run:
        return
    stale.mkdir(exist_ok=True)
    shutil.copy2(d / "POSCAR", stale / "POSCAR.start")
    new_poscar = (d / "CONTCAR").read_bytes() if reseed else None
    for p in list(d.iterdir()):
        if p.name in INPUT_FILES or p.name.startswith("stale_hang_") or p.name == "watchdog.log":
            continue
        shutil.move(str(p), str(stale / p.name))
    if new_poscar is not None:
        (d / "POSCAR").write_bytes(new_poscar)


def resubmit(d, dry_run):
    from submit_job_impl import submit_example_job  # noqa: E402  (imported after variant is set)
    if dry_run:
        log(f"  would resubmit {d.name}")
        return None
    ok, job_id, msg = submit_example_job(d, "bilayer_examples", dry_run=False)
    if ok and job_id:
        log(f"  resubmitted {d.name} as job {job_id}")
        return job_id
    log(f"  RESUBMIT FAILED for {d.name}: {msg}")
    return None


def check_dir(d, args):
    """Returns True when the directory is finished (converged or given up)."""
    if is_converged(d):
        log(f"{d.name}: CONVERGED ({ionic_steps(d)} ionic steps) - done")
        return True

    job_id = latest_job_id(d)
    if job_id is None:
        log(f"{d.name}: no latest_job_id - skipping")
        return False

    state, start = sacct_state(job_id)
    now = datetime.now()

    if state in ("PENDING", "UNKNOWN", "REQUEUED", "SUSPENDED"):
        log(f"{d.name}: job {job_id} {state} - waiting")
        return False

    if state in ("RUNNING", "COMPLETING"):
        last = last_output_time(d, job_id)
        ref = max(t for t in (last, start) if t is not None) if (last or start) else None
        if ref is None:
            log(f"{d.name}: job {job_id} RUNNING, no timing info yet - waiting")
            return False
        silent_min = (now - ref).total_seconds() / 60
        if silent_min < args.stale_minutes:
            log(f"{d.name}: job {job_id} RUNNING, {ionic_steps(d)} steps, last output {silent_min:.0f} min ago - ok")
            return False
        log(f"{d.name}: job {job_id} RUNNING but SILENT for {silent_min:.0f} min "
            f"(> {args.stale_minutes}) - treating as hung")
        if restart_count(d) >= args.max_restarts:
            log(f"  restart cap ({args.max_restarts}) reached - NOT restarting, needs a human")
            return True
        if not args.dry_run:
            subprocess.run(["scancel", job_id], check=False)
            wait_until_gone(job_id)
        else:
            log(f"  would scancel {job_id}")
        archive_and_reseed(d, job_id, args.dry_run)
        resubmit(d, args.dry_run)
        return False

    if state in ENDED_STATES:
        log(f"{d.name}: job {job_id} {state} without convergence ({ionic_steps(d)} steps)")
        if restart_count(d) >= args.max_restarts:
            log(f"  restart cap ({args.max_restarts}) reached - NOT restarting, needs a human")
            return True
        archive_and_reseed(d, job_id, args.dry_run)
        resubmit(d, args.dry_run)
        return False

    log(f"{d.name}: job {job_id} in unexpected state {state} - waiting")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="relaxation directories to watch")
    ap.add_argument("--variant", default=None, help="TWIST_VARIANT to apply (e.g. twisted)")
    ap.add_argument("--stale-minutes", type=float, default=60,
                    help="RUNNING job with no output change for this long is hung (default 60)")
    ap.add_argument("--poll-minutes", type=float, default=10, help="polling interval (default 10)")
    ap.add_argument("--max-restarts", type=int, default=30,
                    help="give up on a directory after this many stale_hang_* archives (default 30)")
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    ap.add_argument("--dry-run", action="store_true", help="inspect and report only; no scancel/mv/sbatch")
    args = ap.parse_args()

    if args.variant:
        os.environ["TWIST_VARIANT"] = args.variant
    sys.path.insert(0, str(WORKFLOW_ROOT / "common"))

    dirs = [Path(p).resolve() for p in args.dirs]
    missing = [d for d in dirs if not d.is_dir()]
    if missing:
        sys.exit(f"not a directory: {', '.join(map(str, missing))}")

    log(f"watchdog start: {len(dirs)} dirs, variant={args.variant!r}, stale>{args.stale_minutes:g} min, "
        f"poll {args.poll_minutes:g} min, max restarts {args.max_restarts}, dry_run={args.dry_run}")
    active = list(dirs)
    while active:
        still = []
        for d in active:
            try:
                done = check_dir(d, args)
            except Exception as e:  # keep the loop alive on any single-dir error
                log(f"{d.name}: ERROR {type(e).__name__}: {e}")
                done = False
            if not done:
                still.append(d)
        active = still
        if args.once:
            log(f"single pass done; {len(active)} dirs still unfinished")
            return
        if not active:
            break
        time.sleep(args.poll_minutes * 60)
    log("all watched directories converged (or hit the restart cap) - watchdog exiting")


if __name__ == "__main__":
    main()
