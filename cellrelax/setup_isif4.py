#!/usr/bin/env python3
"""
Stage 1 of the cell-relaxation campaign (TWIST_VARIANT=cellrelax): ISIF=4 residual-strain
refinement runs for the 16 strained TMD reference cells (cellrelax/materials_16.txt).

Recipe = the workflow's established July-2026 protocol (see data/bilayer_lattice_overrides.json
notes and the *_isif4/ dirs in bilayer_examples/): copy the production ISIF=2 relaxation dir,
start from its CONTCAR, and run a fixed-volume cell-shape + ion relaxation
(ISIF=4, IBRION=1, EDIFFG=-1E-7, NSW=400, same ENCUT/EDIFF/IVDW/KPOINTS). The plateaued in-plane
`a` is then read with common/isif4_lattice_extract.py (stage 2).

Writes only under the variant dirs:
    monolayer_examples_cellrelax/<name>_isif4/   bilayer_examples_cellrelax/<name>_isif4/
and records submitted jobs in cellrelax/isif4_jobs.json. Production dirs are read, never written.
Refuses to run unless TWIST_VARIANT=cellrelax. Dry-run by default; --submit to sbatch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))
from run_variant import generated_dir, variant  # noqa: E402

VARIANT = "cellrelax"
MATERIALS_FILE = HERE / "materials_16.txt"
JOBS_FILE = HERE / "isif4_jobs.json"


def read_materials(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, kind, sigma = line.split()[:3]
        rows.append((name, kind, sigma))
    return rows


def last_stress_kB(outcar: Path) -> str:
    last = ""
    with open(outcar) as f:
        for line in f:
            if "in kB" in line:
                last = line
    parts = last.split()
    return " ".join(parts[2:5]) if len(parts) >= 5 else "n/a"


def make_incar(src_incar: str, name: str) -> str:
    text = src_incar
    if not re.search(r"^\s*ISIF\s*=", text, flags=re.M):
        raise ValueError(f"{name}: production INCAR has no ISIF line")
    text = re.sub(r"^\s*ISIF\s*=.*$", "ISIF = 4 # residual-strain refinement (cellrelax variant, stage 1)",
                  text, flags=re.M)
    text = re.sub(r"^\s*SYSTEM\s*=.*\n", "", text, flags=re.M)
    text = re.sub(r"^# Ionic relaxation.*$", "# Cell-shape relaxation (fixed volume) + ions, quasi-Newton",
                  text, flags=re.M)
    for key, val in (("IBRION", "1"), ("NSW", "400"), ("EDIFFG", "-1E-7"), ("IVDW", "12")):
        m = re.search(rf"^\s*{key}\s*=\s*(\S+)", text, flags=re.M)
        if not m or m.group(1) != val:
            raise ValueError(f"{name}: expected {key} = {val} in production INCAR, found {m.group(0) if m else None}")
    return f"SYSTEM = {name} isif4 residual-strain refinement (cellrelax)\n" + text


def make_bat(src_bat: str, name: str, nodes: int) -> str:
    text = re.sub(r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name={name}_isif4", src_bat, flags=re.M)
    text, n = re.subn(r"^#SBATCH --nodes=\d+$", f"#SBATCH --nodes={nodes}", text, flags=re.M)
    if n != 1:
        raise ValueError(f"{name}: could not set --nodes in bat")
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--materials", nargs="*", default=None, help="subset of names (default: all 16)")
    p.add_argument("--nodes-monolayer", type=int, default=2)
    p.add_argument("--nodes-bilayer", type=int, default=4, help="July isif4 bilayer runs used 4 nodes")
    p.add_argument("--submit", action="store_true", help="sbatch each prepared dir (default: dry run)")
    p.add_argument("--force", action="store_true", help="overwrite an existing <name>_isif4 dir")
    args = p.parse_args()

    if variant() != VARIANT:
        sys.exit(f"Refusing: export TWIST_VARIANT={VARIANT} first (active: {variant() or 'unset'})")

    rows = read_materials(MATERIALS_FILE)
    if args.materials:
        rows = [r for r in rows if r[0] in set(args.materials)]
        missing = set(args.materials) - {r[0] for r in rows}
        if missing:
            sys.exit(f"Not in {MATERIALS_FILE.name}: {sorted(missing)}")

    jobs = json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else {}
    print(f"{'name':18s} {'kind':9s} {'src residual stress (kB)':>32s}  dst")
    for name, kind, sigma in rows:
        base = "monolayer_examples" if kind == "monolayer" else "bilayer_examples"
        src = ROOT / base / name                      # production (TWIST_VARIANT-independent)
        dst = generated_dir(base) / f"{name}_isif4"   # *_cellrelax/<name>_isif4
        for req in ("CONTCAR", "INCAR", "KPOINTS", "POTCAR", "bat", "OUTCAR"):
            if not (src / req).exists():
                sys.exit(f"{name}: missing {src / req}")
        stress = last_stress_kB(src / "OUTCAR")
        print(f"{name:18s} {kind:9s} {stress:>32s}  {dst.relative_to(ROOT)}")
        if dst.exists() and not args.force:
            print(f"    exists, skipping (use --force to rebuild)")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "CONTCAR", dst / "POSCAR")
        shutil.copy(src / "KPOINTS", dst / "KPOINTS")
        shutil.copy(src / "POTCAR", dst / "POTCAR")
        (dst / "INCAR").write_text(make_incar((src / "INCAR").read_text(), name))
        nodes = args.nodes_monolayer if kind == "monolayer" else args.nodes_bilayer
        (dst / "bat").write_text(make_bat((src / "bat").read_text(), name, nodes))
        (dst / "SOURCE.txt").write_text(
            f"POSCAR = {src.relative_to(ROOT)}/CONTCAR (production ISIF=2 result)\n"
            f"production residual stress (kB, VASP sign): {stress}\n"
            f"sigma_PBE+D3_xx at that cell (GPa, ASE sign): {sigma}\n"
        )
        if args.submit:
            out = subprocess.run(["sbatch", "--parsable", "bat"], cwd=dst, capture_output=True, text=True)
            if out.returncode != 0:
                print(f"    sbatch FAILED: {out.stderr.strip()}")
                continue
            job_id = out.stdout.strip().split(";")[0]
            (dst / "latest_job_id").write_text(job_id + "\n")
            jobs[name] = {
                "job_id": job_id, "dir": str(dst.relative_to(ROOT)), "kind": kind, "nodes": nodes,
                "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "src_residual_stress_kB": stress, "sigma_pbe_d3_xx_GPa": sigma, "stage": "isif4",
            }
            print(f"    submitted {job_id} ({nodes} nodes)")
    if args.submit:
        JOBS_FILE.write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"\nrecorded {len(jobs)} jobs in {JOBS_FILE.relative_to(ROOT)}")
    else:
        print("\n(dry run: nothing submitted; add --submit)")


if __name__ == "__main__":
    main()
