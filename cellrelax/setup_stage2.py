#!/usr/bin/env python3
"""
Stages 2-3 of the cell-relaxation campaign (TWIST_VARIANT=cellrelax). See cellrelax/README.md.

  extract     plateaued in-plane `a` of every <name>_isif4 run (mean of the last --window ionic steps;
              10 by default because WSe2 and WS2_bilayer_3R hung after 26/29 steps, so a 50-step window
              would average in the initial transient) -> cellrelax/isif4_lattice.json (all 16) and
              data/bilayer_lattice_overrides_cellrelax.json (the 13 bilayers; read only by this variant).
  monolayers  monolayer_examples_cellrelax/<m>/ for MoS2/WS2/WSe2: POSCAR = ISIF=4 CONTCAR rescaled in-plane to the
              plateau `a` and the cell height reset to 20 A (Cartesian offsets from the layer midpoint kept). Fixed-volume ISIF=4
              stretched c to 20.3-20.5 A, and the bilayer builder reuses monolayer fractional z assuming
              c == 20 A, so leaving c would distort dMX in every bilayer by ~2 %. Inputs (INCAR ISIF=2,
              KPOINTS, POTCAR, bat) copied from production. Also copies the unchanged constituent monolayers
              MoSe2/MoTe2/WTe2 (inputs + CONTCAR only) so the bilayer builder finds them.
  bilayers    build the 13 bilayers with relaxation/bilayer/create_bilayer_example.py (picks up the override
              `a` and the variant monolayers), check a == override, copy production INCAR/KPOINTS/bat.

--submit sbatches the prepared dirs (default: dry run). Writes only variant dirs, cellrelax/ and the
variant override file. Refuses to run unless TWIST_VARIANT=cellrelax.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))
from isif4_lattice_extract import extract_in_plane_a_per_step  # noqa: E402
from run_variant import generated_dir, variant  # noqa: E402

VARIANT = "cellrelax"
MATERIALS_FILE = HERE / "materials_16.txt"
LATTICE_FILE = HERE / "isif4_lattice.json"
JOBS_FILE = HERE / "stage3_jobs.json"
OVERRIDES_FILE = ROOT / "data" / f"bilayer_lattice_overrides_{VARIANT}.json"
UNCHANGED_MONOLAYERS = ("MoSe2", "MoTe2", "WTe2")
MONOLAYER_C = 20.0


def read_materials() -> list[tuple[str, str]]:
    rows = []
    for line in MATERIALS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, kind = line.split()[:2]
            rows.append((name, kind))
    return rows


def isif4_dir(name: str, kind: str) -> Path:
    base = "monolayer_examples" if kind == "monolayer" else "bilayer_examples"
    return generated_dir(base) / f"{name}_isif4"


def last_stress_kB(outcar: Path) -> list[float]:
    last = None
    for line in outcar.read_text().splitlines():
        if "in kB" in line:
            last = line
    return [float(x) for x in last.split()[2:5]]


def cmd_extract(window: int) -> None:
    lattice, overrides = {}, {}
    print(f"{'name':18s} {'steps':>5s} {'a_prod':>8s} {'a_new':>8s} {'std':>7s} {'da%':>6s}  final stress kB")
    for name, kind in read_materials():
        d = isif4_dir(name, kind)
        a = extract_in_plane_a_per_step(d / "OUTCAR")
        tail = a[-window:]
        mean, std = statistics.fmean(tail), statistics.pstdev(tail)
        if len(a) < window + 5:
            sys.exit(f"{name}: only {len(a)} steps, too few for a {window}-step plateau")
        if std > 0.002:
            sys.exit(f"{name}: last-{window} std {std:.4f} A > 0.002 A, not plateaued")
        stress = last_stress_kB(d / "OUTCAR")
        lattice[name] = {"kind": kind, "a": round(mean, 6), "a_std": round(std, 6), "window": window,
                         "n_steps": len(a), "a_production": round(a[0], 6),
                         "final_stress_kB_xx_yy_zz": stress, "source": str(d.relative_to(ROOT))}
        print(f"{name:18s} {len(a):5d} {a[0]:8.4f} {mean:8.4f} {std:7.5f} {100*(mean/a[0]-1):6.2f}  {stress}")
        if kind == "bilayer":
            overrides[name] = {"a": round(mean, 6), "note": (
                f"cellrelax stage 2 (2026-10-09): ISIF=4 residual-strain refinement from the production ISIF=2 "
                f"CONTCAR (a={a[0]:.4f} A). Mean in-plane 'a' over the last {window} of {len(a)} ionic steps "
                f"(std {std:.5f} A); final stress {stress[0]:.2f} kB.")}
    LATTICE_FILE.write_text(json.dumps(lattice, indent=2) + "\n")
    OVERRIDES_FILE.write_text(json.dumps({"bilayers": overrides}, indent=2) + "\n")
    print(f"\nwrote {LATTICE_FILE.relative_to(ROOT)} ({len(lattice)}) and "
          f"{OVERRIDES_FILE.relative_to(ROOT)} ({len(overrides)} bilayers)")


def reset_c(contcar: Path, c_new: float, a_new: float | None = None) -> str:
    lines = contcar.read_text().splitlines()
    scale = float(lines[1])
    lat = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    if abs(lat[2, 0]) > 1e-8 or abs(lat[2, 1]) > 1e-8 or abs(lat[0, 2]) > 1e-8 or abs(lat[1, 2]) > 1e-8:
        sys.exit(f"{contcar}: c not orthogonal to the plane")
    counts = [int(x) for x in lines[6].split()]
    if not lines[7].strip().lower().startswith("d"):
        sys.exit(f"{contcar}: expected Direct coordinates")
    pos = [[float(x) for x in lines[8 + i].split()[:3]] for i in range(sum(counts))]
    if a_new is not None:  # isotropic in-plane rescale to the plateau mean (fractional x,y unchanged)
        lat[:2] *= a_new / np.linalg.norm(lat[0])
    c_old = lat[2, 2]
    zmid = 0.5 * (max(p[2] for p in pos) + min(p[2] for p in pos))
    out = lines[:2]
    out[1] = "   1.00000000000000"
    for row in (lat[0], lat[1], [0.0, 0.0, c_new]):
        out.append("   " + "  ".join(f"{x:20.16f}" for x in row))
    out += lines[5:8]
    for p in pos:
        z = zmid + (p[2] - zmid) * c_old / c_new
        out.append("  " + "  ".join(f"{x:19.16f}" for x in (p[0], p[1], z)))
    return "\n".join(out) + "\n"


def set_job_name(bat: str, name: str) -> str:
    return re.sub(r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name={name}", bat, flags=re.M)


def submit(dst: Path, name: str, kind: str, jobs: dict) -> None:
    out = subprocess.run(["sbatch", "--parsable", "bat"], cwd=dst, capture_output=True, text=True)
    if out.returncode != 0:
        print(f"    sbatch FAILED: {out.stderr.strip()}")
        return
    job_id = out.stdout.strip().split(";")[0]
    (dst / "latest_job_id").write_text(job_id + "\n")
    jobs[name] = {"job_id": job_id, "dir": str(dst.relative_to(ROOT)), "kind": kind, "stage": "isif2",
                  "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    print(f"    submitted {job_id}")


def copy_inputs(src: Path, dst: Path, name: str, files=("INCAR", "KPOINTS", "POTCAR")) -> None:
    for f in files:
        shutil.copy(src / f, dst / f)
    (dst / "bat").write_text(set_job_name((src / "bat").read_text(), f"{name}_cellrelax"))


def cmd_monolayers(do_submit: bool, force: bool, jobs: dict) -> None:
    lattice = json.loads(LATTICE_FILE.read_text())
    mono_root = generated_dir("monolayer_examples")
    for name in UNCHANGED_MONOLAYERS:
        src, dst = ROOT / "monolayer_examples" / name, mono_root / name
        if dst.exists() and not force:
            print(f"{name}: exists, skipping")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("POSCAR", "CONTCAR", "INCAR", "KPOINTS", "POTCAR", "bat", "OUTCAR"):
            shutil.copy(src / f, dst / f)
        (dst / "SOURCE.txt").write_text(f"copied unchanged from {src.relative_to(ROOT)} (not re-relaxed)\n")
        print(f"{name}: copied production monolayer (constituent only)")
    for name, info in lattice.items():
        if info["kind"] != "monolayer":
            continue
        src, isif4, dst = ROOT / "monolayer_examples" / name, ROOT / info["source"], mono_root / name
        if dst.exists() and not force:
            print(f"{name}: exists, skipping")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "POSCAR").write_text(reset_c(isif4 / "CONTCAR", MONOLAYER_C, info["a"]))
        copy_inputs(src, dst, name)
        a_now = float(np.linalg.norm([float(x) for x in (dst / "POSCAR").read_text().splitlines()[2].split()]))
        (dst / "SOURCE.txt").write_text(
            f"POSCAR = {info['source']}/CONTCAR with c reset {MONOLAYER_C} A (Cartesian offsets from midplane "
            f"kept). a = {a_now:.6f} A (plateau mean {info['a']:.6f}). Inputs from {src.relative_to(ROOT)}.\n")
        print(f"{name}: prepared, a={a_now:.5f} (plateau {info['a']:.5f}, production {info['a_production']:.5f})")
        if do_submit:
            submit(dst, name, "monolayer", jobs)


def cmd_bilayers(do_submit: bool, force: bool, jobs: dict) -> None:
    lattice = json.loads(LATTICE_FILE.read_text())
    bi_root = generated_dir("bilayer_examples")
    for name, info in lattice.items():
        if info["kind"] != "bilayer":
            continue
        for m in re.findall(r"[A-Z][a-z]?\d*[A-Z][a-z]?\d*", name.replace("_bilayer", "")):
            if not (generated_dir("monolayer_examples") / m / "CONTCAR").exists():
                sys.exit(f"{name}: variant monolayer {m} has no CONTCAR yet")
        dst = bi_root / name
        if dst.exists() and not force:
            print(f"{name}: exists, skipping")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        r = subprocess.run([sys.executable, str(ROOT / "relaxation/bilayer/create_bilayer_example.py"), name,
                            "--name", name, "--no-mp"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 or not (dst / "POSCAR").exists():
            sys.exit(f"{name}: builder failed\n{r.stdout}\n{r.stderr}")
        a_now = float(np.linalg.norm([float(x) for x in (dst / "POSCAR").read_text().splitlines()[2].split()]))
        if abs(a_now - info["a"]) > 1e-5:
            sys.exit(f"{name}: built a={a_now:.6f} != override {info['a']:.6f}")
        src = ROOT / "bilayer_examples" / name
        copy_inputs(src, dst, name, files=("INCAR", "KPOINTS"))
        pot_ok = (dst / "POTCAR").read_text() == (src / "POTCAR").read_text()
        (dst / "SOURCE.txt").write_text(
            f"built by create_bilayer_example.py (TWIST_VARIANT=cellrelax) at a = {a_now:.6f} A from "
            f"{OVERRIDES_FILE.name}; INCAR/KPOINTS/bat from {src.relative_to(ROOT)}; POTCAR identical: {pot_ok}\n")
        print(f"{name}: built a={a_now:.5f} (production {info['a_production']:.5f}), POTCAR==prod {pot_ok}")
        if not pot_ok:
            sys.exit(f"{name}: POTCAR differs from production")
        if do_submit:
            submit(dst, name, "bilayer", jobs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["extract", "monolayers", "bilayers"])
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if variant() != VARIANT:
        sys.exit(f"Refusing: export TWIST_VARIANT={VARIANT} first (active: {variant() or 'unset'})")
    if args.step == "extract":
        cmd_extract(args.window)
        return
    jobs = json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else {}
    (cmd_monolayers if args.step == "monolayers" else cmd_bilayers)(args.submit, args.force, jobs)
    if args.submit:
        JOBS_FILE.write_text(json.dumps(jobs, indent=2) + "\n")
    else:
        print("(dry run: nothing submitted; add --submit)")


if __name__ == "__main__":
    main()
