#!/usr/bin/env python3
"""Registry x gap scan of the 42 TMD bilayer reference cells: bare-PBE, grid-tight single points.

Purpose (2026-10-26): the Nequix fine-tune sees one Hessian per bilayer at its symmetric equilibrium
registry and nothing about how the interlayer sliding/breathing landscape changes with registry, so it
predicts an imaginary Gamma shear pair in twisted (moire) cells. These configurations sample that
landscape as energy/force/stress labels: the TOP layer of each FINAL_RESULTS_HEALTHY/<bilayer>/POSCAR
(PBE+D3 equilibrium, ISIF=4-refined cells) is shifted rigidly by (u, v) fractions of the in-plane lattice
vectors on an NxN grid and by dz along z, plus a breathing curve at the equilibrium registry.

Settings are deliberately the grid-tight ones (LREAL=.FALSE., ADDGRID, EDIFF 1E-8; egg-box errors are
larger than the meV sliding energies being taught), bare PBE (no IVDW: the deployed model is bare PBE +
explicit D3), ISYM=0 (all registries treated identically), ISIF=2 so the stress is written.

Layout (gitignored, like every *_examples* run dir):
  bilayer_examples_registry_scan/<bilayer>/u0.25_v0.50_dz+0.30/{POSCAR,INCAR,KPOINTS,POTCAR}
  bilayer_examples_registry_scan/<bilayer>/{bat,scan.json}   one resumable SLURM job per bilayer
Templates (host-local, gitignored): common/staticpoint_templates_registry_scan/{INCAR,KPOINTS,bat},
created once with --init-templates from the twisted_gridtight static INCAR and the relaxation KPOINTS/bat.

  python3 scripts/registry_gap_scan.py --init-templates
  python3 scripts/registry_gap_scan.py --pilot --dry-run
  python3 scripts/registry_gap_scan.py --pilot --submit
  python3 scripts/registry_gap_scan.py --all --submit            # all 42 TMD bilayers (v5 split manifests)
  python3 scripts/registry_gap_scan.py --all --status
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "relaxation" / "monolayer"))

RESULTS = WORKFLOW_ROOT / "FINAL_RESULTS_HEALTHY"
OUT_ROOT = WORKFLOW_ROOT / "bilayer_examples_registry_scan"
TEMPLATES = WORKFLOW_ROOT / "common" / "staticpoint_templates_registry_scan"
GRIDTIGHT_INCAR = WORKFLOW_ROOT / "common" / "staticpoint_templates_twisted_gridtight" / "INCAR"
RELAX_TEMPLATES = WORKFLOW_ROOT / "common" / "relaxation_templates"
SPLIT_DIR = WORKFLOW_ROOT / "nequix_datasets" / "v5_tmd_cellrelax"

# Same 4 train-split bilayers as the 2026-09 strain pilot: both metals, three chalcogens, homo + hetero.
PILOT = ["MoS2_bilayer_3R", "WS2_bilayer_2H", "MoS2_MoTe2_2H", "MoTe2_WSe2_2H"]
DEFAULT_DZ = [-0.3, 0.0, 0.3]
DEFAULT_BREATHING = [-0.5, -0.15, 0.15, 0.5, 1.0]


# ----------------------------------------------------------------------------- POSCAR helpers
def read_poscar(path):
    L = Path(path).read_text().splitlines()
    scale = float(L[1])
    cell = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)]) * scale
    species = L[5].split()
    counts = [int(x) for x in L[6].split()]
    i = 7
    if L[i].strip()[0] in "sS":
        i += 1
    direct = L[i].strip()[0] in "dD"
    n = sum(counts)
    pos = np.array([[float(x) for x in L[i + 1 + k].split()[:3]] for k in range(n)])
    if direct:
        pos = pos @ cell
    return L[0].strip(), cell, species, counts, pos


def write_poscar(path, comment, cell, species, counts, pos):
    with open(path, "w") as f:
        f.write(f"{comment}\n1.0\n")
        for v in cell:
            f.write("  %.12f %.12f %.12f\n" % tuple(v))
        f.write(" ".join(species) + "\n" + " ".join(map(str, counts)) + "\nCartesian\n")
        for r in pos:
            f.write("  %.12f %.12f %.12f\n" % tuple(r))


def split_layers(pos):
    """Top layer = atoms above the median z; both layers must hold half the atoms."""
    top = pos[:, 2] > np.median(pos[:, 2])
    if top.sum() != len(pos) // 2:
        raise RuntimeError("layer split by median z failed")
    gap = pos[top, 2].min() - pos[~top, 2].max()
    return top, gap


# ----------------------------------------------------------------------------- materials / templates
def tmd_bilayers():
    names = []
    for split in ("train", "val", "holdout"):
        for line in (SPLIT_DIR / split / "materials.txt").read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#") and s.endswith(("_2H", "_3R")):
                names.append(s)
    return sorted(names)


def init_templates():
    if TEMPLATES.exists():
        sys.exit(f"{TEMPLATES} already exists; delete it by hand if you really want to regenerate it")
    TEMPLATES.mkdir(parents=True)
    incar = GRIDTIGHT_INCAR.read_text()
    incar = re.sub(r"\n# DFT-D3\(BJ\) van der Waals correction\nIVDW = 12\n", "\n", incar)
    assert "IVDW" not in incar, "IVDW line not removed"
    incar = incar.replace(
        "# Static calculation\nNSW = 0\nIBRION = -1\n",
        "# Static calculation; ISIF=2 so the stress tensor is written (energy/force/stress labels)\n"
        "NSW = 0\nIBRION = -1\nISIF = 2\n"
        "# Registry scan: no symmetry so every registry is treated identically\nISYM = 0\n",
    )
    assert "ISYM = 0" in incar and "ISIF = 2" in incar
    incar = (
        "# registry/gap scan (scripts/registry_gap_scan.py, 2026-10-26): bare PBE (no IVDW), grid-tight\n"
        "# settings copied from staticpoint_templates_twisted_gridtight, ISYM=0, ISIF=2.\n" + incar
    )
    (TEMPLATES / "INCAR").write_text(incar)
    shutil.copy2(RELAX_TEMPLATES / "KPOINTS", TEMPLATES / "KPOINTS")
    bat = (RELAX_TEMPLATES / "bat").read_text()
    bat = re.sub(r"#SBATCH --partition=.*", "#SBATCH --partition=short", bat)
    bat = re.sub(r"#SBATCH --time=.*", "#SBATCH --time=12:00:00", bat)
    bat = re.sub(r"#SBATCH --job-name=.*", "#SBATCH --job-name=registry_scan", bat)
    # keep everything up to and including the cd into the submit dir; the per-bilayer loop is appended at build time
    head = bat.split("mpirun")[0].rstrip() + "\n"
    (TEMPLATES / "bat").write_text(head)
    print(f"wrote {TEMPLATES}/{{INCAR,KPOINTS,bat}} (host-local, gitignored)")
    print(incar)


def potcar_for(name, species, workdir):
    """Reuse the bilayer's production POTCAR when its element order matches, else build one."""
    src = WORKFLOW_ROOT / "bilayer_examples" / name / "POTCAR"
    if src.exists():
        titles = re.findall(r"TITEL\s*=\s*PAW_PBE\s+(\S+)", src.read_text())
        if [t.split("_")[0] for t in titles] == species:
            shutil.copy2(src, workdir / "POTCAR")
            return "copied from bilayer_examples/%s/POTCAR (%s)" % (name, " ".join(titles))
    from generate_potcar import generate_potcar  # relaxation/monolayer/generate_potcar.py

    generate_potcar(workdir / "POSCAR", output_path=workdir / "POTCAR")
    titles = re.findall(r"TITEL\s*=\s*PAW_PBE\s+(\S+)", (workdir / "POTCAR").read_text())
    if [t.split("_")[0] for t in titles] != species:
        raise RuntimeError(f"generated POTCAR order {titles} != POSCAR species {species}")
    return "generated (%s)" % " ".join(titles)


# ----------------------------------------------------------------------------- build / submit / status
def configs(grid, dz_list, breathing):
    out = []
    for i in range(grid):
        for j in range(grid):
            for dz in dz_list:
                out.append((i / grid, j / grid, dz))
    for dz in breathing:
        c = (0.0, 0.0, dz)
        if c not in out:
            out.append(c)
    return out


def tag(u, v, dz):
    return f"u{u:.3f}_v{v:.3f}_dz{dz:+.2f}"


def build(name, grid, dz_list, breathing, dry_run):
    if not TEMPLATES.exists():
        sys.exit(f"missing {TEMPLATES}: run --init-templates first")
    poscar = RESULTS / name / "POSCAR"
    comment, cell, species, counts, pos = read_poscar(poscar)
    top, gap = split_layers(pos)
    a1, a2 = cell[0], cell[1]
    cfgs = configs(grid, dz_list, breathing)
    out = OUT_ROOT / name
    print(f"{name}: a = {np.linalg.norm(a1):.4f} A, gap = {gap:.3f} A, {len(cfgs)} configs -> {out.relative_to(WORKFLOW_ROOT)}")
    if dry_run:
        for u, v, dz in cfgs:
            print("   ", tag(u, v, dz))
        return None
    out.mkdir(parents=True, exist_ok=True)
    dirs = []
    potcar_note = None
    for u, v, dz in cfgs:
        wd = out / tag(u, v, dz)
        wd.mkdir(exist_ok=True)
        p = pos.copy()
        p[top] += u * a1 + v * a2 + np.array([0.0, 0.0, dz])
        write_poscar(wd / "POSCAR", f"{comment} | top layer shifted u={u:.3f} v={v:.3f} dz={dz:+.2f} A", cell, species, counts, p)
        shutil.copy2(TEMPLATES / "INCAR", wd / "INCAR")
        shutil.copy2(TEMPLATES / "KPOINTS", wd / "KPOINTS")
        if potcar_note is None:
            potcar_note = potcar_for(name, species, wd)
        else:
            shutil.copy2(dirs[0] / "POTCAR", wd / "POTCAR")
        dirs.append(wd)
    loop = (
        "\n# --- loop over the registry/gap configurations (resumable: finished vasprun.xml are skipped) ---\n"
        "for d in " + " ".join(d.name for d in dirs) + "; do\n"
        '  if [ -s "$d/vasprun.xml" ] && grep -q "</modeling>" "$d/vasprun.xml"; then continue; fi\n'
        '  echo "=== $d  $(date)"\n'
        '  ( cd "$d" && mpirun $VASP_PATH/vasp_std > vasp.out 2>&1 )\n'
        "done\n"
        'echo "=== all done  $(date)"\n'
    )
    (out / "bat").write_text((TEMPLATES / "bat").read_text() + loop)
    meta = {
        "bilayer": name,
        "source_poscar": str(poscar.relative_to(WORKFLOW_ROOT)),
        "a_A": float(np.linalg.norm(a1)),
        "gap_A": float(gap),
        "top_layer_atoms": [int(i) for i in np.where(top)[0]],
        "grid": grid,
        "dz_A": dz_list,
        "breathing_dz_A": breathing,
        "configs": [tag(u, v, dz) for u, v, dz in cfgs],
        "potcar": potcar_note,
        "templates": str(TEMPLATES.relative_to(WORKFLOW_ROOT)),
        "jobs": [],
    }
    existing = out / "scan.json"
    if existing.exists():
        meta["jobs"] = json.loads(existing.read_text()).get("jobs", [])
    existing.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"    POTCAR {potcar_note}")
    return out


def submit(out):
    r = subprocess.run(
        ["sbatch", "--parsable", f"--job-name=rscan-{out.name}", "bat"],
        cwd=out, capture_output=True, text=True, check=True,
    )
    jid = r.stdout.strip().split(";")[0]
    meta = json.loads((out / "scan.json").read_text())
    meta["jobs"].append(jid)
    (out / "scan.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "latest_job_id").write_text(jid + "\n")
    print(f"    submitted job {jid}")


def status(names):
    total_done = total = 0
    for name in names:
        out = OUT_ROOT / name
        if not (out / "scan.json").exists():
            print(f"{name}: not built")
            continue
        meta = json.loads((out / "scan.json").read_text())
        done = 0
        energies = {}
        for c in meta["configs"]:
            vr = out / c / "vasprun.xml"
            if vr.exists() and "</modeling>" in vr.read_text()[-2000:]:
                done += 1
                m = re.findall(r"F=\s*(\S+)\s+E0=\s*(\S+)", (out / c / "OSZICAR").read_text())
                if m:
                    energies[c] = float(m[-1][1])
        total_done += done
        total += len(meta["configs"])
        line = f"{name}: {done}/{len(meta['configs'])} done, jobs {meta['jobs']}"
        e0 = energies.get(tag(0.0, 0.0, 0.0))
        if e0 is not None and len(energies) > 1:
            rel = {c: (e - e0) * 1e3 for c, e in energies.items()}
            line += f"  E-E0 range [{min(rel.values()):+.1f}, {max(rel.values()):+.1f}] meV"
        print(line)
    print(f"total {total_done}/{total}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--materials", nargs="+", help="bilayer names (FINAL_RESULTS_HEALTHY/<name>)")
    g.add_argument("--pilot", action="store_true", help="the 4 pilot bilayers: " + " ".join(PILOT))
    g.add_argument("--all", action="store_true", help="all 42 TMD bilayers listed in the v5 split manifests")
    ap.add_argument("--grid", type=int, default=4, help="NxN registry grid in fractions of a1, a2 (default 4)")
    ap.add_argument("--dz", type=float, nargs="+", default=DEFAULT_DZ, help="gap offsets (A) applied at every registry")
    ap.add_argument("--breathing", type=float, nargs="+", default=DEFAULT_BREATHING,
                    help="extra gap offsets (A) at the equilibrium registry only")
    ap.add_argument("--init-templates", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.init_templates:
        init_templates()
        return
    names = PILOT if a.pilot else tmd_bilayers() if a.all else a.materials
    if not names:
        ap.error("give --materials, --pilot or --all")
    missing = [n for n in names if not (RESULTS / n / "POSCAR").exists()]
    if missing:
        sys.exit(f"no FINAL_RESULTS_HEALTHY POSCAR for: {missing}")
    if a.status:
        status(names)
        return
    for name in names:
        out = build(name, a.grid, a.dz, a.breathing, a.dry_run)
        if out is not None and a.submit:
            submit(out)


if __name__ == "__main__":
    main()
