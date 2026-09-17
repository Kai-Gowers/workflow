#!/usr/bin/env python3
"""
Write the workflow's *pre-relaxation template* starting structure for each material in a
manifest -- the same kind of unrelaxed POSCAR the pipeline hands to VASP -- so a model-side
(Nequix) end-to-end benchmark can relax from a structure that has never seen DFT.

This is a thin wrapper over the existing generators, applying the production (relaxed-path)
parameter precedence: material overrides first, then the Materials Project cache.

  monolayer  -> relaxation/monolayer/generate_monolayer_poscar.generate_poscar(...)
                (a, dMX from mp_material_overrides.json if a full entry exists, else the MP cache)
  bilayer    -> relaxation/bilayer/generate_bilayer_poscar._generate_bilayer_poscar_template(...)
                a   = bilayer_lattice_overrides.json entry if present, else mean of the two
                      materials' a (anchor=None rule)
                dMX = mean of the two materials' dMX (template convention)
                dz  = per-material dz from mp_material_overrides.json if present, else
                      DZ_BILAYER (3.5 A); heterobilayers use the mean (same as _default_params)
                c   = 20 A

The public generate_bilayer_poscar() is not used because its default path stacks DFT-relaxed
monolayer CONTCARs (DFT geometry would leak into the start), and its template path ignores the
bilayer lattice overrides and takes dz from the MP cache (still the 6.5 A placeholder for
MoS2/WS2/WSe2).

Output: <output-dir>/<material>/POSCAR + template.json, and <output-dir>/summary.csv comparing the
template a / interlayer gap with the DFT-relaxed FINAL_RESULTS_HEALTHY geometry.

Usage (from workflow/):
  python3 scripts/generate_template_start_structures.py            # 48 v4_tmd_only materials
  python3 scripts/generate_template_start_structures.py --materials MoS2 MoS2_bilayer_2H
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "relaxation" / "monolayer"))
sys.path.insert(0, str(REPO_ROOT / "relaxation" / "bilayer"))
sys.path.insert(0, str(REPO_ROOT / "common"))

from generate_monolayer_poscar import generate_poscar  # noqa: E402
from generate_bilayer_poscar import (  # noqa: E402
    VACUUM,
    _default_params,
    _generate_bilayer_poscar_template,
    _load_bilayer_overrides,
    parse_bilayer_name,
)
from materials_project_api import get_material_lattice_params  # noqa: E402
from structural_families import STACKING_SUFFIXES  # noqa: E402

FINAL_RESULTS_HEALTHY = REPO_ROOT / "FINAL_RESULTS_HEALTHY"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "nequix_datasets" / "v4_tmd_only"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "template_structures"


def read_manifest_dir(dataset_dir: Path) -> list[str]:
    names = []
    for split in ("train", "val", "holdout", "test"):
        path = dataset_dir / split / "materials.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return sorted(set(names))


def is_bilayer_name(name: str) -> bool:
    return "_bilayer_" in name or any(name.endswith(f"_{s}") for s in STACKING_SUFFIXES)


def poscar_geometry(path: Path) -> dict:
    """In-plane a, c and surface-to-surface interlayer gap (0 for a monolayer) of a POSCAR."""
    from ase.io import read

    atoms = read(str(path))
    z = np.sort(atoms.positions[:, 2])
    gap = float(np.diff(z).max()) if len(atoms) > 3 else 0.0
    return {"a": float(atoms.cell.lengths()[0]), "c": float(atoms.cell.lengths()[2]), "gap": gap}


def material_params(name: str) -> tuple[float, float, float, str]:
    """(a, dMX, dz, source) with the production precedence: override -> MP cache for a/dMX,
    override dz -> DZ_BILAYER for dz."""
    params, meta = get_material_lattice_params(name, use_cache=True)
    if params is None:
        raise RuntimeError(f"{name}: no lattice params (reason={meta.get('reason')}); set MP_API_KEY?")
    a, _c, _dz_mp, dmx = params
    _, _, dz, _ = _default_params(name)  # override dz, else DZ_BILAYER -- never the MP-cache dz
    return float(a), float(dmx), float(dz), meta.get("source", "?")


def build_one(name: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    poscar = out_dir / "POSCAR"

    if is_bilayer_name(name):
        mat1, mat2, stacking = parse_bilayer_name(name)
        a1, dmx1, dz1, src1 = material_params(mat1)
        a2, dmx2, dz2, src2 = material_params(mat2)
        override = _load_bilayer_overrides().get(name)
        if isinstance(override, dict) and "a" in override:
            a, a_source = float(override["a"]), "bilayer_lattice_override"
        else:
            a, a_source = (a1 + a2) / 2.0, f"mean({src1},{src2})"
        dmx = (dmx1 + dmx2) / 2.0
        dz = (dz1 + dz2) / 2.0
        _generate_bilayer_poscar_template(mat1, mat2, stacking, poscar, a, VACUUM, dz, dmx, source=a_source)
        meta = {
            "material": name, "kind": "bilayer", "mat1": mat1, "mat2": mat2, "stacking": stacking,
            "a_template": a, "a_source": a_source, "dMX_template": dmx,
            "dz_template": dz, "dz_source": f"{mat1}:{dz1:.3f},{mat2}:{dz2:.3f} (override dz else 3.5)",
            "c": VACUUM,
        }
    else:
        result = generate_poscar(name, out_dir, filename="POSCAR", use_mp=True)
        a, dmx, _dz, src = material_params(name)
        meta = {
            "material": name, "kind": "monolayer", "a_template": a, "a_source": src,
            "dMX_template": dmx, "dz_template": None, "dz_source": None, "c": VACUUM,
            "generator_source": str(result.get("source", "")),
        }

    tmpl = poscar_geometry(poscar)
    if abs(tmpl["a"] - meta["a_template"]) > 1e-6:
        raise RuntimeError(f"{name}: written a={tmpl['a']:.6f} != resolved a={meta['a_template']:.6f}")
    dft_poscar = FINAL_RESULTS_HEALTHY / name / "POSCAR"
    if dft_poscar.exists():
        dft = poscar_geometry(dft_poscar)
        meta.update({
            "a_dft": dft["a"], "da_pct": 100.0 * (tmpl["a"] - dft["a"]) / dft["a"],
            "gap_template": tmpl["gap"], "gap_dft": dft["gap"], "dgap_A": tmpl["gap"] - dft["gap"],
            "c_dft": dft["c"],
        })
    (out_dir / "template.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--materials", nargs="*", default=None, help="Material names (default: manifest dir)")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR),
                        help="nequix_datasets/<split> dir whose */materials.txt define the set")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    names = args.materials or read_manifest_dir(Path(args.manifest_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        try:
            rows.append(build_one(name, output_dir / name))
        except Exception as e:  # keep going, report at the end
            print(f"  x {name}: {e}", file=sys.stderr)

    if not rows:
        sys.exit("nothing generated")

    fields = ["material", "kind", "a_template", "a_source", "a_dft", "da_pct",
              "gap_template", "gap_dft", "dgap_A", "dz_template", "dMX_template", "c_dft"]
    with open(output_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'material':<20}{'a_tmpl':>9}{'a_dft':>9}{'da%':>8}{'gap_tmpl':>10}{'gap_dft':>9}{'dz_src':>8}  a_source")
    for r in rows:
        print(f"{r['material']:<20}{r['a_template']:>9.4f}{r.get('a_dft', float('nan')):>9.4f}"
              f"{r.get('da_pct', float('nan')):>8.2f}{r.get('gap_template', 0.0):>10.3f}"
              f"{r.get('gap_dft', float('nan')):>9.3f}{(r['dz_template'] or 0.0):>8.3f}  {r['a_source']}")
    print(f"\n{len(rows)}/{len(names)} written under {output_dir}")


if __name__ == "__main__":
    main()
