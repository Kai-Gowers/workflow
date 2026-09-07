#!/usr/bin/env python3
"""
Generate in-plane biaxially strained starting structures for the strain-augmentation
pilot (2026-09-07). Takes each material's canonical relaxed POSCAR from
FINAL_RESULTS_HEALTHY/<material>/POSCAR, scales the two in-plane lattice vectors by
(1 + strain) while leaving fractional (Direct) coordinates and the c-axis vacuum
vector unchanged, and writes a fresh relaxation input directory under
bilayer_examples/strain_<label>_<material>/ (POSCAR + POTCAR + INCAR + KPOINTS +
bat, reusing common/relaxation_templates/ unmodified -- the default ISIF=2 relaxes
ions only, so the strained cell stays fixed and only internal coordinates relax).

Naming keeps the stacking suffix (_2H/_3R) at the END of the directory name
(strain label as a PREFIX, not suffix) so scripts/convert_to_nequix.py's
is_bilayer() (a plain name.endswith("_2H"/"_3R") check) still classifies these
correctly -- see the pre-existing TM_H/TM_H2 suffix bug this same check was fixed
for.
"""

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "relaxation" / "monolayer"))
sys.path.insert(0, str(REPO_ROOT / "common"))

from generate_potcar import generate_potcar
from incar_utils import customize_incar

MATERIALS = [
    "WS2_WSe2_2H", "WS2_WSe2_3R",
    "MoSe2_WSe2_2H", "MoSe2_WSe2_3R",
    "MoTe2_WS2_2H", "MoTe2_WS2_3R",
]
STRAINS = [("m2pct", -0.02), ("m1pct", -0.01), ("p1pct", 0.01), ("p2pct", 0.02)]

TEMPLATE_DIR = REPO_ROOT / "common" / "relaxation_templates"
FINAL_HEALTHY = REPO_ROOT / "FINAL_RESULTS_HEALTHY"
OUT_BASE = REPO_ROOT / "bilayer_examples"


def make_strained_poscar(src_lines: list[str], strain: float) -> list[str]:
    out = list(src_lines)
    for i in (2, 3):  # lines 2,3 = in-plane lattice vectors a1,a2; line 4 (c) untouched
        parts = src_lines[i].split()
        scaled = [f"{float(x) * (1 + strain):.16f}" for x in parts]
        out[i] = "  " + "   ".join(scaled)
    return out


def main():
    created = []
    for mat in MATERIALS:
        src_poscar = FINAL_HEALTHY / mat / "POSCAR"
        if not src_poscar.exists():
            print(f"  SKIP {mat}: no FINAL_RESULTS_HEALTHY/{mat}/POSCAR")
            continue
        src_lines = src_poscar.read_text().splitlines()
        for label, eps in STRAINS:
            name = f"strain_{label}_{mat}"
            outdir = OUT_BASE / name
            outdir.mkdir(parents=True, exist_ok=True)

            strained_lines = make_strained_poscar(src_lines, eps)
            (outdir / "POSCAR").write_text("\n".join(strained_lines) + "\n")

            generate_potcar(outdir / "POSCAR", outdir / "POTCAR")
            customize_incar(TEMPLATE_DIR / "INCAR", outdir / "INCAR", name, suffix="relaxation")
            shutil.copy(TEMPLATE_DIR / "KPOINTS", outdir / "KPOINTS")
            shutil.copy(TEMPLATE_DIR / "bat", outdir / "bat")

            created.append(name)
            print(f"  ✓ {name}")

    print(f"\nCreated {len(created)} strained relaxation inputs under bilayer_examples/")
    return created


if __name__ == "__main__":
    main()
