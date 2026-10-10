#!/usr/bin/env python3
"""
Cell-relax stage 6b: coworker bundle v3 — tmd48_bare_pbe_subtract with the 16 refined cells.

Thin additive wrapper over scripts/export_bare_pbe_subtract_dataset.py (unchanged). Same bundle layout,
labels and README as v2; only the per-material sources of the 16 cellrelax materials are redirected:

    bare-PBE results   FINAL_RESULTS_BARE_PBE_SUBTRACT/<m>  -> FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX/<m>
    displaced vasprun  phonopy_*_examples/<m>_staticpoint   -> phonopy_*_examples_cellrelax/<m>_staticpoint
    relaxation OUTCAR  *_examples/<m>                       -> *_examples_cellrelax/<m>
    split manifests    nequix_datasets/v4_tmd_only          -> nequix_datasets/v5_tmd_cellrelax (same lists)

(the PBE+D3 reference FINAL_RESULTS_HEALTHY/<m> already holds the refined cells since the stage-6 promotion),
summary.csv is merged from the two subtract dirs, and a "v3" section is appended to the README.

Needs simple-dftd3 (nequix uv env):
    cd ../nequix && uv run python ../workflow/cellrelax/export_share_bundle.py [--name tmd48_bare_pbe_subtract_v3] [--full]
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import export_bare_pbe_subtract_dataset as E  # noqa: E402

PROD_SUB = ROOT / "FINAL_RESULTS_BARE_PBE_SUBTRACT"
CELLRELAX_SUB = ROOT / "FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX"
M16 = [l.split()[0] for l in (ROOT / "cellrelax" / "materials_16.txt").read_text().splitlines()
       if l.strip() and not l.startswith("#")]
assert len(M16) == 16

# merged summary.csv: production rows, with the 16 replaced by the cellrelax rows
_prod = list(csv.DictReader((PROD_SUB / "summary.csv").open()))
_cell = {r["material"]: r for r in csv.DictReader((CELLRELAX_SUB / "summary.csv").open())}
assert set(_cell) == set(M16), set(_cell) ^ set(M16)
_merged_rows = [_cell[r["material"]] if r["material"] in M16 else r for r in _prod]
_tmp = Path(tempfile.mkdtemp(prefix="bundle_v3_")) / "summary.csv"
with _tmp.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(_prod[0].keys()))
    w.writeheader()
    w.writerows({k: r.get(k, "") for k in _prod[0].keys()} for r in _merged_rows)
N_FLAGGED = sum(r["hiphive_required"] == "True" for r in _merged_rows)


class _RoutedSource:
    """Stands in for E.SOURCE: `SOURCE / name` and `SOURCE.iterdir()` are all the script uses."""

    def __truediv__(self, name: str) -> Path:
        if name == "summary.csv":
            return _tmp
        return (CELLRELAX_SUB if name in M16 else PROD_SUB) / name

    def iterdir(self):
        return PROD_SUB.iterdir()

    def __str__(self) -> str:
        return f"{PROD_SUB} (+{CELLRELAX_SUB} for {len(M16)} materials)"


_orig_staticpoint_dir, _orig_relaxation_dir, _orig_readme = E.staticpoint_dir, E.relaxation_dir, E.readme


def staticpoint_dir(name: str) -> Path:
    if name in M16:
        for d in (ROOT / "phonopy_monolayer_examples_cellrelax" / f"{name}_staticpoint",
                  ROOT / "phonopy_bilayer_examples_cellrelax" / f"{name}_staticpoint"):
            if d.is_dir():
                return d
        sys.exit(f"{name}: no cellrelax staticpoint directory on this host")
    return _orig_staticpoint_dir(name)


def relaxation_dir(name: str):
    if name in M16:
        for d in (ROOT / "monolayer_examples_cellrelax" / name, ROOT / "bilayer_examples_cellrelax" / name):
            if (d / "OUTCAR").exists():
                return d
        return None
    return _orig_relaxation_dir(name)


V3_NOTE = f"""
## v3 (2026-10-10): 16 materials re-referenced at ISIF=4-refined cells

Same 48 names, same split, same labels and training protocol as `tmd48_bare_pbe_subtract_v2`; what changed:

- **16 materials** — MoS2, WS2, WSe2, MoS2_bilayer_2H, WS2_bilayer_2H, WS2_bilayer_3R, MoS2_WS2_2H, MoS2_MoSe2_3R,
  MoSe2_WS2_2H, MoSe2_WS2_3R, WS2_WSe2_2H, WS2_WSe2_3R, WSe2_WTe2_2H, WSe2_WTe2_3R, MoTe2_WSe2_2H, MoTe2_WSe2_3R —
  previously sat at a Materials-Project (monolayers, homobilayers) or lattice-mismatch-compromise (heterobilayers)
  in-plane `a` that left **+0.7 to +2.1 GPa residual PBE+D3 in-plane stress** in the relaxed cell. They were
  re-relaxed with a fixed-volume cell+ion relaxation (VASP ISIF=4, same settings), then ISIF=2 at the plateaued `a`
  (Δa = −0.5 to −1.2 %), and the phonons recomputed (4×4×1, 0.01 Å, same settings). Residual PBE+D3 in-plane
  stress is now +0.03 to +0.25 GPa; the bare-PBE unit-cell stress in `equilibrium.extxyz` / `materials.csv`
  moved accordingly (it is still −0.9 to −2.1 GPa in-plane: that is the D3 contribution, as explained above).
- Every file of those 16 is new: `materials/<name>/*`, their frames in `displacements.extxyz`, their rows in
  `equilibrium.extxyz` and `materials.csv`. The other 32 materials are unchanged from v2.
- Refined-cell phonons differ from the v2 references by 0.05–0.24 THz RMS over the band path (top optical modes
  +0.10 to +0.16 THz). Four of the 16 (MoSe2_WS2_2H, MoSe2_WS2_3R, WS2_WSe2_3R, WSe2_WTe2_3R) showed the
  doubly-degenerate Γ shear artefact at the new cells (−0.2 to −0.5 THz) that the tensile strain had masked, and
  went through the same hiphive rotational-sum-rule refit as the other hiphive materials; MoS2, MoS2_MoSe2_3R and
  WSe2_WTe2_2H got the forced refit for small flexural dips. The hiphive counts in "How bare PBE was obtained"
  include these.
- The PBE+D3 parent bundle `tmd48_phonon_dataset_v1` still has these 16 at the old strained cells.
"""


def readme(n_mat, n_frames, n_hiphive, forced, full):
    txt = _orig_readme(n_mat, n_frames, n_hiphive, forced, full)
    txt = txt.replace("the 20 whose PBE+D3 reference had itself been hiphive-corrected",
                      f"the {N_FLAGGED} whose PBE+D3 reference had itself been hiphive-corrected")
    marker = "## Contact"
    return txt.replace(marker, V3_NOTE.lstrip("\n") + "\n" + marker) if marker in txt else txt + V3_NOTE


E.SOURCE = _RoutedSource()
E.SPLIT_ROOT = ROOT / "nequix_datasets" / "v5_tmd_cellrelax"
E.staticpoint_dir = staticpoint_dir
E.relaxation_dir = relaxation_dir
E.readme = readme

if __name__ == "__main__":
    if "--name" not in sys.argv:
        sys.argv += ["--name", "tmd48_bare_pbe_subtract_v3"]
    E.main()
