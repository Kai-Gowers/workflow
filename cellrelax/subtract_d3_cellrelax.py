#!/usr/bin/env python3
"""
Cell-relax stage 5a: bare-PBE-by-subtraction phonons for the 16 refined cells.

Thin, additive wrapper over the two production scripts of the subtraction pipeline — nothing in them
is changed; only their module-level input/output roots are redirected:

    reference  FINAL_RESULTS_HEALTHY          -> FINAL_RESULTS_CELLRELAX   (the 16 refined cells)
    output     FINAL_RESULTS_BARE_PBE_SUBTRACT -> FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX

so FINAL_RESULTS_BARE_PBE_SUBTRACT (the e20 training targets at the old strained cells) is never
touched. Materials = cellrelax/materials_16.txt.

    stage1  phonopy/subtract_d3_force_sets.py      (needs simple-dftd3 -> nequix uv env)
            cd ../nequix && uv run python ../workflow/cellrelax/subtract_d3_cellrelax.py stage1 [--overwrite]
    stage2  phonopy/postprocess_bare_pbe_subtract.py (phonopy + hiphive -> workflow conda env)
            python3 cellrelax/subtract_d3_cellrelax.py stage2 [--force-hiphive M ...] [--only M ...] [--summary-only]

Stage 2 applies the hiphive constraint-based refit wherever the reference FORCE_CONSTANTS was
hiphive-corrected (the 4 Γ-shear bilayers), plus --force-hiphive on named materials (same rule as
production 2026-10-06: clear the small ZA dips the plain bare result carries).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phonopy"))

REFERENCE = ROOT / "FINAL_RESULTS_CELLRELAX"
OUT_ROOT = ROOT / "FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX"
MATERIALS = [l.split()[0] for l in (ROOT / "cellrelax" / "materials_16.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]


def stage1(overwrite: bool) -> None:
    import subtract_d3_force_sets as S
    S.FINAL_RESULTS_HEALTHY = REFERENCE
    S.OUT_ROOT = OUT_ROOT
    OUT_ROOT.mkdir(exist_ok=True)
    rows = []
    for m in MATERIALS:
        info = S.process(m, overwrite)
        rows.append(info)
        print(f"{m:22s} ndisp={info['n_displacements']:3d} |F_D3(eq)|max={info['max_abs_F_d3_eq_eV_per_A']:.3f} "
              f"|dF_D3|max={info['max_abs_dF_d3_eV_per_A']:.4f} eV/A  ref_plain={info['reference_fc_is_plain_phonopy_rebuild']}",
              flush=True)
    (OUT_ROOT / "stage1_subtraction_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} materials to {OUT_ROOT}")


def stage2(force: list[str], only: list[str] | None, summary_only: bool, cutoff, lambdas) -> None:
    import postprocess_bare_pbe_subtract as P
    P.OUT_ROOT = OUT_ROOT
    P.FINAL_RESULTS_HEALTHY = REFERENCE
    lam = [float(x) for x in lambdas.split(",")] if lambdas else P.DEFAULT_LAMBDAS
    mats = only or MATERIALS
    rows = []
    for m in mats:
        rows.append(P.summarize(m) if summary_only else P.process(m, cutoff, lam, False, m in force))
    if only:
        return
    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_h = sum(r["hiphive_applied"] for r in rows)
    forced = [r["material"] for r in rows if r["hiphive_applied"] and not r["hiphive_required"]]
    worst = min(rows, key=lambda r: r["min_freq_final_THz"])
    (OUT_ROOT / "README.md").write_text(f"""# FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX — bare-PBE phonons at the 16 refined cells

Generated {date.today().isoformat()} by `cellrelax/subtract_d3_cellrelax.py` (stage1 / stage2), which runs the unchanged
`phonopy/subtract_d3_force_sets.py` and `phonopy/postprocess_bare_pbe_subtract.py` with the reference redirected to
`FINAL_RESULTS_CELLRELAX/` (ISIF=4-refined PBE+D3 cells, `cellrelax/README.md`) and the output to this directory.
Same definitions, file layout and columns as `FINAL_RESULTS_BARE_PBE_SUBTRACT/README.md`; the other 32 TMDs of the
48-material set stay there (their cells were already at the PBE+D3 equilibrium).

{len(rows)} materials; hiphive refit applied to {n_h} ({', '.join(r['material'] for r in rows if r['hiphive_applied'])})
{'incl. forced on ' + ', '.join(forced) if forced else ''}.
Lowest band-path frequency after post-processing: {worst['min_freq_final_THz']:+.4f} THz ({worst['material']}).
`rmse_vs_pbed3_THz` compares against `FINAL_RESULTS_CELLRELAX/<mat>/band.yaml` (the refined-cell PBE+D3 reference).
See `summary.csv`.
""")
    print(f"\nwrote {OUT_ROOT / 'summary.csv'} and README.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("stage1")
    s1.add_argument("--overwrite", action="store_true")
    s2 = sub.add_parser("stage2")
    s2.add_argument("--force-hiphive", nargs="*", default=[], metavar="MATERIAL")
    s2.add_argument("--only", nargs="*", default=None, metavar="MATERIAL",
                    help="(re)process only these; summary.csv/README are not rewritten — rerun with --summary-only")
    s2.add_argument("--summary-only", action="store_true")
    s2.add_argument("--cutoff", type=float, default=None)
    s2.add_argument("--lambdas", type=str, default=None)
    args = ap.parse_args()
    if args.stage == "stage1":
        stage1(args.overwrite)
    else:
        stage2(args.force_hiphive, args.only, args.summary_only, args.cutoff, args.lambdas)


if __name__ == "__main__":
    main()
