# Cell-relaxation campaign (`TWIST_VARIANT=cellrelax`) — 16 strained TMD reference cells

**Why (2026-10-08).** Of the 48 TMD cells in `FINAL_RESULTS_HEALTHY`, 32 sit at the PBE+D3 equilibrium
(|σ_xx| < 0.3 GPa) but 16 carry +0.7…+2.1 GPa residual in-plane stress (ISIF=2 relaxations at a Materials-Project
`a` for the monolayers/homobilayers, or at the lattice-mismatch compromise `a` for heterobilayers). Those 16 never
showed a Γ dip (tensile strain stiffens the flexural mode) so they never went through the July ISIF=4 fix. A model
with correct stress (Nequix `-baresub-e20` + D3) relaxes them by Δa ≈ −1 % and its phonons leave the reference, so
an all-48 cell-relaxation benchmark is unfair. Goal: references for all 48 at the PBE+D3 equilibrium cell, for a
paper. Numbers: `nequix/explicit_dispersion/data/v4_tmd_only_bare_subtract/summary.csv`, list: `materials_16.txt`.

**Additive only.** New files: `cellrelax/{README.md,materials_16.txt,setup_isif4.py,isif4_jobs.json}`, host-local
template copies `common/{relaxation,staticpoint}_templates_cellrelax/` (gitignored, identical to production).
Zero edits to existing scripts, templates, registries or data files; with `TWIST_VARIANT` unset nothing changes.
Production `monolayer_examples/`, `bilayer_examples/`, `FINAL_RESULTS*/` are read, never written.

**Stages**
1. `TWIST_VARIANT=cellrelax python3 cellrelax/setup_isif4.py --submit` → `*_examples_cellrelax/<name>_isif4/`
   (POSCAR = production CONTCAR; ISIF=4, IBRION=1, EDIFFG=-1E-7, NSW=400, IVDW=12, same ENCUT/KPOINTS; 2 nodes
   monolayers, 4 nodes bilayers as in July). Expect ZBRENT near the end; CONTCAR is still the last good step.
2. Extract the plateaued `a` with `common/isif4_lattice_extract.py <dir> --window 50` (check std and c drift);
   write bilayer `a` to `data/bilayer_lattice_overrides_cellrelax.json` (variant-aware, read only by this variant);
   for the 3 monolayers start the ISIF=2 production re-relaxation directly from the ISIF=4 CONTCAR
   (`mp_material_overrides.json` is deliberately not variant-aware — do not edit it).
3. ISIF=2 re-relaxation in the variant (`monolayer_examples_cellrelax/<m>/`, `bilayer_examples_cellrelax/<b>/`;
   the unchanged constituent monolayers MoSe2/MoTe2/WTe2 are copied from production so the bilayer builder finds them).
4. phonopy `--dim "4 4 1"` as production → `FINAL_RESULTS_CELLRELAX/<name>/` (tracked, commit like `FINAL_RESULTS/`).
5. Curate: 32 from `FINAL_RESULTS_HEALTHY` + 16 from `FINAL_RESULTS_CELLRELAX` → new Nequix dataset version
   (never overwrite `v4_tmd_only`), retrain e20, rerun the template-relax evals.
6. **Promotion to production (after the bands at the new cells are verified stable; user 2026-10-08: the correct lattice
   constants belong in production).** Write the 3 monolayer `(a, c, dMX)` into `data/mp_material_overrides.json` (as MoSe2/
   MoTe2/WTe2 were in July) and the 13 bilayer `a` into `data/bilayer_lattice_overrides.json`; replace the 16 entries in
   `FINAL_RESULTS_HEALTHY/` with the `FINAL_RESULTS_CELLRELAX/` ones (archive the old 16 untracked + git history); note in
   `CLAUDE.md` that structures generated after that date use the refined cells while all earlier results (incl. twisted
   m1–m3 at a = 3.1922 Å) sit at the old ones. Regenerating an old template is then no longer byte-identical — intended.

**Stage 2-3 as run (2026-10-09), `cellrelax/setup_stage2.py {extract,monolayers,bilayers} [--submit]`.**
- `a` = mean of the last **10** ISIF=4 steps (not 50): WSe2 and WS2_bilayer_3R hung after 26/29 steps, so a 50-step
  window would include the initial transient; elsewhere 10 vs 50 agree to ≤0.0005 Å. All std < 0.001 Å.
  Five runs (MoS2, WS2, WSe2, MoS2/WS2 homobilayers) hit silent VASP hangs, but `a` had plateaued first.
  Results: `cellrelax/isif4_lattice.json`; bilayers → `data/bilayer_lattice_overrides_cellrelax.json`.
- Monolayer POSCAR = ISIF=4 CONTCAR rescaled in-plane to the plateau `a` **and c reset to 20 Å** (Cartesian offsets
  from the midplane kept, so dMX is unchanged). Fixed-volume ISIF=4 grows c to 20.3–20.5 Å, and the bilayer builder
  reuses monolayer fractional z assuming c = 20 Å; using the raw CONTCAR would distort dMX in the bilayers by ~2 %.
- Jobs: `cellrelax/stage3_jobs.json`.

**Plan additions (2026-10-09), beyond the original stages.** The 16 also live in two more tracked dirs:
- `FINAL_RESULTS/` (raw source of `FINAL_RESULTS_HEALTHY`): at stage 6, replace the 16 there too (old copies archived
  untracked + git history), so the two dirs never disagree.
- `FINAL_RESULTS_BARE_PBE_SUBTRACT/` (e20 training targets) was built at the old strained cells: at stage 5, rerun the
  D3 subtraction on the 16 `FINAL_RESULTS_CELLRELAX` FORCE_SETS (into a variant/new dir, not over the old one) and
  build the new dataset from that, before retraining e20.
- `FINAL_RESULTS_BARE_PBE/` (retired IVDW-off campaign, 6 of the 16) stays frozen.
- Stage 4 as run: 153 statics via `TWIST_VARIANT=cellrelax phonopy/prepare_and_submit.py --{monolayer,bilayer} <name>
  --dim "4 4 1"` per name (not `--all`: the variant dirs also hold `*_isif4` and the copied MoSe2/MoTe2/WTe2).

**Stage 4 as run (2026-10-10).** All 153 statics COMPLETED and converged; `TWIST_VARIANT=cellrelax
phonopy/postprocess_results.py --{monolayer,bilayer} --all` → `FINAL_RESULTS_CELLRELAX/` (16 dirs, tracked).
Plain phonopy: 12/16 clean; 4 bilayers (MoSe2_WS2_2H/3R, WS2_WSe2_3R, WSe2_WTe2_3R) showed the doubly-degenerate Γ
shear dip (−0.21…−0.51 THz) that the old tensile cells had masked. `hiphive_fit_force_constants.py --bilayer` (the
production method, 20/48 in `FINAL_RESULTS_HEALTHY`) cleared all four to exactly 0.0000 THz at every λ (fit RMSE
≤ 0.0025 eV/Å, cutoff 5.8–6.1 Å; pre-fix files in `backups/pre_hiphive_fit_correction_*_20261010*`). Verified all 16 on a
Γ-centred 96×96×1 mesh and a fine Γ→K / Γ→M scan (0–6 %): worst residual −0.033 THz (MoS2_MoSe2_3R), MoS2 −0.019,
WSe2_WTe2_2H −0.013 — same ZA-noise class the production references carry, left as plain phonopy (not force-hiphive'd).
Effect of the −0.5…−1.2 % cell change vs `FINAL_RESULTS_HEALTHY`: RMS band shift 0.05–0.24 THz, top optical modes
+0.1…+0.16 THz. Next: stage 5 (D3-subtract the 16 FORCE_SETS into a new dir, build dataset, retrain e20).

**Stage 6 (promotion) as run 2026-10-10 — PARTIAL, see below.**
- `data/mp_material_overrides.json`: MoS2 / WS2 / WSe2 now carry full `(a, c=20, dMX)` entries (a from `isif4_lattice.json`,
  dMX from `monolayer_examples_cellrelax/<m>/CONTCAR`); `data/bilayer_lattice_overrides.json`: the 13 bilayer `a` added
  (copied from the variant file, 52 entries now). `template_structures/` regenerated (generator reproduces the previous
  tracked files byte-for-byte before the change; after it, 33 of 48 templates differ: the 16 + 17 more bilayers built
  from MoS2/WS2/WSe2 whose template dMX moves by ≤ 0.025 Å, and MoS2_MoSe2_2H whose template `a` goes from +0.72 % to
  +0.14 % off its DFT reference — same ripple as the July MoSe2/MoTe2/WTe2 overrides). `twisted/build_twisted_bilayer.py`
  also reads these overrides, so new MoS2 moiré cells will be built at a = 3.1551 Å (m1–m3 were 3.1922 Å).
- **Not yet done (tool permission denied, user to run):** replace the 16 `FINAL_RESULTS_HEALTHY/<m>` and `FINAL_RESULTS/<m>`
  entries with `FINAL_RESULTS_CELLRELAX/<m>` (old copies → `backups/pre_cellrelax_promotion_2026-10-10/`). Until then the two
  production reference dirs still hold the old strained cells for the 16 and every eval that reads `FINAL_RESULTS_HEALTHY`
  compares against them.

**Stage 5 as run 2026-10-10.**
- `cellrelax/subtract_d3_cellrelax.py stage1|stage2` (wrapper; runs the unchanged subtraction scripts with reference →
  `FINAL_RESULTS_CELLRELAX`, output → `FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX/`, tracked). hiphive refit on the 4 flagged
  + forced on MoS2 (plain bare −0.098), MoS2_MoSe2_3R (−0.061), WSe2_WTe2_2H (−0.032) by the 2026-10-06 rule; all 16 final
  ≥ −1e-4 THz; RMS vs refined-cell PBE+D3 0.006–0.040 THz.
- `nequix_datasets/v5_tmd_cellrelax/` = v4 manifests (39/3/6) + README row; training db built in nequix by
  `explicit_dispersion/build_bare_subtract_dataset_cellrelax.py` → `data/v5_tmd_cellrelax_bare_subtract/` (32 rows
  byte-identical to v4; the 16 at the refined cells, PBE+D3 σ_xx now +0.03…+0.25 GPa, was +0.7…+2.1).
- Retrain: `nequix-healthy-2d-pft-v5-tmd-cellrelax-omat-baresub-e20.yml`, job 3124312. Evals after it finishes.
