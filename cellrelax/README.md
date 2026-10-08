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
