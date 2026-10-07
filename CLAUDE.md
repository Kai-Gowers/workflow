# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational materials science workflow for automated phonon dispersion calculations of 2D materials (monolayers and bilayers). It integrates with the Materials Project API to fetch crystal structures, generates VASP inputs, submits jobs to a SLURM HPC cluster, and post-processes phonopy results into band structures.

The full pipeline for a material is:
1. Fetch/generate structure → 2. Relax with VASP → 3. Displace atoms (phonopy) → 4. Run static VASP on each displacement → 5. Post-process into band structure

## Environment

```bash
export MP_API_KEY=<your_key>   # Required for Materials Project API calls
```

Dependencies: `phonopy` (CLI tool), `mp-api`, `pymatgen`, VASP (external), SLURM (external).

Host-specific settings (POTCAR paths, etc.) live in `common/host_config.py` and
auto-detect NERSC vs Andromeda. Override with `TWIST_HOST=nersc|andromeda` or
`TWIST_POTCAR_PATH=/path/to/potpaw_PBE.64`. SLURM/`bat` templates remain
host-local under `common/*_templates/` (gitignored).

## Common Commands

**Monolayer relaxation:**
```bash
cd relaxation/monolayer
python3 new_monolayer_relax.py                              # Interactive single job
python3 generate_monolayer_poscar.py MoS2 --structure      # Generate POSCAR only
```

**Bilayer relaxation:**
```bash
cd relaxation/bilayer
python3 create_bilayer_example.py MoS2_MoS2_3R             # Single bilayer
python3 create_all_bilayers.py --max 10                    # Batch creation
```

**Phonopy (all-in-one orchestrator):**
```bash
cd phonopy
python3 prepare_and_submit.py --monolayer MoS2             # Single material
python3 prepare_and_submit.py --monolayer --all --no-submit # All, no submit
python3 prepare_and_submit.py --bilayer MoS2_MoS2_3R --dry-run
```

**Phonopy (step-by-step):**
```bash
python3 phonopy/monolayer/prepare_staticpoint.py MoS2 --dim "3 3 1"
python3 phonopy/monolayer/setup_displacements.py MoS2_staticpoint
python3 phonopy/postprocess_results.py --monolayer MoS2_staticpoint
```

**Batch management:**
```bash
python3 scripts/batch_management/create_batches.py --require-p63mmc --summary
python3 scripts/batch_management/submit_batch.py 1         # Submit batch 1
python3 scripts/batch_management/report_symmetry_eligibility.py
```

## Batch Workflow (in order)

Run these three steps in sequence to go from relaxed structures to final band structures:

**Step 1 — Submit relaxation jobs:**
```bash
python3 scripts/batch_management/submit_batch.py --<monolayer/bilayer> <batch_number>
```

**Step 2 — Create displacements and submit staticpoint calculations:**
```bash
python3 phonopy/submit_batch.py --<monolayer/bilayer> --batch <batch_number>
```

**Step 3 — Post-process into band structures (output goes to `FINAL_RESULTS/`):**
```bash
python3 phonopy/postprocess_batch.py --<monolayer/bilayer> --batch <batch_number>
```

**Step 4 — Convert to Nequix training format:**
```bash
python3 scripts/convert_to_nequix.py \
    --output nequix_datasets/vN_<material-count>materials/dataset.aselmdb
```
Each version directory is a permanent, immutable snapshot — never regenerate into an
existing version dir. The script auto-writes a `materials.txt` manifest alongside
`dataset.aselmdb` listing every material included. See `nequix_datasets/README.md`.

**Template start structures (for Nequix end-to-end evals):**
```bash
python3 scripts/generate_template_start_structures.py      # 48 v4_tmd_only materials -> template_structures/
```
Writes the workflow's *pre-relaxation* POSCAR for each material (the same kind of unrelaxed
structure VASP is given), using the production parameter precedence (material/bilayer overrides
first, then the MP cache; dz from overrides else 3.5 Å). `template_structures/` is tracked;
`summary.csv` compares template vs DFT-relaxed `a` and interlayer gap. Consumed by
`nequix/scripts/eval_healthy2d_finetune.py --relax positions|cell`. Structure generation of any
kind belongs here in `workflow/`, not in `nequix/`.

**Twisted bilayers (model-only phonon inputs for Nequix):**
```bash
python3 twisted/build_twisted_bilayer.py            # MoS2, (m, m+1) series m=1..5, near0 + near60 -> twisted/structures/
python3 twisted/build_twisted_bilayer.py --m 2 --family near0 --png
```
Builds commensurate twisted homobilayers from the existing 1x1 generators (`get_monolayer_coords` +
`apply_bilayer_stacking`), rotating the top layer by +θ(m, n) about the layer-1 metal with an exact
commensurability assert. Families: `near0` (3R-seeded), `near60` (2H-seeded). One in-plane `a` for
both layers from the MP cache (the stacking-specific bilayer lattice override is deliberately not
applied); interlayer gap from the DFT-relaxed homobilayer POSCAR. Each `twist.json` records θ, atom
count and the recommended phonopy `k` (`ceil(25 Å / a_moiré)`, capped at 900 supercell atoms).
Consumed by `nequix/scripts/eval_twisted_phonons.py`. `twisted/structures/` is tracked.

**Cleanup:**
```bash
python3 scripts/maintenance/cleanup_all.py                 # Remove all generated dirs
python3 scripts/maintenance/list_jobs.py                   # Show tracked SLURM jobs
```

## Architecture

### Module Layout

- **`common/`** — shared library used by all other modules
- **`relaxation/`** — VASP input generation and job submission (monolayer + bilayer subdirs)
- **`phonopy/`** — phonopy displacement setup, job submission, and post-processing
- **`scripts/`** — batch management and maintenance utilities
- **`data/`** — JSON data files (caches, job registry, batch assignments)
- **`FINAL_RESULTS/`** — final band structures and force constants per material

### Key Source Files

**`common/materials_project_api.py`** is the core library. It handles:
- Fetching structures from the MP API with caching (`data/mp_structure_cache.json`, `data/mp_lattice_params_cache.json`)
- Validating hexagonal P6₃/mmc symmetry
- Extracting lattice parameters (a, c, dz, dMX) for POSCAR generation
- Applying per-material overrides from `data/mp_material_overrides.json`

**`common/structural_families.py`** classifies materials into families (TMD, binary honeycomb like BN/GaN, single-element like graphene) and defines which stacking configurations are valid for each bilayer pair. Valid stackings: `3R`, `2H`, `AB`, `BA`, `TM_H`, `TM_H2`.

**`phonopy/prepare_and_submit.py`** is the main entry point for phonon calculations — it coordinates staticpoint preparation, phonopy displacement generation, and optional SLURM submission.

**`phonopy/postprocess_results.py`** collects `vasprun.xml` from displacement folders, builds `FORCE_SETS`, calls phonopy to generate the band structure, and copies outputs to `FINAL_RESULTS/`.

**Band-path gotcha (fixed 2026-09-17):** POSCARs are 60° cells, but phonopy's CLI default `PRIMITIVE_AXES = AUTO`
re-expresses band q-points in a 120° standardized primitive cell (`primitive_matrix` in `phonopy.yaml`). In that basis
K = (1/3, 1/3, 0), not (2/3, 1/3, 0). `band.conf` now uses the 120° coordinates; all `FINAL_RESULTS*/*/band.{yaml,pdf}`
were regenerated from the stored `FORCE_CONSTANTS` with `phonopy/regenerate_band_structures.py` (force constants and
stability verdicts unchanged). Files produced before that date have a mislabelled "K" tick and never sampled true K.

### Data Files

- **`data/batches/`** — batch assignments, one JSON file per batch (`monolayer_batch_<N>.json`, `bilayer_batch_<N>.json`); add a new batch by dropping in a new file
- **`data/job_registry.json`** — SLURM job tracking (IDs, timestamps, paths); modified frequently
- **`data/symmetry_eligible_materials.json`** — materials passing the P6₃/mmc filter
- **`data/mp_material_overrides.json`** — manual lattice param overrides for problematic MP entries

### Generated Directories (gitignored)

Running the workflow creates these top-level directories:
- `monolayer_examples/` — relaxation inputs per monolayer
- `bilayer_examples/` — relaxation inputs per bilayer stacking
- `phonopy_monolayer_examples/` — phonopy staticpoint + disp-XXX dirs
- `phonopy_bilayer_examples/` — same for bilayers

Each `disp-XXX/` folder inside a staticpoint dir is a separate VASP calculation.

### Keep the `workflow/` top level clean

Keep the top level of `workflow/` to the core folders:
- code: `common/`, `phonopy/`, `relaxation/`, `scripts/`, `twisted/`
- inputs: `data/`, `template_structures/`
- results: `FINAL_RESULTS*/`, `nequix_datasets/`
- the standard `*_examples*/` run directories above
- `backups/`

Do not create new top-level folders for anything else:
- superseded or archived runs and one-off previews go in `backups/<campaign>/` (gitignored)
- scratch analyses and plots go in a scratch directory outside `workflow/`
- campaign scripts go under `scripts/`

Ask before adding a new top-level folder.

### VASP Templates

Templates for INCAR, KPOINTS, and SLURM batch scripts live in (gitignored; host-local):
- `common/relaxation_templates/` — for structural relaxation (ionic + cell DOF)
- `common/staticpoint_templates/` — for static calculations (IBRION=-1, NSW=0)

Perlmutter defaults (`vasp/6.6.1-cpu`, account `m5370`, queue `regular`,
5h walltime, 21 MPI ranks/node × 6 OpenMP):
- relaxation: 2 nodes, 42 ranks, `KPAR=7`, `NCORE=6`
- staticpoint: 6 nodes, 126 ranks, `KPAR=21`, `NCORE=6`

### Material List

`common/materials_list.txt` is the master list of 16 materials. Bilayer combinations are generated from all valid pairs using `generate_bilayer_combinations.py`.

## Run variants (`TWIST_VARIANT`) and the bare-PBE campaign

`common/run_variant.py` lets a second, independent campaign run from this same checkout. With the variable
unset the pipeline behaves exactly as before (verified byte-for-byte on a 23-command regression suite on
2026-09-19). With `export TWIST_VARIANT=<name>` (lowercase `[a-z0-9_]`), every generated path gets a suffix:

| unset | `TWIST_VARIANT=bare_pbe` |
|---|---|
| `monolayer_examples/`, `bilayer_examples/` | `monolayer_examples_bare_pbe/`, `bilayer_examples_bare_pbe/` |
| `phonopy_monolayer_examples/`, `phonopy_bilayer_examples/` | `..._bare_pbe/` |
| `FINAL_RESULTS/` (tracked) | `FINAL_RESULTS_BARE_PBE/` (tracked) |
| `common/relaxation_templates/`, `common/staticpoint_templates/` | `common/relaxation_templates_bare_pbe/`, `common/staticpoint_templates_bare_pbe/` (host-local, gitignored) |
| `data/job_registry.json` | `data/job_registry_bare_pbe.json` |

Exception: bilayer in-plane `a` corrections are variant-aware. A variant reads only
`data/bilayer_lattice_overrides_<variant>.json` and falls back to the relaxed-monolayer `a`, never to the PBE+D3
file, because D3-fitted `a` values leave +15-17 kB stress in bare PBE (2026-09-30).

Not variant-aware on purpose: `FINAL_RESULTS_HEALTHY/` (hand-curated), `data/batches/`, other override/cache JSONs,
`template_structures/`, `nequix_datasets/` (pass an explicit `--output`). The active variant is announced once
on stderr by every script, so a stray export is visible. Every entry point resolves paths through
`generated_dir()` / `templates_dir()` / `registry_path()` from `run_variant.py`; when adding a new script, use
those instead of `WORKFLOW_ROOT / "monolayer_examples"`.

**Bare-PBE campaign (IVDW off), started 2026-09-19 at the PI's request** — re-run relaxation + phonons for the 48
`v4_tmd_only` TMDs with **no dispersion correction**, to get genuinely D3-free reference data (the earlier
`nequix/explicit_dispersion` stage-1 check only *subtracted* D3 at the PBE+D3 geometries).

- Templates: `common/*_templates_bare_pbe/` are copies of the PBE+D3 templates with the `IVDW = 12` line removed
  (both relaxation and staticpoint INCAR); nothing else differs. On a new host, copy your templates and delete
  that line again — they are gitignored.
- Batches (tracked): `monolayer_batch_5.json` (6 TMD monolayers) and `bilayer_batch_8..11.json` (42 bilayers,
  11/11/10/10, alphabetical); each carries a `note`. Running them without the variant is harmless (the PBE+D3
  example dirs already exist → skipped) but pointless.
- Order: monolayers first — bilayers are built from the relaxed monolayer `CONTCAR`s in
  `monolayer_examples_bare_pbe/`, so bare-PBE bilayers need bare-PBE monolayers.
- Commands are the usual ones with the variable exported, e.g.
  `TWIST_VARIANT=bare_pbe python3 scripts/batch_management/submit_batch.py --monolayer 5`, then
  `phonopy/submit_batch.py --monolayer --batch 5`, `phonopy/postprocess_batch.py --monolayer --batch 5`
  (→ `FINAL_RESULTS_BARE_PBE/`), then bilayer batches 8–11. Commit `FINAL_RESULTS_BARE_PBE/` like `FINAL_RESULTS/`.
- **2026-10-06: the bare-PBE relaxation route below is retired.** Bare-PBE forces are going back to the subtraction
  method (PBE+D3 forces minus the additive D3 term, at the PBE+D3 geometries). The scripts and run outputs of the
  route (gap/slide/a-scan, ISIF=4, intralayer, tight phonopy, ADDGRID test) are archived untracked in
  `backups/bare_pbe/gap_scan_bare_pbe/`; the scripts are also in git history (last tracked at 2262a0f). They
  resolve `WORKFLOW_ROOT` from their own location, so move them back under `workflow/` before re-running any.
  Batch 8 results in `FINAL_RESULTS_BARE_PBE/` (93c8327) stay as they are.
- Bilayer protocol (from batch 8, 2026-09-30; retired 2026-10-06, kept for reference). The PI expects NO imaginary modes, so every bilayer goes through:
  1. ISIF=2 relaxation (the gap stalls 0.25-0.8 Å short of the minimum; don't trust it).
  2. ISIF=4 for `a`: `gap_scan_bare_pbe/setup_isif4.py <names>` → `<name>_isif4/`. Keep only the plateaued `a`
     (`common/isif4_lattice_extract.py`) after checking `c` drift < ~2-3%, and write it to
     `data/bilayer_lattice_overrides_bare_pbe.json`.
  3. Rebuild at that `a` (`apply_isif4_a.py`: ISIF=4 CONTCAR rescaled to the override `a`, c reset to 20 Å; old files →
     `<name>/pre_isif4a/`), then gap scan (`make_scan.py`, `apply_dmin.py`; batch 8 used `--archive=pre_gapscan2`).
  4. Fixed-gap intralayer relaxation (`setup_intralayer.py`). Target max |F| ≲ 1e-3 eV/Å, including the 2 frozen
     chalcogens, and in-plane stress < ~3 kB.
  5. Phonopy (`setup_phonopy_tight.py`): 0.03 Å displacements, EDIFF 1E-8, LREAL .FALSE. Batch 8 skipped disp-000
     because its residual forces after step 4 were ≤ 7e-4 eV/Å. Then run the normal `postprocess_batch.py`, and
     check Γ, a fine near-Γ scan and a Γ-centred odd mesh.
     Do NOT hiphive bare-PBE bilayers. At a ~4.3 Å gap the interlayer FCs extend past the 4x4 cutoff (≤5.85 Å),
     so truncating them makes breathing negative and swings the shear by ±0.6-0.8 THz (tested 2026-10-03).
     Batch 8 result: residual negatives ≤ 0.11 THz, either a doubly-degenerate Γ shear or (Te pairs) a small
     acoustic dip near Γ. A rigid slide scan (`make_slide_scan.py` / `analyze_slide_scan.py`) gave a positive
     sliding curvature in all 11 (+0.06..+0.18 THz), so the negative Γ shear is force noise.
  A dip that survives all 5 steps is kept in `FINAL_RESULTS_BARE_PBE/`, flagged as genuinely unstable and reported
  to the PI, not excluded (user decision 2026-09-30).
  Expect softer interlayer (shear/breathing) modes and larger gaps in bare-PBE bilayers; that is physics.

**Bare PBE by D3 subtraction (`FINAL_RESULTS_BARE_PBE_SUBTRACT/`, 2026-10-06)** — now the primary bare-PBE route
(the relaxation campaign above was retired the same day): bare-PBE force constants for the same 48 TMDs *at the PBE+D3 geometries*, obtained by subtracting the
explicit DFT-D3(BJ) force (VASP `IVDW = 12` parameters via simple-dftd3, `nequix/explicit_dispersion/d3.py`) from every
displaced-supercell force in `FINAL_RESULTS_HEALTHY/<mat>/FORCE_SETS`, *including the equilibrium D3 force*
(`F_bare(disp) = F_ref(disp) - [F_D3(disp) - F_D3(eq)]`, phonopy's `--fz` treatment). Not variant-aware; two new
scripts, nothing existing changed:

- `phonopy/subtract_d3_force_sets.py` (stage 1, **run from the nequix uv env**: `cd ../nequix && uv run python
  ../workflow/phonopy/subtract_d3_force_sets.py`) writes `<mat>/{FORCE_SETS,POSCAR,phonopy_disp.yaml,band.conf,
  d3_subtraction.npz,d3_subtraction.json}`; the JSON flags whether the reference `FORCE_CONSTANTS` was a plain phonopy
  rebuild (28) or hiphive-corrected (20); the flag is decided by the RMS band-frequency change of the
  plain rebuild vs the shipped FC (> 0.002 THz), not by the FC max-diff alone (that mis-flagged MoTe2_WSe2_3R).
- `phonopy/postprocess_bare_pbe_subtract.py` (stage 2, workflow conda env) runs the usual `phonopy -p -s --writefc` for
  all 48 (copy kept in `<mat>/plain_phonopy/`) and re-applies the `hiphive_fit_force_constants.py` constraint-based fit
  to the 20 flagged materials from the bare FORCE_SETS (no vasprun needed), plus `--force-hiphive` on MoS2, MoTe2,
  WS2_WSe2_2H, WS2_WSe2_3R, WTe2_bilayer_2H (user request 2026-10-06, to clear −0.002…−0.03 THz ZA dips their plain references also
  carry; costs 0.01–0.04 THz RMS vs the plain reference) → final `FORCE_CONSTANTS/band.yaml/band.pdf`,
  `hiphive_fit.json`, `bare_vs_pbed3.json` (incl. a plain-vs-plain column = pure D3 effect, separate from hiphive-method
  mismatch); `summary.csv` + `README.md` at the top level (`--summary-only` rebuilds them). Commit the directory like
  `FINAL_RESULTS/`.
- Label any number from it "bare-PBE curvature at the PBE+D3 geometry". `FINAL_RESULTS_BARE_PBE/` (6 monolayers +
  batch 8's 11 MoS2_* bilayers, retired route) is the only true-bare-PBE-minimum data; comparing the two per material
  isolates the geometry-relaxation effect. Note the retired protocol's finding that hiphive truncation damages
  interlayer modes at the bare-PBE gap (~4.3 Å); the subtraction set sits at the PBE+D3 gap, where the refits were benign.

**Twisted-bilayer DFT campaign (`TWIST_VARIANT=twisted`), started 2026-10-01** — run the moiré cells from
`twisted/build_twisted_bilayer.py` through the standard relax → phonopy → postprocess pipeline (first batch:
`MoS2_twist_m{1,2,3}_near0`, θ = 21.8/13.2/9.4°, 42/114/222 atoms). Additive only: no production script or
template was changed; the default dry-run was verified byte-identical before/after.

- Templates (host-local, gitignored — recreate on a new host): `common/relaxation_templates_twisted/` = PBE+D3
  relaxation templates with `EDIFFG = -1E-4` (the production `-1E-7` never converges and burns all 400 steps even
  on 6-atom cells; a 100–200-atom cell would exceed 24 h); `common/staticpoint_templates_twisted/` = production
  static templates with `KPOINTS` Gamma `5 5 1` (7x7x1 is sized for a 12.8 Å supercell; moiré supercells are
  14–19 Å). Everything else (ENCUT 520, EDIFF 1e-6, IVDW=12, ISIF=2, 2 nodes × 44, medium/24 h) is unchanged.
- `twisted/prepare_dft_inputs.py` writes `bilayer_examples_twisted/<name>/` (POSCAR copied verbatim, POTCAR,
  INCAR, per-structure `KPOINTS` n×n×1 with n = ceil(67 Å / a_moiré) matching the 21x21x1@3.19 Å production
  density, `bat` with `--nodes=4` above 200 atoms, and `phonopy_dim.txt` = the `--dim` to use: smallest k with
  k·a_moiré ≥ 12.8 Å → `2 2 1` for m1, `1 1 1` for m2/m3). It refuses to run unless `TWIST_VARIANT=twisted`.
- Per structure, with the variable exported: `relaxation/bilayer/submit_bilayer_job.py bilayer_examples_twisted/<name>`;
  after relaxation `phonopy/prepare_and_submit.py --bilayer bilayer_examples_twisted/<name> --dim "$(cat
  bilayer_examples_twisted/<name>/phonopy_dim.txt)" --no-submit`, **check the `POSCAR-XXX` count equals the atom
  count** (P321 is kept by VASP's ISYM; a numerically broken CONTCAR gives P1 and 3× the displacements — symmetrize
  it with spglib 1e-3 and re-prepare instead), for m3 set `--nodes=4` in `common/staticpoint_templates_twisted/bat`
  first (the bat is copied from the template at setup time, so edit the template, not `disp-XXX/bat`), then
  `phonopy/bilayer/setup_displacements.py <name>` (bare name; the script appends `_staticpoint` itself), then `phonopy/postprocess_results.py --bilayer
  <name>_staticpoint --dim "..."` → `FINAL_RESULTS_TWISTED/<name>/` (commit it like `FINAL_RESULTS/`).
- Cost: displacements = atoms (42/114/222) on 168/114/222-atom supercells; m3 dominates (~1,200 node-hours total,
  comparable to the whole 48-TMD dataset). Run m1 → m2 → m3 so m3 can be dropped.
- Not in scope here: comparing to Nequix. That is a separate step (the twisted `band.yaml` is in the same
  `FINAL_RESULTS` layout as everything else, so `scripts/share_bundle/evaluate_phonons.py`-style tooling applies).
