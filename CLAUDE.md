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

### VASP Templates

Templates for INCAR, KPOINTS, and SLURM batch scripts live in (gitignored; host-local):
- `common/relaxation_templates/` — for structural relaxation (ionic + cell DOF)
- `common/staticpoint_templates/` — for static calculations (IBRION=-1, NSW=0)

Perlmutter defaults (`vasp/6.6.0-cpu`, account `m5370`, queue `regular`,
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

Not variant-aware on purpose: `FINAL_RESULTS_HEALTHY/` (hand-curated), `data/batches/`, override/cache JSONs,
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
- Known caveat: relaxation is `ISIF=2` at the in-plane `a` from the overrides/MP cache, several of which were
  refined under PBE+D3. Bare PBE prefers a slightly different `a`, so check residual stress in each `OUTCAR` after
  relaxation and apply the ISIF=4 protocol (see memory / `common/isif4_lattice_extract.py`) where it is large.
  Expect softer interlayer (shear/breathing) modes and larger gaps in bare-PBE bilayers; that is physics.
