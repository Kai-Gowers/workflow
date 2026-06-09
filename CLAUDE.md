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

### Data Files

- **`data/batches.json`** — batch assignments (13 monolayers in 1 batch; 138 bilayers in 10 batches of ~15)
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

Templates for INCAR, KPOINTS, and SLURM batch scripts live in:
- `common/relaxation_templates/` — for structural relaxation (ionic + cell DOF)
- `common/staticpoint_templates/` — for static calculations (IBRION=-1, NSW=0)

Key VASP settings used: DFT-D3 van der Waals corrections, KPAR=6, NCORE=4.

### Material List

`common/materials_list.txt` is the master list of 16 materials. Bilayer combinations are generated from all valid pairs using `generate_bilayer_combinations.py`.
