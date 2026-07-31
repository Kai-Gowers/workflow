# Workflow

Automated VASP + phonopy pipeline for phonon calculations of 2D materials (monolayers and bilayers) on a SLURM HPC cluster. Structures come from the Materials Project API (with template fallbacks); results land in `FINAL_RESULTS/` and can be converted to a Nequix `.aselmdb` training set.

## Pipeline

```
structure → VASP relax → phonopy displacements → VASP statics → FORCE_SETS / band.pdf → Nequix DB
```

1. **Relaxation** — ionic (+ optional cell) relaxation with VASP
2. **Phonopy prep** — generate displaced supercells (`3×3×1` by default)
3. **Static calculations** — single-point forces on each displacement
4. **Post-process** — assemble `FORCE_SETS`, run phonopy, write `band.pdf` / `FORCE_CONSTANTS` / `POSCAR` to `FINAL_RESULTS/`
5. **Convert** — build a Nequix PFT `.aselmdb` from `FINAL_RESULTS/`

## Requirements

```bash
export MP_API_KEY=<your_key>   # required for Materials Project API calls
```

| Dependency | Role |
|---|---|
| `phonopy` (CLI + Python) | Displacements, force sets, band structures |
| `mp-api`, `pymatgen` | Structure fetch and validation |
| `ase` | Nequix database conversion |
| VASP | DFT calculations (external) |
| SLURM | Job submission (external) |

## Quick start

### Single material (end-to-end phonon after relaxation)

```bash
# From a finished relaxation in monolayer_examples/MoS2/
python3 phonopy/prepare_and_submit.py --monolayer MoS2
# ... wait for SLURM static jobs ...
python3 phonopy/postprocess_results.py --monolayer MoS2_staticpoint
```

### Batch workflow (recommended)

```bash
# Step 1 — submit relaxations
python3 scripts/batch_management/submit_batch.py --monolayer 1

# Step 2 — displacements + static VASP jobs (after relaxations finish)
python3 phonopy/submit_batch.py --monolayer --batch 1

# Step 3 — band structures → FINAL_RESULTS/
python3 phonopy/postprocess_batch.py --monolayer --batch 1

# Step 4 — Nequix training database
python3 scripts/convert_to_nequix.py --output nequix_dataset.aselmdb
# or from a curated subset:
python3 scripts/convert_to_nequix.py --source FINAL_RESULTS_HEALTHY --output nequix_dataset_healthy.aselmdb
```

Use `--bilayer` instead of `--monolayer` for bilayer batches. Add `--dry-run` / `--no-submit` to inspect without submitting.

### Symmetry filter

Batch lists in `data/batches.json` are built from materials with hexagonal **P6₃/mmc** symmetry (Materials Project). Regenerate after editing `common/materials_list.txt`:

```bash
python3 scripts/batch_management/report_symmetry_eligibility.py
python3 relaxation/bilayer/generate_bilayer_combinations.py --require-p63mmc
python3 scripts/batch_management/create_batches.py --require-p63mmc --summary
```

## Repository layout

```
workflow/
├── common/                  # Shared library + VASP templates
│   ├── materials_project_api.py
│   ├── structural_families.py
│   ├── materials_list.txt
│   ├── relaxation_templates/
│   └── staticpoint_templates/
├── relaxation/              # VASP relaxation (monolayer/, bilayer/)
├── phonopy/                 # Displacements, submission, post-processing
├── scripts/
│   ├── batch_management/    # Batches, eligibility, SLURM submit
│   ├── maintenance/         # Cleanup, job listing
│   └── convert_to_nequix.py
├── data/                    # batches.json, job_registry.json, MP caches
├── FINAL_RESULTS/           # Per-material band.pdf, FORCE_CONSTANTS, …
├── monolayer_examples/      # Generated (gitignored)
├── bilayer_examples/        # Generated (gitignored)
├── phonopy_*_examples/      # Generated (gitignored)
└── CLAUDE.md                # Agent-oriented reference
```

More detail: [`relaxation/README.md`](relaxation/README.md), [`phonopy/README.md`](phonopy/README.md), [`common/README.md`](common/README.md).

## Materials and stackings

Master list: `common/materials_list.txt` (graphene, TMDs, BN, plus OOD test set).

`common/structural_families.py` classifies families (TMD, binary honeycomb, single-element) and defines valid bilayer stackings: `3R`, `2H`, `AB`, `BA`, `TM_H`, `TM_H2`.

Current batches (`data/batches.json`): ~13 monolayers (1 batch) and ~138 bilayers (10 batches).

## Key modules

| Module | Role |
|---|---|
| `common/materials_project_api.py` | MP fetch + cache, P6₃/mmc checks, lattice params, overrides |
| `common/structural_families.py` | Family classification and valid stackings |
| `phonopy/prepare_and_submit.py` | Single-material phonon orchestrator |
| `phonopy/postprocess_results.py` | `FORCE_SETS` → band structure → `FINAL_RESULTS/` |
| `scripts/convert_to_nequix.py` | `FINAL_RESULTS/` → `.aselmdb` with Hessian for PFT |

VASP templates use DFT-D3 vdW corrections, `KPAR=6`, `NCORE=4`. Relaxation templates allow ionic/cell DOF; staticpoint templates use `IBRION=-1`, `NSW=0`.

## Outputs

Each `FINAL_RESULTS/<material>/` typically contains:

- `band.pdf` / `band.yaml` — phonon dispersion
- `FORCE_SETS` / `FORCE_CONSTANTS` — phonopy force data
- `phonopy.yaml` / `POSCAR` — structure metadata used by conversion

Nequix conversion writes one ASE DB entry per material with `atoms.info["hessian"]` (Cartesian force-constant matrix) for phonon fine-tuning.

## Maintenance

```bash
python3 scripts/maintenance/list_jobs.py      # tracked SLURM jobs
python3 scripts/maintenance/cleanup_all.py    # remove generated example dirs
```
