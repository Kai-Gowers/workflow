# Relaxation Workflow

This module contains workflows for initial structure relaxation using VASP.

## Structure

```
relaxation/
├── monolayer/                    # Monolayer relaxation scripts
│   ├── generate_monolayer_poscar.py    # Generate POSCAR files
│   ├── generate_potcar.py              # Generate POTCAR files
│   ├── create_monolayer_example.py     # Create single example
│   ├── new_monolayer_relax.py          # Create and submit example
│   └── submit_monolayer_job.py         # Submit jobs to cluster
│
└── bilayer/                      # Bilayer relaxation scripts
    ├── generate_bilayer_combinations.py  # Generate all combinations
    ├── generate_bilayer_poscar.py        # Generate bilayer POSCARs
    ├── create_bilayer_example.py         # Create single bilayer example
    ├── create_all_bilayers.py            # Create all bilayer examples
    ├── submit_bilayer_job.py             # Submit bilayer jobs
    └── bilayer_combinations.txt          # Generated combinations list
```

## Output Structure

Relaxed structures are stored in:

- `../monolayer_examples/<material_name>/` (e.g., `MoS2/`, `graphene/`)
- `../bilayer_examples/<bilayer_name>/` (e.g., `MoS2_bilayer_3R/`, `MoS2_WS2_2H/`)

Each example directory contains:
- `POSCAR` - Initial structure
- `POTCAR` - Pseudopotentials
- `INCAR` - VASP parameters
- `KPOINTS` - K-point mesh
- `bat` - SLURM batch script
- `CONTCAR` - Final relaxed structure (after completion)

## Quick Start

### Monolayer Relaxation

```bash
cd relaxation/monolayer

# Create and submit a new training example
python3 new_monolayer_relax.py
```

### Bilayer Relaxation

```bash
cd relaxation/bilayer

# Create a specific bilayer example
python3 create_bilayer_example.py MoS2_bilayer_3R

# Create all bilayer examples
python3 create_all_bilayers.py --max 10
```

## Shared Resources

All workflows use shared resources from `../common/`:
- `common/materials_list.txt` - Master materials database
- `common/relaxation_templates/` - VASP input templates (INCAR, KPOINTS, bat)

## Materials Project Integration (Monolayers)

Monolayer POSCAR generation now supports an MP-first flow:
- Try to fetch full structure from Materials Project.
- Validate stoichiometry/geometry.
- Fallback to internal coordinate templates when MP is missing/invalid.

Environment/deps:
- Install `mp-api`: `pip install mp-api`
- Set API key: `export MP_API_KEY=...`

Useful flags (monolayer scripts):
- `--no-mp` - disable MP lookup and force template generation
- `--mp-api-key <key>` - override API key for this run
- `--mp-refresh` - refresh cached MP structure entries
- `--mp-verbose` - print MP material-id and fallback reasons
- `--strict-validation` - fail instead of fallback on MP validation errors

Data files:
- `data/mp_structure_cache.json` - cached MP structure payloads
- `data/mp_material_overrides.json` - optional per-material overrides

