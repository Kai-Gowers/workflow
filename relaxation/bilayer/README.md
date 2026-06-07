# Bilayer Relaxation Workflow

This directory contains scripts for systematically generating and relaxing bilayer structures with both 3R (0-degree) and 2H (180-degree) stacking.

## Structure

```
bilayer_relax/
├── generate_bilayer_combinations.py  # Generate all bilayer combinations
├── generate_bilayer_poscar.py         # Generate POSCAR from relaxed monolayers
├── relaxed_monolayer.py               # Load CONTCAR from monolayer_examples
├── create_bilayer_example.py          # Create single bilayer example
├── create_all_bilayers.py             # Create all bilayer examples
├── submit_bilayer_job.py              # Submit bilayer jobs to cluster
├── (templates in ../common/relaxation_templates/)  # Shared VASP input templates
└── bilayer_combinations.txt           # Generated list of all combinations
```

## Relaxed monolayer prerequisite

By default, bilayer POSCARs are built from **relaxed monolayer geometries** in
`monolayer_examples/<material>/CONTCAR` (POSCAR fallback). Run monolayer relaxation
first (`relaxation/monolayer/create_monolayer_example.py`).

- **All bilayers (default)**: stack two layers from CONTCAR with 3R/2H rules;
  shared in-plane `a` from relaxed cells (optional `--anchor` for heterobilayers).

Materials with relaxed CONTCAR in this repo (as of last update): MoS2, MoSe2, MoTe2,
WS2, WSe2, WTe2, NbS2, NbSe2, TaS2, TaSe2, BN, GaN, graphene. Other combinations
will fail until the corresponding monolayer is relaxed.

Use `--use-templates` to build from ideal template coordinates instead (MP/fallback
lattice parameters).

## Workflow

### Step 1: Generate Bilayer Combinations

First, generate the list of all possible bilayer combinations:

```bash
python3 generate_bilayer_combinations.py
```

This creates `bilayer_combinations.txt` with all combinations, each with both 3R and 2H stacking:
- Homostructures: same material bilayers (e.g., `MoS2_bilayer_3R`, `MoS2_bilayer_2H`)
- Heterostructures: different material bilayers (e.g., `MoS2_WS2_3R`, `MoS2_WS2_2H`)

### Step 2: Create Bilayer Examples

#### Option A: Create All Bilayers Systematically

```bash
# Create all bilayer examples
python3 create_all_bilayers.py

# Create first 10 examples (for testing)
python3 create_all_bilayers.py --max 10
```

#### Option B: Create Individual Bilayer Example

```bash
# Create specific bilayer
python3 create_bilayer_example.py MoS2_bilayer_3R

# Create with specific ID
python3 create_bilayer_example.py MoS2_WS2_2H --id 5
```

### Step 3: Submit Jobs

```bash
# Submit single job
python3 submit_bilayer_job.py 5

# Submit multiple jobs
python3 submit_bilayer_job.py 5 6 7

# Dry run
python3 submit_bilayer_job.py 5 --dry-run
```

## Output Structure

Bilayer examples are stored in `../bilayer_examples/` with naming:
- `bilayer_<id>_3R/` - 3R stacking (0-degree twist)
- `bilayer_<id>_2H/` - 2H stacking (180-degree twist)

Each directory contains:
- `POSCAR` - Generated bilayer structure
- `POTCAR` - Pseudopotentials
- `INCAR` - VASP input (customized with bilayer name)
- `KPOINTS` - k-point mesh
- `bat` - Batch submission script

## Stacking Types

- **3R stacking (0-degree)**: Layers are aligned (AA-like stacking)
- **2H stacking (180-degree)**: Second layer rotated 180 degrees (AB stacking)

## Notes

- All scripts can use functions from `../monolayer/` for shared functionality (e.g., POTCAR generation)
- Bilayer examples use separate numbering from monolayer examples
- Each bilayer combination generates 2 examples (one for each stacking type)

