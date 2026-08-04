# Phonopy Workflow

This module contains workflows for phonon calculations using phonopy.

## Structure

```
phonopy/
├── prepare_and_submit.py         # Master script: prepare + setup + submit (all-in-one)
├── postprocess_results.py         # Post-process completed calculations: build FORCE_SETS, generate phonon bands
├── monolayer/                    # Monolayer phonopy scripts
│   ├── prepare_staticpoint.py    # Prepare static-point calculation directories
│   └── setup_displacements.py    # Set up and submit displacement calculations
│
└── bilayer/                      # Bilayer phonopy scripts
    ├── prepare_staticpoint.py    # Prepare static-point calculation directories
    └── setup_displacements.py    # Set up and submit displacement calculations
```
## Workflow

The phonopy workflow operates on relaxed structures from the relaxation workflow:

1. **Prepare Static-Point**: Convert relaxed CONTCAR to static-point calculation directories
   - Input: CONTCAR files from `monolayer_examples/` or `bilayer_examples/`
   - Output: Static-point calculation directories in `phonopy_monolayer_examples/` or `phonopy_bilayer_examples/`
   - Creates clean directories with: POSCAR (from CONTCAR), POTCAR, KPOINTS, INCAR (static-point template), bat
   - **Automatically generates phonopy displacements** using `phonopy --dim="3 3 1" -d -c POSCAR`
   - Creates displaced structures (POSCAR-001, POSCAR-002, etc.) and phonopy_disp.yaml

2. **Setup Displacements**: Set up and submit VASP calculations for each displaced structure
   - Input: Static-point directories with POSCAR-001, POSCAR-002, etc.
   - For each POSCAR-XXX, creates a `disp-XXX/` subdirectory
   - Copies POSCAR-XXX → disp-XXX/POSCAR
   - Copies INCAR, KPOINTS, POTCAR, bat to each disp-XXX folder
   - Optionally submits jobs using `sbatch bat` for each displacement

3. **Post-Process Results**: After all displacement calculations complete, collect forces and generate phonon dispersion
   - Input: Completed displacement calculations (vasprun.xml files in disp-*/ folders)
   - Builds FORCE_SETS from all vasprun.xml files
   - Generates band.conf configuration file
   - Runs phonopy to compute phonon dispersion (band.pdf, band.yaml, FORCE_CONSTANTS)
   - Copies final results to FINAL_RESULTS/<material_name>/

## Quick Start: All-in-One Script

The easiest way to run the full workflow is using the master script `prepare_and_submit.py`:

```bash
cd phonopy

# Process a single monolayer (prepare + setup + submit)
python3 prepare_and_submit.py --monolayer MoS2

# Process a single bilayer
python3 prepare_and_submit.py --bilayer MoS2_TaTe2_3R

# Process all monolayers
python3 prepare_and_submit.py --monolayer --all

# Process all bilayers
python3 prepare_and_submit.py --bilayer --all

# Process both monolayers and bilayers
python3 prepare_and_submit.py --monolayer --bilayer --all

# Set up only (don't submit jobs)
python3 prepare_and_submit.py --monolayer MoS2 --no-submit

# Custom supercell dimensions
python3 prepare_and_submit.py --monolayer MoS2 --dim "4 4 1"

# Dry run
python3 prepare_and_submit.py --monolayer MoS2 --dry-run
```

This script automatically:
1. Prepares the static-point directory from the relaxed CONTCAR
2. Generates phonopy displacements
3. Sets up displacement folders (disp-001, disp-002, etc.)
4. Submits jobs for each displacement

## Batch symmetry filter (hexagonal P6₃/mmc)

Relaxation and phonopy batch submission use [`data/batches/`](../data/batches/) (one JSON file per batch), which is built only from materials with **hexagonal** crystal system and space group **P6₃/mmc** (Materials Project metadata). Phosphorene and several TMDs without an MP P6₃/mmc entry are excluded.

Regenerate the manifest and batches after changing [`common/materials_list.txt`](../common/materials_list.txt):

```bash
# Report eligibility and write data/symmetry_eligible_materials.json
python3 scripts/batch_management/report_symmetry_eligibility.py

# Regenerate bilayer list and data/batches/
python3 relaxation/bilayer/generate_bilayer_combinations.py --require-p63mmc
python3 scripts/batch_management/create_batches.py --require-p63mmc --summary
```

`submit_batch.py` (relaxation and phonopy) skips any batch entry not listed in `symmetry_eligible_materials.json` when that file exists. Use `--no-symmetry-filter` on the batch-creation scripts to include all materials from the master list.

## Step-by-Step Workflow

If you prefer to run each step separately:

## Prepare Static-Point Calculations

Before running phonopy, you need to prepare static-point calculation directories from relaxed structures.

### Monolayer

```bash
cd phonopy/monolayer

# Prepare a single example (with default 3x3x1 supercell)
python3 prepare_staticpoint.py MoS2

# Or with full path
python3 prepare_staticpoint.py ../monolayer_examples/MoS2

# Custom supercell dimensions
python3 prepare_staticpoint.py MoS2 --dim "4 4 1"

# Prepare all examples
python3 prepare_staticpoint.py --all

# Skip displacement generation (if needed)
python3 prepare_staticpoint.py MoS2 --no-displacements
```

### Bilayer

```bash
cd phonopy/bilayer

# Prepare a single example (with default 3x3x1 supercell)
python3 prepare_staticpoint.py MoS2_bilayer_3R

# Or with full path
python3 prepare_staticpoint.py ../bilayer_examples/MoS2_bilayer_3R

# Custom supercell dimensions
python3 prepare_staticpoint.py MoS2_bilayer_3R --dim "4 4 1"

# Prepare all examples
python3 prepare_staticpoint.py --all

# Skip displacement generation (if needed)
python3 prepare_staticpoint.py MoS2_bilayer_3R --no-displacements
```

**Note**: The script requires `phonopy` to be installed and available in your PATH. By default, it uses a 3×3×1 supercell for 2D materials.

## Setup and Submit Displacement Calculations

After preparing static-point calculations and generating displacements, set up individual VASP calculations for each displacement.

### Monolayer

```bash
cd phonopy/monolayer

# Set up and submit jobs for a specific staticpoint directory
python3 setup_displacements.py MoS2_staticpoint

# Set up only (don't submit)
python3 setup_displacements.py MoS2_staticpoint --no-submit

# Dry run
python3 setup_displacements.py MoS2_staticpoint --dry-run

# Process all staticpoint directories
python3 setup_displacements.py --all
```

### Bilayer

```bash
cd phonopy/bilayer

# Set up and submit jobs for a specific staticpoint directory
python3 setup_displacements.py MoS2_TaTe2_3R_staticpoint

# Set up only (don't submit)
python3 setup_displacements.py MoS2_TaTe2_3R_staticpoint --no-submit

# Dry run
python3 setup_displacements.py MoS2_TaTe2_3R_staticpoint --dry-run

# Process all staticpoint directories
python3 setup_displacements.py --all
```

**What it does:**
- Finds all `POSCAR-001`, `POSCAR-002`, etc. files in the staticpoint directory
- Creates `disp-001/`, `disp-002/`, etc. subdirectories
- Copies `POSCAR-XXX` → `disp-XXX/POSCAR`
- Copies `INCAR`, `KPOINTS`, `POTCAR`, `bat` to each displacement folder
- Submits jobs using `sbatch bat` for each displacement (unless `--no-submit` is used)

## Post-Process Results

After all displacement calculations complete, collect forces and generate phonon dispersion curves.

### Usage

```bash
cd phonopy

# Process a single monolayer staticpoint
python3 postprocess_results.py --monolayer HfSe2_staticpoint

# Process all monolayer staticpoints
python3 postprocess_results.py --monolayer --all

# Process a single bilayer staticpoint
python3 postprocess_results.py --bilayer MoS2_TaTe2_3R_staticpoint

# Process all bilayer staticpoints
python3 postprocess_results.py --bilayer --all
```

**What it does:**
1. Collects all `vasprun.xml` files from `disp-*/` folders in each staticpoint directory
2. Runs `phonopy --vasp -f disp-001/vasprun.xml disp-002/vasprun.xml ...` to build **FORCE_SETS**
3. Creates `band.conf` in the staticpoint directory with:
   - `ATOM_NAME` (auto-detected from POSCAR, e.g., `Mo Te`)
   - `DIM = 3 3 1`
   - `BAND = 0 0 0  0.6667 0.3333 0  0.5 0 0  0 0 0`
   - `BAND_LABELS = Γ K M Γ`
4. Runs `phonopy -p band.conf --save` to generate:
   - `band.pdf` (phonon dispersion plot)
   - `band.yaml` (phonon data in YAML format)
   - `FORCE_CONSTANTS` (force constant matrix, generated from FORCE_SETS)
5. Copies `band.pdf`, `band.yaml`, and `FORCE_SETS` into `FINAL_RESULTS/<material_name>/` (e.g., `FINAL_RESULTS/HfSe2/`)

**Note**: The script requires `phonopy` to be installed and available in your PATH. All displacement calculations must be completed (vasprun.xml files must exist) before running this script.

## Output Structure

After preparing static-point calculations:

```
phonopy_monolayer_examples/
└── <material_name>_staticpoint/    # e.g., MoS2_staticpoint
    ├── POSCAR                       # Copied from CONTCAR
    ├── POTCAR                       # Copied from original
    ├── KPOINTS                      # Copied from original
    ├── INCAR                        # Static-point template (customized)
    ├── bat                          # Batch script template
    ├── POSCAR-001                   # Displaced structure 1 (generated by phonopy)
    ├── POSCAR-002                   # Displaced structure 2 (generated by phonopy)
    ├── POSCAR-003                   # ... (more displacements)
    ├── phonopy_disp.yaml            # Displacement information
    └── ...

phonopy_bilayer_examples/
└── <bilayer_name>_staticpoint/     # e.g., MoS2_bilayer_3R_staticpoint
    ├── POSCAR
    ├── POTCAR
    ├── KPOINTS
    ├── INCAR
    ├── bat
    ├── POSCAR-001                   # Displaced structures (generated by phonopy)
    ├── POSCAR-002
    ├── phonopy_disp.yaml
    └── ...
```

After running `setup_displacements.py`:

```
phonopy_monolayer_examples/
└── <material_name>_staticpoint/    # e.g., MoS2_staticpoint
    ├── POSCAR                       # Original structure
    ├── POTCAR
    ├── KPOINTS
    ├── INCAR
    ├── bat
    ├── POSCAR-001                   # Displaced structures
    ├── POSCAR-002
    ├── phonopy_disp.yaml
    ├── disp-001/                    # Displacement calculation 1
    │   ├── POSCAR                   # Copied from POSCAR-001
    │   ├── POTCAR                   # Copied from parent
    │   ├── KPOINTS                  # Copied from parent
    │   ├── INCAR                    # Copied from parent
    │   └── bat                      # Copied from parent
    ├── disp-002/                    # Displacement calculation 2
    │   ├── POSCAR                   # Copied from POSCAR-002
    │   ├── POTCAR
    │   ├── KPOINTS
    │   ├── INCAR
    │   └── bat
    └── ...
```

Each `disp-XXX/` folder contains a complete VASP calculation setup for that displacement, ready to run and collect forces.

After running `postprocess_results.py`:

```
FINAL_RESULTS/
├── HfSe2/                         # Monolayer example
│   ├── band.pdf                   # Phonon dispersion plot
│   ├── band.yaml                  # Phonon data (YAML format)
│   └── FORCE_SETS                 # Force sets (from phonopy --vasp -f)
├── MoS2_TaTe2_3R/                 # Bilayer example
│   ├── band.pdf
│   ├── band.yaml
│   └── FORCE_SETS
└── ...
```

The `band.conf` file is also created in each staticpoint directory for reference.

