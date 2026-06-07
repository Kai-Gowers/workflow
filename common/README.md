# Common Resources

This directory contains shared resources used across all workflow modules.

## Contents

- **`materials_list.txt`**: Master list of 2D materials used for generating training examples
- **`relaxation_templates/`**: Shared VASP input file templates
  - `INCAR`: VASP input parameters
  - `KPOINTS`: k-point mesh settings
  - `bat`: SLURM batch submission script

## Usage

All workflow modules (`relaxation/monolayer/`, `relaxation/bilayer/`, `phonopy/monolayer/`, `phonopy/bilayer/`, etc.) reference these shared resources, ensuring consistency across different calculation types.

## Modifying Templates

When updating templates:
- Changes will affect all workflows that use them
- Test changes with a single example before running large batches
- Consider creating workflow-specific templates if significant differences are needed

