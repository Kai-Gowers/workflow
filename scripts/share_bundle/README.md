# 48-material Mo/W TMD phonon dataset (DFT, PBE+D3(BJ))

Harmonic phonon reference data for 48 dynamically stable two-dimensional transition-metal
dichalcogenides built from MoS2, MoSe2, MoTe2, WS2, WSe2 and WTe2:

| Type | Count | Description |
|---|---|---|
| monolayers | 6 | 1H MX2 (space group P-6m2) |
| homobilayers | 12 | each MX2 in 2H and 3R stacking |
| heterobilayers | 30 | all 15 MX2/M'X'2 pairs in 2H and 3R stacking |

All 48 have no imaginary modes along Γ–K–M–Γ in DFT (see `min_freq_THz` in `materials.csv`).
No train/validation/test split is imposed; `materials.csv` has `kind`, `stacking` and `layers`
columns so you can define your own (e.g. hold out a stacking, a compound or a chalcogen).

## Contents

```
materials.csv            one row per material: formula, kind, stacking, supercell, #atoms,
                         #displacements, min/max DFT frequency, force-constant provenance
displacements.extxyz     all DFT displaced supercells with forces (ASE extended XYZ)
evaluate_phonons.py      reference benchmark script for any ASE calculator (see below)
requirements.txt
SHA256SUMS
materials/<name>/
    POSCAR               DFT-relaxed unit cell (VASP 5 format, 60° hexagonal cell, 20 Å c-axis)
    phonopy.yaml         phonopy metadata: unit cell, supercell matrix, primitive matrix, symmetry
    FORCE_SETS           raw DFT forces for every symmetry-inequivalent displacement (0.01 Å)
    FORCE_CONSTANTS      harmonic force constants used for the reference dispersion (eV/Å²)
    band.yaml            reference dispersion: frequencies (THz) on 153 q-points along Γ–K–M–Γ
    band.pdf             plot of band.yaml
```

Material names: `MoS2` (monolayer), `MoS2_bilayer_2H` (homobilayer), `MoS2_WSe2_3R`
(heterobilayer, first compound is the bottom layer). Stacking labels follow the bulk-polytype
convention: `2H` = second layer rotated 180° (AA'), `3R` = translated, same orientation (AB).

## How the data were generated

- **Code**: VASP 6.5.1, PAW PBE potentials `Mo_sv`, `W_sv`, `S`, `Se`, `Te`; phonopy 4.1.0.
- **Functional**: PBE + Grimme D3 with Becke-Johnson damping (`IVDW = 12`), spin-unpolarized.
- **Numerics**: `ENCUT = 520` eV, `PREC = Accurate`, `EDIFF = 1e-6`, Gaussian smearing
  `ISMEAR = 0`, `SIGMA = 0.05`, `LREAL = Auto`, `ALGO = Normal`. No DFT+U, no SOC.
- **Relaxation**: atomic positions only (`ISIF = 2`), fixed in-plane lattice constant and 20 Å
  cell height, `EDIFFG = -1e-7` eV/Å, Γ-centered 21×21×1 k-mesh. Bilayers start from the two
  relaxed monolayers at the stated stacking; their in-plane lattice constant is the mismatch-
  weighted average used in the workflow (all pairs are < 5 % mismatch). A few cells were
  additionally cell-relaxed in-plane (`ISIF = 4`) when the first dispersion showed a Γ-point
  acoustic dip from residual strain; the shipped POSCAR is always the geometry the phonons
  were computed on.
- **Phonons**: finite displacements of 0.01 Å in a 4×4×1 supercell by default, enlarged to
  5×5×1 or 6×6×1 where the smaller cell left a residual near-Γ artifact (per-material size in
  `materials.csv`), Γ-centered 7×7×1 k-mesh for the supercell statics, `fc_symmetry = .TRUE.`
  (translational + permutation symmetrization). Frequencies are in THz (phonopy VASP units).
- **Force-constant provenance**: `force_constants_source` in `materials.csv` is `phonopy` when
  `FORCE_CONSTANTS` is a plain symmetrized phonopy fit of `FORCE_SETS` (reproduced to < 0.01 THz
  by `phonopy.load(..., force_sets_filename=...)` + `symmetrize_force_constants()`). It is `hiphive-corrected` when the
  plain fit showed a small spurious negative acoustic branch at or very near Γ caused by the
  numerical violation of the rotational sum rules in a 2D slab (not a real instability), and
  the shipped `FORCE_CONSTANTS`/`band.yaml` were regenerated with hiphive enforcing the
  translational and rotational sum rules. `FORCE_SETS` are always the untouched DFT forces,
  so you can always refit yourself.

The dispersion q-path is Γ (0,0,0) → K (1/3,1/3,0) → M (1/2,0,0) → Γ in reduced coordinates
of **phonopy's standardized 120° primitive cell**, 51 points per segment. phonopy applies the
`primitive_matrix` in `phonopy.yaml` to the 60° POSCAR; if you build your own q-path from the
POSCAR directly (60° cell) K is (2/3,1/3,0) instead.

## Loading the data

```python
import phonopy, yaml
from ase.io import read

# reference dispersion
ph = phonopy.load("materials/MoS2/phonopy.yaml", force_constants_filename="materials/MoS2/FORCE_CONSTANTS")
band = yaml.safe_load(open("materials/MoS2/band.yaml"))
q = [p["q-position"] for p in band["phonon"]]                     # 153 q-points
f = [[b["frequency"] for b in p["band"]] for p in band["phonon"]]  # THz, shape (153, 3N)

# raw DFT displaced supercells + forces (for training / force-error metrics)
frames = read("displacements.extxyz", index=":")
frames[0].info["material"], frames[0].get_forces()
```

Every frame in `displacements.extxyz` carries `material`, `displacement_index`,
`displaced_atom` and `displacement` in `atoms.info`. No energies or stresses are included
(only forces were needed for the phonons); forces are in eV/Å.

## Benchmarking a model

`evaluate_phonons.py` runs the same finite-displacement protocol with any ASE calculator and
compares to `band.yaml` on exactly the DFT q-points:

```bash
pip install -r requirements.txt   # + your model's package
python evaluate_phonons.py --calc "mace.calculators:mace_mp" --calc-kwargs '{"model":"medium"}' --plots
python evaluate_phonons.py --calc "nequix.calculator:NequixCalculator" --calc-kwargs '{"model_name":"nequix-omat-1"}'
python evaluate_phonons.py --calc "my_pkg.my_module:make_calculator" --relax positions
```

Per-material RMSE / MAE (THz), acoustic/optical breakdown and minimum predicted frequency land
in `results/<label>/results.csv`. Two evaluation modes, please state which one you report:

- `--relax none` (default): phonons on the DFT geometry as-is.
- `--relax positions`: atoms relaxed with the model first (cell fixed), closer to how a model
  would be deployed.

For reference, on this exact protocol a mean whole-band RMSE of ~0.05 THz over the 48 materials
is "near-DFT"; off-the-shelf foundation models typically sit at 0.15–0.3 THz with occasional
imaginary modes.

## Citation / contact

Dataset produced by Kai Gowers (Boston College). Please get in touch before publishing results
derived from it.
