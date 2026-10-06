# FINAL_RESULTS_BARE_PBE_SUBTRACT — bare-PBE phonons by explicit D3 subtraction

Generated 2026-10-06 by `phonopy/subtract_d3_force_sets.py` (stage 1, nequix uv env) and
`phonopy/postprocess_bare_pbe_subtract.py` (stage 2, workflow conda env). 48 Mo/W TMDs, same set and
geometries as `FINAL_RESULTS_HEALTHY` restricted to `nequix_datasets/v4_tmd_only`.

**What these are.** Bare-PBE harmonic phonons *at the PBE+D3-relaxed geometry*. The DFT-D3(BJ) term (VASP
`IVDW = 12` parameters, evaluated with simple-dftd3; see `nequix/explicit_dispersion/README.md`) is subtracted
from the VASP forces on every displaced supercell, together with the equilibrium D3 force, so the result is what
phonopy would give for PBE-only forces on a structure that *is* a PBE minimum in the harmonic sense:

    F_bare(disp) = F_ref(disp) - [ F_D3(disp) - F_D3(eq) ]

The structures were **not** re-relaxed with bare PBE (that is the separate `TWIST_VARIANT=bare_pbe` DFT campaign,
`FINAL_RESULTS_BARE_PBE/`). The residual bare-PBE force -F_D3(eq) (0.10-0.17 eV/A, mostly interlayer / chalcogen z)
is recorded per material and the stress placeholder is unchanged from the reference.

**Pipeline per material** (`<material>/`):

| file | content |
|---|---|
| `POSCAR`, `phonopy_disp.yaml`, `band.conf` | unchanged geometry / cell setup / band path from the reference run |
| `FORCE_SETS` | bare-PBE displaced forces (above) |
| `d3_subtraction.npz/.json` | every D3 quantity used (F_D3 per displacement, F_D3(eq), E_D3, stress) and sanity numbers |
| `plain_phonopy/` | step 1: plain `phonopy -p -s --writefc` result (FC_SYMMETRY on), kept for all 48 |
| `FORCE_CONSTANTS`, `band.yaml`, `band.pdf`, `phonopy.yaml` | final result: = plain for 24 materials, hiphive-fit for 24 |
| `hiphive_fit.json` | (24 materials) cutoff, lambda sweep, chosen lambda — where the PBE+D3 reference had been hiphive-corrected, plus forced (--force-hiphive, user request 2026-10-06) on MoS2, MoTe2, WS2_WSe2_3R, WTe2_bilayer_2H to remove the small ZA dips their plain references also carry |
| `bare_vs_pbed3.json` | band-path / 40x40x1-mesh minima and RMS difference to the PBE+D3 reference on its own q-points |

**Headline.** RMS(bare - PBE+D3) over all bands: mean 0.0124 THz, range 0.0061-0.0413 THz.
The pure D3 effect at fixed geometry is `rmse_plain_bare_vs_plain_pbed3_THz` (plain phonopy on both sides); for the
24 hiphive materials `rmse_vs_pbed3_THz` additionally contains the mismatch between this refit and the reference's own
hiphive step (`rmse_pbed3_ref_vs_plain_pbed3_THz` shows how large that step was), so use the plain-vs-plain column for
"what does D3 do to the bands".
Lowest band-path frequency after post-processing: -0.0024 THz (WS2_WSe2_2H).
See `summary.csv` for every material.
