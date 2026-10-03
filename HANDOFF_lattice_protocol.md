# Bilayer equilibrium + negative-frequency protocol (handoff)

Distilled from batches 1–7 and the bare-PBE campaign. Everything below is run from
`workflow/`. For the bare-PBE campaign prepend `TWIST_VARIANT=bare_pbe` to every
command (dirs become `bilayer_examples_bare_pbe/`, `FINAL_RESULTS_BARE_PBE/`, and the
host-local `common/*_templates_bare_pbe/` must exist with the `IVDW = 12` line removed).

## 0. Normal relaxation first

```bash
python3 scripts/batch_management/submit_batch.py --bilayer <N>     # ISIF=2, IBRION=1, NSW=400, EDIFFG=-1E-7
```

Bilayers are built from the relaxed monolayer `CONTCAR`s in `monolayer_examples*/`, in-plane
`a` = mean of the two monolayers' `a` (or the per-bilayer override, see §3), starting
interlayer separation dz = mean of the two materials' dz (3.5 Å default, per-material
override in `data/mp_material_overrides.json`).

Policy: if the run hits NSW=400 without "reached required accuracy", leave the CONTCAR
as-is and continue. Flat vdW surfaces do this routinely; do not extend NSW or switch IBRION.

## 1. Two post-relaxation checks (cheap, no new jobs)

**(a) Interlayer gap** — catches the trapped-relaxation bug (layers stuck at ~2–3× the true
spacing; only seen for light-element families: graphene, BN, silicene, germanene, TiS2 and
their heteropairs; never for Mo/W TMD–TMD pairs):

```bash
python3 common/interlayer_check.py bilayer_examples/<mat>/CONTCAR     # expect 2.9–3.3 Å (TMD-TMD), 3.5–5.0 Å (GaSe/InSe)
```

Gap > 5.5 Å → structure is wrong regardless of what the phonons say. Fix = set a smaller
per-material `dz` in `data/mp_material_overrides.json` (found by a rigid-separation PES
scan of the homobilayer), rebuild, re-relax.

**(b) Residual in-plane stress** — decides whether the lattice constant needs refining:

```bash
grep "external pressure" bilayer_examples/<mat>/OUTCAR | tail -1
```

| |P| at end of ISIF=2 | Action |
|---|---|
| ≳ 5 kB (typically 10–20 kB) | Residual strain is real. Do the ISIF=4 protocol (§2) **before** spending phonon jobs. |
| ≲ 3 kB | ISIF=4 is a no-op (shifts `a` by ~0.03%). Skip it, go to phonons (§5); if a small dip appears use hiphive (§6). |
| in between | Either; historically went to phonons first and only did ISIF=4 if a dip appeared. |

Historical calibration: every bilayer with 5–20 kB that showed a Γ or near-Γ acoustic dip
(−0.3 to −0.7 THz) was fixed by ISIF=4; the cases with < 2 kB and a dip were not.

## 2. ISIF=4 refinement of the in-plane lattice constant

Fixed-volume shape+ion relaxation lets `a` float while the vacuum `c` compensates.

```bash
cd bilayer_examples
mkdir <mat>_isif4
cp <mat>/{INCAR,KPOINTS,POTCAR,bat} <mat>_isif4/
cp <mat>/CONTCAR <mat>_isif4/POSCAR            # start from the ISIF=2 result
sed -i 's/^ISIF *= *2.*/ISIF = 4   # residual-strain refinement/' <mat>_isif4/INCAR
# keep IBRION=1, NSW=400, EDIFF=1E-6, EDIFFG=-1E-7 unchanged
cd <mat>_isif4 && sbatch bat
```

Expected endings, all fine: hits NSW=400 (usual), or a ZBRENT crash near the end (CONTCAR
still holds the last completed step). If the job hangs mid-SCF with flat output, that is
the known cluster hang; scancel, copy CONTCAR→POSCAR, resubmit (4 nodes helped).

Extract the plateaued `a`:

```bash
python3 common/isif4_lattice_extract.py bilayer_examples/<mat>_isif4      # averages last 50 ionic steps
grep "external pressure" bilayer_examples/<mat>_isif4/OUTCAR | tail -1    # should now be ~0–3 kB
```

Accept when: std of `a` over the last 50 steps ≲ 1e-3 Å, stress dropped, and `c` drifted
< ~1% (check first vs last "direct lattice vectors" block). Typical correction is −0.5 to
−1.5% in `a`.

If `a` does not plateau or `c` collapses (ISIF=4 paying for a large correction by eating the
vacuum), fall back to a fixed-a energy scan: `common/fixed_a_scan.py` builds several
rescaled copies at trial `a` values, each relaxed ISIF=2; fit E(a) for the minimum.

## 3. Persist the refined `a`

Add an entry to `data/bilayer_lattice_overrides.json` under `"bilayers"`, keyed by the
example-dir name, with a `note` recording old `a`, old stress, new `a`, new stress:

```json
"MoS2_WSe2_2H": {"a": 3.218258, "note": "ISIF=4 correction: -12.76 kB at a=3.256085 -> a=3.218258, re-relaxed ISIF=2 -0.87 kB; clean 5x5x1 band"}
```

`generate_bilayer_poscar.py` reads this file first for both homo- and heterobilayers, so any
future rebuild of that bilayer picks it up. Commit the JSON.

**Bare-PBE caveat:** this file is shared by the PBE+D3 default and the bare-PBE variant (it is
deliberately not variant-aware). A bare-PBE `a` written here would change the D3 build of the
same bilayer. For bare-PBE, either do the in-place rescale in §4 without writing the override,
or agree on a variant-specific override file first. Do not silently change the shared file.

## 4. Re-relax ISIF=2 at the refined `a`

```bash
mkdir -p backups/<mat>_strain_fix_$(date +%Y%m%d)
cp -r bilayer_examples/<mat> bilayer_examples/<mat>_staticpoint backups/<mat>_strain_fix_$(date +%Y%m%d)/ 2>/dev/null
```

Then either regenerate the example dir from scratch (`relaxation/bilayer/create_bilayer_example.py`
or re-running `submit_batch.py` after deleting the dir; the override is applied automatically),
or rescale in place, which is what was done historically: multiply only the two in-plane
lattice vectors of `CONTCAR` by `a_new/a_old`, leave `c` and the Direct coordinates untouched,
write it as `POSCAR`, keep the normal ISIF=2 INCAR, `sbatch bat`.

Afterwards confirm `external pressure` is ~ ≤ 1.5 kB and re-run `interlayer_check.py`.

## 5. Phonons

```bash
python3 phonopy/prepare_and_submit.py --bilayer <mat>                  # 4x4x1 default
python3 phonopy/prepare_and_submit.py --bilayer <mat> --dim "5 5 1"    # used for the batch-1/2 fixes
# ... wait for statics ...
python3 phonopy/postprocess_results.py --bilayer <mat>_staticpoint [--dim "5 5 1"]
python3 scripts/check_phonon_stability.py --bilayer [--batch N]
```

Gotcha when regenerating a staticpoint dir at a new supercell size on top of an old run:
delete only `FORCE_CONSTANTS`, `band.yaml`, `phonopy.yaml`, `band.pdf`, never
`phonopy_disp.yaml`/`POSCAR-*`, and pass the matching `--dim` to postprocess.

Verification is band path **plus** a dense mesh. Any manual mesh check must be Γ-centred
with an odd grid (e.g. `MP = 21 21 1`, `GAMMA_CENTER = .TRUE.`); an even grid without
GAMMA_CENTER never samples Γ and gave a false "stable". If `band.pdf` shows a near-Γ dip
that a mesh calls clean, refine the mesh (24→96) or scan q-points along that direction.
Stable threshold used throughout: min frequency ≥ −0.10 THz, and ideally ≈ 0.

## 6. Small residual dip after §2–5: hiphive rotational sum rule

Free (reuses the existing `disp-*/vasprun.xml`), needs `hiphive` importable in the shell that
runs it:

```bash
python3 phonopy/hiphive_fit_force_constants.py --bilayer <mat>_staticpoint     # fit-based, stronger; λ sweep 1e-3..1e2
python3 phonopy/hiphive_fix_force_constants.py --bilayer <mat>_staticpoint     # quick post-processing variant, weaker
```

Both back up the old FORCE_CONSTANTS/band files, write corrected ones, copy to
`FINAL_RESULTS*/<mat>/`, and print a dense-mesh scan. Reconstruction error should stay
< 1–2% (else raise `--cutoff`).

Reading the result: dip → 0.0000 THz with no plateau across λ = numerical artifact, promote.
Dip plateaus at a nonzero value even at λ ~ 100–1000 = genuine instability (or a wrong
geometry; re-check §1a). hiphive cannot fix BZ-wide instabilities or large dips with small
stress.

## 7. Promote

Copy the material's `FINAL_RESULTS*/<mat>/` into `FINAL_RESULTS_HEALTHY/` (hand-curated),
commit results + override JSON + a daily-log entry.

## Order of levers, summarised

1. interlayer-gap check → dz override if trapped
2. residual stress ≳ 5 kB → ISIF=4 `a` refinement → persist → re-relax ISIF=2
3. phonons at 4x4x1 (5x5x1 if a dip survives and looks finite-size)
4. small residual dip → hiphive fit; plateau = genuine
5. large dip with small stress → not a lattice problem; don't ISIF=4 it
