"""Phonopy setup for the fixed-gap intralayer-relaxed bare-PBE bilayers, with tightened settings (protocol step 5).

Isolated on purpose: it calls the normal pipeline functions unchanged and only edits its own output dirs, so the
shared code and templates (common/, phonopy/, *_templates*/) are untouched. Differences from the normal pipeline:
  - "Selective dynamics" is stripped from <name>/CONTCAR (the original is kept as CONTCAR.seldyn)
  - the staticpoint INCAR gets EDIFF = 1E-8 and LREAL = .FALSE. (template: 1E-6 / Auto)
  - displacements are generated with --amplitude 0.03 (phonopy default 0.01 Å)
  - no disp-000 job (the residual forces after the intralayer relax are <= 7e-4 eV/Å)
The existing phonopy_bilayer_examples_bare_pbe/<name>_staticpoint is moved to
phonopy_superseded_bare_pbe_pre_intralayer/ first. Postprocessing is unchanged: phonopy_disp.yaml carries the amplitude.

Usage (dry run is the default; nothing is submitted without --submit):
  TWIST_VARIANT=bare_pbe python3 setup_phonopy_tight.py [--submit] <name> ...
First used 2026-10-01 for the 11 batch-8 bilayers.
"""
import os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import yaml
from ase.io import read, write

WF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WF / "common"))
sys.path.insert(0, str(WF / "phonopy" / "bilayer"))
from run_variant import generated_dir, variant  # noqa: E402
from prepare_staticpoint import prepare_staticpoint  # noqa: E402
from setup_displacements import setup_and_submit_displacements  # noqa: E402

AMPLITUDE = 0.03
SRC = WF / "bilayer_examples_bare_pbe"
SUPERSEDED = WF / "phonopy_superseded_bare_pbe_pre_intralayer"

assert variant() == "bare_pbe", "run with TWIST_VARIANT=bare_pbe"
submit = "--submit" in sys.argv[1:]

for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
    ex = SRC / name
    con = ex / "CONTCAR"
    if "selective" in con.read_text().lower():
        shutil.copy2(con, ex / "CONTCAR.seldyn")
        atoms = read(con)
        del atoms.constraints
        write(con, atoms, format="vasp", direct=True, sort=False)
        assert np.allclose(read(con).positions, read(ex / "CONTCAR.seldyn").positions, atol=1e-8)
        assert "selective" not in con.read_text().lower()

    sp = generated_dir("phonopy_bilayer_examples") / f"{name}_staticpoint"
    if sp.exists():
        SUPERSEDED.mkdir(exist_ok=True)
        assert not (SUPERSEDED / sp.name).exists(), f"{SUPERSEDED / sp.name} already exists"
        shutil.move(str(sp), SUPERSEDED / sp.name)

    sp = Path(prepare_staticpoint(ex, generate_displacements=False)["output_dir"])

    incar = (sp / "INCAR").read_text()
    for key, val in [("EDIFF", "1E-8"), ("LREAL", ".FALSE.")]:
        incar, n = re.subn(rf"(?m)^{key}\s*=.*$", f"{key} = {val}", incar)
        assert n == 1, f"{name}: {key} not found exactly once in INCAR"
    (sp / "INCAR").write_text(incar)

    subprocess.run(["phonopy-init", "--dim", "4", "4", "1", "-d", "--amplitude", str(AMPLITUDE), "-c", "POSCAR"],
                   cwd=sp, check=True, capture_output=True, text=True)
    disps = yaml.safe_load(open(sp / "phonopy_disp.yaml"))["displacements"]
    norms = [np.linalg.norm(d["displacement"]) for d in disps]
    assert np.allclose(norms, AMPLITUDE, atol=1e-6), f"{name}: displacement norms {norms}"

    res = setup_and_submit_displacements(sp, submit=submit)
    print(f"{name}: {len(disps)} displacements of {AMPLITUDE} Å, set up {res['displacements_setup']}, "
          f"submitted {res['jobs_submitted']} {res['job_ids']}")
