#!/usr/bin/env python3
"""
Post-relaxation sanity check for bilayer interlayer separation.

See feedback_trapped_bilayer_relaxation_bug memory: bilayers in a handful of
light-element families (graphene, BN, silicene, germanene, TiS2, and their
heteropairs) can satisfy VASP's force-convergence criterion while still
sitting at ~2-3x the true interlayer separation, because the vdW attraction
at the shared starting dz guess is too flat there to pull the layers all the
way in. The resulting "converged" CONTCAR looks fine to VASP but is
physically two decoupled monolayers, not a bonded bilayer -- and a decoupled
bilayer trivially passes the phonon-stability check, producing a false
"healthy" verdict.
"""

from pathlib import Path
import sys

WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW_ROOT / "phonopy"))
from poscar_utils import read_poscar  # noqa: E402

# Observed range across every correctly-relaxed bilayer in this project:
# ~2.9-3.3 A for TMD-TMD pairs, ~3.5-5.0 A for monochalcogenide (GaSe/InSe)
# pairs. Anything above this is the trapped-relaxation signature -- roughly
# 2-3x the true separation, not a small deviation.
MAX_PLAUSIBLE_INTERLAYER_GAP = 5.5


def compute_interlayer_gap(contcar_path):
    """Return the physical interlayer separation (Angstroms) from a relaxed
    bilayer CONTCAR/POSCAR.

    Sorts every atom by z, computes consecutive gaps (with periodic wrap
    across the cell boundary), and returns the second-largest gap -- the
    largest gap is the intentional vacuum padding, not the interlayer bond.
    """
    data = read_poscar(contcar_path)
    lattice = [[c * data.scale for c in row] for row in data.lattice]
    c_length = lattice[2][2]

    def z_cartesian(pos):
        if data.coord_type == "Direct":
            return (
                pos[0] * lattice[0][2]
                + pos[1] * lattice[1][2]
                + pos[2] * lattice[2][2]
            )
        return pos[2] * data.scale

    zs = sorted(z_cartesian(p) % c_length for p in data.positions)

    gaps = []
    for i in range(len(zs)):
        nxt = zs[(i + 1) % len(zs)] + (c_length if i == len(zs) - 1 else 0)
        gaps.append(nxt - zs[i])
    gaps.sort(reverse=True)

    return gaps[1]


def check_bilayer_interlayer_gap(contcar_path, material_name):
    """Raise RuntimeError if a relaxed bilayer looks trapped (see module
    docstring). No-op for anything within the normal vdW contact range.

    Returns the measured gap (Angstroms) on success.
    """
    gap = compute_interlayer_gap(contcar_path)
    if gap > MAX_PLAUSIBLE_INTERLAYER_GAP:
        raise RuntimeError(
            f"{material_name}: relaxed CONTCAR shows interlayer separation of "
            f"{gap:.3f} A (expected ~2.9-5.0 A for a correctly-relaxed "
            "bilayer). This looks like a trapped relaxation (see "
            "feedback_trapped_bilayer_relaxation_bug memory), not a genuinely "
            "converged bilayer -- check the starting dz in "
            "mp_material_overrides.json and re-relax from a closer initial "
            "separation before proceeding."
        )
    return gap
