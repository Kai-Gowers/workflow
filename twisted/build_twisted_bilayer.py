#!/usr/bin/env python3
"""
Build commensurate twisted homobilayer supercells (the (m, m+1) series) for a TMD, as
model-only phonon inputs for nequix/scripts/eval_twisted_phonons.py.

Structure generation lives here in workflow/ (nequix/ is model-only). The 1x1 layer geometry
comes from the existing generators (get_monolayer_coords + apply_bilayer_stacking in
relaxation/bilayer/generate_bilayer_poscar.py); only the supercell/rotation is new (ASE).

Geometry conventions
  * 60 deg hexagonal cell from _hexagonal_lattice_matrix(a, c), c = 20 A (VACUUM), pbc TTT.
  * a, dMX from get_material_lattice_params(<mat>) -- material overrides first, then the MP cache.
    For MoS2 that is the MP/DFT monolayer value a = 3.19224 A. The stacking-specific bilayer
    lattice override (MoS2_bilayer_3R, ISIF=4, 3.154 A) is deliberately NOT applied: a moire
    cell has one a for both layers and its local stacking varies across the cell (user decision
    2026-09-17).
  * Interlayer surface-to-surface gap taken from the DFT-relaxed homobilayer POSCAR in
    FINAL_RESULTS_HEALTHY/<mat>_bilayer_<3R|2H>/POSCAR (largest z gap) -- gap only, nothing else.
  * Families: near0  = 3R-seeded (layer 2 = layer 1 shifted by (1/3, 1/3)), twist +theta
              near60 = 2H-seeded (layer 2 = layer 1 rotated 180 deg), twist +theta
    so the twist angle is measured from the untwisted 3R (0 deg) / 2H (60 deg) stacking.
  * Commensurate cell for (m, n=m+1): N = m^2 + mn + n^2 unit cells per layer,
    cos(theta) = (m^2 + 4mn + n^2) / (2N); layer 1 supercell P1 = [[m, n, 0], [-n, m+n, 0], [0, 0, 1]];
    layer 2 is rotated by +theta about the z axis through the layer-1 metal at the origin (positions
    and cell), then P2 = rint(cell1_super @ inv(cell2_rot)) with an exact-commensurability assert.
  * Recommended phonopy supercell k = ceil(25 A / a_m), reduced until the phonopy supercell has
    <= MAX_SUPERCELL_ATOMS atoms (k = 1 is Gamma-exact but off-Gamma-approximate).

Output: <out>/<mat>_twist_m<m>_<family>/POSCAR + twist.json, and <out>/summary.csv.

Usage (from workflow/):
  python3 twisted/build_twisted_bilayer.py                      # MoS2, m = 1..5, both families
  python3 twisted/build_twisted_bilayer.py --m 2 --family near0 --png
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import write as ase_write

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "relaxation" / "monolayer"))
sys.path.insert(0, str(REPO_ROOT / "relaxation" / "bilayer"))
sys.path.insert(0, str(REPO_ROOT / "common"))

from generate_bilayer_poscar import (  # noqa: E402
    VACUUM,
    _hexagonal_lattice_matrix,
    apply_bilayer_stacking,
    get_material_elements,
    get_monolayer_coords,
)
from materials_project_api import get_material_lattice_params  # noqa: E402

FINAL_RESULTS_HEALTHY = REPO_ROOT / "FINAL_RESULTS_HEALTHY"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "structures"

FAMILY_SEED = {"near0": "3R", "near60": "2H"}
PHONOPY_TARGET_LENGTH_A = 25.0
MAX_SUPERCELL_ATOMS = 900


def twist_angle_deg(m: int, n: int) -> float:
    N = m * m + m * n + n * n
    return math.degrees(math.acos((m * m + 4 * m * n + n * n) / (2.0 * N)))


def recommended_phonopy_k(a_moire: float, n_atoms: int) -> int:
    k = max(1, math.ceil(PHONOPY_TARGET_LENGTH_A / a_moire))
    while k > 1 and n_atoms * k * k > MAX_SUPERCELL_ATOMS:
        k -= 1
    return k


def surface_gap_from_poscar(path: Path) -> float:
    """Largest z gap between consecutive atomic planes (A) -- the surface-to-surface gap."""
    from ase.io import read as ase_read

    atoms = ase_read(str(path), format="vasp")
    z = np.sort(atoms.positions[:, 2])
    return float(np.diff(z).max())


def seed_bilayer(material: str, stacking: str, gap: float):
    """Untwisted 1x1 bilayer (ASE Atoms) from the workflow generators; returns (atoms, a, dMX)."""
    params, meta = get_material_lattice_params(material, use_cache=True)
    if params is None:
        raise RuntimeError(f"lattice params unavailable for {material}: {meta}")
    a, c, _dz_ignored, dMX = params
    c = VACUUM
    mat = get_material_elements(material)
    coords1, species1 = get_monolayer_coords(a, c, dMX, mat)
    coords2, species2 = get_monolayer_coords(a, c, dMX, mat)
    coords, species = apply_bilayer_stacking(
        stacking, coords1, species1, coords2, species2, c, hetero=False, dz=gap
    )
    cell = _hexagonal_lattice_matrix(a, c)
    atoms = Atoms(symbols=species, scaled_positions=np.array(coords), cell=cell, pbc=True)
    return atoms, float(a), float(dMX)


def split_layers(atoms: Atoms):
    z = atoms.positions[:, 2]
    n_layer = len(atoms) // 2
    order = np.argsort(z)
    bottom = atoms[order[:n_layer]]
    top = atoms[order[n_layer:]]
    return bottom, top


def build_twisted(seed: Atoms, m: int, n: int):
    """Twist the top layer of `seed` by +theta(m, n) about the z axis through the origin."""
    theta = twist_angle_deg(m, n)
    layer1, layer2 = split_layers(seed)
    P1 = np.array([[m, n, 0], [-n, m + n, 0], [0, 0, 1]])
    sc1 = make_supercell(layer1, P1)

    rot = layer2.copy()
    rot.rotate(theta, "z", center=(0.0, 0.0, 0.0), rotate_cell=True)
    P2 = np.rint(sc1.cell[:] @ np.linalg.inv(rot.cell[:]))
    residual = np.abs(P2 @ rot.cell[:] - sc1.cell[:]).max()
    assert residual < 1e-8, f"layer 2 not commensurate after +{theta:.4f} deg (residual {residual:.2e})"
    sc2 = make_supercell(rot, P2.astype(int))

    twisted = sc1 + sc2
    twisted.set_cell(sc1.cell[:])
    twisted.pbc = True
    twisted.wrap(eps=1e-10)
    return twisted, theta


def check_structure(twisted: Atoms, seed: Atoms, N: int, gap: float, metal: str, chalc: str):
    n_seed = len(seed)
    assert len(twisted) == N * n_seed, (len(twisted), N * n_seed)
    symbols = np.array(twisted.get_chemical_symbols())
    assert (symbols == metal).sum() == N * (n_seed // 3), "metal count"
    # in-layer geometry must be untouched: every metal has 6 chalcogens at the seed's M-X distance
    d_seed = seed.get_all_distances(mic=True)
    seed_sym = np.array(seed.get_chemical_symbols())
    ref_mx = np.min(d_seed[seed_sym == metal][:, seed_sym == chalc])
    d = twisted.get_all_distances(mic=True)
    mx = d[symbols == metal][:, symbols == chalc]
    nearest6 = np.sort(mx, axis=1)[:, :6]
    assert np.abs(nearest6 - ref_mx).max() < 1e-6, "in-layer M-X distances distorted"
    # interlayer gap preserved
    z = np.sort(twisted.positions[:, 2])
    zgap = float(np.diff(z).max())
    assert abs(zgap - gap) < 1e-6, (zgap, gap)
    # no unphysically close pairs
    np.fill_diagonal(d, np.inf)
    return {"min_pair_distance_A": float(d.min()), "MX_distance_A": float(ref_mx)}


def build_one(material: str, m: int, family: str, out_root: Path, png: bool = False) -> dict:
    n = m + 1
    N = m * m + m * n + n * n
    stacking = FAMILY_SEED[family]
    homobilayer = f"{material}_bilayer_{stacking}"
    gap = surface_gap_from_poscar(FINAL_RESULTS_HEALTHY / homobilayer / "POSCAR")
    seed, a, dMX = seed_bilayer(material, stacking, gap)
    metal, chalc = get_material_elements(material)[:2]

    twisted, theta = build_twisted(seed, m, n)
    checks = check_structure(twisted, seed, N, gap, metal, chalc)
    a_moire = float(np.linalg.norm(twisted.cell[0]))
    k = recommended_phonopy_k(a_moire, len(twisted))

    name = f"{material}_twist_m{m}_{family}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ase_write(str(out_dir / "POSCAR"), twisted, format="vasp", direct=True, sort=True)
    info = {
        "name": name,
        "material": material,
        "family": family,
        "seed_stacking": stacking,
        "m": m,
        "n": n,
        "N_cells_per_layer": N,
        "n_atoms": len(twisted),
        "theta_deg": theta,
        "a_layer_A": a,
        "a_source": "get_material_lattice_params (overrides-first, MP cache); stacking-specific bilayer "
        "lattice override deliberately not applied",
        "dMX_A": dMX,
        "c_A": VACUUM,
        "gap_A": gap,
        "gap_source": f"FINAL_RESULTS_HEALTHY/{homobilayer}/POSCAR (gap only)",
        "a_moire_A": a_moire,
        "phonopy_k_recommended": k,
        "phonopy_supercell_atoms": len(twisted) * k * k,
        "rotation_center": "layer-1 metal at origin, +theta about z, positions and cell",
        **checks,
    }
    (out_dir / "twist.json").write_text(json.dumps(info, indent=2) + "\n")
    if png:
        ase_write(str(out_dir / "top_view.png"), twisted, rotation="0x,0y,0z", show_unit_cell=2)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material", default="MoS2", help="Homobilayer material (default: MoS2)")
    parser.add_argument("--m", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="m of the (m, m+1) series")
    parser.add_argument("--family", nargs="+", choices=sorted(FAMILY_SEED), default=["near0", "near60"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output root (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--png", action="store_true", help="Also write a top-view PNG per structure")
    args = parser.parse_args()

    rows = []
    for family in args.family:
        for m in args.m:
            info = build_one(args.material, m, family, args.out, png=args.png)
            rows.append(info)
            print(
                f"{info['name']:<28} theta={info['theta_deg']:6.2f} deg  atoms={info['n_atoms']:4d}"
                f"  a_m={info['a_moire_A']:6.2f} A  gap={info['gap_A']:.3f} A  k={info['phonopy_k_recommended']}"
                f" ({info['phonopy_supercell_atoms']} atoms)  min d={info['min_pair_distance_A']:.3f} A"
            )

    keys = [
        "name", "family", "seed_stacking", "m", "n", "N_cells_per_layer", "n_atoms", "theta_deg",
        "a_layer_A", "gap_A", "a_moire_A", "phonopy_k_recommended", "phonopy_supercell_atoms",
        "min_pair_distance_A",
    ]
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print(f"\nWrote {len(rows)} structures under {args.out}")


if __name__ == "__main__":
    main()
