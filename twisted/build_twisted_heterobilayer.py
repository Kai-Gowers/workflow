#!/usr/bin/env python3
"""
Build commensurate twisted HETEROBILAYER supercells (the (m, m+1) series) — additive companion of
build_twisted_bilayer.py (whose rotation / supercell / angle machinery is imported unchanged).

Differences from the homobilayer builder
  * two materials: layer 1 = <mat1>, layer 2 = <mat2>, each with its own dMX from
    get_material_lattice_params (overrides first, then the MP cache);
  * ONE common in-plane lattice constant a = mean(a1, a2) for both layers (the untwisted
    heterobilayer convention of this workflow; the stacking-specific bilayer lattice override is
    deliberately not applied, as for the homobilayers). Both layers are strained by |a1 - a2| / 2,
    so the twisted cell is exactly commensurate — only sensible for near-matched pairs (MoS2/WS2:
    0.15 %; the strain per layer is recorded in twist.json);
  * interlayer surface-to-surface gap from the DFT-relaxed untwisted heterobilayer
    FINAL_RESULTS_HEALTHY/<mat1>_<mat2>_<3R|2H>/POSCAR (gap only);
  * the structure check is per layer (each layer's metal keeps 6 chalcogens at its own seed M–X distance).

Output: <out>/<mat1>_<mat2>_twist_m<m>_<family>/POSCAR + twist.json (same keys as the homobilayer
builder plus materials / a_layer1_A / a_layer2_A / strain_pct_layer1 / strain_pct_layer2), so
twisted/prepare_dft_inputs.py consumes them unchanged.

Usage (from workflow/):
  python3 twisted/build_twisted_heterobilayer.py --materials MoS2 WS2 --m 1 2 3 --family near0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_twisted_bilayer as H  # noqa: E402
from build_twisted_bilayer import (  # noqa: E402
    FAMILY_SEED, FINAL_RESULTS_HEALTHY, VACUUM, apply_bilayer_stacking, build_twisted, get_material_elements,
    get_material_lattice_params, get_monolayer_coords, recommended_phonopy_k, surface_gap_from_poscar,
    _hexagonal_lattice_matrix,
)


def seed_heterobilayer(mat1: str, mat2: str, stacking: str, gap: float):
    p1, meta1 = get_material_lattice_params(mat1, use_cache=True)
    p2, meta2 = get_material_lattice_params(mat2, use_cache=True)
    if p1 is None or p2 is None:
        raise RuntimeError(f"lattice params unavailable: {mat1}: {meta1}; {mat2}: {meta2}")
    a1, _c1, _dz1, dMX1 = p1
    a2, _c2, _dz2, dMX2 = p2
    a = 0.5 * (a1 + a2)
    c = VACUUM
    coords1, species1 = get_monolayer_coords(a, c, dMX1, get_material_elements(mat1))
    coords2, species2 = get_monolayer_coords(a, c, dMX2, get_material_elements(mat2))
    coords, species = apply_bilayer_stacking(stacking, coords1, species1, coords2, species2, c, hetero=True, dz=gap)
    atoms = Atoms(symbols=species, scaled_positions=np.array(coords), cell=_hexagonal_lattice_matrix(a, c), pbc=True)
    return atoms, float(a), float(a1), float(a2), float(dMX1), float(dMX2)


def check_hetero(twisted: Atoms, seed: Atoms, N: int, gap: float, mats: tuple[str, str]) -> dict:
    assert len(twisted) == N * len(seed), (len(twisted), N * len(seed))
    sym = np.array(twisted.get_chemical_symbols())
    z = twisted.positions[:, 2]
    mid = 0.5 * (z.min() + z.max())
    d = twisted.get_all_distances(mic=True)
    seed_sym = np.array(seed.get_chemical_symbols())
    zs = seed.positions[:, 2]
    smid = 0.5 * (zs.min() + zs.max())
    d_seed = seed.get_all_distances(mic=True)
    out = {}
    for li, (mat, sel, ssel) in enumerate(((mats[0], z < mid, zs < smid), (mats[1], z > mid, zs > smid)), start=1):
        metal, chalc = get_material_elements(mat)[:2]
        ref = np.min(d_seed[ssel & (seed_sym == metal)][:, ssel & (seed_sym == chalc)])
        mx = d[sel & (sym == metal)][:, sel & (sym == chalc)]
        assert (sel & (sym == metal)).sum() == N * 1, f"layer {li} metal count"
        nearest6 = np.sort(mx, axis=1)[:, :6]
        assert np.abs(nearest6 - ref).max() < 1e-6, f"layer {li} ({mat}) in-layer M-X distances distorted"
        out[f"MX_distance_layer{li}_A"] = float(ref)
    zgap = float(np.diff(np.sort(z)).max())
    assert abs(zgap - gap) < 1e-6, (zgap, gap)
    np.fill_diagonal(d, np.inf)
    out["min_pair_distance_A"] = float(d.min())
    return out


def build_one(mat1: str, mat2: str, m: int, family: str, out_root: Path, png: bool = False) -> dict:
    n = m + 1
    N = m * m + m * n + n * n
    stacking = FAMILY_SEED[family]
    ref_bilayer = f"{mat1}_{mat2}_{stacking}"
    gap = surface_gap_from_poscar(FINAL_RESULTS_HEALTHY / ref_bilayer / "POSCAR")
    seed, a, a1, a2, dMX1, dMX2 = seed_heterobilayer(mat1, mat2, stacking, gap)
    twisted, theta = build_twisted(seed, m, n)
    checks = check_hetero(twisted, seed, N, gap, (mat1, mat2))
    a_moire = float(np.linalg.norm(twisted.cell[0]))
    k = recommended_phonopy_k(a_moire, len(twisted))
    name = f"{mat1}_{mat2}_twist_m{m}_{family}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ase_write(str(out_dir / "POSCAR"), twisted, format="vasp", direct=True, sort=True)
    info = {
        "name": name, "material": f"{mat1}_{mat2}", "materials": [mat1, mat2], "family": family,
        "seed_stacking": stacking, "m": m, "n": n, "N_cells_per_layer": N, "n_atoms": len(twisted),
        "theta_deg": theta, "a_layer_A": a, "a_layer1_A": a1, "a_layer2_A": a2,
        "strain_pct_layer1": 100.0 * (a - a1) / a1, "strain_pct_layer2": 100.0 * (a - a2) / a2,
        "a_source": "mean of the two monolayers' get_material_lattice_params a (overrides-first, MP cache); "
                    "bilayer lattice override deliberately not applied",
        "dMX_A": [dMX1, dMX2], "c_A": VACUUM, "gap_A": gap,
        "gap_source": f"FINAL_RESULTS_HEALTHY/{ref_bilayer}/POSCAR (gap only)",
        "a_moire_A": a_moire, "phonopy_k_recommended": k, "phonopy_supercell_atoms": len(twisted) * k * k,
        "rotation_center": "layer-1 metal at origin, +theta about z, positions and cell",
        **checks,
    }
    (out_dir / "twist.json").write_text(json.dumps(info, indent=2) + "\n")
    if png:
        ase_write(str(out_dir / "top_view.png"), twisted, rotation="0x,0y,0z", show_unit_cell=2)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materials", nargs=2, required=True, metavar=("MAT1", "MAT2"), help="bottom and top layer")
    ap.add_argument("--m", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--family", nargs="+", choices=sorted(FAMILY_SEED), default=["near0"])
    ap.add_argument("--out", type=Path, default=H.DEFAULT_OUTPUT_DIR)
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    for m in args.m:
        for fam in args.family:
            i = build_one(args.materials[0], args.materials[1], m, fam, args.out, png=args.png)
            print(f"{i['name']:<28} theta={i['theta_deg']:6.2f} deg  atoms={i['n_atoms']:4d}  a={i['a_layer_A']:.4f} "
                  f"(strain {i['strain_pct_layer1']:+.3f}/{i['strain_pct_layer2']:+.3f} %)  a_m={i['a_moire_A']:6.2f} A  "
                  f"gap={i['gap_A']:.3f} A  k={i['phonopy_k_recommended']} ({i['phonopy_supercell_atoms']} atoms)  "
                  f"min d={i['min_pair_distance_A']:.3f} A")


if __name__ == "__main__":
    main()
