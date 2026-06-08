#!/usr/bin/env python3
"""
Generate POSCAR files for bilayer structures with family-appropriate stacking.

Stacking labels depend on pair type (TMD/TMD, honeycomb/honeycomb, TMD/honeycomb).
By default, layers are taken from relaxed monolayer CONTCAR files in monolayer_examples.
Use ``use_relaxed_monolayers=False`` (CLI: ``--use-templates``) for ideal template geometry.
"""

import sys
import math
import numpy as np
from pathlib import Path

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.vasp import Poscar
except ImportError:
    print("Error: pymatgen is not installed. Please install it with: pip install pymatgen")
    sys.exit(1)

# Import lattice parameters and functions from monolayer
MONOLAYER_DIR = Path(__file__).parent.parent / "monolayer"
sys.path.insert(0, str(MONOLAYER_DIR))
from generate_monolayer_poscar import (
    LATTICE_PARAMS,
    DZ_BILAYER,
    get_tmd_monolayer_coords,
    get_binary_monolayer_coords,
    get_single_element_coords,
    get_ternary_coords,
)

COMMON_DIR = Path(__file__).parent.parent.parent / "common"
sys.path.insert(0, str(COMMON_DIR))
try:
    from materials_project_api import get_material_lattice_params
    MP_HELPERS_AVAILABLE = True
except ImportError:
    MP_HELPERS_AVAILABLE = False

from structural_families import validate_stacking, STACKING_SUFFIXES

try:
    sys.path.insert(0, str(MONOLAYER_DIR))
    from validate_structure import validate_bilayer_structure
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

VACUUM = 20.0

# Material structure type classification
MATERIAL_STRUCTURE_TYPES = {
    # Single element 2D materials (2 atoms per unit cell)
    'single_element': ['graphene', 'phosphorene', 'silicene', 'germanene', 'stanene'],
    
    # TMDs (3 atoms per unit cell: metal + 2 chalcogens)
    'tmd': ['MoS2', 'MoSe2', 'MoTe2', 'WS2', 'WSe2', 'WTe2', 'NbS2', 'NbSe2', 'NbTe2',
            'TaS2', 'TaSe2', 'TaTe2', 'ReS2', 'ReSe2', 'SnS2', 'SnSe2', 'TiS2', 'TiSe2',
            'ZrS2', 'ZrSe2', 'HfS2', 'HfSe2'],
    
    # Binary compounds (2 atoms per unit cell)
    'binary': ['BN', 'GaN', 'InSe', 'GaSe'],
    
    # Ternary / alloy monolayers
    'ternary': ['MoSSe', 'WSSe', 'MoWSe2', 'MoWTe2'],
}


def get_material_structure_type(material_name):
    """
    Get the structure type of a material.
    
    Parameters:
    -----------
    material_name : str
        Material name
    
    Returns:
    --------
    str : Structure type ('single_element', 'tmd', 'binary', or None)
    """
    for struct_type, materials in MATERIAL_STRUCTURE_TYPES.items():
        if material_name in materials:
            return struct_type
    return None


def are_materials_compatible(mat1, mat2):
    """
    Check if two materials have compatible structures for stacking.
    
    Parameters:
    -----------
    mat1 : str
        First material name
    mat2 : str
        Second material name
    
    Returns:
    --------
    bool : True if materials can be stacked together
    """
    type1 = get_material_structure_type(mat1)
    type2 = get_material_structure_type(mat2)
    
    # Both must have known structure types
    if type1 is None or type2 is None:
        return False
    
    # Materials must have the same structure type
    return type1 == type2


def get_monolayer_coords(a, c, dMX, mat):
    """
    Get monolayer coordinates, automatically selecting the right function based on material type.
    """
    if len(mat) == 1:
        return get_single_element_coords(a, c, mat)
    if len(mat) == 2:
        if mat == ['B', 'N'] or mat[0] in ['Ga', 'In']:
            return get_binary_monolayer_coords(a, c, dMX, mat)
        return get_tmd_monolayer_coords(a, c, dMX, mat)
    if len(mat) == 3:
        return get_ternary_coords(a, c, dMX, mat)
    return get_tmd_monolayer_coords(a, c, dMX, mat[:2])


def _shift_frac(x, y, dx=1/3, dy=1/3):
    """Shift fractional coordinates by (dx, dy) and wrap to [0, 1)."""
    return ((x + dx) % 1.0, (y + dy) % 1.0)


def _flip_frac_180(x, y):
    """
    Rotate (x, y) by 180° in the hexagonal plane (fractional coords).
    Maps (0, 0) <-> (1/3, 1/3) so that second layer chalcogens sit beneath first-layer metal.
    """
    return ((1/3 - x) % 1.0, (1/3 - y) % 1.0)


def _stack_bilayer_shift(
    coords1, species1, coords2, species2, c, dx=1/3, dy=1/3, hetero=False, dz=None
):
    """Apply a rigid in-plane fractional shift (dx, dy) to layer 2, with vertical separation."""
    if dz is None:
        dz = DZ_BILAYER
    dz_frac = dz / c

    if hetero:
        z1_center = sum(c[2] for c in coords1) / len(coords1)
        z2_center = sum(c[2] for c in coords2) / len(coords2)
        z2_target_center = z1_center + dz_frac
        z_shift = z2_target_center - z2_center

        coords2_out = [
            [*_shift_frac(c[0], c[1], dx, dy), c[2] + z_shift] for c in coords2
        ]
        return coords1 + coords2_out, species1 + species2

    coords2_out = [
        [*_shift_frac(c[0], c[1], dx, dy), c[2] + dz_frac] for c in coords1
    ]
    return coords1 + coords2_out, species1 + species1


def _stack_bilayer_3R(coords1, species1, coords2, species2, c, hetero=False, dz=None):
    """Apply 3R stacking to pre-built layer fractional coordinates (unchanged semantics)."""
    return _stack_bilayer_shift(
        coords1, species1, coords2, species2, c, dx=1/3, dy=1/3, hetero=hetero, dz=dz
    )


def _stack_bilayer_2H(coords1, species1, coords2, species2, c, dz=None):
    """Apply 2H stacking to pre-built layer fractional coordinates (unchanged semantics)."""
    if dz is None:
        dz = DZ_BILAYER
    dz_frac = dz / c
    z1_center = sum(c[2] for c in coords1) / len(coords1)
    z2_center = sum(c[2] for c in coords2) / len(coords2)
    z2_target_center = z1_center + dz_frac
    z_shift = z2_target_center - z2_center
    coords2_out = [[*_flip_frac_180(c[0], c[1]), c[2] + z_shift] for c in coords2]
    return coords1 + coords2_out, species1 + species2


def _stack_bilayer_identity(coords1, species1, coords2, species2, c, hetero=False, dz=None):
    """Apply identity in-plane stacking (AA, TM_TX): vertical offset only."""
    if dz is None:
        dz = DZ_BILAYER
    dz_frac = dz / c

    if hetero:
        z1_center = sum(c[2] for c in coords1) / len(coords1)
        z2_center = sum(c[2] for c in coords2) / len(coords2)
        z2_target_center = z1_center + dz_frac
        z_shift = z2_target_center - z2_center
        coords2_out = [[c[0], c[1], c[2] + z_shift] for c in coords2]
        return coords1 + coords2_out, species1 + species2

    coords2_out = [[c[0], c[1], c[2] + dz_frac] for c in coords1]
    return coords1 + coords2_out, species1 + species1


def apply_bilayer_stacking(
    stacking, coords1, species1, coords2, species2, c, hetero=False, dz=None
):
    """Dispatch stacking label to the appropriate layer-2 transform."""
    if stacking in ("3R", "TM_H"):
        return _stack_bilayer_shift(
            coords1, species1, coords2, species2, c,
            dx=1/3, dy=1/3, hetero=hetero, dz=dz,
        )
    if stacking == "AB":
        # Homobilayer Bernal: (+1/3, +1/3).  Heterobilayer honeycombs share the same
        # cation-at-origin registry, so (+1/3, +1/3) eclipses cations on anions;
        # (-1/3, -1/3) places layer-2 anion over layer-1 cation and cation over hollow.
        dx, dy = (-1/3, -1/3) if hetero else (1/3, 1/3)
        return _stack_bilayer_shift(
            coords1, species1, coords2, species2, c,
            dx=dx, dy=dy, hetero=hetero, dz=dz,
        )
    if stacking in ("2H", "AA_prime"):
        return _stack_bilayer_2H(coords1, species1, coords2, species2, c, dz=dz)
    if stacking in ("AA", "TM_TX"):
        return _stack_bilayer_identity(
            coords1, species1, coords2, species2, c, hetero=hetero, dz=dz
        )
    raise ValueError(f"Unknown stacking type: {stacking}")


def get_bilayer_coords_3R(a, c, dMX, mat1, mat2=None, dz=None):
    """
    Get fractional coordinates for 3R stacking: second layer shifted by (1/3, 1/3).
    """
    if mat2 is None:
        mat2 = mat1
    coords1, species1 = get_monolayer_coords(a, c, dMX, mat1)
    if mat1 != mat2:
        coords2, species2 = get_monolayer_coords(a, c, dMX, mat2)
        return _stack_bilayer_3R(
            coords1, species1, coords2, species2, c, hetero=True, dz=dz
        )
    return _stack_bilayer_3R(
        coords1, species1, coords1, species1, c, hetero=False, dz=dz
    )


def get_bilayer_coords_2H(a, c, dMX, mat1, mat2=None, dz=None):
    """Get fractional coordinates for 2H stacking."""
    if mat2 is None:
        mat2 = mat1
    coords1, species1 = get_monolayer_coords(a, c, dMX, mat1)
    coords2, species2 = get_monolayer_coords(a, c, dMX, mat2)
    return _stack_bilayer_2H(coords1, species1, coords2, species2, c, dz=dz)


def get_bilayer_coords(a, c, dMX, mat1, mat2, stacking, dz=None):
    """Get fractional coordinates for any supported stacking label."""
    coords1, species1 = get_monolayer_coords(a, c, dMX, mat1)
    hetero = mat1 != mat2
    if hetero:
        coords2, species2 = get_monolayer_coords(a, c, dMX, mat2)
    else:
        coords2, species2 = coords1, species1
    return apply_bilayer_stacking(
        stacking, coords1, species1, coords2, species2, c, hetero=hetero, dz=dz
    )


def _parse_material_pair(materials):
    """Parse material1_material2 prefix from a bilayer name."""
    import re

    match = re.match(
        r"^(graphene|[A-Z][a-z]?[A-Z]?[0-9]?)_(graphene|[A-Z][a-z]?[A-Z]?[0-9]?)$",
        materials,
    )
    if match:
        return match.group(1), match.group(2)
    mat1, mat2 = materials.split("_", 1)
    return mat1, mat2


def parse_bilayer_name(bilayer_name):
    """
    Parse bilayer name to extract materials and stacking type.
    """
    if "_bilayer_" in bilayer_name:
        base_material, stacking = bilayer_name.rsplit("_bilayer_", 1)
        if stacking not in STACKING_SUFFIXES:
            raise ValueError(f"Unknown stacking suffix in bilayer name: {bilayer_name}")
        return base_material, base_material, stacking

    for suffix in STACKING_SUFFIXES:
        token = f"_{suffix}"
        if bilayer_name.endswith(token):
            materials = bilayer_name[: -len(token)]
            mat1, mat2 = _parse_material_pair(materials)
            return mat1, mat2, suffix

    raise ValueError(f"Could not parse bilayer name: {bilayer_name}")


def get_material_elements(material_name):
    """Extract element list from material name (including ternaries)."""
    if material_name in ['graphene', 'phosphorene', 'silicene', 'germanene', 'stanene']:
        return {
            'graphene': ['C'],
            'phosphorene': ['P'],
            'silicene': ['Si'],
            'germanene': ['Ge'],
            'stanene': ['Sn'],
        }[material_name]

    if material_name == 'MoSSe':
        return ['Mo', 'S', 'Se']
    if material_name == 'WSSe':
        return ['W', 'S', 'Se']
    if material_name == 'MoWSe2':
        return ['Mo', 'W', 'Se']
    if material_name == 'MoWTe2':
        return ['Mo', 'W', 'Te']
    if material_name == 'BN':
        return ['B', 'N']
    if material_name == 'GaN':
        return ['Ga', 'N']
    if material_name == 'InSe':
        return ['In', 'Se']
    if material_name == 'GaSe':
        return ['Ga', 'Se']

    if material_name.startswith('Mo'):
        metal = 'Mo'
    elif material_name.startswith('W'):
        metal = 'W'
    elif material_name.startswith('Nb'):
        metal = 'Nb'
    elif material_name.startswith('Ta'):
        metal = 'Ta'
    elif material_name.startswith('Re'):
        metal = 'Re'
    elif material_name.startswith('Sn'):
        metal = 'Sn'
    elif material_name.startswith('Ti'):
        metal = 'Ti'
    elif material_name.startswith('Zr'):
        metal = 'Zr'
    elif material_name.startswith('Hf'):
        metal = 'Hf'
    else:
        metal = 'Mo'

    if 'S2' in material_name:
        chalcogen = 'S'
    elif 'Se2' in material_name:
        chalcogen = 'Se'
    elif 'Te2' in material_name:
        chalcogen = 'Te'
    else:
        chalcogen = 'S'

    return [metal, chalcogen]


def _hexagonal_lattice_matrix(a, c):
    return np.array([
        [a, 0, 0],
        [a / 2, a * math.sqrt(3) / 2, 0],
        [0, 0, c],
    ])


def _wrap_frac_xy(x, y):
    """Wrap in-plane fractional coordinates to [0, 1)."""
    return x % 1.0, y % 1.0


def _write_bilayer_poscar(a, c, coords, species, output_path, comment=None):
    """Write bilayer POSCAR in Direct coordinates (fractional, species-grouped)."""
    PHONOPY_DIR = Path(__file__).parent.parent.parent / "phonopy"
    if str(PHONOPY_DIR) not in sys.path:
        sys.path.insert(0, str(PHONOPY_DIR))
    from poscar_utils import PoscarData, sort_by_species, write_poscar

    lattice_matrix = _hexagonal_lattice_matrix(a, c)
    direct_coords = []
    for fc in coords:
        x, y = _wrap_frac_xy(fc[0], fc[1])
        direct_coords.append([x, y, fc[2]])
    sorted_symbols, sorted_positions, _, _ = sort_by_species(species, direct_coords)
    if comment is None:
        comment = "Bilayer structure"
    poscar_data = PoscarData(
        comment=comment,
        scale=1.0,
        lattice=lattice_matrix.tolist(),
        symbols=sorted_symbols,
        positions=sorted_positions,
        coord_type="Direct",
    )
    write_poscar(poscar_data, output_path)
    return Structure(Lattice(lattice_matrix), species, coords)


def _generate_bilayer_from_relaxed_monolayers(
    mat1,
    mat2,
    stacking,
    output_path,
    vacuum,
    dz,
    monolayer_examples_dir,
    anchor=None,
):
    """Build bilayer from relaxed monolayer CONTCAR Direct coordinates.

    Workflow: read fractional coords from each CONTCAR, apply stacking in fractional
    space (rigid shifts/flips with modulo-1 wrapping in-plane), and write the POSCAR
    under a Direct header without Cartesian conversion.
    """
    from relaxed_monolayer import (
        default_monolayer_examples_dir,
        hexagonal_lattice_a,
        layer_in_vacuum_cell,
        load_relaxed_monolayer,
    )

    mono_dir = monolayer_examples_dir or default_monolayer_examples_dir()
    struct1 = load_relaxed_monolayer(mat1, mono_dir)
    a1 = hexagonal_lattice_a(struct1)

    validate_stacking(mat1, mat2, stacking)

    if mat1 == mat2:
        a = a1
        coords1, species1, _ = layer_in_vacuum_cell(struct1, vacuum, in_plane_a=a)
        coords, species = apply_bilayer_stacking(
            stacking, coords1, species1, coords1, species1, vacuum, hetero=False, dz=dz
        )
    else:
        struct2 = load_relaxed_monolayer(mat2, mono_dir)
        a2 = hexagonal_lattice_a(struct2)
        if anchor == 1:
            a = a1
        elif anchor == 2:
            a = a2
        else:
            a = (a1 + a2) / 2
        coords1, species1, _ = layer_in_vacuum_cell(struct1, vacuum, in_plane_a=a)
        coords2, species2, _ = layer_in_vacuum_cell(struct2, vacuum, in_plane_a=a)
        coords, species = apply_bilayer_stacking(
            stacking, coords1, species1, coords2, species2, vacuum, hetero=True, dz=dz
        )

    comment = f"Bilayer {mat1}/{mat2} {stacking} (relaxed monolayers)"
    structure = _write_bilayer_poscar(a, vacuum, coords, species, output_path, comment=comment)
    return structure, {
        "source": "relaxed_monolayer",
        "a": a,
        "c": vacuum,
        "dz": dz,
        "poscar_method": "relaxed",
        "monolayer_examples_dir": str(Path(mono_dir).resolve()),
    }


def _default_params(material_name):
    a, c, dmx = LATTICE_PARAMS.get(material_name, (3.2, 20.0, 1.58))
    return a, c, DZ_BILAYER, dmx


def _resolve_bilayer_lattice_params(
    mat1,
    mat2,
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
    anchor=None,
):
    """Resolve (a, c, dz, dMX) for bilayer template generation.

    Parameters
    ----------
    anchor : int or None, optional
        Which material's in-plane lattice constant ``a`` to use for the bilayer.
        ``1`` locks to mat1, ``2`` locks to mat2, ``None`` uses the arithmetic mean.
    """
    if anchor not in (None, 1, 2):
        raise ValueError(f"anchor must be None, 1, or 2, got {anchor!r}")
    a1, c1, dz1, dmx1 = _default_params(mat1)
    a2, _, dz2, dmx2 = _default_params(mat2)
    meta = {"mat1_id": None, "mat2_id": None}
    source = "fallback"

    if use_mp and MP_HELPERS_AVAILABLE:
        p1, m1 = get_material_lattice_params(
            mat1,
            api_key=mp_api_key,
            use_cache=True,
            refresh_cache=mp_refresh,
            verbose=mp_verbose,
        )
        p2, m2 = get_material_lattice_params(
            mat2,
            api_key=mp_api_key,
            use_cache=True,
            refresh_cache=mp_refresh,
            verbose=mp_verbose,
        )
        if p1 is not None and p2 is not None:
            a1, c1, dz1, dmx1 = p1
            a2, _, dz2, dmx2 = p2
            meta = {
                "mat1_id": m1.get("material_id"),
                "mat2_id": m2.get("material_id"),
            }
            source = "mp"
        elif mp_verbose:
            reason = m1.get("reason") if p1 is None else m2.get("reason")
            print(f"[MP] bilayer fallback: {reason}")

    if anchor == 1:
        a = a1
    elif anchor == 2:
        a = a2
    else:
        a = (a1 + a2) / 2
    c = c1
    dz = (dz1 + dz2) / 2
    dmx = (dmx1 + dmx2) / 2
    return a, c, dz, dmx, meta, source


def _generate_bilayer_poscar_template(
    mat1, mat2, stacking, output_path, a, c, dz, dMX, source="fallback"
):
    """Template-based bilayer generation from resolved lattice parameters."""
    validate_stacking(mat1, mat2, stacking)
    elem1 = get_material_elements(mat1)
    elem2 = get_material_elements(mat2)

    coords, species = get_bilayer_coords(a, c, dMX, elem1, elem2, stacking, dz=dz)

    structure = _write_bilayer_poscar(a, c, coords, species, output_path)
    return structure, {"source": source, "a": a, "c": c, "dz": dz, "dMX": dMX}


def generate_bilayer_poscar(
    bilayer_name,
    output_dir,
    filename="POSCAR",
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
    strict_validation=False,
    anchor=None,
    vacuum=None,
    metadata_out=None,
    monolayer_examples_dir=None,
    use_relaxed_monolayers=True,
):
    """
    Generate POSCAR file for a bilayer structure.

    By default, layers are built from relaxed monolayer CONTCAR files in
    ``monolayer_examples/<material>/``. Set ``use_relaxed_monolayers=False`` to use
    ideal templates.

    Parameters
    ----------
    monolayer_examples_dir : path, optional
        Root directory of relaxed monolayer examples (default: workflow/monolayer_examples).
    use_relaxed_monolayers : bool
        If True (default), require CONTCAR/POSCAR per material in monolayer_examples.
    metadata_out : dict, optional
        If provided, filled with generation metadata (e.g. ``poscar_method``, ``a``, ``dz``).
    """
    mat1, mat2, stacking = parse_bilayer_name(bilayer_name)
    validate_stacking(mat1, mat2, stacking)
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if vacuum is None:
        vacuum = VACUUM

    if mat1 != mat2 and anchor is not None and mp_verbose and not use_relaxed_monolayers:
        print(f"[bilayer] heterobilayer legacy template, anchor={anchor}")

    if use_relaxed_monolayers:
        from relaxed_monolayer import default_monolayer_examples_dir

        mono_dir = monolayer_examples_dir or default_monolayer_examples_dir()
        _, _, dz1, _ = _default_params(mat1)
        _, _, dz2, _ = _default_params(mat2)
        dz = (dz1 + dz2) / 2
        structure, metadata = _generate_bilayer_from_relaxed_monolayers(
            mat1,
            mat2,
            stacking,
            output_path,
            vacuum,
            dz,
            mono_dir,
            anchor=anchor,
        )
        mp_meta = {}
        source = metadata.get("source", "relaxed_monolayer")
        a = metadata["a"]
        c = metadata["c"]
        dMX = None
    else:
        a, c, dz, dMX, mp_meta, source = _resolve_bilayer_lattice_params(
            mat1,
            mat2,
            use_mp=use_mp,
            mp_api_key=mp_api_key,
            mp_refresh=mp_refresh,
            mp_verbose=mp_verbose,
            anchor=anchor,
        )
        structure, metadata = _generate_bilayer_poscar_template(
            mat1, mat2, stacking, output_path, a, c, dz, dMX, source=source
        )

    if metadata_out is not None:
        metadata_out.update(
            {
                "poscar_method": metadata.get("poscar_method", "template"),
                "a": a,
                "c": c,
                "dz": dz if use_relaxed_monolayers else metadata.get("dz", dz),
                "dMX": dMX,
                "source": source,
                "structure": structure,
                **mp_meta,
                **{k: v for k, v in metadata.items() if k != "structure"},
            }
        )

    if VALIDATION_AVAILABLE:
        valid, messages = validate_bilayer_structure(structure, strict=strict_validation)
        if not valid and strict_validation:
            raise ValueError(
                f"Bilayer validation failed for {mat1}/{mat2}: {'; '.join(messages)}"
            )
        if not valid and mp_verbose:
            print(f"Warning: bilayer validation: {'; '.join(messages)}")

    mid1 = mp_meta.get("mat1_id")
    mid2 = mp_meta.get("mat2_id")
    ids = f", material_ids={mid1}/{mid2}" if mid1 and mid2 else ""
    dz_print = metadata.get("dz", dz) if use_relaxed_monolayers else dz
    dmx_print = f" dMX={dMX:.4f}" if dMX is not None else ""
    print(
        f"Generated: {output_path} (source={source}{ids}, "
        f"a={a:.4f} c={c:.1f} dz={dz_print:.2f}{dmx_print})"
    )
    return output_path


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate POSCAR file for bilayer structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate POSCAR for MoS2 bilayer with 3R stacking
  python3 generate_bilayer_poscar.py MoS2_bilayer_3R -o output_dir

  # Generate POSCAR for BN bilayer with AB stacking
  python3 generate_bilayer_poscar.py BN_bilayer_AB -o output_dir

  # Generate POSCAR for MoS2/GaN heterostructure with TM_TX stacking
  python3 generate_bilayer_poscar.py MoS2_GaN_TM_TX -o output_dir
        """
    )
    parser.add_argument(
        "bilayer_name",
        type=str,
        help="Bilayer name (e.g., 'MoS2_bilayer_3R', 'BN_bilayer_AB', 'MoS2_GaN_TM_TX')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Output directory for POSCAR"
    )
    parser.add_argument(
        "-f", "--filename",
        type=str,
        default="POSCAR",
        help="Output filename (default: POSCAR)"
    )
    parser.add_argument(
        "--no-mp",
        action="store_true",
        help="Disable Materials Project lookup and use template-only generation",
    )
    parser.add_argument(
        "--mp-api-key",
        type=str,
        default=None,
        help="Materials Project API key (default: MP_API_KEY environment variable)",
    )
    parser.add_argument(
        "--mp-refresh",
        action="store_true",
        help="Refresh MP cache entries for queried materials",
    )
    parser.add_argument(
        "--mp-verbose",
        action="store_true",
        help="Print MP selection/fallback details",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Treat bilayer validation failures as hard errors",
    )
    parser.add_argument(
        "--anchor",
        type=int,
        choices=[1, 2],
        default=None,
        help="Lock bilayer in-plane lattice a to mat1 (1) or mat2 (2); default: average",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=None,
        help="c-axis cell height for bilayers (default: 20 Å)",
    )
    parser.add_argument(
        "--monolayer-examples-dir",
        type=str,
        default=None,
        help="Root directory of relaxed monolayer examples (default: monolayer_examples)",
    )
    parser.add_argument(
        "--use-templates",
        action="store_true",
        help="Use ideal template monolayer geometry instead of relaxed CONTCAR files",
    )

    args = parser.parse_args()

    try:
        output_path = generate_bilayer_poscar(
            bilayer_name=args.bilayer_name,
            output_dir=args.output_dir,
            filename=args.filename,
            use_mp=not args.no_mp,
            mp_api_key=args.mp_api_key,
            mp_refresh=args.mp_refresh,
            mp_verbose=args.mp_verbose,
            strict_validation=args.strict_validation,
            anchor=args.anchor,
            vacuum=args.vacuum,
            monolayer_examples_dir=args.monolayer_examples_dir,
            use_relaxed_monolayers=not args.use_templates,
        )
        print(f"Generated POSCAR: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
