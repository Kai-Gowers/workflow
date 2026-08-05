#!/usr/bin/env python3
"""
Script to generate POSCAR files for 2D materials from common/materials_list.txt
Uses pymatgen for structure generation
"""

import os
import math
import sys
import numpy as np
from pathlib import Path

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.vasp import Poscar
except ImportError:
    print("Error: pymatgen is not installed. Please install it with: pip install pymatgen")
    exit(1)

COMMON_DIR = Path(__file__).parent.parent.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

try:
    from materials_project_api import get_material_lattice_params, _load_overrides
    MP_HELPERS_AVAILABLE = True
except ImportError:
    MP_HELPERS_AVAILABLE = False

try:
    from validate_structure import validate_structure
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

from cli_helpers import add_mp_args

# Lattice parameters for common 2D materials (in Angstrom)
# Format: {material_name: (a, c, dMX)}
# a: in-plane lattice constant
# c: c-axis length (for vacuum)
# dMX: vertical spacing between metal and chalcogen (for TMDs)
LATTICE_PARAMS = {
    # Single element 2D materials
    'graphene': (2.46, 20.0, 0.0),
    'phosphorene': (3.32, 20.0, 0.0),  # a=3.32, b=4.58
    'silicene': (3.86, 20.0, 0.4864),
    'germanene': (4.02, 20.0, 0.7088),
    'stanene': (4.70, 20.0, 0.8405),
    
    # TMDs - typical lattice parameter ~3.2-3.3 Angstrom, dMX ~1.58
    'MoS2': (3.16, 20.0, 1.58),
    'MoSe2': (3.29, 20.0, 1.58),
    'MoTe2': (3.52, 20.0, 1.58),
    'WS2': (3.15, 20.0, 1.58),
    'WSe2': (3.28, 20.0, 1.58),
    'WTe2': (3.52, 20.0, 1.58),
    'NbS2': (3.31, 20.0, 1.58),
    'NbSe2': (3.44, 20.0, 1.58),
    'NbTe2': (3.65, 20.0, 1.58),
    'TaS2': (3.31, 20.0, 1.58),
    'TaSe2': (3.43, 20.0, 1.58),
    'TaTe2': (3.65, 20.0, 1.58),
    'ReS2': (3.32, 20.0, 1.58),
    'ReSe2': (3.35, 20.0, 1.58),
    
    # Binary compounds
    'BN': (2.50, 20.0, 0.0),
    'GaN': (3.19, 20.0, 1.5),
    'InSe': (4.05, 20.0, 1.5),
    'GaSe': (3.75, 20.0, 1.5),
    'SnS2': (3.65, 20.0, 1.58),
    'SnSe2': (3.81, 20.0, 1.58),
    'TiS2': (3.41, 20.0, 1.58),
    'TiSe2': (3.54, 20.0, 1.58),
    'ZrS2': (3.66, 20.0, 1.58),
    'ZrSe2': (3.77, 20.0, 1.58),
    'HfS2': (3.63, 20.0, 1.58),
    'HfSe2': (3.74, 20.0, 1.58),
    
    # Ternary compounds
    'MoSSe': (3.22, 20.0, 1.58),
    'WSSe': (3.21, 20.0, 1.58),
    'MoWSe2': (3.28, 20.0, 1.58),
    'MoWTe2': (3.52, 20.0, 1.58),
}

# Interlayer spacing for bilayers (in Angstrom)
DZ_BILAYER = 6.5


def get_tmd_monolayer_coords(a, c, dMX, mat):
    """
    Get fractional coordinates for TMD monolayer.
    Parameters:
    a (float): In-plane lattice constant
    c (float): c-axis length
    dMX (float): Vertical spacing between metal and chalcogen
    mat (list): [metal, chalcogen] elements
    Returns:
    coords (list): Fractional coordinates
    species (list): Element species
    """
    # Convert dMX to fractional coordinate
    dMX_frac = dMX / c
    z_center = 0.5  # Center in fractional coordinates
    
    coords = [
        [0, 0, z_center],  # Metal at center
        [1/3, 1/3, z_center - dMX_frac],  # Chalcogen bottom
        [1/3, 1/3, z_center + dMX_frac],  # Chalcogen top
    ]
    
    species = [mat[0], mat[1], mat[1]]
    return coords, species

def get_binary_monolayer_coords(a, c, dMX, mat):
    """Get fractional coordinates for a stable flat binary monolayer."""
    z_center = 0.5

    # All true 2D binary monolayers (BN, GaN, etc.) share the honeycomb lattice
    coords = [
        [0.0, 0.0, z_center],
        [1/3, 1/3, z_center],
    ]
    species = [mat[0], mat[1]]

    return coords, species


def get_monochalcogenide_dimer_coords(a, c, dMX, dMM, mat):
    """Get fractional coordinates for a post-transition-metal monochalcogenide
    (GaSe, InSe): a 4-atom X-M-M-X layer with a vertically-bonded metal-metal
    dimer, NOT the 2-atom flat honeycomb get_binary_monolayer_coords builds.

    mat = [metal, chalcogen]. dMX is the metal-chalcogen bond length, dMM is
    the metal-metal dimer bond length -- both confirmed against the real MP
    structure (mp-1943/mp-20485), see the GaSe/InSe entries in
    data/mp_material_overrides.json.
    """
    dMX_frac = dMX / c
    dMM_frac = dMM / c
    z_center = 0.5

    coords = [
        [0.0, 0.0, z_center - dMM_frac / 2],  # Metal, bottom
        [0.0, 0.0, z_center + dMM_frac / 2],  # Metal, top (dimer partner)
        [1/3, 1/3, z_center - dMM_frac / 2 - dMX_frac],  # Chalcogen, bottom
        [1/3, 1/3, z_center + dMM_frac / 2 + dMX_frac],  # Chalcogen, top
    ]
    species = [mat[0], mat[0], mat[1], mat[1]]

    return coords, species


def get_single_element_coords(a, c, mat, dMX=0.0):
    """Get fractional coordinates for single element 2D materials"""
    z_center = 0.5

    if mat[0] == 'C':  # graphene
        # Honeycomb structure
        coords = [
            [0, 0, z_center],
            [1/3, 1/3, z_center],
        ]
        species = ['C', 'C']
    elif mat[0] == 'P':  # phosphorene
        # Simplified orthorhombic
        coords = [
            [0, 0, z_center],
            [0.5, 0.5, z_center],
        ]
        species = ['P', 'P']
    else:
        # Other single-element honeycombs (Si, Ge, Sn, ...) are only dynamically
        # stable when buckled -- a flat (dMX=0) seed can't spontaneously break
        # the mirror symmetry during relaxation, so the offset must be baked
        # into the template itself.
        dMX_frac = dMX / c
        coords = [
            [0, 0, z_center + dMX_frac / 2],
            [1/3, 1/3, z_center - dMX_frac / 2],
        ]
        species = [mat[0], mat[0]]

    return coords, species


def get_ternary_coords(a, c, dMX, mat):
    """Get fractional coordinates for ternary compounds"""
    dMX_frac = dMX / c
    z_center = 0.5
    
    if len(mat) == 3:  # MoSSe, WSSe
        coords = [
            [0, 0, z_center],  # Metal
            [1/3, 1/3, z_center - dMX_frac],  # First chalcogen
            [1/3, 1/3, z_center + dMX_frac],  # Second chalcogen
        ]
        species = mat
    else:  # MoWSe2, MoWTe2 - alloy in metal layer
        coords = [
            [0, 0, z_center],  # Mo
            [1/3, 1/3, z_center],  # W
            [1/6, 1/3, z_center - dMX_frac],  # Chalcogen 1
            [1/2, 5/6, z_center - dMX_frac],  # Chalcogen 2
            [1/3, 1/3, z_center + dMX_frac],  # Chalcogen 3
            [5/6, 1/3, z_center + dMX_frac],  # Chalcogen 4
        ]
        species = [mat[0], mat[1], mat[2], mat[2], mat[2], mat[2]]
    
    return coords, species


def get_bilayer_coords(a, c, dMX, mat, stacking='AA', dz=None):
    """
    Get fractional coordinates for bilayer structure.
    Parameters:
    a, c, dMX: lattice parameters
    mat: [metal, chalcogen] elements
    stacking: 'AA' or 'AB'
    dz: interlayer spacing in Angstroms (default: DZ_BILAYER)
    """
    if dz is None:
        dz = DZ_BILAYER
    # Get monolayer coordinates
    coords1, species1 = get_tmd_monolayer_coords(a, c, dMX, mat)
    
    # Convert interlayer spacing to fractional
    dz_frac = dz / c
    
    if stacking == 'AA':
        # AA stacking: same x,y positions
        coords2 = [[c[0], c[1], c[2] + dz_frac] for c in coords1]
    else:  # AB stacking
        # AB stacking: shift by (1/3, 1/3, dz)
        coords2 = [[(c[0] + 1/3) % 1.0, (c[1] + 1/3) % 1.0, c[2] + dz_frac] for c in coords1]
    
    # Combine coordinates
    coords = coords1 + coords2
    species = species1 + species1
    
    return coords, species


def get_heterostructure_coords(
    mat1,
    mat2,
    c,
    dz=None,
    a1=None,
    a2=None,
    dMX1=None,
    dMX2=None,
):
    """Get fractional coordinates for heterostructure"""
    if dz is None:
        dz = DZ_BILAYER
    if a1 is None or dMX1 is None:
        a1, _, dMX1 = LATTICE_PARAMS.get(mat1, (3.2, 20.0, 1.58))
    if a2 is None or dMX2 is None:
        a2, _, dMX2 = LATTICE_PARAMS.get(mat2, (3.2, 20.0, 1.58))
    
    # Use average lattice parameter
    a = (a1 + a2) / 2
    
    # Parse elements
    if mat1 in ['MoS2', 'MoSe2', 'MoTe2', 'WS2', 'WSe2', 'WTe2', 'NbS2', 'NbSe2', 'NbTe2',
                'TaS2', 'TaSe2', 'TaTe2', 'ReS2', 'ReSe2', 'SnS2', 'SnSe2', 'TiS2', 'TiSe2',
                'ZrS2', 'ZrSe2', 'HfS2', 'HfSe2']:
        if mat1.startswith('Mo'):
            elem1 = ['Mo']
        elif mat1.startswith('W'):
            elem1 = ['W']
        elif mat1.startswith('Nb'):
            elem1 = ['Nb']
        elif mat1.startswith('Ta'):
            elem1 = ['Ta']
        elif mat1.startswith('Re'):
            elem1 = ['Re']
        elif mat1.startswith('Sn'):
            elem1 = ['Sn']
        elif mat1.startswith('Ti'):
            elem1 = ['Ti']
        elif mat1.startswith('Zr'):
            elem1 = ['Zr']
        elif mat1.startswith('Hf'):
            elem1 = ['Hf']
        else:
            elem1 = ['Mo']
        
        if 'S2' in mat1:
            elem1.append('S')
        elif 'Se2' in mat1:
            elem1.append('Se')
        elif 'Te2' in mat1:
            elem1.append('Te')
        else:
            elem1.append('S')
    else:
        elem1 = ['X', 'Y']
    
    if mat2 in ['MoS2', 'MoSe2', 'MoTe2', 'WS2', 'WSe2', 'WTe2', 'NbS2', 'NbSe2', 'NbTe2',
                'TaS2', 'TaSe2', 'TaTe2', 'ReS2', 'ReSe2', 'SnS2', 'SnSe2', 'TiS2', 'TiSe2',
                'ZrS2', 'ZrSe2', 'HfS2', 'HfSe2']:
        if mat2.startswith('Mo'):
            elem2 = ['Mo']
        elif mat2.startswith('W'):
            elem2 = ['W']
        elif mat2.startswith('Nb'):
            elem2 = ['Nb']
        elif mat2.startswith('Ta'):
            elem2 = ['Ta']
        elif mat2.startswith('Re'):
            elem2 = ['Re']
        elif mat2.startswith('Sn'):
            elem2 = ['Sn']
        elif mat2.startswith('Ti'):
            elem2 = ['Ti']
        elif mat2.startswith('Zr'):
            elem2 = ['Zr']
        elif mat2.startswith('Hf'):
            elem2 = ['Hf']
        else:
            elem2 = ['Mo']
        
        if 'S2' in mat2:
            elem2.append('S')
        elif 'Se2' in mat2:
            elem2.append('Se')
        elif 'Te2' in mat2:
            elem2.append('Te')
        else:
            elem2.append('S')
    elif mat2 == 'graphene':
        elem2 = ['C']
    elif mat2 == 'BN':
        elem2 = ['B', 'N']
    else:
        elem2 = ['X', 'Y']
    
    # Get coordinates for each layer
    if len(elem1) == 2:
        coords1, species1 = get_tmd_monolayer_coords(a, c, dMX1, elem1)
    else:
        coords1, species1 = get_single_element_coords(a, c, elem1)
    
    if len(elem2) == 2:
        coords2, species2 = get_tmd_monolayer_coords(a, c, dMX2, elem2)
    elif len(elem2) == 1:
        coords2, species2 = get_single_element_coords(a, c, elem2)
    else:
        coords2, species2 = get_binary_monolayer_coords(a, c, dMX2, elem2)
    
    # Stack layers
    dz_frac = dz / c
    max_z = max(c[2] for c in coords1)
    coords2 = [[c[0], c[1], max_z + dz_frac] for c in coords2]
    
    coords = coords1 + coords2
    species = species1 + species2
    
    return coords, species, a


def poscar_writer(a, c, dMX, mat, coords, species, filename):
    """
    Create a POSCAR file using pymatgen.
    Parameters:
    a (float): In-plane lattice constant in Angstroms
    c (float): c-axis length in Angstroms (for vacuum)
    dMX (float): Vertical spacing between metal and chalcogen in Angstroms (not used if coords provided)
    mat: Material name (for reference)
    coords (list): Fractional coordinates
    species (list): Element species
    filename (str): Output filename
    """
    # Create the hexagonal lattice
    lattice_matrix = np.array([
        [a, 0, 0],
        [a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c]
    ])
    
    lattice = Lattice(lattice_matrix)
    
    # Create the structure
    structure = Structure(lattice, species, coords)
    
    # Create the Poscar object
    poscar = Poscar(structure)
    
    # Write the POSCAR file
    poscar.write_file(filename)


def _default_lattice_params(material_name):
    """Local fallback (a, c, dz, dMX) when MP is disabled or unavailable."""
    a, c, dmx = LATTICE_PARAMS.get(material_name, (3.2, 20.0, 1.58))
    return a, c, DZ_BILAYER, dmx


def resolve_lattice_params(
    material_name,
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
):
    """
    Resolve (a, c, dz, dMX) from MP or local LATTICE_PARAMS.

    Returns (params_tuple, metadata, source) where source is 'mp' or 'fallback'.
    """
    a, c, dz, dmx = _default_lattice_params(material_name)
    meta = {}
    if use_mp and MP_HELPERS_AVAILABLE:
        params, meta = get_material_lattice_params(
            material_name,
            api_key=mp_api_key,
            use_cache=True,
            refresh_cache=mp_refresh,
            verbose=mp_verbose,
        )
        if params is not None:
            return params, meta, "mp"
        if mp_verbose:
            print(f"[MP] fallback for {material_name}: {meta.get('reason')}")
    return (a, c, dz, dmx), meta, "fallback"


def _build_single_material_template(material_name, a, c, dMX):
    """Build template coordinates/species for a single monolayer material."""

    if material_name in ['graphene', 'phosphorene', 'silicene', 'germanene', 'stanene']:
        if material_name == 'graphene':
            mat = ['C']
        elif material_name == 'phosphorene':
            mat = ['P']
        elif material_name == 'silicene':
            mat = ['Si']
        elif material_name == 'germanene':
            mat = ['Ge']
        else:
            mat = ['Sn']
        coords, species = get_single_element_coords(a, c, mat, dMX)

    elif material_name in ['BN', 'GaN']:
        mat = ['B', 'N'] if material_name == 'BN' else ['Ga', 'N']
        coords, species = get_binary_monolayer_coords(a, c, dMX, mat)

    elif material_name in ['InSe', 'GaSe']:
        mat = ['In', 'Se'] if material_name == 'InSe' else ['Ga', 'Se']
        overrides = _load_overrides()
        dMM = overrides.get(material_name, {}).get('dMM')
        if dMM is None:
            raise ValueError(
                f"{material_name} needs a 'dMM' (metal-metal dimer bond length) entry "
                f"in data/mp_material_overrides.json -- it cannot use the flat "
                f"binary-honeycomb template."
            )
        coords, species = get_monochalcogenide_dimer_coords(a, c, dMX, dMM, mat)

    elif material_name in ['MoSSe', 'WSSe', 'MoWSe2', 'MoWTe2']:
        if material_name == 'MoSSe':
            mat = ['Mo', 'S', 'Se']
        elif material_name == 'WSSe':
            mat = ['W', 'S', 'Se']
        elif material_name == 'MoWSe2':
            mat = ['Mo', 'W', 'Se']
        else:
            mat = ['Mo', 'W', 'Te']
        coords, species = get_ternary_coords(a, c, dMX, mat)

    else:
        # Default to TMD structure
        if material_name.startswith('Mo'):
            mat = ['Mo']
        elif material_name.startswith('W'):
            mat = ['W']
        elif material_name.startswith('Nb'):
            mat = ['Nb']
        elif material_name.startswith('Ta'):
            mat = ['Ta']
        elif material_name.startswith('Re'):
            mat = ['Re']
        elif material_name.startswith('Sn'):
            mat = ['Sn']
        elif material_name.startswith('Ti'):
            mat = ['Ti']
        elif material_name.startswith('Zr'):
            mat = ['Zr']
        elif material_name.startswith('Hf'):
            mat = ['Hf']
        else:
            mat = ['Mo']

        if 'S2' in material_name:
            mat.append('S')
        elif 'Se2' in material_name:
            mat.append('Se')
        elif 'Te2' in material_name:
            mat.append('Te')
        else:
            mat.append('S')

        coords, species = get_tmd_monolayer_coords(a, c, dMX, mat)

    return a, c, dMX, coords, species


def generate_poscar(
    material_name,
    output_dir,
    filename=None,
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
    strict_validation=False,
    a_override=None,
):
    """
    Generate POSCAR file for a given material
    
    Parameters:
    -----------
    material_name : str
        Name of the material
    output_dir : Path or str
        Directory where POSCAR will be written
    filename : str, optional
        Name of the output file. If None, uses "POSCAR_<material_name>"
    """
    # Parse material name
    output_dir = Path(output_dir)

    if '_bilayer_' in material_name:
        base_material = material_name.split('_bilayer_')[0]
        stacking = material_name.split('_bilayer_')[1]
        (a, c, dz, dMX), _, _ = resolve_lattice_params(
            base_material, use_mp, mp_api_key, mp_refresh, mp_verbose
        )
        
        # Parse elements
        if base_material.startswith('Mo'):
            mat = ['Mo']
        elif base_material.startswith('W'):
            mat = ['W']
        else:
            mat = ['Mo']
        
        if 'S2' in base_material:
            mat.append('S')
        elif 'Se2' in base_material:
            mat.append('Se')
        elif 'Te2' in base_material:
            mat.append('Te')
        else:
            mat.append('S')
        
        coords, species = get_bilayer_coords(a, c, dMX, mat, stacking, dz=dz)
        
    elif '_twist_' in material_name:
        base_material = material_name.split('_twist_')[0]
        angle_str = material_name.split('_twist_')[1]
        angle_deg = int(angle_str.replace('deg', ''))
        (a, c, dz, dMX), _, _ = resolve_lattice_params(
            base_material, use_mp, mp_api_key, mp_refresh, mp_verbose
        )
        
        # Parse elements
        if base_material.startswith('Mo'):
            mat = ['Mo']
        elif base_material.startswith('W'):
            mat = ['W']
        else:
            mat = ['Mo']
        
        if 'S2' in base_material:
            mat.append('S')
        elif 'Se2' in base_material:
            mat.append('Se')
        elif 'Te2' in base_material:
            mat.append('Te')
        else:
            mat.append('S')
        
        # For twisted, use AB stacking (0deg) or approximate
        stacking = 'AB' if angle_deg == 0 else 'AB'
        coords, species = get_bilayer_coords(a, c, dMX, mat, stacking, dz=dz)
        
    elif '_' in material_name and material_name.count('_') == 1 and material_name not in ['MoSSe', 'WSSe']:
        # Heterostructure
        mat1, mat2 = material_name.split('_')
        (a1, c, dz, dmx1), _, _ = resolve_lattice_params(
            mat1, use_mp, mp_api_key, mp_refresh, mp_verbose
        )
        (a2, _, _, dmx2), _, _ = resolve_lattice_params(
            mat2, use_mp, mp_api_key, mp_refresh, mp_verbose
        )
        coords, species, a = get_heterostructure_coords(
            mat1, mat2, c, dz=dz, a1=a1, a2=a2, dMX1=dmx1, dMX2=dmx2
        )
        dMX = (dmx1 + dmx2) / 2
        
    else:
        # Single material: MP lattice params + template geometry.
        output_path = output_dir / (filename if filename is not None else f"POSCAR_{material_name}")
        (a, c, dz, dMX), meta, source = resolve_lattice_params(
            material_name, use_mp, mp_api_key, mp_refresh, mp_verbose
        )
        if a_override is not None:
            a = float(a_override)
        a, c, dMX, coords, species = _build_single_material_template(
            material_name, a, c, dMX
        )
        poscar_writer(a, c, dMX, material_name, coords, species, str(output_path))
        mid = meta.get("material_id")
        src_label = f"{source}" + (f", material_id={mid}" if mid else "")
        print(
            f"Generated: {output_path} (source={src_label}, "
            f"a={a:.4f} c={c:.1f} dz={dz:.2f} dMX={dMX:.4f})"
        )
        return {
            "material": material_name,
            "source": source,
            "material_id": mid,
            "output_path": output_path,
        }
    
    # Non-single-material branch keeps legacy behavior.
    if filename is None:
        output_path = output_dir / f"POSCAR_{material_name}"
    else:
        output_path = output_dir / filename
    
    poscar_writer(a, c, dMX, material_name, coords, species, str(output_path))
    print(f"Generated: {output_path} (source=fallback)")
    return {
        "material": material_name,
        "source": "fallback",
        "output_path": output_path,
    }


def load_materials_list(materials_file=None):
    """Load materials from materials_list.txt"""
    if materials_file is None:
        materials_file = Path(__file__).parent.parent.parent / "common" / "materials_list.txt"
    else:
        materials_file = Path(materials_file)
    
    if not materials_file.exists():
        raise FileNotFoundError(f"Materials list not found: {materials_file}")
    
    materials = []
    with open(materials_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                materials.append(line)
    
    return materials


def generate_random_poscar(
    output_dir=None,
    materials_file=None,
    seed=None,
    filename="POSCAR",
    use_mp=True,
    mp_api_key=None,
    mp_refresh=False,
    mp_verbose=False,
    strict_validation=False,
):
    """
    Randomly select a material and generate a POSCAR file ready for relaxation.
    
    Parameters:
    -----------
    output_dir : str or Path, optional
        Directory where POSCAR will be written. Default: "input" folder in workflow directory.
    materials_file : str or Path, optional
        Path to materials list file. Default: common/materials_list.txt
    seed : int, optional
        Random seed for reproducibility.
    filename : str, optional
        Name of the POSCAR file (default: "POSCAR" for VASP)
    
    Returns:
    --------
    tuple : (material_name, output_path)
        The selected material name and the path where POSCAR was written.
    """
    import random
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Load materials list
    materials = load_materials_list(materials_file)
    
    if not materials:
        raise ValueError("No materials found in materials list!")
    
    # Randomly select a material
    selected_material = random.choice(materials)
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "input"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate POSCAR
    output_path = output_dir / filename
    
    try:
        generate_poscar(
            selected_material,
            output_dir,
            filename=filename,
            use_mp=use_mp,
            mp_api_key=mp_api_key,
            mp_refresh=mp_refresh,
            mp_verbose=mp_verbose,
            strict_validation=strict_validation,
        )
        return selected_material, output_path
    except Exception as e:
        raise RuntimeError(f"Error generating POSCAR for {selected_material}: {e}")


def main():
    """Main function to process materials list"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate POSCAR files for 2D materials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all POSCAR files (batch mode)
  python3 generate_monolayer_poscar.py
  
  # Generate a random POSCAR for workflow
  python3 generate_monolayer_poscar.py --random
  
  # Generate random POSCAR in specific directory
  python3 generate_monolayer_poscar.py --random -o ./my_job
  
  # Use specific seed for reproducibility
  python3 generate_monolayer_poscar.py --random -s 42
        """
    )
    parser.add_argument(
        "-r", "--random",
        action="store_true",
        help="Randomly select one material and generate POSCAR (for workflow automation)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: 'input' for random mode, 'initial_structures' for batch mode)"
    )
    parser.add_argument(
        "-m", "--materials-file",
        type=str,
        default=None,
        help="Path to materials list file (default: ../common/materials_list.txt)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (only used with --random)"
    )
    parser.add_argument(
        "-f", "--filename",
        type=str,
        default=None,
        help="Output filename (default: 'POSCAR' for random mode, 'POSCAR_<material>' for batch mode)"
    )
    add_mp_args(parser)

    args = parser.parse_args()
    
    # Random mode
    if args.random:
        try:
            material, poscar_path = generate_random_poscar(
                output_dir=args.output_dir,
                materials_file=args.materials_file,
                seed=args.seed,
                filename=args.filename or "POSCAR",
                use_mp=not args.no_mp,
                mp_api_key=args.mp_api_key,
                mp_refresh=args.mp_refresh,
                mp_verbose=args.mp_verbose,
                strict_validation=args.strict_validation,
            )
            print(f"Selected material: {material}")
            print(f"POSCAR written to: {poscar_path}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return
    
    # Batch mode (original behavior)
    materials_file = args.materials_file or (Path(__file__).parent.parent / "common" / "materials_list.txt")
    
    if not Path(materials_file).exists():
        print(f"Error: {materials_file} not found!")
        sys.exit(1)
    
    output_dir = args.output_dir or (Path(__file__).parent / "initial_structures")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    materials = load_materials_list(materials_file)
    
    print(f"Found {len(materials)} materials to process")
    print(f"Output directory: {output_dir}\n")
    
    # Generate POSCAR files
    for material in materials:
        try:
            generate_poscar(
                material,
                output_dir,
                use_mp=not args.no_mp,
                mp_api_key=args.mp_api_key,
                mp_refresh=args.mp_refresh,
                mp_verbose=args.mp_verbose,
                strict_validation=args.strict_validation,
            )
        except Exception as e:
            print(f"Error generating POSCAR for {material}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nDone! Generated {len(materials)} POSCAR files in {output_dir}")


if __name__ == "__main__":
    import sys
    main()
