#!/usr/bin/env python3
"""Load relaxed monolayer structures from monolayer_examples for bilayer building."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from pymatgen.core import Structure
except ImportError as exc:
    raise ImportError(
        "pymatgen is required for relaxed monolayer loading. "
        "Install with: pip install pymatgen"
    ) from exc


def default_monolayer_examples_dir() -> Path:
    """Workflow-level ``monolayer_examples`` directory."""
    return Path(__file__).resolve().parent.parent.parent / "monolayer_examples"


def relaxed_monolayer_path(
    material_name: str,
    monolayer_examples_dir: Optional[Path | str] = None,
) -> Path:
    """Return path to CONTCAR (or POSCAR fallback) for a relaxed monolayer."""
    base = Path(monolayer_examples_dir) if monolayer_examples_dir else default_monolayer_examples_dir()
    material_dir = base / material_name
    contcar = material_dir / "CONTCAR"
    if contcar.exists():
        return contcar
    poscar = material_dir / "POSCAR"
    if poscar.exists():
        return poscar
    raise FileNotFoundError(
        f"No relaxed monolayer structure for {material_name!r}. "
        f"Expected {contcar} or {poscar}. "
        "Run monolayer relaxation first (create_monolayer_example.py)."
    )


def load_relaxed_monolayer(
    material_name: str,
    monolayer_examples_dir: Optional[Path | str] = None,
) -> Structure:
    """Load a relaxed monolayer as a pymatgen ``Structure``."""
    path = relaxed_monolayer_path(material_name, monolayer_examples_dir)
    return Structure.from_file(str(path))


def hexagonal_lattice_a(structure: Structure) -> float:
    """In-plane hex lattice parameter ``a`` (length of first lattice vector)."""
    return float(np.linalg.norm(structure.lattice.matrix[0]))


def structure_frac_coords_species(
    structure: Structure,
) -> Tuple[List[List[float]], List[str]]:
    """Fractional coordinates and species strings from a structure."""
    coords = [list(fc) for fc in structure.frac_coords]
    species = [str(s) for s in structure.species]
    return coords, species


def layer_in_vacuum_cell(
    structure: Structure,
    vacuum: float,
    in_plane_a: Optional[float] = None,
) -> Tuple[List[List[float]], List[str], float]:
    """
    Fractional (Direct) coordinates for a layer in the bilayer supercell.

    Reads the pristine fractional coordinates from the relaxed monolayer (CONTCAR
    Direct positions via ``structure.frac_coords``). Coordinates remain in fractional
    space through stacking and POSCAR output.

    Parameters
    ----------
    vacuum : float
        Supercell height (Å). Used by the caller when building ``A_super``; must match
        the monolayer cell ``c`` if fractional ``z`` from the CONTCAR is reused as-is.
    in_plane_a : float, optional
        Shared supercell in-plane lattice parameter ``a``. Default: from the structure.
    """
    del vacuum  # caller sets c on A_super; z fractions assume monolayer c == vacuum
    a_tgt = in_plane_a if in_plane_a is not None else hexagonal_lattice_a(structure)
    coords, species = structure_frac_coords_species(structure)
    return coords, species, a_tgt
