#!/usr/bin/env python3
"""Validation helpers for MP-derived monolayer structures."""

from __future__ import annotations

from typing import List, Tuple

try:
    from pymatgen.core.composition import Composition
except ImportError:
    Composition = None


FORMULA_MAP = {
    "graphene": "C",
    "phosphorene": "P",
    "silicene": "Si",
    "germanene": "Ge",
    "stanene": "Sn",
    "MoS2": "MoS2",
    "MoSe2": "MoSe2",
    "MoTe2": "MoTe2",
    "WS2": "WS2",
    "WSe2": "WSe2",
    "WTe2": "WTe2",
    "NbS2": "NbS2",
    "NbSe2": "NbSe2",
    "NbTe2": "NbTe2",
    "TaS2": "TaS2",
    "TaSe2": "TaSe2",
    "TaTe2": "TaTe2",
    "ReS2": "ReS2",
    "ReSe2": "ReSe2",
    "SnS2": "SnS2",
    "SnSe2": "SnSe2",
    "TiS2": "TiS2",
    "TiSe2": "TiSe2",
    "ZrS2": "ZrS2",
    "ZrSe2": "ZrSe2",
    "HfS2": "HfS2",
    "HfSe2": "HfSe2",
    "BN": "BN",
    "GaN": "GaN",
    "InSe": "InSe",
    "GaSe": "GaSe",
    "MoSSe": "MoSSe",
    "WSSe": "WSSe",
    "MoWSe2": "MoWSe2",
    "MoWTe2": "MoWTe2",
}


def _species_counts_from_formula(formula: str) -> dict:
    if Composition is None:
        return {}
    comp = Composition(formula).get_el_amt_dict()
    return {k: int(v) if abs(v - int(v)) < 1e-8 else v for k, v in comp.items()}


def validate_structure(material_name: str, structure, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate basic stoichiometry/geometry for MP-derived structures.

    Returns (is_valid, messages). In non-strict mode callers may still fallback.
    """
    messages: List[str] = []
    ok = True

    if structure is None:
        return False, ["Structure is None"]

    # Basic atom count and lattice checks
    if len(structure.sites) < 2:
        ok = False
        messages.append("Too few atoms (<2 sites)")

    if structure.lattice.a < 2.0 or structure.lattice.b < 2.0:
        ok = False
        messages.append(
            f"In-plane lattice too small (a={structure.lattice.a:.3f}, b={structure.lattice.b:.3f})"
        )

    # Reasonable nearest-neighbor distance
    if len(structure.sites) >= 2:
        dmat = structure.distance_matrix
        min_nonzero = min(
            dmat[i][j]
            for i in range(len(dmat))
            for j in range(len(dmat))
            if i != j
        )
        if min_nonzero < 0.8:
            ok = False
            messages.append(f"Unphysical short bond detected (dmin={min_nonzero:.3f} Å)")

    # Layer spread (we expect a quasi-2D slab)
    z_vals = [site.coords[2] for site in structure.sites]
    z_span = max(z_vals) - min(z_vals) if z_vals else 0.0
    if z_span > 12.0:
        ok = False
        messages.append(f"Layer too thick/unwrapped (z span={z_span:.3f} Å)")

    # Stoichiometry check against expected formula
    formula = FORMULA_MAP.get(material_name)
    if formula and Composition is not None:
        expected = _species_counts_from_formula(formula)
        actual = _species_counts_from_formula(structure.composition.reduced_formula)
        if expected and actual and expected != actual:
            ok = False
            messages.append(
                f"Stoichiometry mismatch (expected {expected}, got {actual})"
            )

    if strict and not ok:
        messages.insert(0, "Strict validation failed")

    return ok, messages


def validate_bilayer_structure(structure, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate geometry for assembled bilayer structures (MP or template).

    Returns (is_valid, messages). Non-strict callers may still fallback.
    """
    messages: List[str] = []
    ok = True

    if structure is None:
        return False, ["Structure is None"]

    if len(structure.sites) < 4:
        ok = False
        messages.append("Too few atoms for bilayer (<4 sites)")

    if structure.lattice.a < 2.0 or structure.lattice.b < 2.0:
        ok = False
        messages.append(
            f"In-plane lattice too small (a={structure.lattice.a:.3f}, b={structure.lattice.b:.3f})"
        )

    if len(structure.sites) >= 2:
        dmat = structure.distance_matrix
        min_nonzero = min(
            dmat[i][j]
            for i in range(len(dmat))
            for j in range(len(dmat))
            if i != j
        )
        if min_nonzero < 0.8:
            ok = False
            messages.append(f"Unphysical short contact (dmin={min_nonzero:.3f} Å)")

    z_vals = [site.coords[2] for site in structure.sites]
    z_span = max(z_vals) - min(z_vals) if z_vals else 0.0
    if z_span > 25.0:
        ok = False
        messages.append(f"Bilayer z span too large ({z_span:.3f} Å)")

    if strict and not ok:
        messages.insert(0, "Strict bilayer validation failed")

    return ok, messages
