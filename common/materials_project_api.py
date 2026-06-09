#!/usr/bin/env python3
"""
Materials Project API integration for retrieving material properties.

Primary use-case: fetch lattice parameters (a, c, dz, dMX) from MP structures
in space group P6₃/mmc, then build POSCARs from template geometry.
"""

import json
import os
import sys
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

# Try to import mp-api + pymatgen
try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    print("Warning: mp-api not installed. Install with: pip install mp-api", file=sys.stderr)

try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.core.composition import Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    Structure = None
    Composition = None
    print("Warning: pymatgen not installed. Install with: pip install pymatgen", file=sys.stderr)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_FILE = DATA_DIR / "mp_structure_cache.json"
PARAMS_CACHE_FILE = DATA_DIR / "mp_lattice_params_cache.json"
OVERRIDES_FILE = DATA_DIR / "mp_material_overrides.json"
SYMMETRY_ELIGIBLE_FILE = DATA_DIR / "symmetry_eligible_materials.json"
TARGET_SPACEGROUP_SYMBOL = "P6₃/mmc"
TARGET_CRYSTAL_SYSTEM = "Hexagonal"

# Workflow defaults when MP does not define slab vacuum / interlayer spacing
DEFAULT_VACUUM_C = 20.0
DEFAULT_DZ = 6.5

# Bump when dMX extraction logic changes (invalidates mp_lattice_params_cache entries).
DMX_EXTRACTION_VERSION = 2

METAL_SYMBOLS = frozenset(
    {"Mo", "W", "Nb", "Ta", "Re", "Sn", "Ti", "Zr", "Hf", "Ga", "In", "B"}
)
CHALCOGEN_OR_ANION_SYMBOLS = frozenset({"S", "Se", "Te", "N", "P"})

# Shared mapping used by both lattice-only and structure lookup paths
MATERIAL_FORMULA_MAP = {
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


def get_api_key() -> Optional[str]:
    """
    Get Materials Project API key from environment variable.
    
    Returns:
    --------
    str or None : API key if found, None otherwise
    """
    return os.environ.get('MP_API_KEY') or os.environ.get('MATERIALS_PROJECT_API_KEY')


def get_material_formula(material_name: str) -> Optional[str]:
    """Map workflow material name to formula used for MP queries."""
    return MATERIAL_FORMULA_MAP.get(material_name)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _structure_to_dict(structure) -> dict:
    if hasattr(structure, "as_dict"):
        return structure.as_dict()
    raise ValueError("Structure object cannot be serialized")


def _structure_from_dict(payload: dict):
    if not PYMATGEN_AVAILABLE:
        return None
    return Structure.from_dict(payload)


def _load_cache() -> dict:
    return _load_json(CACHE_FILE, {})


def _save_cache(cache: dict) -> None:
    _save_json(CACHE_FILE, cache)


def _load_overrides() -> dict:
    data = _load_json(OVERRIDES_FILE, {})
    if not isinstance(data, dict):
        return {}
    mats = data.get("materials", {})
    if isinstance(mats, dict):
        return mats
    return data


def _score_doc(doc, target_formula: str) -> tuple:
    """Lower score is better."""
    eah = getattr(doc, "energy_above_hull", 999.0)
    if eah is None:
        eah = 999.0
    nsites = getattr(doc, "nsites", 9999)
    if nsites is None:
        nsites = 9999
    formula_pretty = getattr(doc, "formula_pretty", "") or ""
    exact_formula_penalty = 0 if formula_pretty == target_formula else 1
    stable_bonus = 0 if bool(getattr(doc, "is_stable", False)) else 1
    return (exact_formula_penalty, stable_bonus, float(eah), int(nsites))


def _normalize_spacegroup_symbol(symbol: Optional[str]) -> str:
    """Normalize space-group symbol variants to a comparable ASCII form."""
    if not symbol:
        return ""
    # Convert unicode subscripts used in e.g. P6₃/mmc to ascii
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    normalized = symbol.translate(subscript_map)
    normalized = normalized.replace(" ", "")
    # Strip any non essential punctuation differences
    normalized = re.sub(r"[^A-Za-z0-9/\-]", "", normalized)
    return normalized.lower()


def _get_doc_spacegroup_symbol(doc) -> Optional[str]:
    """Extract space-group symbol from an MP summary doc."""
    return _get_doc_symmetry(doc).get("spacegroup_symbol")


def _get_doc_symmetry(doc) -> dict:
    """Extract space-group and crystal-system fields from an MP summary doc."""
    symmetry = getattr(doc, "symmetry", None)
    if symmetry is None:
        return {}
    def _coerce_crystal_system(value):
        if value is None:
            return None
        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return str(enum_value)
        text = str(value)
        if "." in text:
            text = text.split(".")[-1]
        return text
    if isinstance(symmetry, dict):
        return {
            "spacegroup_symbol": symmetry.get("symbol"),
            "spacegroup_number": symmetry.get("number"),
            "crystal_system": _coerce_crystal_system(symmetry.get("crystal_system")),
        }
    return {
        "spacegroup_symbol": getattr(symmetry, "symbol", None),
        "spacegroup_number": getattr(symmetry, "number", None),
        "crystal_system": _coerce_crystal_system(getattr(symmetry, "crystal_system", None)),
    }


def _symmetry_result(
    material_name: str,
    *,
    source: str,
    spacegroup_symbol: Optional[str] = None,
    spacegroup_number: Optional[int] = None,
    crystal_system: Optional[str] = None,
    material_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> dict:
    """Build a normalized symmetry lookup result dict."""
    return {
        "material": material_name,
        "source": source,
        "spacegroup_symbol": spacegroup_symbol,
        "spacegroup_number": spacegroup_number,
        "crystal_system": crystal_system,
        "material_id": material_id,
        "reason": reason,
    }


def _symmetry_from_params_cache(material_name: str) -> Optional[dict]:
    """Return symmetry metadata from mp_lattice_params_cache when structure cache is absent."""
    entry = _load_params_cache().get(material_name)
    if not isinstance(entry, dict):
        return None
    sg = entry.get("spacegroup_symbol")
    if not sg:
        return None
    cs = None
    if _normalize_spacegroup_symbol(sg) == _normalize_spacegroup_symbol(TARGET_SPACEGROUP_SYMBOL):
        cs = TARGET_CRYSTAL_SYSTEM
    return _symmetry_result(
        material_name,
        source="params-cache",
        spacegroup_symbol=sg,
        crystal_system=cs,
        material_id=entry.get("material_id"),
    )


def get_material_symmetry(
    material_name: str,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Return MP symmetry metadata for a workflow material name.

    Cache-first via mp_structure_cache.json, then mp_lattice_params_cache.json;
    on miss queries MP for a P6₃/mmc entry.
    """
    if get_material_formula(material_name) is None:
        return _symmetry_result(material_name, source="missing", reason="unknown_material")

    overrides = _load_overrides()
    override = overrides.get(material_name)
    if isinstance(override, dict) and bool(override.get("force_template")):
        return _symmetry_result(material_name, source="missing", reason="forced_template_via_override")
    if override is None and material_name in overrides:
        return _symmetry_result(material_name, source="missing", reason="forced_template_via_override")

    cache = _load_cache() if use_cache else {}
    if use_cache and not refresh_cache and material_name in cache:
        entry = cache[material_name]
        sg = entry.get("spacegroup_symbol")
        cs = entry.get("crystal_system")
        if cs is None and _normalize_spacegroup_symbol(sg) == _normalize_spacegroup_symbol(
            TARGET_SPACEGROUP_SYMBOL
        ):
            cs = TARGET_CRYSTAL_SYSTEM
        return _symmetry_result(
            material_name,
            source="cache",
            spacegroup_symbol=sg,
            spacegroup_number=entry.get("spacegroup_number"),
            crystal_system=cs,
            material_id=entry.get("material_id"),
        )

    params_sym = _symmetry_from_params_cache(material_name)
    if params_sym is not None:
        return params_sym

    if not MP_API_AVAILABLE:
        return _symmetry_result(material_name, source="missing", reason="mp_api_not_available")

    if api_key is None:
        api_key = get_api_key()
    if api_key is None:
        return _symmetry_result(material_name, source="missing", reason="missing_api_key")

    force_mpid = override.get("material_id") if isinstance(override, dict) else None
    try:
        docs, _query_meta = _query_structure_docs(material_name, api_key, force_mpid=force_mpid)
        if not docs:
            return _symmetry_result(material_name, source="missing", reason="no_docs")

        doc = _select_best_doc(material_name, docs)
        if doc is None:
            return _symmetry_result(
                material_name,
                source="missing",
                reason=f"no_{TARGET_SPACEGROUP_SYMBOL.replace('/', '')}_entry",
            )

        sym = _get_doc_symmetry(doc)
        result = _symmetry_result(
            material_name,
            source="mp-api",
            spacegroup_symbol=sym.get("spacegroup_symbol"),
            spacegroup_number=sym.get("spacegroup_number"),
            crystal_system=sym.get("crystal_system"),
            material_id=str(getattr(doc, "material_id", "")) or None,
        )
        if verbose:
            print(
                f"[symmetry] {material_name}: sg={result['spacegroup_symbol']} "
                f"cs={result['crystal_system']} id={result['material_id']}"
            )
        return result
    except Exception as e:  # noqa: BLE001
        return _symmetry_result(material_name, source="missing", reason=f"query_error: {e}")


def is_symmetry_eligible(
    material_name: str,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> bool:
    """True if material has hexagonal crystal system and space group P6₃/mmc (MP metadata)."""
    info = get_material_symmetry(
        material_name,
        api_key=api_key,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )
    target_sg = _normalize_spacegroup_symbol(TARGET_SPACEGROUP_SYMBOL)
    sg_ok = _normalize_spacegroup_symbol(info.get("spacegroup_symbol")) == target_sg
    cs = (info.get("crystal_system") or "").strip().lower()
    cs_ok = cs == TARGET_CRYSTAL_SYSTEM.lower()
    return sg_ok and cs_ok


def load_symmetry_eligible_set() -> Optional[frozenset]:
    """
    Load eligible material names from data/symmetry_eligible_materials.json.

    Returns None if the manifest is missing (callers should not filter).
    """
    data = _load_json(SYMMETRY_ELIGIBLE_FILE, {})
    if not isinstance(data, dict):
        return None
    eligible = data.get("eligible_materials")
    if isinstance(eligible, list) and eligible:
        return frozenset(eligible)
    return None


def parse_bilayer_components(bilayer_name: str) -> list:
    """
    Parse workflow bilayer names into constituent material names.

    Examples: MoS2_bilayer_3R -> [MoS2]; graphene_MoS2_2H -> [graphene, MoS2]
    """
    known = sorted(MATERIAL_FORMULA_MAP.keys(), key=len, reverse=True)
    for suffix in ("_bilayer_3R", "_bilayer_2H", "_3R", "_2H"):
        if not bilayer_name.endswith(suffix):
            continue
        core = bilayer_name[: -len(suffix)]
        if suffix.startswith("_bilayer"):
            homo = core.replace("_bilayer", "")
            return [homo] if homo else []
        materials = []
        rest = core
        while rest:
            matched = None
            for mat in known:
                if rest == mat:
                    matched = mat
                    rest = ""
                    break
                if rest.startswith(mat + "_"):
                    matched = mat
                    rest = rest[len(mat) + 1 :]
                    break
            if matched is None:
                return []
            materials.append(matched)
        return materials
    return []


def filter_materials(
    materials: list,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> Tuple[list, list]:
    """
    Split materials into (eligible, excluded) for hexagonal P6₃/mmc batch processing.

    Preserves input order within each list.
    """
    eligible = []
    excluded = []
    for name in materials:
        if is_symmetry_eligible(
            name,
            api_key=api_key,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            verbose=verbose,
        ):
            eligible.append(name)
        else:
            excluded.append(name)
            if verbose:
                info = get_material_symmetry(
                    name, api_key=api_key, use_cache=use_cache, refresh_cache=refresh_cache
                )
                print(
                    f"[symmetry] exclude {name}: sg={info.get('spacegroup_symbol')} "
                    f"cs={info.get('crystal_system')} ({info.get('reason') or info.get('source')})"
                )
    return eligible, excluded


def _query_structure_docs(material_name: str, api_key: str, force_mpid: Optional[str] = None):
    formula = get_material_formula(material_name)
    if not formula:
        return [], {"reason": "unknown_material"}

    with MPRester(api_key) as mpr:
        fields = [
            "material_id",
            "formula_pretty",
            "energy_above_hull",
            "is_stable",
            "nsites",
            "symmetry",
            "structure",
        ]
        if force_mpid:
            docs = mpr.summary.search(material_ids=[force_mpid], fields=fields)
            return docs, {"formula": formula, "forced_material_id": force_mpid}

        docs = mpr.summary.search(formula=formula, fields=fields)
        return docs, {"formula": formula}


def _load_params_cache() -> dict:
    return _load_json(PARAMS_CACHE_FILE, {})


def _save_params_cache(cache: dict) -> None:
    _save_json(PARAMS_CACHE_FILE, cache)


def rebuild_lattice_params_cache(verbose: bool = False) -> int:
    """
    Recompute mp_lattice_params_cache from cached MP structures (no API calls).

    Returns the number of materials updated.
    """
    struct_cache = _load_cache()
    params_cache = _load_params_cache()
    updated = 0
    for material_name, entry in struct_cache.items():
        if not isinstance(entry, dict):
            continue
        structure = _structure_from_dict(entry.get("structure"))
        if structure is None:
            continue
        formula = get_material_formula(material_name) or entry.get("formula")
        params = extract_lattice_params_from_structure(structure, material_name)
        params_cache[material_name] = {
            "a": params[0],
            "c": params[1],
            "dz": params[2],
            "dMX": params[3],
            "dmx_extraction_version": DMX_EXTRACTION_VERSION,
            "material_id": entry.get("material_id"),
            "formula": formula,
            "spacegroup_symbol": entry.get("spacegroup_symbol"),
            "fetched_at": _now_iso(),
        }
        updated += 1
        if verbose:
            print(
                f"  {material_name}: a={params[0]:.4f} dMX={params[3]:.4f} "
                f"(id={entry.get('material_id')})"
            )
    _save_params_cache(params_cache)
    return updated


def _inplane_lattice_a(structure) -> float:
    """In-plane lattice constant from a pymatgen Structure (Å)."""
    lat = structure.lattice
    if abs(lat.gamma - 120.0) < 5.0 or abs(lat.gamma - 60.0) < 5.0:
        return float((lat.a + lat.b) / 2.0)
    return float(lat.a)


def _infer_metal_symbol(material_name: str, species: set) -> Optional[str]:
    """Pick metal element for TMD/binary materials from name and composition."""
    for sym in ("Mo", "W", "Nb", "Ta", "Re", "Sn", "Ti", "Zr", "Hf", "Ga", "In"):
        if material_name.startswith(sym) and sym in species:
            return sym
    for sym in METAL_SYMBOLS:
        if sym in species:
            return sym
    return None


def _z_distance_along_c(z1: float, z2: float, c_axis: Optional[float]) -> float:
    """Shortest distance along c between two Cartesian z coordinates (Å)."""
    d = abs(z1 - z2)
    if c_axis is not None and c_axis > 0:
        d = min(d, abs(c_axis - d))
    return float(d)


def _mean_min_cross_species_z_distance(
    z_list_a: list,
    z_list_b: list,
    c_axis: Optional[float],
) -> float:
    """
    For each site in z_list_b, minimum z-separation to any site in z_list_a; return the mean.
    Used for monolayer dMX so bulk cells do not mix inter-layer Mo–X distances.
    """
    if not z_list_a or not z_list_b:
        return 0.0
    distances = [
        min(_z_distance_along_c(z_b, z_a, c_axis) for z_a in z_list_a) for z_b in z_list_b
    ]
    return float(sum(distances) / len(distances))


def _extract_dmx(structure, material_name: str) -> float:
    """
    In-plane monolayer metal–chalcogen (or cation–anion) spacing in Å from MP coordinates.

    Uses the mean of each anion's nearest metal (or opposite species) along c, with periodic
    wrapping from the MP unit cell. This avoids inflating dMX when the MP entry is a bulk
    multilayer cell (e.g. MoTe2 mp-602).
    Returns 0.0 for single-element / flat monolayers.
    """
    if not PYMATGEN_AVAILABLE or structure is None:
        return 0.0

    by_symbol = {}
    for site in structure.sites:
        sym = site.specie.symbol
        by_symbol.setdefault(sym, []).append(float(site.coords[2]))

    species = set(by_symbol)
    if material_name in ("graphene", "phosphorene", "silicene", "germanene", "stanene", "BN"):
        return 0.0

    c_axis = float(structure.lattice.c) if structure.lattice is not None else None

    metal = _infer_metal_symbol(material_name, species)
    anions = [s for s in species if s in CHALCOGEN_OR_ANION_SYMBOLS]

    if metal and anions:
        metal_zs = by_symbol[metal]
        distances = []
        for an in anions:
            distances.append(
                _mean_min_cross_species_z_distance(metal_zs, by_symbol[an], c_axis)
            )
        if distances:
            return float(sum(distances) / len(distances))

    if len(species) == 2:
        s1, s2 = sorted(species)
        return _mean_min_cross_species_z_distance(
            by_symbol[s1], by_symbol[s2], c_axis
        )

    return 0.0


def extract_lattice_params_from_structure(
    structure,
    material_name: str,
    vacuum_c: float = DEFAULT_VACUUM_C,
    default_dz: float = DEFAULT_DZ,
) -> Tuple[float, float, float, float]:
    """
    Extract workflow lattice parameters from an MP structure.

    Returns (a, c, dz, dMX) in Angstroms.
    c and dz use workflow defaults (slab vacuum and bilayer spacing).
    """
    a = _inplane_lattice_a(structure)
    c = float(vacuum_c)
    dz = float(default_dz)
    dmx = _extract_dmx(structure, material_name)
    return a, c, dz, dmx


def get_material_lattice_params(
    material_name: str,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
    vacuum_c: float = DEFAULT_VACUUM_C,
    default_dz: float = DEFAULT_DZ,
) -> Tuple[Optional[Tuple[float, float, float, float]], dict]:
    """
    Fetch (a, c, dz, dMX) from MP for template-based POSCAR generation.

    Returns ((a, c, dz, dMX), metadata) or (None, metadata) on failure.
    """
    meta = {
        "material": material_name,
        "source": None,
        "reason": None,
        "material_id": None,
    }

    if not MP_API_AVAILABLE or not PYMATGEN_AVAILABLE:
        meta["reason"] = "mp_api_not_available" if not MP_API_AVAILABLE else "pymatgen_not_available"
        return None, meta

    formula = get_material_formula(material_name)
    if not formula:
        meta["reason"] = "unknown_material"
        return None, meta

    overrides = _load_overrides()
    override = overrides.get(material_name)
    if isinstance(override, dict) and bool(override.get("force_template")):
        meta["reason"] = "forced_template_via_override"
        return None, meta

    if isinstance(override, dict):
        if all(k in override for k in ("a", "c", "dz", "dMX")):
            params = (
                float(override["a"]),
                float(override["c"]),
                float(override["dz"]),
                float(override["dMX"]),
            )
            meta.update({"source": "override", "material_id": override.get("material_id")})
            return params, meta

    target_sg = _normalize_spacegroup_symbol(TARGET_SPACEGROUP_SYMBOL)
    params_cache = _load_params_cache() if use_cache else {}
    if use_cache and not refresh_cache and material_name in params_cache:
        entry = params_cache[material_name]
        dmx_version = entry.get("dmx_extraction_version", 1)
        if (
            dmx_version >= DMX_EXTRACTION_VERSION
            and _normalize_spacegroup_symbol(entry.get("spacegroup_symbol")) == target_sg
        ):
            params = (
                float(entry["a"]),
                float(entry["c"]),
                float(entry["dz"]),
                float(entry["dMX"]),
            )
            meta.update(
                {
                    "source": "params-cache",
                    "material_id": entry.get("material_id"),
                    "spacegroup_symbol": entry.get("spacegroup_symbol"),
                }
            )
            return params, meta

    structure, struct_meta = get_monolayer_structure(
        material_name,
        api_key=api_key,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )
    meta.update(struct_meta)
    if structure is None:
        return None, meta

    params = extract_lattice_params_from_structure(
        structure, material_name, vacuum_c=vacuum_c, default_dz=default_dz
    )
    meta.update(
        {
            "source": struct_meta.get("source", "mp-api"),
            "spacegroup_symbol": struct_meta.get("spacegroup_symbol"),
        }
    )
    if verbose:
        print(
            f"[MP] {material_name}: params a={params[0]:.4f} c={params[1]:.1f} "
            f"dz={params[2]:.2f} dMX={params[3]:.4f} Å "
            f"(id={meta.get('material_id')}, sg={meta.get('spacegroup_symbol')})"
        )

    if use_cache:
        params_cache[material_name] = {
            "a": params[0],
            "c": params[1],
            "dz": params[2],
            "dMX": params[3],
            "dmx_extraction_version": DMX_EXTRACTION_VERSION,
            "material_id": meta.get("material_id"),
            "formula": formula,
            "spacegroup_symbol": meta.get("spacegroup_symbol"),
            "fetched_at": _now_iso(),
        }
        _save_params_cache(params_cache)

    return params, meta


def _select_best_doc(material_name: str, docs):
    formula = get_material_formula(material_name)
    if not docs:
        return None
    target_sg = _normalize_spacegroup_symbol(TARGET_SPACEGROUP_SYMBOL)
    docs_with_target_sg = [
        d for d in docs
        if _normalize_spacegroup_symbol(_get_doc_spacegroup_symbol(d)) == target_sg
    ]
    ranked = sorted(docs_with_target_sg, key=lambda d: _score_doc(d, formula))
    for doc in ranked:
        if hasattr(doc, "structure") and doc.structure is not None:
            return doc
    return None


def normalize_structure_vacuum(structure, vacuum: float = 20.0):
    """
    Normalize structure to keep in-plane lattice and set c-axis vacuum.

    Assumes 2D layers are oriented normal to z.
    """
    if not PYMATGEN_AVAILABLE:
        return structure
    s = structure.copy()
    a_vec = s.lattice.matrix[0]
    b_vec = s.lattice.matrix[1]
    lattice = Lattice([a_vec, b_vec, [0.0, 0.0, float(vacuum)]])
    s = Structure(
        lattice=lattice,
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True,
        to_unit_cell=True,
    )
    z_vals = [site.frac_coords[2] for site in s.sites]
    z_center = 0.5 * (min(z_vals) + max(z_vals))
    s.translate_sites(range(len(s)), [0.0, 0.0, 0.5 - z_center], frac_coords=True, to_unit_cell=True)
    return s


def get_monolayer_structure(
    material_name: str,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> Tuple[Optional["Structure"], dict]:
    """
    Fetch full structure from MP for monolayer generation.

    Returns (structure, metadata). Structure is None on lookup failure.
    """
    meta = {
        "material": material_name,
        "source": None,
        "reason": None,
        "material_id": None,
    }

    if not MP_API_AVAILABLE:
        meta["reason"] = "mp_api_not_available"
        return None, meta
    if not PYMATGEN_AVAILABLE:
        meta["reason"] = "pymatgen_not_available"
        return None, meta

    formula = get_material_formula(material_name)
    if not formula:
        meta["reason"] = "unknown_material"
        return None, meta

    overrides = _load_overrides()
    override = overrides.get(material_name)
    force_template = isinstance(override, dict) and bool(override.get("force_template"))
    force_mpid = override.get("material_id") if isinstance(override, dict) else None
    if override is None and material_name in overrides:
        meta["reason"] = "forced_template_via_override"
        return None, meta
    if force_template:
        meta["reason"] = "forced_template_via_override"
        return None, meta

    cache = _load_cache() if use_cache else {}
    if use_cache and not refresh_cache and material_name in cache:
        entry = cache[material_name]
        payload = entry.get("structure")
        cached_sg = _normalize_spacegroup_symbol(entry.get("spacegroup_symbol"))
        target_sg = _normalize_spacegroup_symbol(TARGET_SPACEGROUP_SYMBOL)
        if cached_sg != target_sg:
            payload = None
            if verbose:
                print(
                    f"[MP] cache miss for {material_name}: "
                    f"spacegroup {entry.get('spacegroup_symbol')} != {TARGET_SPACEGROUP_SYMBOL}"
                )
        if payload:
            structure = _structure_from_dict(payload)
            if structure is not None:
                meta.update(
                    {
                        "source": "cache",
                        "material_id": entry.get("material_id"),
                        "formula": entry.get("formula", formula),
                    }
                )
                return structure, meta

    if api_key is None:
        api_key = get_api_key()
    if api_key is None:
        meta["reason"] = "missing_api_key"
        return None, meta

    try:
        docs, query_meta = _query_structure_docs(material_name, api_key, force_mpid=force_mpid)
        meta.update(query_meta)
        if not docs:
            meta["reason"] = "no_docs"
            return None, meta

        doc = _select_best_doc(material_name, docs)
        if doc is None:
            meta["reason"] = "no_structure_in_docs"
            return None, meta

        structure = doc.structure
        material_id = str(getattr(doc, "material_id", ""))
        sym = _get_doc_symmetry(doc)
        sg_symbol = sym.get("spacegroup_symbol")
        meta.update(
            {
                "source": "mp-api",
                "material_id": material_id,
                "formula_pretty": getattr(doc, "formula_pretty", None),
                "energy_above_hull": getattr(doc, "energy_above_hull", None),
                "is_stable": getattr(doc, "is_stable", None),
                "nsites": getattr(doc, "nsites", None),
                "spacegroup_symbol": sg_symbol,
                "spacegroup_number": sym.get("spacegroup_number"),
                "crystal_system": sym.get("crystal_system"),
            }
        )
        if verbose:
            print(
                f"[MP] {material_name}: selected {material_id} "
                f"(formula={meta.get('formula_pretty')}, sg={sg_symbol}, "
                f"e_hull={meta.get('energy_above_hull')})"
            )

        if use_cache:
            cache[material_name] = {
                "material_id": material_id,
                "formula": formula,
                "fetched_at": _now_iso(),
                "spacegroup_symbol": sg_symbol,
                "spacegroup_number": sym.get("spacegroup_number"),
                "crystal_system": sym.get("crystal_system"),
                "structure": _structure_to_dict(structure),
            }
            _save_cache(cache)

        return structure, meta
    except Exception as e:  # noqa: BLE001
        meta["reason"] = f"query_error: {e}"
        return None, meta


def write_structure_poscar(
    structure,
    output_path: Path,
    vacuum: float = 20.0,
) -> None:
    """Normalize structure vacuum and write POSCAR."""
    if not PYMATGEN_AVAILABLE:
        raise RuntimeError("pymatgen is required to write MP-derived structures")
    from pymatgen.io.vasp import Poscar

    normalized = normalize_structure_vacuum(structure, vacuum=vacuum)
    Poscar(normalized).write_file(str(output_path))


def get_layer_lattice_info(structure) -> dict:
    """
    Compact lattice/geometry summary for a normalized monolayer layer.

    Returns dict with keys: a, b, gamma, c, z_span, nsites.
    """
    if structure is None or not PYMATGEN_AVAILABLE:
        return {}
    lat = structure.lattice
    z_vals = [site.frac_coords[2] for site in structure.sites]
    z_span = max(z_vals) - min(z_vals) if z_vals else 0.0
    return {
        "a": float(lat.a),
        "b": float(lat.b),
        "gamma": float(lat.gamma),
        "c": float(lat.c),
        "z_span": float(z_span),
        "nsites": len(structure),
    }


def structure_to_layer_fractional(structure, vacuum: float = 20.0):
    """
    Normalize MP structure and return fractional coords + species for bilayer stacking.

    Returns (coords, species, lattice_info) or (None, None, {}) on failure.
    """
    if not PYMATGEN_AVAILABLE or structure is None:
        return None, None, {}
    normalized = normalize_structure_vacuum(structure, vacuum=vacuum)
    coords = [list(site.frac_coords) for site in normalized]
    species = [site.specie.symbol for site in normalized]
    return coords, species, get_layer_lattice_info(normalized)


def get_normalized_monolayer_layer(
    material_name: str,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
    vacuum: float = 20.0,
) -> Tuple[Optional[list], Optional[list], dict, dict]:
    """
    Fetch MP monolayer and return layer fractional data for bilayer assembly.

    Returns (coords, species, lattice_info, metadata). coords/species are None on failure.
    """
    structure, meta = get_monolayer_structure(
        material_name,
        api_key=api_key,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )
    if structure is None:
        return None, None, {}, meta
    coords, species, lattice_info = structure_to_layer_fractional(structure, vacuum=vacuum)
    if coords is None:
        meta["reason"] = meta.get("reason") or "normalization_failed"
        return None, None, {}, meta
    return coords, species, lattice_info, meta


def get_material_lattice_constant(material_name: str, api_key: Optional[str] = None) -> Optional[float]:
    """
    Get the in-plane lattice constant 'a' for a 2D material from The Materials Project.
    """
    if not MP_API_AVAILABLE:
        print(f"Error: mp-api not available. Cannot query lattice constant for {material_name}", file=sys.stderr)
        return None

    params, meta = get_material_lattice_params(material_name, api_key=api_key, use_cache=True, verbose=False)
    if params is not None:
        return float(params[0])
    if meta.get("reason"):
        print(f"Warning: Could not retrieve lattice constant for {material_name} ({meta.get('reason')})", file=sys.stderr)
    return None


@lru_cache(maxsize=128)
def get_cached_lattice_constant(material_name: str, api_key: Optional[str] = None) -> Optional[float]:
    """
    Get lattice constant with caching to avoid repeated API calls.
    
    Parameters:
    -----------
    material_name : str
        Material name
    api_key : str, optional
        API key (for cache key purposes)
    
    Returns:
    --------
    float or None : Lattice constant 'a' in Angstroms
    """
    return get_material_lattice_constant(material_name, api_key)


def are_lattice_constants_compatible(a1: float, a2: float, tolerance: float = 0.20) -> bool:
    """
    Check if two lattice constants are within the specified tolerance of each other.
    
    Parameters:
    -----------
    a1 : float
        First lattice constant (Angstroms)
    a2 : float
        Second lattice constant (Angstroms)
    tolerance : float
        Maximum relative difference (default: 0.20 = 20%)
    
    Returns:
    --------
    bool : True if |a1 - a2| / min(a1, a2) <= tolerance
    """
    if a1 <= 0 or a2 <= 0:
        return False
    
    min_a = min(a1, a2)
    max_a = max(a1, a2)
    relative_diff = (max_a - min_a) / min_a
    
    return relative_diff <= tolerance


def are_materials_compatible_by_lattice(
    mat1: str,
    mat2: str,
    api_key: Optional[str] = None,
    tolerance: float = 0.20,
    verbose: bool = False,
    lattice_constants: Optional[Dict[str, Optional[float]]] = None,
) -> bool:
    """
    Check if two materials have compatible lattice constants for bilayer stacking.
    
    Parameters:
    -----------
    mat1 : str
        First material name
    mat2 : str
        Second material name
    api_key : str, optional
        Materials Project API key
    tolerance : float
        Maximum relative difference in lattice constants (default: 0.20 = 20%)
    verbose : bool
        Print detailed information about compatibility check
    
    Returns:
    --------
    bool : True if materials have compatible lattice constants
    """
    # Get lattice constants (optional pre-fetched dict avoids redundant API calls)
    if lattice_constants is not None:
        a1 = lattice_constants.get(mat1)
        a2 = lattice_constants.get(mat2)
    else:
        a1 = get_cached_lattice_constant(mat1, api_key)
        a2 = get_cached_lattice_constant(mat2, api_key)
    
    if a1 is None:
        if verbose:
            print(f"  Could not retrieve lattice constant for {mat1}")
        return False
    
    if a2 is None:
        if verbose:
            print(f"  Could not retrieve lattice constant for {mat2}")
        return False
    
    compatible = are_lattice_constants_compatible(a1, a2, tolerance)
    
    if verbose:
        relative_diff = abs(a1 - a2) / min(a1, a2) * 100
        status = "compatible" if compatible else "incompatible"
        print(f"  {mat1}: a = {a1:.4f} Å, {mat2}: a = {a2:.4f} Å, diff = {relative_diff:.2f}% ({status})")
    
    return compatible


def get_all_lattice_constants(materials: list, api_key: Optional[str] = None, 
                              verbose: bool = False) -> Dict[str, Optional[float]]:
    """
    Get lattice constants for a list of materials.
    
    Parameters:
    -----------
    materials : list
        List of material names
    api_key : str, optional
        Materials Project API key
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict : Dictionary mapping material names to lattice constants (or None if not found)
    """
    results = {}
    
    for i, mat in enumerate(materials, 1):
        if verbose:
            print(f"Querying {mat} ({i}/{len(materials)})...", end=' ', flush=True)
        
        a = get_cached_lattice_constant(mat, api_key)
        results[mat] = a
        
        if verbose:
            if a is not None:
                print(f"a = {a:.4f} Å")
            else:
                print("not found")
    
    return results


if __name__ == "__main__":
    # Test the API connection
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Materials Project API connection")
    parser.add_argument("material", nargs="?", default="MoS2", help="Material name to query")
    parser.add_argument("--list", action="store_true", help="Test with materials list")
    parser.add_argument("--tolerance", type=float, default=0.20, help="Tolerance for compatibility (default: 0.20)")
    parser.add_argument("--structure", action="store_true", help="Fetch full structure and print selected MP id")
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh MP structure cache for --structure")
    parser.add_argument(
        "--rebuild-params-cache",
        action="store_true",
        help="Recompute mp_lattice_params_cache from cached structures (no API)",
    )
    
    args = parser.parse_args()
    
    if args.rebuild_params_cache:
        n = rebuild_lattice_params_cache(verbose=True)
        print(f"Rebuilt lattice params for {n} materials -> {PARAMS_CACHE_FILE}")
        sys.exit(0)
    
    api_key = get_api_key()
    if api_key is None:
        print("Error: No API key found. Set MP_API_KEY environment variable.")
        sys.exit(1)
    
    if args.list:
        # Load materials and test all
        from pathlib import Path
        materials_file = Path(__file__).parent / "materials_list.txt"
        if materials_file.exists():
            materials = []
            with open(materials_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        materials.append(line)
            
            print(f"Querying lattice constants for {len(materials)} materials...")
            results = get_all_lattice_constants(materials, api_key, verbose=True)
            
            print("\nSummary:")
            found = sum(1 for a in results.values() if a is not None)
            print(f"  Found: {found}/{len(materials)}")
            
            # Test compatibility for a few pairs
            print("\nTesting compatibility (tolerance = {:.0%}):".format(args.tolerance))
            test_pairs = [
                ('MoS2', 'WS2'),
                ('MoS2', 'graphene'),
                ('BN', 'graphene'),
            ]
            for mat1, mat2 in test_pairs:
                if mat1 in results and mat2 in results:
                    compatible = are_materials_compatible_by_lattice(
                        mat1, mat2, api_key, args.tolerance, verbose=True
                    )
        else:
            print(f"Materials list not found: {materials_file}")
    elif args.structure:
        print(f"Querying lattice parameters for {args.material}...")
        params, meta = get_material_lattice_params(
            args.material,
            api_key=api_key,
            use_cache=True,
            refresh_cache=args.refresh_cache,
            verbose=True,
        )
        if params is None:
            print(f"  Could not retrieve parameters ({meta.get('reason')})")
            sys.exit(1)
        a, c, dz, dmx = params
        print(
            f"  Selected: {meta.get('material_id')} "
            f"(source={meta.get('source')}, sg={meta.get('spacegroup_symbol')})"
        )
        print(f"  a={a:.4f} Å, c={c:.1f} Å, dz={dz:.2f} Å, dMX={dmx:.4f} Å")
    else:
        print(f"Querying lattice parameters for {args.material}...")
        params, meta = get_material_lattice_params(args.material, api_key=api_key, verbose=True)
        if params is not None:
            a, c, dz, dmx = params
            print(f"  a={a:.4f} Å, c={c:.1f} Å, dz={dz:.2f} Å, dMX={dmx:.4f} Å")
        else:
            print(f"  Could not retrieve parameters ({meta.get('reason')})")

