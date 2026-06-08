"""
Structural family classification for bilayer stacking.

Each material belongs to one of three families:
  - TMD (tri-layer sandwiches)
  - binary_honeycomb (BN, GaN)
  - single_element_honeycomb (graphene)

Pair type determines which two stacking configurations are allowed.
"""

TMD_MATERIALS = frozenset([
    "MoS2", "MoSe2", "MoTe2",
    "WS2", "WSe2", "WTe2",
    "NbS2", "NbSe2",
    "TaS2", "TaSe2",
])

BINARY_HONEYCOMB = frozenset(["BN", "GaN"])

SINGLE_ELEMENT_HONEYCOMB = frozenset(["graphene"])

HONEYCOMB_MATERIALS = BINARY_HONEYCOMB | SINGLE_ELEMENT_HONEYCOMB

# Longest suffix first for parsing bilayer names
STACKING_SUFFIXES = (
    "AA_prime",
    "TM_TX",
    "TM_H",
    "3R",
    "2H",
    "AA",
    "AB",
)

PAIR_STACKINGS = {
    "tmd_tmd": ("3R", "2H"),
    "honeycomb_honeycomb": ("AA_prime", "AB"),
    "tmd_honeycomb": ("TM_TX", "TM_H"),
}


def get_structural_family(material: str) -> str:
    """Return structural family for a material name."""
    if material in TMD_MATERIALS:
        return "tmd"
    if material in BINARY_HONEYCOMB:
        return "binary_honeycomb"
    if material in SINGLE_ELEMENT_HONEYCOMB:
        return "single_element_honeycomb"
    raise ValueError(f"Unknown material for structural family: {material!r}")


def is_honeycomb(material: str) -> bool:
    return material in HONEYCOMB_MATERIALS


def is_tmd(material: str) -> bool:
    return material in TMD_MATERIALS


def get_pair_type(mat1: str, mat2: str) -> str:
    """Return pair type: tmd_tmd, honeycomb_honeycomb, or tmd_honeycomb."""
    f1 = get_structural_family(mat1)
    f2 = get_structural_family(mat2)

    if f1 == "tmd" and f2 == "tmd":
        return "tmd_tmd"
    if is_honeycomb(mat1) and is_honeycomb(mat2):
        return "honeycomb_honeycomb"
    if (f1 == "tmd" and is_honeycomb(mat2)) or (is_honeycomb(mat1) and f2 == "tmd"):
        return "tmd_honeycomb"
    raise ValueError(f"Incompatible pair: {mat1!r} / {mat2!r}")


def get_allowed_stackings(mat1: str, mat2: str) -> tuple[str, ...]:
    """Return exactly two allowed stacking labels for a material pair."""
    pair_type = get_pair_type(mat1, mat2)
    return PAIR_STACKINGS[pair_type]


def format_stacking_label(mat1: str, mat2: str, stacking: str) -> str:
    """Map AA_prime to AA for pure graphene/graphene homobilayers."""
    if stacking == "AA_prime" and mat1 == mat2 == "graphene":
        return "AA"
    return stacking


def normalize_stacking_for_validation(mat1: str, mat2: str, stacking: str) -> str:
    """Map display label AA back to canonical AA_prime for validation."""
    if stacking == "AA" and mat1 == mat2 == "graphene":
        return "AA_prime"
    return stacking


def validate_stacking(mat1: str, mat2: str, stacking: str) -> None:
    """Raise ValueError if stacking is not allowed for this pair."""
    canonical = normalize_stacking_for_validation(mat1, mat2, stacking)
    allowed = get_allowed_stackings(mat1, mat2)
    if canonical not in allowed:
        raise ValueError(
            f"Stacking {stacking!r} not allowed for {mat1}/{mat2}. "
            f"Allowed: {', '.join(format_stacking_label(mat1, mat2, s) for s in allowed)}"
        )
