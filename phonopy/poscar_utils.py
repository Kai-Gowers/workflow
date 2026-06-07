#!/usr/bin/env python3
"""Utilities for preparing POSCAR files for phonopy + VASP workflows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class PoscarData:
    comment: str
    scale: float
    lattice: List[List[float]]
    symbols: List[str]
    positions: List[List[float]]
    coord_type: str = "Direct"
    selective_dynamics: Optional[List[List[bool]]] = None


def _argsort_stable(keys: Sequence[int]) -> List[int]:
    return sorted(range(len(keys)), key=keys.__getitem__)


def sort_by_species(
    symbols: Sequence[str],
    positions: Sequence[Sequence[float]],
    selective_dynamics: Optional[Sequence[Sequence[bool]]] = None,
) -> Tuple[List[str], List[List[float]], Optional[List[List[bool]]], List[int]]:
    """
    Group atoms by species using phonopy's stable sort convention.

    Species blocks appear in order of first occurrence in the input list,
    matching phonopy.interface.vasp.sort_positions_by_symbols.
    """
    reduced_symbols = list(dict.fromkeys(symbols))
    sort_keys = [reduced_symbols.index(symbol) for symbol in symbols]
    perm = _argsort_stable(sort_keys)

    sorted_symbols = [symbols[i] for i in perm]
    sorted_positions = [list(positions[i]) for i in perm]
    sorted_sd = None
    if selective_dynamics is not None:
        sorted_sd = [list(selective_dynamics[i]) for i in perm]

    return sorted_symbols, sorted_positions, sorted_sd, perm


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _parse_species_and_counts(line_symbols: str, line_counts: str) -> Tuple[List[str], List[int]]:
    symbol_tokens = line_symbols.split()
    count_tokens = line_counts.split()

    if len(symbol_tokens) == len(count_tokens):
        species = symbol_tokens
        counts = [int(c) for c in count_tokens]
        return species, counts

    if len(count_tokens) == 1 and len(symbol_tokens) > 1:
        total = int(count_tokens[0])
        if total != len(symbol_tokens):
            raise ValueError(
                "Could not parse POSCAR species/count lines: "
                f"{line_symbols!r} / {line_counts!r}"
            )
        return symbol_tokens, [1] * total

    raise ValueError(
        "Could not parse POSCAR species/count lines: "
        f"{line_symbols!r} / {line_counts!r}"
    )


def _expand_symbols(species: Sequence[str], counts: Sequence[int]) -> List[str]:
    symbols: List[str] = []
    for symbol, count in zip(species, counts):
        symbols.extend([symbol] * count)
    return symbols


def read_poscar(path: Path | str) -> PoscarData:
    """Read a VASP POSCAR/CONTCAR file."""
    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR too short: {path}")

    comment = lines[0]
    scale = float(lines[1].split()[0])
    lattice = [
        [float(x) for x in lines[i].split()[:3]]
        for i in range(2, 5)
    ]

    species, counts = _parse_species_and_counts(lines[5], lines[6])
    symbols = _expand_symbols(species, counts)
    num_atoms = len(symbols)

    idx = 7
    selective_dynamics = None
    if lines[idx].strip().lower().startswith("s"):
        idx += 1

    coord_line = lines[idx].strip()
    coord_type = "Direct" if coord_line.lower().startswith("d") else "Cartesian"
    idx += 1

    positions: List[List[float]] = []
    for _ in range(num_atoms):
        parts = lines[idx].split()
        positions.append([float(x) for x in parts[:3]])
        idx += 1

    if idx < len(lines) and lines[idx].strip() == "":
        idx += 1

    if idx < len(lines):
        sd_rows: List[List[bool]] = []
        for _ in range(num_atoms):
            if idx >= len(lines):
                break
            parts = lines[idx].split()
            if len(parts) >= 3 and not all(_is_float(p) for p in parts[:3]):
                sd_rows.append([p.upper().startswith("T") for p in parts[:3]])
                idx += 1
            else:
                break
        if len(sd_rows) == num_atoms:
            selective_dynamics = sd_rows

    return PoscarData(
        comment=comment,
        scale=scale,
        lattice=lattice,
        symbols=symbols,
        positions=positions,
        coord_type=coord_type,
        selective_dynamics=selective_dynamics,
    )


def write_poscar(data: PoscarData, path: Path | str) -> None:
    """Write a grouped-species VASP POSCAR/CONTCAR file."""
    path = Path(path)

    reduced_symbols = list(dict.fromkeys(data.symbols))
    counts = [Counter(data.symbols)[symbol] for symbol in reduced_symbols]

    lines = [
        data.comment.rstrip("\n"),
        f"{data.scale:19.16f}",
    ]
    for vec in data.lattice:
        lines.append(f"{vec[0]:20.16f} {vec[1]:20.16f} {vec[2]:20.16f}")
    lines.append("   " + "   ".join(reduced_symbols))
    lines.append("     " + "     ".join(str(c) for c in counts))
    if data.selective_dynamics is not None:
        lines.append("Selective dynamics")
    lines.append(data.coord_type)
    for pos in data.positions:
        lines.append(f"{pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f}")
    if data.selective_dynamics is not None:
        lines.append("")
        for flags in data.selective_dynamics:
            lines.append(
                " ".join(" T" if flag else " F" for flag in flags)
            )

    path.write_text("\n".join(lines) + "\n")


def needs_species_grouping(symbols: Sequence[str]) -> bool:
    """Return True if any species block is split in the current atom ordering."""
    if not symbols:
        return False
    reduced_symbols = list(dict.fromkeys(symbols))
    sort_keys = [reduced_symbols.index(symbol) for symbol in symbols]
    return sort_keys != sorted(sort_keys)


def reorder_poscar_for_phonopy(input_path: Path | str, output_path: Path | str) -> bool:
    """
    Group POSCAR atoms by species for phonopy/VASP compatibility.

    Returns True if the file was reordered, False if it was already grouped.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = read_poscar(input_path)
    if not needs_species_grouping(data.symbols):
        if input_path.resolve() != output_path.resolve():
            output_path.write_text(input_path.read_text())
        return False

    sorted_symbols, sorted_positions, sorted_sd, _ = sort_by_species(
        data.symbols,
        data.positions,
        data.selective_dynamics,
    )
    grouped = PoscarData(
        comment=data.comment,
        scale=data.scale,
        lattice=data.lattice,
        symbols=sorted_symbols,
        positions=sorted_positions,
        coord_type=data.coord_type,
        selective_dynamics=sorted_sd,
    )
    write_poscar(grouped, output_path)
    return True
