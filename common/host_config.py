#!/usr/bin/env python3
"""
Host-specific paths and settings for NERSC Perlmutter vs Andromeda.

Detection order:
1. TWIST_HOST env ("nersc" | "andromeda")
2. NERSC_HOST env (set on Perlmutter)
3. Existing potpaw directory (path probe)
4. Hostname heuristics

Override any single path with env vars (e.g. TWIST_POTCAR_PATH) without
changing the detected host.
"""

import os
import socket
from pathlib import Path

# potpaw_PBE.64 roots used by generate_potcar.py
POTCAR_PATHS = {
    "nersc": Path(
        "/global/common/software/nersc9/vasp/dependencies/pseudopotentials/"
        "PBE/potpaw_PBE.64"
    ),
    "andromeda": Path("/projects/twist2d/modules/vasp/potpaw_PBE.64"),
}

_VALID_HOSTS = frozenset(POTCAR_PATHS)


def detect_host():
    """Return 'nersc' or 'andromeda'."""
    explicit = os.environ.get("TWIST_HOST", "").strip().lower()
    if explicit in _VALID_HOSTS:
        return explicit

    nersc_host = os.environ.get("NERSC_HOST", "").strip().lower()
    if nersc_host:
        return "nersc"

    # Prefer a path that actually exists on this machine.
    existing = [name for name, path in POTCAR_PATHS.items() if path.is_dir()]
    if len(existing) == 1:
        return existing[0]
    if "nersc" in existing and Path("/global/common/software/nersc9").is_dir():
        return "nersc"
    if "andromeda" in existing:
        return "andromeda"

    hostname = socket.gethostname().lower()
    if "perlmutter" in hostname or hostname.startswith("login") and Path("/pscratch").is_dir():
        return "nersc"
    if "andromeda" in hostname:
        return "andromeda"

    # Last resort: NERSC-style filesystems are a strong signal.
    if Path("/pscratch").is_dir() and Path("/global/common/software").is_dir():
        return "nersc"
    if Path("/projects/twist2d").is_dir():
        return "andromeda"

    raise RuntimeError(
        "Could not detect HPC host (nersc vs andromeda). "
        "Set TWIST_HOST=nersc or TWIST_HOST=andromeda, "
        "or set TWIST_POTCAR_PATH to your potpaw_PBE.64 directory."
    )


def get_potcar_path():
    """
    Return the potpaw_PBE.64 directory for the current host.

    Override with TWIST_POTCAR_PATH or VASP_POTCAR_PATH if set.
    """
    override = os.environ.get("TWIST_POTCAR_PATH") or os.environ.get("VASP_POTCAR_PATH")
    if override:
        return Path(override)

    path = POTCAR_PATHS[detect_host()]
    if not path.is_dir():
        raise FileNotFoundError(
            f"POTCAR directory not found for host '{detect_host()}': {path}\n"
            "Set TWIST_POTCAR_PATH to the correct potpaw_PBE.64 directory, "
            "or set TWIST_HOST explicitly."
        )
    return path
