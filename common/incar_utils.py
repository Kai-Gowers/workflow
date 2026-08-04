#!/usr/bin/env python3
"""Shared INCAR customization used by both relaxation and phonopy staticpoint prep."""


def customize_incar(template_path, output_path, name, suffix="relaxation"):
    """
    Copy INCAR template and customize it for the specific material/bilayer.

    Parameters:
    -----------
    template_path : Path
        Path to INCAR template file
    output_path : Path
        Path where customized INCAR will be written
    name : str
        Material/bilayer name to use in SYSTEM line
    suffix : str
        Trailing word in the SYSTEM line -- "relaxation" for relaxation INCARs,
        "phonon" for phonopy staticpoint INCARs.
    """
    with open(template_path, 'r') as f:
        lines = f.readlines()

    # Customize the INCAR
    customized_lines = []
    system_found = False

    for line in lines:
        # Update SYSTEM line with name (replace first occurrence, skip duplicates)
        stripped = line.strip()
        if stripped.startswith('SYSTEM'):
            if not system_found:
                # First SYSTEM line: use name
                customized_lines.append(f"SYSTEM = {name} {suffix}\n")
                system_found = True
            # Skip duplicate SYSTEM lines
        else:
            customized_lines.append(line)

    # If no SYSTEM line found, add one at the beginning
    if not system_found:
        customized_lines.insert(0, f"SYSTEM = {name} {suffix}\n")

    # Write customized INCAR
    with open(output_path, 'w') as f:
        f.writelines(customized_lines)
