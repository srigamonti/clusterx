# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import os
from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.utils import list_integer_named_folders

commands = ["build_structures_set"]


@plac.pos(
    "basepath",
    help="""
        Path to a folder containing integer-named subfolders (see :func:`clusterx.utils.list_integer_named_folders`).
    """,
)
@plac.pos(
    "plat_filepath",
    help="""
        Path to a folder containing a serialized parent lattice object. All structures to be included must derive from this parent lattice.
    """,
)
@plac.opt(
    "structure_fname",
    abbrev="sf",
    help="Name of the structure file inside every folder (default: structure.json)",
    type=str,
)
@plac.opt(
    "sset_filepath",
    abbrev="ssfp",
    help="""
        file path for writing the StructuresSet database file.
    """,
)
def build_structures_set(
    basepath: str,
    plat_filepath: str,
    sset_filepath: Optional[str] = "sset.db",
    structure_fname: str = "structure.json",
):
    """Build structres set from a folder containing integer-named subfolders, each with a CELL's structure file."""
    cmd_message("head")

    parent_lattice = ParentLattice(filepath=os.path.join(basepath, plat_filepath))
    sset = StructuresSet(parent_lattice=parent_lattice)

    flist = list_integer_named_folders(basepath)

    # If flist contains integers, join with basepath; otherwise assume strings
    if len(flist) == 0:
        print(f"No integer-named folders found in {basepath}")

    for i, f in enumerate(flist):
        folder_name = str(f)
        folder = os.path.join(basepath, folder_name)
        struct_path = os.path.join(folder, structure_fname)

        print(f"Processing folder {i + 1}/{len(flist)}: {folder_name}")

        try:
            s = Structure.from_file(struct_path)
        except Exception as exc:
            print(f"Failed to load structure from {struct_path}: {exc}")
            continue

        try:
            splat = s.get_parent_lattice()
        except Exception as exc:
            print(
                f"Failed to obtain parent lattice from structure {struct_path}: {exc}"
            )
            continue

        # Only add structures that derive from the provided parent lattice
        if parent_lattice is not None and splat != parent_lattice:
            print(
                f"Skipping {struct_path}: parent lattice does not match {plat_filepath}"
            )
            continue

        sset.add_structure(s, folder=folder)
    # Serialize the created structures set
    if sset_filepath is None:
        sset_filepath = os.path.join(basepath, "sset.db")
    sset.serialize(filepath=sset_filepath, overwrite=True)
    print(f"StructuresSet serialized to {sset_filepath}")
