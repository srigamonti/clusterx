# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import os
from typing import Optional, Sequence, Union, cast

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
        Either:
        - path to a folder containing integer-named subfolders
                    (see :func:`clusterx.utils.list_integer_named_folders`), or
        - a list/tuple of explicit structure file paths to add directly.

        When a list/tuple is provided, folder discovery is skipped and the provided
        file paths are used as-is.
    """,
)
@plac.pos(
    "plat_filepath",
    help="""
        Path to a serialized parent lattice object.

        If ``basepath`` is a folder path, this path is resolved relative to it.
        If ``basepath`` is a list/tuple of structure file paths, this path is used
        directly.
        All structures to be included must derive from this parent lattice.
    """,
)
@plac.opt(
    "structure_fname",
    abbrev="sf",
    help=(
        "Name of the structure file inside every discovered folder "
        "(default: structure.json). Ignored when basepath is a list/tuple "
        "of explicit structure file paths."
    ),
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
    basepath: Union[str, Sequence[str]],
    plat_filepath: str,
    sset_filepath: Optional[str] = "sset.db",
    structure_fname: str = "structure.json",
):
    """Build a StructuresSet from either discovered folders or explicit structure files.

    Supported modes:

    1. ``basepath`` is a folder path: discover integer-named subfolders and load
       ``structure_fname`` from each one.
    2. ``basepath`` is a list/tuple of structure file paths: skip discovery and
       load each provided file path directly (``structure_fname`` is ignored).
    """
    cmd_message("head")

    using_explicit_structure_paths = isinstance(basepath, (list, tuple))
    output_basepath = "."

    if using_explicit_structure_paths:
        parent_lattice = ParentLattice(filepath=plat_filepath)
        structure_paths = [str(path) for path in basepath]
    else:
        basepath_str = cast(str, basepath)
        parent_lattice = ParentLattice(filepath=os.path.join(basepath_str, plat_filepath))
        flist = list_integer_named_folders(basepath_str)
        structure_paths = [os.path.join(basepath_str, str(folder_name), structure_fname) for folder_name in flist]
        output_basepath = basepath_str

    sset = StructuresSet(parent_lattice=parent_lattice)

    if len(structure_paths) == 0:
        if using_explicit_structure_paths:
            print("No structure file paths were provided")
        else:
            print(f"No integer-named folders found in {basepath}")

    for i, struct_path in enumerate(structure_paths):
        folder = os.path.dirname(struct_path)

        print(f"Processing structure {i + 1}/{len(structure_paths)}: {struct_path}")

        try:
            s = Structure.from_file(struct_path)
        except Exception as exc:
            print(f"Failed to load structure from {struct_path}: {exc}")
            continue

        try:
            splat = s.get_parent_lattice()
        except Exception as exc:
            print(f"Failed to obtain parent lattice from structure {struct_path}: {exc}")
            continue

        # Only add structures that derive from the provided parent lattice
        if parent_lattice is not None and splat != parent_lattice:
            print(f"Skipping {struct_path}: parent lattice does not match {plat_filepath}")
            continue

        sset.add_structure(s, folder=folder)
    # Serialize the created structures set
    if sset_filepath is None:
        sset_filepath = os.path.join(output_basepath, "sset.db")
    sset.serialize(filepath=sset_filepath, overwrite=True)
    print(f"StructuresSet serialized to {sset_filepath}")
