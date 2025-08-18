# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional
import plac
from clusterx.parent_lattice import ParentLattice
from clusterx.cli.config_utils import cmd_message


commands = ["print_plat_info"]


@plac.annotations(
    plat_filepath=(
        "Path to serialized ParentLattice object.",
        "positional",
        None,
        str,
    ),
)
def print_plat_info(plat_filepath: Optional[str] = None):
    """Print ParentLattice info"""
    cmd_message("head")

    if plat_filepath is not None:
        plat = ParentLattice(filepath=plat_filepath)
    else:
        raise ValueError("Either sset_filepath or plat_filepath must be provided.")

    plat.print_sublattice_types()
