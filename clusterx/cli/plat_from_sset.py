# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac
from clusterx.structures_set import StructuresSet
from clusterx.cli.config_utils import cmd_message

commands = ["plat_from_sset"]


@plac.pos(
    "sset_filepath",
    help="Path to a serialized structures set json file to get the parent lattice from it.",
)
@plac.opt(
    "plat_filepath",
    help="File path to store the parent lattice.",
)
def plat_from_sset(sset_filepath: str, plat_filepath: str = "sset.json"):
    """Build a pool of clusters"""
    cmd_message("head")

    sset = StructuresSet(filepath=sset_filepath)
    plat = sset.get_parent_lattice()

    plat.serialize(filepath=plat_filepath)
