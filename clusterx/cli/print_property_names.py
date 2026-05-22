# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.structures_set import StructuresSet

commands = ["print_property_names"]


@plac.annotations(
    sset_filepath=(
        "Path to serialized StructuresSet object.",
        "positional",
        None,
        str,
    ),
)
def print_property_names(sset_filepath: Optional[str] = None):
    """Print property names available in a StructuresSet serialized object."""
    cmd_message("head")

    if sset_filepath is not None:
        sset = StructuresSet(filepath=sset_filepath)
    else:
        raise ValueError("sset_filepath must be provided.")

    print(sset.get_property_names())
