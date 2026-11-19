# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.structures_set import StructuresSet

commands = ["print_property_values"]


@plac.pos(
    "sset_filepath",
    help="Path to serialized StructuresSet object.",
)
@plac.pos(
    "property_names",
    help="List of property names (comma-separated).",
)
@plac.opt(
    "show_index",
    abbrev="i",
    help="Show row index/number (default: True)",
    type=bool,
)
def print_property_values(
    sset_filepath: str,
    property_names: str,
    show_index: bool = True,
):
    """Print property values for structures in a StructuresSet."""
    cmd_message("head")

    sset = StructuresSet(filepath=sset_filepath)

    # Parse comma-separated property names
    props_list = [p.strip() for p in property_names.split(",")]

    # Retrieve property values
    properties = {}
    for prop_name in props_list:
        try:
            properties[prop_name] = sset.get_property_values(prop_name)
        except Exception as exc:
            print(f"Warning: Failed to get property '{prop_name}': {exc}")
            properties[prop_name] = [None] * len(sset)

    # Print header
    header = []
    if show_index:
        header.append("Index")
    header.extend(props_list)
    print("\t".join(header))

    # Print rows
    for i in range(len(sset)):
        row = []
        if show_index:
            row.append(str(i))
        for prop_name in props_list:
            val = properties[prop_name][i]
            row.append(str(val) if val is not None else "")
        print("\t".join(row))