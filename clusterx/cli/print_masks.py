# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DerivativeStructures

commands = ["print_masks"]


@plac.annotations(
    dss_filepath=(
        "Path to serialized DerivativeStructures object.",
        "positional",
        None,
        str,
    ),
)
def print_masks(dss_filepath: Optional[str] = None):
    """Print all mask names and their contents from a serialized DerivativeStructures."""
    cmd_message("head")

    if dss_filepath is None:
        raise ValueError("dss_filepath must be provided.")

    dss = DerivativeStructures.from_file(dss_filepath)
    mask_names = sorted(dss.masks.keys())

    if not mask_names:
        print("No masks found.")
        return

    max_name_width = 24
    name_column_width = min(max(len(name) for name in mask_names), max_name_width)

    def _format_mask_name(mask_name: str) -> str:
        if len(mask_name) <= name_column_width:
            return mask_name.ljust(name_column_width)
        if name_column_width <= 3:
            return mask_name[:name_column_width]
        return mask_name[: name_column_width - 3] + "..."

    for mask_name in mask_names:
        formatted_name = _format_mask_name(mask_name)
        print(f"{formatted_name} : {list(dss.masks[mask_name])}")
