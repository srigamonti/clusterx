# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DerivativeStructures

commands = ["print_mask_names"]


@plac.annotations(
    dss_filepath=(
        "Path to serialized DerivativeStructures object.",
        "positional",
        None,
        str,
    ),
)
def print_mask_names(dss_filepath: Optional[str] = None):
    """Print mask names available in a serialized DerivativeStructures."""
    cmd_message("head")

    if dss_filepath is None:
        raise ValueError("dss_filepath must be provided.")

    dss = DerivativeStructures.from_file(dss_filepath)
    print(sorted(dss.masks.keys()))
