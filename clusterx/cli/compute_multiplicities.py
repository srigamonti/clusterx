# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DerivativeStructures

commands = ["compute_multiplicities"]


@plac.pos(
    "dss_filepath",
    help="Path to serialized DerivativeStructures object.",
)
@plac.flg(
    "skip_existing",
    abbrev="s",
    help="Do not recompute if the multiplicity column already exists.",
)
@plac.flg(
    "quiet",
    abbrev="q",
    help="Disable the multiplicity-computation progress bar.",
)
def compute_multiplicities(
    dss_filepath: str,
    skip_existing: bool = False,
    quiet: bool = False,
):
    """Compute DerivativeStructures multiplicities and persist them to the same file."""
    cmd_message("head")

    dss = DerivativeStructures.from_file(dss_filepath)
    dss.add_multiplicity(recompute=not skip_existing, show_progress=not quiet)
    dss.serialize(filepath=dss_filepath)

    print(
        "Serialized multiplicities for "
        f"{len(dss.configurations)} configurations to '{dss_filepath}'."
    )
