# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DerivativeStructures

commands = ["convert_sset_to_dss"]


@plac.pos(
    "sset_filepath",
    help="Path to a serialized StructuresSet object.",
)
@plac.pos(
    "dss_filepath",
    help="Output path for the serialized DerivativeStructures pickle.",
)
@plac.opt(
    "quick_parse",
    abbrev="q",
    help="Use StructuresSet quick_parse mode while reading the input file.",
    type=bool,
)
def convert_sset_to_dss(
    sset_filepath: str,
    dss_filepath: str,
    quick_parse: bool = False,
):
    """Convert a serialized StructuresSet file to a serialized DerivativeStructures."""
    cmd_message("head")

    dss = DerivativeStructures.from_structures_set_file(
        sset_filepath,
        quick_parse=quick_parse,
    )
    dss.serialize(filepath=dss_filepath)
    print(f"Info(convert_sset_to_dss): wrote DerivativeStructures to '{dss_filepath}'.")
