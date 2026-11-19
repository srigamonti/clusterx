# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DerivativeStructures

commands = ["convert_dss_to_sset"]


def load_dss_as_structures_set(dss_filepath: str, mask_name: Optional[str] = None):
    """Load a serialized DerivativeStructures and convert it to a StructuresSet."""
    dss = DerivativeStructures.from_file(dss_filepath)
    return dss.to_structures_set(mask_name=mask_name)


@plac.pos(
    "dss_filepath",
    help="Path to a serialized DerivativeStructures object.",
)
@plac.pos(
    "sset_filepath",
    help="Output path for the serialized StructuresSet object.",
)
@plac.opt(
    "mask_name",
    abbrev="m",
    help="Optional DerivativeStructures mask name used to select configurations.",
    type=str,
)
@plac.opt(
    "ase_db_type",
    abbrev="t",
    help="Optional ASE database type passed to StructuresSet.serialize.",
    type=str,
)
def convert_dss_to_sset(
    dss_filepath: str,
    sset_filepath: str,
    mask_name: Optional[str] = None,
    ase_db_type: Optional[str] = None,
):
    """Convert a serialized DerivativeStructures file to a serialized StructuresSet."""
    cmd_message("head")

    sset = load_dss_as_structures_set(dss_filepath, mask_name=mask_name)
    sset.serialize(
        filepath=sset_filepath,
        ase_db_type=ase_db_type,
        overwrite=True,
        rm_vac=False,
    )
    print(f"Info(convert_dss_to_sset): wrote StructuresSet to '{sset_filepath}'.")
