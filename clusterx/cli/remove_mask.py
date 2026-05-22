# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DSGenerator

commands = ["remove_mask"]


@plac.pos(
    "dss_filepath",
    help="Path to serialized DSGenerator object.",
)
@plac.pos(
    "mask_name",
    help="Name of the mask to remove.",
)
@plac.flg(
    "force",
    abbrev="f",
    help="Remove the mask without asking for confirmation.",
)
def remove_mask(
    dss_filepath: str,
    mask_name: str,
    force: bool = False,
):
    """Remove a mask from a serialized DSGenerator and persist the change."""
    cmd_message("head")

    dss = DSGenerator.from_file(dss_filepath)
    mask_names = sorted(dss.masks.keys())

    if mask_name not in dss.masks:
        raise ValueError(f"Mask '{mask_name}' not found. Available masks: {mask_names}")

    n_config_ids = len(dss.masks[mask_name])
    print(
        f"Warning: this will permanently remove mask '{mask_name}' "
        f"({n_config_ids} config_ids) from '{dss_filepath}'."
    )

    if not force:
        confirmation = input("Type 'yes' to confirm mask removal: ").strip().lower()
        if confirmation != "yes":
            print("Mask removal cancelled.")
            return

    del dss.masks[mask_name]
    dss.serialize(filepath=dss_filepath)
    print(f"Removed mask '{mask_name}' from '{dss_filepath}'.")
