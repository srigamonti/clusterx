# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.structures_set import StructuresSet

commands = ["find_lowest"]


@plac.annotations(
    sset_filepath=("Path to the serialized StructuresSet object.", "option", None, str),
    property_name=(
        "Name of the property used to determine the lowest structure(s).",
        "option",
        None,
        str,
    ),
    sset_higher_filepath=(
        "Path to store StructuresSet with non-lowest structures.",
        "option",
        "ssh",
        str,
    ),
    sset_lowest_filepath=(
        "Path to store StructuresSet with lowest structures.",
        "option",
        "ssl",
        str,
    ),
)
def find_lowest(
    sset_filepath: str = "sset.json",
    property_name: Optional[str] = None,
    sset_higher_filepath: Optional[str] = None,
    sset_lowest_filepath: Optional[str] = None,
):
    """Find structure with lowest property value at each concentration"""
    cmd_message("head")

    sset = StructuresSet(filepath=sset_filepath)
    conc = np.array(sset.get_concentrations(site_type=0, sigma=1))
    pvals = sset.get_property_values(property_name=property_name)
    sidxs = np.lexsort((conc, pvals))
    conc_unique_sorted = np.unique(conc)

    idxs_lowest = []
    for i, c in enumerate(conc_unique_sorted):
        for sidx in sidxs:
            if conc[sidx] == c:
                idxs_lowest.append(sidx)
                break

    idxs_higher = []
    for sidx in sidxs:
        if sidx not in idxs_lowest:
            idxs_higher.append(sidx)

    sset_lowest = sset.get_subset(idxs_lowest, transfer_properties=True)
    if sset_lowest_filepath is not None:
        sset_lowest.serialize(sset_lowest_filepath, overwrite=True)

    sset_higher = sset.get_subset(idxs_higher, transfer_properties=True)
    if sset_higher_filepath is not None:
        sset_higher.serialize(sset_higher_filepath, overwrite=True)

    return sset_lowest, sset_higher
