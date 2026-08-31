# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message, remove_trailing_extension
from clusterx.structures_set import StructuresSet

commands = ["compute_concentrations"]


@plac.opt(
    "sset_filepath",
    abbrev="ssf",
    help="""
        Path to a serialized StructuresSet object. All cluster correlations 
        of all structures in this object are computed and saved to a
        two dimensional array.
    """,
)
@plac.opt(
    "site_type",
    abbrev="st",
    help="""
        Site type as output by command print_plat_info
    """,
)
@plac.opt(
    "sigma",
    abbrev="si",
    help="Integer, substitutional species. 0 pristine. 1 first substitutional species. 2 second substitutional species. etc.",
)
@plac.opt(
    "conc_filepath",
    abbrev="cof",
    help="""
        Path to store the computed concentrations array. The file is saved in npz format.
    """,
)
def compute_concentrations(
    sset_filepath: str = "sset.json",
    site_type: int = 0,
    sigma: int = 1,
    conc_filepath: str = "conc.npz",
):
    """Compute concentrations"""
    cmd_message("head")

    sset = StructuresSet(filepath=sset_filepath)

    frconc = sset.get_concentrations(site_type, sigma)

    filepath = remove_trailing_extension(conc_filepath)

    np.savez(f"{filepath}.npz", conc=frconc)
