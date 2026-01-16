# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message, remove_trailing_extension
from clusterx.structures_set import StructuresSet

commands = ["compute_concentrations"]


@plac.opt(
    "basepath",
    abbrev="bp",
    help="""
        Path to a folder containing integer-named subfolders (see :func:`clusterx.utils.list_integer_named_folders`).
    """,
)
@plac.opt(
    "site_type",
    abbrev="st",
    help="""
        Path to store correlations matrix. Extension is removed. File is 
        saved in formats txt and npz
    """,
)
@plac.opt(
    "sigma",
    abbrev="si",
    help="Name of the property",
)
@plac.opt(
    "conc_filepath",
    abbrev="cof",
    help="""
        Overwrite the file 'ccalc_filepath' to remember the cluster orbits of the 
        supercells in contained in 'sset_filepath'. This leads to faster correlation 
        evaluations on future calculator use.
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
