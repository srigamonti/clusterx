# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional
import plac
import numpy as np
from clusterx.correlations import CorrelationsCalculator
from clusterx.structures_set import StructuresSet
from clusterx.cli.config_utils import cmd_message
from clusterx.cli.config_utils import remove_trailing_extension

commands = ["compute_comat"]


@plac.opt(
    "ccalc_filepath",
    abbrev="ccf",
    help="Path to the pickle file of a serialized CorrelationsCalculator object.",
)
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
    "comat_filepath",
    abbrev="cmf",
    help="""
        Path to store correlations matrix. Extension is removed. File is
        saved in formats txt and npz.
    """,
)
@plac.opt("property_name", abbrev="pn", help="Name of the property")
@plac.flg(
    "update_ccalc",
    abbrev="u",
    help="""
        Overwrite  file 'ccalc_filepath' to keep the cluster orbits of the 
        supercells contained in 'sset_filepath'. This leads to faster correlation 
        evaluations on future calculator use.
    """,
)
def compute_comat(
    ccalc_filepath: str = "ccalc.pickle",
    sset_filepath: str = "sset.json",
    comat_filepath: str = "comat",
    property_name: Optional[str] = None,
    update_ccalc: bool = False,
):
    """Compute matrix of correlations"""
    cmd_message("head")

    ccalc = CorrelationsCalculator(filepath=ccalc_filepath)
    sset = StructuresSet(filepath=sset_filepath)

    filepath = remove_trailing_extension(comat_filepath)

    comat = ccalc.get_correlation_matrix(sset, f"{filepath}.txt")

    arrays_dict = {"comat": comat}

    if property_name is not None:
        pvals = sset.get_property_values(property_name=property_name)

        arrays_dict[f"property_{property_name}"] = (
            pvals  # dynamically pass keyword to savez from string property_name
        )

    np.savez(f"{filepath}.npz", **arrays_dict)

    if update_ccalc:
        ccalc.serialize(filepath=ccalc_filepath)
