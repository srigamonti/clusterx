# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import pickle
from typing import Optional

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message, remove_trailing_extension
from clusterx.correlations import CorrelationsCalculator
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell

commands = ["compute_comat"]


@plac.annotations(
    ccalc_filepath=("Path to the pickle file of a serialized CorrelationsCalculator object.", "option", "ccf", str),
    sset_filepath=(
        "Path to a serialized StructuresSet object. Correlations of all structures are computed and saved as a 2D array.",
        "option",
        "ssf",
        str,
    ),
    dss_filepath=(
        "Path to a serialized DerivativeStructures object. Correlations of all structures are computed and saved as a 2D array.",
        "option",
        "dsf",
        str,
    ),
    comat_filepath=(
        "Path to store correlations matrix (extension removed, saved as .txt and .npz).",
        "option",
        "cmf",
        str,
    ),
    property_name=("Name of the property.", "option", "pn", str),
    update_ccalc=("Overwrite ccalc file to keep supercell cluster orbits for future reuse.", "flag", "u", bool),
)
def compute_comat(
    ccalc_filepath: str = "ccalc.pickle",
    sset_filepath: str = None,
    dss_filepath: str = None,
    comat_filepath: str = "comat",
    property_name: Optional[str] = None,
    update_ccalc: bool = False,
):
    """Compute matrix of correlations"""
    cmd_message("head")

    if sset_filepath is not None and dss_filepath is not None:
        raise ValueError("Only one of sset_filepath or dss_filepath should be provided, not both.")

    filepath = remove_trailing_extension(comat_filepath)

    ccalc = CorrelationsCalculator(filepath=ccalc_filepath)

    if sset_filepath is not None:
        sset = StructuresSet(filepath=sset_filepath)
        comat = ccalc.get_correlation_matrix(sset, f"{filepath}.txt")

    if dss_filepath is not None:
        with open(dss_filepath, "rb") as f:
            dss = pickle.load(f)

        sset = StructuresSet(parent_lattice=dss.plat)
        for row in dss.configurations.itertuples(index=False):
            sigma = row.sigma
            sc_shape = dss.get_scell_shape(row.shape_id)
            scell = SuperCell(dss.plat, sc_shape)
            sset.add_structure(Structure(scell, sigmas=sigma))

        comat = ccalc.get_correlation_matrix(sset, f"{filepath}.txt")

    arrays_dict = {"comat": comat}

    if property_name is not None:
        if sset_filepath is not None:
            pvals = sset.get_property_values(property_name=property_name)

        if dss_filepath is not None:
            pvals = dss.configurations[property_name].to_list()

        arrays_dict[f"property_{property_name}"] = pvals  # dynamically pass keyword to savez from string property_name

    if not filepath.endswith(".npz"):
        filepath += ".npz"

    np.savez(filepath, **arrays_dict)

    if update_ccalc:
        ccalc.serialize(filepath=ccalc_filepath)
