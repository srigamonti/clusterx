# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from os import PathLike
from typing import List, Optional

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message, remove_trailing_extension
from clusterx.cli.convert_dss_to_sset import load_dss_as_structures_set
from clusterx.correlations import CorrelationsCalculator
from clusterx.structures_set import StructuresSet

commands = ["compute_comat"]


def _coerce_filepath(filepath, argument_name):
    if isinstance(filepath, PathLike):
        filepath = str(filepath)
    if not isinstance(filepath, str):
        raise TypeError(f"{argument_name} must contain string filepaths.")
    if not filepath:
        raise ValueError(f"{argument_name} must not contain empty filepaths.")
    return filepath


def _coerce_filepaths(filepaths, argument_name):
    if isinstance(filepaths, (str, PathLike)):
        return [_coerce_filepath(filepaths, argument_name)]

    try:
        paths = list(filepaths)
    except TypeError as exc:
        raise TypeError(f"{argument_name} must be a list of filepaths.") from exc

    if len(paths) == 0:
        raise ValueError(f"{argument_name} must contain at least one filepath.")

    return [_coerce_filepath(path, argument_name) for path in paths]


def _select_source_filepaths(
    sset_filepath,
    sset_filepaths,
    dss_filepath,
    dss_filepaths,
):
    sources = {
        "sset_filepath": sset_filepath,
        "sset_filepaths": sset_filepaths,
        "dss_filepath": dss_filepath,
        "dss_filepaths": dss_filepaths,
    }
    provided = [name for name, value in sources.items() if value is not None]

    if len(provided) != 1:
        raise ValueError(
            "Exactly one of sset_filepath, sset_filepaths, dss_filepath, "
            "or dss_filepaths should be provided."
        )

    source_name = provided[0]
    source_value = sources[source_name]
    if source_name.endswith("_filepaths"):
        return source_name, _coerce_filepaths(source_value, source_name)

    return source_name, [_coerce_filepath(source_value, source_name)]


def _load_structures_sets(source_name, filepaths):
    if source_name.startswith("sset_"):
        return [StructuresSet.from_file(filepath) for filepath in filepaths]

    return [load_dss_as_structures_set(filepath) for filepath in filepaths]


def _write_correlation_matrix(filepath, comat):
    with open(filepath, "w+", encoding="utf-8") as f:
        for covec in comat:
            for co in covec:
                f.write("%2.12f\t" % co)
            f.write("\n")


@plac.annotations(
    ccalc_filepath=(
        "Path to the pickle file of a serialized CorrelationsCalculator object.",
        "option",
        "ccf",
        str,
    ),
    sset_filepath=(
        "Path to a serialized StructuresSet object. Correlations of all structures are computed and saved as a 2D array.",
        "option",
        "ssf",
        str,
    ),
    sset_filepaths=(
        "List of paths to serialized StructuresSet objects. Correlations are concatenated in the given order.",
        "option",
        "ssfs",
        list,
    ),
    dss_filepath=(
        "Path to a serialized DerivativeStructures object. Correlations of all structures are computed and saved as a 2D array.",
        "option",
        "dsf",
        str,
    ),
    dss_filepaths=(
        "List of paths to serialized DSGenerator objects. Correlations are concatenated in the given order.",
        "option",
        "dsfs",
        list,
    ),
    comat_filepath=(
        "Path to store correlations matrix (extension removed, saved as .txt and .npz).",
        "option",
        "cmf",
        str,
    ),
    property_name=("Name of the property.", "option", "pn", str),
    update_ccalc=(
        "Overwrite ccalc file to keep supercell cluster orbits for future reuse.",
        "flag",
        "u",
        bool,
    ),
)
def compute_comat(
    ccalc_filepath: str = "ccalc.pickle",
    sset_filepath: str = None,
    sset_filepaths: Optional[List[str]] = None,
    dss_filepath: str = None,
    dss_filepaths: Optional[List[str]] = None,
    comat_filepath: str = "comat",
    property_name: Optional[str] = None,
    update_ccalc: bool = False,
):
    """Compute matrix of correlations"""
    cmd_message("head")

    source_name, source_filepaths = _select_source_filepaths(
        sset_filepath,
        sset_filepaths,
        dss_filepath,
        dss_filepaths,
    )

    filepath = remove_trailing_extension(comat_filepath)

    ccalc = CorrelationsCalculator(filepath=ccalc_filepath)
    ssets = _load_structures_sets(source_name, source_filepaths)
    comat = np.vstack([ccalc.get_correlation_matrix(sset) for sset in ssets])
    _write_correlation_matrix(f"{filepath}.txt", comat)

    arrays_dict = {"comat": comat}

    if property_name is not None:
        pvals = np.concatenate(
            [
                np.asarray(sset.get_property_values(property_name=property_name))
                for sset in ssets
            ]
        )
        arrays_dict[f"property_{property_name}"] = (
            pvals  # dynamically pass keyword to savez from string property_name
        )

    if not filepath.endswith(".npz"):
        filepath += ".npz"

    np.savez(filepath, **arrays_dict)

    if update_ccalc:
        ccalc.serialize(filepath=ccalc_filepath)
