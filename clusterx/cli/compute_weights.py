# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message, remove_trailing_extension
from clusterx.cli.find_lowest import find_lowest
from clusterx.structures_set import StructuresSet

commands = ["compute_weights"]


@plac.annotations(
    sset_filepath=("Path to serialized StructuresSet object.", "option", None, str),
    sset_gss_filepath=(
        "Path to serialized StructuresSet object for GS structures.",
        "option",
        "gss",
        str,
    ),
    weights_filepath=(
        "Output path for weights (e.g., weights.npz).",
        "option",
        None,
        str,
    ),
    property_name=("Name of the property to use for weights.", "option", None, str),
    temperature=(
        "Temperature value used for Boltzmann weighting.",
        "option",
        None,
        float,
    ),
    kind=("Weighting method ID (e.g., 1 for Boltzmann).", "option", None, int),
)
def compute_weights(
    sset_filepath: str = "sset.json",
    sset_gss_filepath: Optional[str] = None,
    weights_filepath: str = "weights.npz",
    property_name: Optional[str] = None,
    temperature: float = 300.0,
    kind: int = 1,
):
    """Compute weights"""
    cmd_message("head")

    sset = StructuresSet(filepath=sset_filepath)
    conc = sset.get_concentrations(site_type=0, sigma=1)
    prop = sset.get_property_values(property_name=property_name)

    weights = np.ones(len(sset))

    if kind == 1:
        if sset_gss_filepath is not None:
            sset_gss = StructuresSet(filepath=sset_gss_filepath)
        else:
            sset_gss, _ = find_lowest(sset_filepath, property_name=property_name)

        conc_gss = sset_gss.get_concentrations(site_type=0, sigma=1)
        prop_gss = sset_gss.get_property_values(property_name=property_name)

        k_boltzmann = 8.617333262e-2  # Boltzmann constant in meV / Kelvin

        for i, (c, p) in enumerate(zip(conc, prop)):
            for i_gss, (c_gss, p_gss) in enumerate(zip(conc_gss, prop_gss)):
                if c == c_gss:
                    weights[i] = np.exp(-(p - p_gss) / (k_boltzmann * temperature))
                    break

    if kind == 2:
        for i, (c, p) in enumerate(zip(conc, prop)):
            if c < 0.025:
                weights[i] = 0
            elif c < 0.08:
                weights[i] = _inverse_sigmoid(p - 5, 0.5)
            elif c < 0.14:
                weights[i] = _inverse_sigmoid(p + 10, 0.5)
            else:
                weights[i] = _inverse_sigmoid(p + 15, 0.5)

            # weights[i] *= _inverse_sigmoid(c + 0.15, 0.4)

            print(c, p, weights[i])

    filepath = remove_trailing_extension(weights_filepath)
    np.savez(f"{filepath}.npz", weights=weights)


def _inverse_sigmoid(x, a=1):
    """
    Returns the value of the inverse sigmoid function.

    Parameters:
    - x: Input value or array.
    - a: Controls the steepness of the curve. Default is 1.

    Returns:
    - The value of the inverse sigmoid function at x.
    """
    return 1 / (1 + np.exp(a * x))
