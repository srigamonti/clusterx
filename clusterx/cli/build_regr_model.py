# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import importlib
from typing import Optional

import numpy as np
import plac
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clusterx.cli.config_utils import cmd_message, get_command_name
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model

commands = ["build_regr_model"]


@plac.annotations(
    property_name=("Property to be modelled. Must be present in the StructuresSet object.", "positional", None, str),
    ccalc_filepath=("Path to the pickle file of a serialized CorrelationsCalculator object.", "option", None, str),
    xp_filepath=("Path to the npz file of a serialized correlation matrix (from compute_comat).", "option", None, str),
    model_filepath=("Path to serialize the created Model object.", "option", "mof", str),
    regression_model=("Dictionary of estimator options.", "option", "rm", dict),
    weights_filepath=("Sample weights for fitting and evaluating the weighted MSE.", "option", None, str),
    standardize=("Standardize the input data.", "flag", "std", bool),
)
def build_regr_model(
    property_name: str,
    ccalc_filepath: str = "ccalc.pickle",
    xp_filepath: str = "xp.npz",
    model_filepath: str = "model.pickle",
    regression_model: Optional[dict] = None,
    weights_filepath: Optional[str] = None,
    standardize: bool = False,
):
    """Build CE model with arbitrary estimator."""
    cmd_message("head")

    print(f"Info({get_command_name()}): Initialization")

    ccalc = CorrelationsCalculator(filepath=ccalc_filepath)
    comat = np.load(xp_filepath)["comat"]
    pvals = np.load(xp_filepath)[f"property_{property_name}"]

    if regression_model is None:
        regression_model = {"kind": "LinearRegression", "args": {}}

    print(f"Info({get_command_name()}): Computing model")

    reg = _build_pipeline(
        regression_model["module"],
        regression_model["class"],
        regression_model["args"],
        standardize=standardize,
    )

    if weights_filepath is not None:
        weights = np.load(weights_filepath)["weights"]
        kind = regression_model["kind"].lower()
        kwargs = {f"{kind}__sample_weight": weights}
        reg.fit(comat, pvals, **kwargs)
    else:
        reg.fit(comat, pvals)

    print(f"Info({get_command_name()}): Building and serializing Model object")

    nlmodel = Model(corrc=ccalc, property_name=property_name, estimator=reg)
    nlmodel.serialize(model_filepath)


def _build_pipeline(
    rm_module: str = "sklearn.linear_model",
    rm_class: str = "LinearRegression",
    rm_args: Optional[dict] = None,
    standardize: bool = False,
):
    """Create pipeline

    Parameters:

        rm_conf(dict): Regression model configuration

        ft_conf(dict): Function transformer configuration
    """
    if rm_args is None:
        rm_args = {"kind": "linreg", "alpha": 0.0}

    module = importlib.import_module(rm_module)
    rm_class = getattr(module, rm_class)
    rm = rm_class(**rm_args)

    pipeline_steps = [rm]

    if standardize:
        pipeline_steps.insert(0, StandardScaler())

    _pipeline = make_pipeline(*pipeline_steps)

    return _pipeline
