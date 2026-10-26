# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import List, Union, cast

import numpy as np
import plac
from sklearn.linear_model import LinearRegression

from clusterx.cli.config_utils import cmd_message, convert_to_float_list
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model

commands = ["build_model_from_coefficients"]


@plac.annotations(
    ccalc_filepath=(
        "Path to the pickle file of a serialized CorrelationsCalculator object.",
        "option",
        None,
        str,
    ),
    property_name=(
        "Name of the property represented by the model.",
        "option",
        None,
        str,
    ),
    model_filepath=(
        "Path to serialize the created Model object.",
        "option",
        None,
        str,
    ),
    model_coefficients=(
        "Comma-separated model coefficients, optionally surrounded by brackets.",
        "option",
        None,
        str,
    ),
    model_intercept=("Model intercept.", "option", None, float),
)
def build_model_from_coefficients(
    ccalc_filepath: str = "ccalc.pickle",
    property_name: str = "custom_property",
    model_filepath: str = "custom_model.pickle",
    model_coefficients: Union[str, List[float]] = "",
    model_intercept: float = 0.0,
):
    """Build and serialize a linear model from supplied coefficients."""
    cmd_message("head")

    if isinstance(model_coefficients, str):
        try:
            coefficients = convert_to_float_list(model_coefficients)
        except ValueError as error:
            raise ValueError("model_coefficients must be a comma-separated list of numbers.") from error
    else:
        coefficients = [float(value) for value in model_coefficients]

    ccalc = cast(
        CorrelationsCalculator,
        CorrelationsCalculator(filepath=ccalc_filepath),
    )
    cpool = ccalc.get_cpool()
    if cpool is None:
        raise ValueError("The serialized CorrelationsCalculator does not contain a cluster pool.")
    n_features = len(cpool)
    if len(coefficients) != n_features:
        raise ValueError(
            "The number of model coefficients must match the number of features "
            f"in the cluster pool: expected {n_features}, got {len(coefficients)}."
        )

    estimator = LinearRegression()
    estimator.coef_ = np.asarray(coefficients, dtype=float)
    estimator.intercept_ = float(model_intercept)

    cemodel = Model(
        corrc=ccalc,
        property_name=property_name,
        estimator=estimator,
    )

    cemodel.serialize(filepath=model_filepath)
