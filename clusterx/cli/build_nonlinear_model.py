# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional
import importlib
import plac
import numpy as np
from clusterx.correlations import CorrelationsCalculator
from clusterx.cli.config_utils import cmd_message
from clusterx.cli.config_utils import get_command_name
from clusterx.model import Model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
)


commands = ["build_nonlinear_model"]


@plac.pos(
    "property_name",
    help="Property to be modelled. Must be present in the StructuresSet object.",
)
@plac.pos(
    "ccalc_filepath",
    help="Path to the pickle file of a serialized CorrelationsCalculator object.",
)
@plac.pos(
    "xp_filepath",
    help="Path to the npz file of a serialized matrix of correlations built with compute_comat().",
)
@plac.opt(
    "model_filepath",
    abbrev="mof",
    help=("Path to serialize the created Model object.",),
)
@plac.opt(
    "regression_model",
    abbrev="rm",
    help=("Estimator options.",),
    type=dict,
)
@plac.opt(
    "nonlinear_transformation",
    abbrev="nt",
    help=("Estimator options.",),
    type=dict,
)
@plac.opt(
    "weights",
    help=(
        "Sample weights for fitting the model and evaluating the weighted mean squared error."
    ),
)
@plac.flg("standardize", abbrev="std", help="Standardize the input data.")
def build_nonlinear_model(
    property_name: str,
    ccalc_filepath: str = "ccalc.pickle",
    xp_filepath: str = "xp.npz",
    model_filepath: str = "model.pickle",
    regression_model: Optional[dict] = None,
    nonlinear_transformation: Optional[dict] = None,
    weights_filepath: Optional[str] = None,
    standardize: bool = False,
):
    """Compute nonlinear CE model"""
    cmd_message("head")

    print(f"Info({get_command_name()}): Initialization")

    ccalc = CorrelationsCalculator(filepath=ccalc_filepath)
    comat = np.load(xp_filepath)["comat"]
    pvals = np.load(xp_filepath)[f"property_{property_name}"]

    if regression_model is None:
        regression_model = {"kind": "LinearRegression", "args": {}}
    if nonlinear_transformation is None:
        nonlinear_transformation = {"kind": "PolynomialFeatures", "args": {"degree": 1}}

    print(f"Info({get_command_name()}): Computing model")

    reg = _build_pipeline(
        regression_model["kind"],
        regression_model["args"],
        nonlinear_transformation["kind"],
        nonlinear_transformation["args"],
        standardize=standardize,
    )

    if weights_filepath is not None:
        weights = np.load(weights_filepath)["weights"]
        kind = regression_model["kind"].lower()
        kwargs = {f"{kind}__sample_weight": weights}
        reg.fit(comat, pvals, **kwargs)
    else:
        reg.fit(comat, pvals)

    if regression_model["kind"] == "LassoCV":
        lasso_cv = reg.named_steps["lassocv"]
        cv_scores = lasso_cv.mse_path_
        alphas = lasso_cv.alphas_
        # Mean cross-validation scores for each alpha
        mean_cv_scores = np.mean(cv_scores, axis=1)
        rmse_cv_scores = np.sqrt(mean_cv_scores)
        # Print the mean cross-validation scores
        print("Mean CV scores for each alpha:", rmse_cv_scores)
        for a, cv in zip(alphas, rmse_cv_scores):
            print(f"{a:>10.5f}\t{cv:>15.5f}")

        print("Dual gap", lasso_cv.dual_gap_)
        print("Optimal alpha", lasso_cv.alpha_)
        print("Model coefficients", lasso_cv.coef_)
        print("Number of non-zero coefficients", np.count_nonzero(lasso_cv.coef_))
        print("Number of initial features", lasso_cv.n_features_in_)

    print(f"Info({get_command_name()}): Building and serializing Model object")

    nlmodel = Model(corrc=ccalc, property_name=property_name, estimator=reg)
    nlmodel.serialize(model_filepath)


def _build_pipeline(
    rm_kind: str = "LinearRegression",
    rm_args: Optional[dict] = None,
    ft_kind: str = "PolynomialFeatures",
    ft_args: Optional[dict] = None,
    standardize: bool = False,
):
    """Create pipeline

    Parameters:

        rm_conf(dict): Regression model configuration

        ft_conf(dict): Function transformer configuration
    """
    if rm_args is None:
        rm_args = {"kind": "linreg", "alpha": 0.0}
    if ft_args is None:
        ft_args = {"degree": 2, "include_bias": False}

    module = importlib.import_module("sklearn.linear_model")
    rm_class = getattr(module, rm_kind)

    module = importlib.import_module("sklearn.preprocessing")
    ft_class = getattr(module, ft_kind)

    rm = rm_class(**rm_args)
    ft = ft_class(**ft_args)

    pipeline_steps = [ft, rm]

    if standardize:
        pipeline_steps.insert(0, StandardScaler())

    _pipeline = make_pipeline(*pipeline_steps)

    return _pipeline
