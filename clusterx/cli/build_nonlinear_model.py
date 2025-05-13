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

commands = ["build_nonlinear_model"]


@plac.annotations(
    property_name=("Property to be modelled. Must be present in the StructuresSet object.", "positional", None, str),
    ccalc_filepath=("Path to the pickle file of a serialized CorrelationsCalculator object.", "option", None, str),
    xp_filepath=("Path to the npz file of a serialized correlation matrix (from compute_comat).", "option", None, str),
    model_filepath=("Path to serialize the created Model object.", "option", "mof", str),
    regression_model=("Dictionary of estimator options.", "option", "rm", dict),
    nonlinear_transformation=("Dictionary of nonlinear transformation settings.", "option", "nt", dict),
    weights_filepath=("Sample weights for fitting and evaluating the weighted MSE.", "option", None, str),
    standardize=("Standardize the input data.", "flag", "std", bool),
)
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
        regression_model = {
            "module": "sklearn.linear_model",
            "class": "LinearRegression",
            "args": {}}
    if nonlinear_transformation is None:
        nonlinear_transformation = {
            "module": "sklearn.preprocessing",
            "class": "PolynomialFeatures",
            "args": {"degree": 1}}

    print(f"Info({get_command_name()}): Computing model")

    reg = _build_pipeline(
        regression_model,
        nonlinear_transformation,
        standardize=standardize,
    )

    if weights_filepath is not None:
        weights = np.load(weights_filepath)["weights"]
        kind = regression_model["kind"].lower()
        kwargs = {f"{kind}__sample_weight": weights}
        reg.fit(comat, pvals, **kwargs)
    else:
        reg.fit(comat, pvals)

    if regression_model["class"] == "LassoCV":
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
    regression_model: dict,
    nonlinear_transformation: dict,
    standardize: bool = False,
):
    """Create pipeline

    Parameters:

        regression_model(dict): Regression model configuration
        nonlinear_transformation(dict): nonlinear feature configuration
    """
    module = importlib.import_module(regression_model["module"])
    rm_class = getattr(module, regression_model["class"])
    rm = rm_class(**regression_model["args"])

    module = importlib.import_module(nonlinear_transformation["module"])
    ft_class = getattr(module, nonlinear_transformation["class"])
    ft = ft_class(**nonlinear_transformation["args"])

    pipeline_steps = [ft, rm]

    if standardize:
        pipeline_steps.insert(0, StandardScaler())

    _pipeline = make_pipeline(*pipeline_steps)

    return _pipeline
