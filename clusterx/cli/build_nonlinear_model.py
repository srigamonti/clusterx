# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import importlib
from typing import Optional

import numpy as np
import plac
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from clusterx.cli.config_utils import cmd_message, get_command_name
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model

commands = ["build_nonlinear_model"]


@plac.annotations(
    property_name=(
        "Property to be modelled. Must be present in the StructuresSet object.",
        "positional",
        None,
        str,
    ),
    ccalc_filepath=(
        "Path to the pickle file of a serialized CorrelationsCalculator object.",
        "option",
        None,
        str,
    ),
    xp_filepath=(
        "Path to the npz file of a serialized correlation matrix (from compute_comat).",
        "option",
        None,
        str,
    ),
    model_filepath=(
        "Path to serialize the created Model object.",
        "option",
        "mof",
        str,
    ),
    regression_model=("Dictionary of estimator options.", "option", "rm", dict),
    selection_model=(
        "Dictionary of estimator options for feature selection model (e.g. LASSO).",
        "option",
        "sm",
        dict,
    ),
    selection_threshold=(
        "Threshold to interactions.",
        "option",
        "thr",
        float,
    ),
    nonlinear_transformation=(
        "Dictionary of nonlinear transformation settings.",
        "option",
        "nt",
        dict,
    ),
    weights_filepath=(
        "Sample weights for fitting and evaluating the weighted MSE.",
        "option",
        None,
        str,
    ),
    standardize=("Standardize the input data.", "flag", "std", bool),
)
def build_nonlinear_model(
    property_name: str,
    ccalc_filepath: str = "ccalc.pickle",
    xp_filepath: str = "xp.npz",
    model_filepath: str = "model.pickle",
    regression_model: Optional[dict] = None,
    selection_model: Optional[dict] = None,
    selection_threshold: Optional[float] = None,
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

    default_regression_model = {
        "module": "sklearn.linear_model",
        "class": "LinearRegression",
        "args": {},
    }
    if regression_model is None:
        regression_model = {}
    regression_model = {**default_regression_model, **regression_model}

    default_selection_model = {
        "module": "sklearn.linear_model",
        "class": "LassoCV",
        "args": {"alphas": 10, "cv": 10},
    }
    if selection_model is not None:
        selection_model = {**default_selection_model, **selection_model}

    default_nonlinear_transformation = {
        "module": "sklearn.preprocessing",
        "class": "PolynomialFeatures",
        "args": {"degree": 1, "include_bias": True},
    }
    if nonlinear_transformation is None:
        nonlinear_transformation = {}
    nonlinear_transformation = {
        **default_nonlinear_transformation,
        **nonlinear_transformation,
    }

    print(f"Info({get_command_name()}): Computing model")

    reg = _build_pipeline(
        regression_model,
        nonlinear_transformation,
        selection_model=selection_model,
        selection_threshold=selection_threshold,
        standardize=standardize,
    )

    if weights_filepath is not None:
        weights = np.load(weights_filepath)["weights"]
        kind = regression_model["kind"].lower()
        kwargs = {f"{kind}__sample_weight": weights}
        reg.fit(comat, pvals, **kwargs)
    else:
        reg.fit(comat, pvals)

    preds = reg.predict(comat)
    fit_error = np.sqrt(mean_squared_error(pvals, preds))
    print(f"Fit RMSE: {fit_error:.5f}")

    if selection_model is not None:
        summarize_cv_model(
            reg.named_steps["selector"], selection_model["class"], label="SELECTOR"
        )

    summarize_cv_model(
        reg.named_steps["regressor"], regression_model["class"], label="REGRESSOR"
    )

    print(f"Info({get_command_name()}): Building and serializing Model object")

    nlmodel = Model(corrc=ccalc, property_name=property_name, estimator=reg)
    nlmodel.serialize(model_filepath)


def summarize_cv_model(regressor, model_type: str, label: str = "MODEL"):
    """
    Print CV performance summary for a fitted pipeline containing LassoCV or RidgeCV.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline with a 'regressor' step.
    X : array-like
        Input features to predict on (e.g., training set).
    y : array-like
        Ground truth values.
    model_type : str
        One of 'LassoCV' or 'RidgeCV'.
    """
    # Predict and compute fit error
    print("\n" + "=" * 72)
    print(f"CV Model Summary Report — {model_type}/{label}")
    print("=" * 72)

    # Collect cross-validation scores
    if model_type == "LassoCV":
        if not hasattr(regressor, "mse_path_"):
            print("mse_path_ not available. Did you forget to fit the model?")
            return
        cv_scores = regressor.mse_path_  # shape: (n_alphas, n_folds)
        mean_cv_scores = np.mean(cv_scores, axis=1)  # mean over folds
        alphas = regressor.alphas_

    elif model_type == "RidgeCV":
        if not hasattr(regressor, "cv_values_"):
            print("cv_values_ not available. Did you set store_cv_values=True?")
            return
        cv_scores = regressor.cv_values_  # shape: (n_samples, n_alphas)
        mean_cv_scores = np.mean(cv_scores, axis=0)  # mean over samples
        alphas = regressor.alphas

    else:
        print(f"Unsupported model type: {model_type}")
        return

    # Compute RMSE for each alpha
    rmse_cv_scores = np.sqrt(mean_cv_scores)
    print("\nCross-Validation RMSE by Alpha:")
    for a, rmse in zip(alphas, rmse_cv_scores):
        print(f"  alpha = {a:10.5f} -> RMSE = {rmse:10.5f}")

    # Summary of fitted model
    print("\nOptimal alpha:", regressor.alpha_)
    print("Number of features:", regressor.n_features_in_)
    print("Number of non-zero coefficients:", np.count_nonzero(regressor.coef_))
    print("First 10 coefficients:\n", regressor.coef_[:10])
    print("=" * 72 + "\n")


def _build_pipeline(
    regression_model: dict,
    nonlinear_transformation: dict,
    selection_model: dict | None = None,
    selection_threshold: str | float = "mean",
    standardize: bool = False,
):
    """
    Build a regression pipeline with optional feature selection.

    Parameters
    ----------
    regression_model : dict
        Final regression model config:
        {
            "module": "sklearn.linear_model",
            "class": "RidgeCV",
            "args": {"alphas": [0.1, 1.0, 10.0]}
        }

    nonlinear_transformation : dict
        Feature transformation config (e.g., PolynomialFeatures):
        {
            "module": "sklearn.preprocessing",
            "class": "PolynomialFeatures",
            "args": {"degree": 2, "include_bias": False}
        }

    selection_model : dict or None, default None
        Optional selection model config for SelectFromModel:
        {
            "module": "sklearn.linear_model",
            "class": "LassoCV",
            "args": {"cv": 5, "max_iter": 1000}
        }

    selection_threshold : str or float, default "mean"
        Threshold for SelectFromModel.

    standardize : bool, default False
        If True, inserts StandardScaler after the nonlinear transformation.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fully configured pipeline.
    """

    # Import and instantiate the nonlinear transformer
    module = importlib.import_module(nonlinear_transformation["module"])
    ft_class = getattr(module, nonlinear_transformation["class"])
    ft = ft_class(**nonlinear_transformation["args"])

    # Import and instantiate the final regression model
    module = importlib.import_module(regression_model["module"])
    rm_class = getattr(module, regression_model["class"])
    rm = rm_class(**regression_model["args"])

    pipeline_steps = [("transformer", ft)]

    if standardize:
        pipeline_steps.append(("scaler", StandardScaler()))

    # Add feature selection if configured
    if selection_model is not None:
        module = importlib.import_module(selection_model["module"])
        sel_class = getattr(module, selection_model["class"])
        sel_model = sel_class(**selection_model["args"])
        selector = SelectFromModel(sel_model, threshold=selection_threshold)
        pipeline_steps.append(("selector", selector))

    pipeline_steps.append(("regressor", rm))

    # return make_pipeline(*pipeline_steps)
    return Pipeline(steps=pipeline_steps)


def _build_pipeline_3(
    regression_model: dict,
    nonlinear_transformation: dict,
    standardize: bool = False,
    use_lassocv_selection: bool = True,
    selection_threshold: str | float = 1.0e-2,
    selection_model_args: dict | None = None,
):
    """Create pipeline with optional LassoCV-based feature selection followed by LassoCV refit.

    Parameters
    ----------
    regression_model : dict
        Final regression model configuration:
        {"module": "...", "class": "LassoCV", "args": {...}}
        (Should be LassoCV for this setup.)
    nonlinear_transformation : dict
        Nonlinear feature configuration (e.g., PolynomialFeatures):
        {"module": "...", "class": "PolynomialFeatures", "args": {...}}
    standardize : bool, default False
        If True, insert StandardScaler **after** the nonlinear transformation.
    use_lassocv_selection : bool, default True
        If True, insert SelectFromModel(LassoCV) before the final estimator.
    selection_threshold : {"mean","median"} or float, default "mean"
        Threshold for SelectFromModel.
    selection_model_args : dict or None, default None
        Args for the LassoCV used inside SelectFromModel. If None, reuse
        regression_model["args"].

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    # Final estimator (should be LassoCV for this specific workflow)
    module = importlib.import_module(regression_model["module"])
    rm_class = getattr(module, regression_model["class"])
    rm = rm_class(**regression_model["args"])

    # Nonlinear transformer (e.g., PolynomialFeatures)
    module = importlib.import_module(nonlinear_transformation["module"])
    ft_class = getattr(module, nonlinear_transformation["class"])
    ft = ft_class(**nonlinear_transformation["args"])

    pipeline_steps = [ft]

    # Standardize AFTER expansion so penalties act on comparable scales
    if standardize:
        pipeline_steps.append(StandardScaler())

    # Optional LassoCV-based feature selection, then refit with final LassoCV
    if use_lassocv_selection:
        sel_args = (
            selection_model_args
            if selection_model_args is not None
            else regression_model["args"]
        )
        lasso_for_selection = LassoCV(**sel_args)
        sfm = SelectFromModel(lasso_for_selection, threshold=selection_threshold)
        pipeline_steps.extend([sfm, rm])
    else:
        pipeline_steps.append(rm)

    _pipeline = make_pipeline(*pipeline_steps)
    return _pipeline


def _build_pipeline_2(
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
