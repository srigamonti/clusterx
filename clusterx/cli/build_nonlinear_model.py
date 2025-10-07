# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import importlib
from functools import partial
from typing import Optional

import numpy as np
import plac
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
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
    selection_must_include: list[int] = [],
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
        "args": {"degree": 1, "include_bias": False},
    }
    if nonlinear_transformation is not None:
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
        selection_must_include=selection_must_include,
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
        print(reg.named_steps["selector"].estimator_)
        summarize_model(
            reg.named_steps["selector"].estimator_,
            comat,
            pvals,
            selection_model["class"],
            label="SELECTOR",
        )
    print(reg.named_steps["regressor"])
    summarize_model(
        reg.named_steps["regressor"],
        comat,
        pvals,
        regression_model["class"],
        label="REGRESSOR",
    )

    print(f"Info({get_command_name()}): Building and serializing Model object")

    nlmodel = Model(corrc=ccalc, property_name=property_name, estimator=reg)
    nlmodel.serialize(model_filepath)


def summarize_model(regressor, x, y, model_type: str, label: str = "MODEL"):
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
    print(f"Model Summary Report — {model_type}/{label}")
    print("=" * 72)
    print_cv_info = False
    # Collect cross-validation scores
    if model_type == "LassoCV":
        if not hasattr(regressor, "mse_path_"):
            print("mse_path_ not available. Did you forget to fit the model?")
            return
        cv_scores = regressor.mse_path_  # shape: (n_alphas, n_folds)
        mean_cv_scores = np.mean(cv_scores, axis=1)  # mean over folds
        alphas = regressor.alphas_
        print_cv_info = True

    elif model_type == "RidgeCV":
        if not hasattr(regressor, "cv_results_"):
            print(
                "cv_results_ not available. Set store_cv_results to True, and cv to None."
            )
            return
        cv_scores = regressor.cv_results_  # shape: (n_samples, n_alphas)
        mean_cv_scores = np.mean(cv_scores, axis=0)  # mean over samples
        alphas = regressor.alphas
        print_cv_info = True

    elif model_type == "Lasso":
        cvs = cross_val_score(
            regressor,
            x,
            y,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
        )
        pred_cv = cross_val_predict(regressor, x, y, cv=LeaveOneOut())

        absolute_errors = np.sqrt(-cvs)
        cv = np.sqrt(-np.mean(cvs))
        maxae = np.amax(absolute_errors)
        mae = np.mean(absolute_errors)

        print(
            "RMSE-CV",
            cv,
            "MAE-CV",
            mae,
            "MaxAE-CV",
            maxae,
        )

        sorted_indices = np.argsort(absolute_errors)[::-1]
        top10_indices = sorted_indices[:10]
        top10_values = absolute_errors[top10_indices]
        print("Absolute errors (sorted, first 10)")
        for i, v in zip(top10_indices, top10_values):
            print(
                f"Structure index: {i}, Absolute error: {v}, Prediction (CV): {pred_cv[i]}, ab-initio: {y[i]}, Diff(ai-pred): {y[i] - pred_cv[i]}"
            )

    else:
        print(f"Unsupported model type: {model_type}")
        return

    if print_cv_info:
        # Compute RMSE for each alpha
        rmse_cv_scores = np.sqrt(mean_cv_scores)
        print("\nCross-Validation RMSE by Alpha:")
        for a, rmse in zip(alphas, rmse_cv_scores):
            print(f"  alpha = {a:10.5f} -> RMSE = {rmse:10.5f}")

        # Summary of fitted model
        print("\nOptimal alpha:", regressor.alpha_)
        print("Number of features:", regressor.n_features_in_)

    print(
        f"Number of non-zero coefficients: {np.count_nonzero(regressor.coef_)} out of {len(regressor.coef_)}"
    )
    print("First 10 coefficients:\n", regressor.coef_[:10])

    print("=" * 72 + "\n")


def _selection_importance(estimator, selection_must_include=[]):
    coef = estimator.coef_
    if coef.ndim == 2:
        base = np.max(np.abs(coef), axis=0)  # or np.linalg.norm(coef, axis=0)
    else:
        base = np.abs(coef)

    base = base.astype(float).copy()

    boost_val = max(1.0, base.max())
    base[selection_must_include] = boost_val
    return base


def _build_pipeline(
    regression_model: dict,
    nonlinear_transformation: dict | None = None,
    selection_model: dict | None = None,
    selection_threshold: str | float = "mean",
    selection_must_include: list[int] = [],
    standardize: bool = False,
):
    """
    Build a regression pipeline with optional feature selection.

    REGRESSOR < SELECTOR(None) < SCALER(None) < TRANSFORMER(None)

    Parameters
    ----------
    regression_model : dict
        Final regression model configuration:
        {
            "module": "sklearn.linear_model",
            "class": "RidgeCV",
            "args": {"alphas": [0.1, 1.0, 10.0]}
            "store_cv_results": True
        }

    nonlinear_transformation : dict
        Feature transformation configuration (e.g., PolynomialFeatures):
        {
            "module": "sklearn.preprocessing",
            "class": "PolynomialFeatures",
            "args": {"degree": 2, "include_bias": False}
        }

    selection_model : dict or None, default None
        Optional selection model configuration for SelectFromModel:
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

    pipeline_steps = []

    if nonlinear_transformation is not None:
        # Import and instantiate the nonlinear transformer
        module = importlib.import_module(nonlinear_transformation["module"])
        ft_class = getattr(module, nonlinear_transformation["class"])
        ft = ft_class(**nonlinear_transformation["args"])
        pipeline_steps.append(("transformer", ft))

    if standardize:
        pipeline_steps.append(("scaler", StandardScaler()))

    # Add feature selection if configured
    if selection_model is not None:
        module = importlib.import_module(selection_model["module"])
        sel_class = getattr(module, selection_model["class"])
        sel_model = sel_class(**selection_model["args"])
        getter = partial(
            _selection_importance, selection_must_include=selection_must_include
        )
        selector = SelectFromModel(
            sel_model, threshold=selection_threshold, importance_getter=getter
        )
        pipeline_steps.append(("selector", selector))

    # Import and instantiate the final regression model
    module = importlib.import_module(regression_model["module"])
    rm_class = getattr(module, regression_model["class"])
    rm = rm_class(**regression_model["args"])

    pipeline_steps.append(("regressor", rm))

    # return make_pipeline(*pipeline_steps)
    return Pipeline(steps=pipeline_steps)
