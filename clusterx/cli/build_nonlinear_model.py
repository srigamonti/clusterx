# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import importlib
from functools import partial
from typing import Optional

import numpy as np
import plac
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_predict,
    cross_val_score,
    train_test_split,
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
    test_size=(
        "Fraction of samples to reserve as a hold-out test set. If not provided or <= 0, no split is performed.",
        "option",
        "ts",
        float,
    ),
    random_state=(
        "Random seed used for the optional train/test split.",
        "option",
        "rs",
        int,
    ),
    shuffle=(
        "Shuffle samples before the optional train/test split.",
        "flag",
        "sh",
        bool,
    ),
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
    test_size: Optional[float] = None,
    random_state: int = 42,
    shuffle: bool = True,
):
    """
    Build and serialize a nonlinear cluster expansion (CE) model.

    This function constructs a nonlinear model for a given property based on a
    precomputed correlation matrix and correlation calculator. It combines
    optional nonlinear transformations, feature selection, and regression
    estimators to fit and evaluate a CE model. The final model is serialized
    to disk.

    The function is designed to be invoked both from the command line and programmatically.

    Parameters
    ----------
    property_name : str
        Name of the property to be modeled. Must exist in the ``StructuresSet``
        object used in correlation computations.
    ccalc_filepath : str, optional
        Path to the pickle file of a serialized
        :class:`~clusterx.calculators.CorrelationsCalculator` object.
        Default is ``"ccalc.pickle"``.
    xp_filepath : str, optional
        Path to the ``.npz`` file containing the serialized correlation matrix
        (as produced by ``compute_comat``). Default is ``"xp.npz"``.
    model_filepath : str, optional
        Path to serialize the resulting
        :class:`~clusterx.model.Model` object. Default is ``"model.pickle"``.
    regression_model : dict, optional
        Dictionary specifying the regression estimator configuration, with
        keys ``"module"``, ``"class"``, and ``"args"``.  
        Defaults to ``{"module": "sklearn.linear_model",
        "class": "LinearRegression", "args": {}}``.
    selection_model : dict, optional
        Dictionary specifying a feature selection model, typically a LASSO
        variant. Defaults to
        ``{"module": "sklearn.linear_model", "class": "LassoCV",
        "args": {"alphas": 10, "cv": 10}}``.
    selection_threshold : float, optional
        Threshold applied to the selected interactions for filtering.
    selection_must_include : list of int, optional
        List of feature indices that must always be included in the model.
    nonlinear_transformation : dict, optional
        Dictionary specifying the nonlinear transformation settings.
        Defaults to
        ``{"module": "sklearn.preprocessing",
        "class": "PolynomialFeatures",
        "args": {"degree": 1, "include_bias": False}}``.
    weights_filepath : str, optional
        Path to an ``.npz`` file containing sample weights for fitting and
        evaluating the weighted mean squared error (MSE).
    standardize : bool, optional
        Whether to standardize the input data prior to fitting.
        Default is ``False``.
    test_size : float, optional
        Fraction of samples reserved for a final hold-out test set. If
        ``None`` or ``<= 0``, no split is performed.
    random_state : int, optional
        Random seed for the optional train/test split.
    shuffle : bool, optional
        Whether to shuffle before splitting.

    Returns
    -------
    None
        The trained model is serialized to the specified ``model_filepath``.

    Notes
    -----
    - The function prints progress and diagnostic information to stdout.
    - If ``weights_filepath`` is provided, weighted regression is performed.
    - Model summaries are printed for both the selector and regressor stages.

    Examples
    --------
    Run from command line::

        $ clusterx build-nonlinear-model energy --ccalc ccalc.pickle \
              --xp xp.npz --model model.pickle --standardize

    Or use programmatically::

        build_nonlinear_model(
            property_name="energy",
            ccalc_filepath="ccalc.pickle",
            xp_filepath="xp.npz",
            regression_model={"class": "Ridge", "args": {"alpha": 1.0}},
            standardize=True
        )
    """
    cmd_message("head")

    print(f"Info({get_command_name()}): Initialization")

    ccalc = CorrelationsCalculator(filepath=ccalc_filepath)
    xp = np.load(xp_filepath)
    comat = xp["comat"]
    pvals = xp[f"property_{property_name}"]

    weights = None
    if weights_filepath is not None:
        weights = np.load(weights_filepath)["weights"]
        if len(weights) != len(pvals):
            raise ValueError(
                f"weights has length {len(weights)}, but pvals has length {len(pvals)}."
            )

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

    do_split = test_size is not None and test_size > 0.0

    if do_split:
        if not (0.0 < test_size < 1.0):
            raise ValueError(
                f"test_size must be between 0 and 1 (exclusive), got {test_size}."
            )

        print(f"Info({get_command_name()}): Splitting data into train/test sets")
        print(
            f"  test_size = {test_size}, random_state = {random_state}, shuffle = {shuffle}"
        )

        if weights is not None:
            (
                x_train,
                x_test,
                y_train,
                y_test,
                w_train,
                w_test,
            ) = train_test_split(
                comat,
                pvals,
                weights,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                comat,
                pvals,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
            )
            w_train = None
            w_test = None

        print(f"  Total samples: {len(pvals)}")
        print(f"  Train samples: {len(y_train)}")
        print(f"  Test samples : {len(y_test)}")
    else:
        x_train = comat
        y_train = pvals
        x_test = None
        y_test = None
        w_train = weights
        w_test = None

        print(f"Info({get_command_name()}): No train/test split requested")
        print(f"  Total samples used for training: {len(y_train)}")

    print(f"Info({get_command_name()}): Computing model")

    reg = _build_pipeline(
        regression_model,
        nonlinear_transformation,
        selection_model=selection_model,
        selection_threshold=selection_threshold,
        selection_must_include=selection_must_include,
        standardize=standardize,
    )

    if w_train is not None:
        regressor_step_name = reg.steps[-1][
            0
        ]  # This is the regression_model at the end of the pipeline
        fit_kwargs = {f"{regressor_step_name}__sample_weight": w_train}
        reg.fit(x_train, y_train, **fit_kwargs)
    else:
        reg.fit(x_train, y_train)

    # ------------------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------------------

    train_preds = reg.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)
    train_maxae = np.max(np.abs(y_train - train_preds))

    print("\n" + "=" * 72)
    print("Training-set performance")
    print("=" * 72)
    print(f"Train RMSE : {train_rmse:.8f}")
    print(f"Train MAE  : {train_mae:.8f}")
    print(f"Train MaxAE: {train_maxae:.8f}")
    print(f"Train R^2  : {train_r2:.8f}")

    if w_train is not None:
        train_wrmse = np.sqrt(
            mean_squared_error(y_train, train_preds, sample_weight=w_train)
        )
        print(f"Train wRMSE: {train_wrmse:.8f}")

    # ------------------------------------------------------------------
    # Hold-out test metrics
    # ------------------------------------------------------------------
    if x_test is not None:
        test_preds = reg.predict(x_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_maxae = np.max(np.abs(y_test - test_preds))

        print("\n" + "=" * 72)
        print("Hold-out test-set performance")
        print("=" * 72)
        print(f"Test RMSE : {test_rmse:.8f}")
        print(f"Test MAE  : {test_mae:.8f}")
        print(f"Test MaxAE: {test_maxae:.8f}")
        print(f"Test R^2  : {test_r2:.8f}")

        if w_test is not None:
            test_wrmse = np.sqrt(
                mean_squared_error(y_test, test_preds, sample_weight=w_test)
            )
            print(f"Test wRMSE: {test_wrmse:.8f}")

        abs_errors = np.abs(y_test - test_preds)
        sorted_indices = np.argsort(abs_errors)[::-1]
        topn = min(10, len(sorted_indices))

        print(f"\nLargest absolute errors on hold-out test set (top {topn})")
        for rank, idx in enumerate(sorted_indices[:topn], start=1):
            print(
                f"[{rank:2d}] local_test_index={idx}, "
                f"abs_err={abs_errors[idx]:.8f}, "
                f"pred={test_preds[idx]:.8f}, "
                f"target={y_test[idx]:.8f}, "
                f"diff(target-pred)={(y_test[idx] - test_preds[idx]):.8f}"
            )

    # ------------------------------------------------------------------
    # Model summaries on training data only
    # ------------------------------------------------------------------
    if selection_model is not None:
        print(reg.named_steps["selector"].estimator_)
        summarize_model(
            reg.named_steps["selector"].estimator_,
            x_train,
            y_train,
            selection_model["class"],
            label="SELECTOR",
        )

    print(reg.named_steps["regressor"])
    summarize_model(
        reg.named_steps["regressor"],
        x_train,
        y_train,
        regression_model["class"],
        label="REGRESSOR",
    )

    print(f"Info({get_command_name()}): Building and serializing Model object")

    nlmodel = Model(corrc=ccalc, property_name=property_name, estimator=reg)
    nlmodel.serialize(model_filepath)
    print(f"Info({get_command_name()}): Model serialized to '{model_filepath}'")


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
    elif model_type == "Ridge":
        pass
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

    if model_type in ["Ridge", "LinearRegression"]:
        print(f"Number of non-zero coefficients: {np.count_nonzero(regressor.coef_)}.")
    else:
        print(
            f"Number of non-zero coefficients: {np.count_nonzero(regressor.coef_)} out of {len(regressor.coef_)}."
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
