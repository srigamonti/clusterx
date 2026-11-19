# Copyright (c) 2015-2026, CELL Developers.
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

commands = ["build_nonlinear_model_bagging"]


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
        "Random seed used for the optional train/test split and, if enabled, for bagging.",
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
    bagging_model=(
        "Dictionary of bagging ensemble options. If provided, wraps the whole base pipeline in a bagging regressor.",
        "option",
        "bm",
        dict,
    ),
)
def build_nonlinear_model_bagging(
    property_name: str,
    ccalc_filepath: str = "ccalc.pickle",
    xp_filepath: str = "xp.npz",
    model_filepath: str = "model.pickle",
    regression_model: Optional[dict] = None,
    selection_model: Optional[dict] = None,
    selection_threshold: Optional[float] = None,
    selection_must_include: Optional[list[int]] = None,
    nonlinear_transformation: Optional[dict] = None,
    weights_filepath: Optional[str] = None,
    standardize: bool = False,
    test_size: Optional[float] = None,
    random_state: int = 42,
    shuffle: bool = False,
    bagging_model: Optional[dict] = None,
):
    """
    Build and serialize a nonlinear cluster expansion (CE) model.

    Optionally:
    - split data into train/test at the beginning
    - wrap the whole pipeline in a bagging ensemble

    The hold-out test set, if requested, is never used during fitting.
    """
    cmd_message("head")

    if selection_must_include is None:
        selection_must_include = []

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

    default_bagging_model = {
        "module": "sklearn.ensemble",
        "class": "BaggingRegressor",
        "args": {
            "n_estimators": 10,
            "max_samples": 1.0,
            "max_features": 1.0,
            "bootstrap": True,
            "bootstrap_features": False,
        },
    }
    if bagging_model is not None:
        bagging_model = {**default_bagging_model, **bagging_model}
        bagging_model["args"] = {
            **default_bagging_model["args"],
            **bagging_model.get("args", {}),
        }

    # ------------------------------------------------------------------
    # Optional hold-out split performed before any fitting
    # ------------------------------------------------------------------
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

    base_reg = _build_pipeline(
        regression_model,
        nonlinear_transformation,
        selection_model=selection_model,
        selection_threshold=selection_threshold,
        selection_must_include=selection_must_include,
        standardize=standardize,
    )

    reg = _wrap_with_bagging_if_requested(
        base_estimator=base_reg,
        bagging_model=bagging_model,
        random_state=random_state,
    )

    if bagging_model is not None:
        print(f"Info({get_command_name()}): Bagging enabled")
        print(reg)

    if w_train is not None:
        if bagging_model is not None:
            print(
                f"Warning({get_command_name()}): sample weights were provided, "
                "but weighted fitting is not passed through BaggingRegressor in this implementation. "
                "Weights will still be used for weighted metric reporting."
            )
            reg.fit(x_train, y_train)
        else:
            regressor_step_name = reg.steps[-1][0]
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
    # Model summaries
    # ------------------------------------------------------------------
    if bagging_model is None:
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
    else:
        summarize_bagging_model(
            reg,
            selection_model=selection_model,
            regression_model=regression_model,
        )

    print(f"Info({get_command_name()}): Building and serializing Model object")

    nlmodel = Model(corrc=ccalc, property_name=property_name, estimator=reg)
    nlmodel.serialize(model_filepath)
    print(f"Info({get_command_name()}): Model serialized to '{model_filepath}'")


def summarize_model(regressor, x, y, model_type: str, label: str = "MODEL"):
    """
    Print CV performance summary for a fitted estimator.
    """
    print("\n" + "=" * 72)
    print(f"Model Summary Report — {model_type}/{label}")
    print("=" * 72)

    print_cv_info = False

    if model_type == "LassoCV":
        if not hasattr(regressor, "mse_path_"):
            print("mse_path_ not available. Did you forget to fit the model?")
            return
        cv_scores = regressor.mse_path_
        mean_cv_scores = np.mean(cv_scores, axis=1)
        alphas = regressor.alphas_
        print_cv_info = True

    elif model_type == "RidgeCV":
        if not hasattr(regressor, "cv_results_"):
            print(
                "cv_results_ not available. Set store_cv_results to True, and cv to None."
            )
            return
        cv_scores = regressor.cv_results_
        mean_cv_scores = np.mean(cv_scores, axis=0)
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

        print("RMSE-CV", cv, "MAE-CV", mae, "MaxAE-CV", maxae)

        sorted_indices = np.argsort(absolute_errors)[::-1]
        top10_indices = sorted_indices[:10]
        top10_values = absolute_errors[top10_indices]
        print("Absolute errors (sorted, first 10)")
        for i, v in zip(top10_indices, top10_values):
            print(
                f"Structure index: {i}, Absolute error: {v}, Prediction (CV): {pred_cv[i]}, "
                f"ab-initio: {y[i]}, Diff(ai-pred): {y[i] - pred_cv[i]}"
            )

    elif model_type in ["Ridge", "LinearRegression"]:
        pass

    else:
        print(f"Unsupported model type: {model_type}")
        return

    if print_cv_info:
        rmse_cv_scores = np.sqrt(mean_cv_scores)
        print("\nCross-Validation RMSE by Alpha:")
        for a, rmse in zip(alphas, rmse_cv_scores):
            print(f"  alpha = {a:10.5f} -> RMSE = {rmse:10.5f}")

        print("\nOptimal alpha:", regressor.alpha_)
        print("Number of features:", regressor.n_features_in_)

    if hasattr(regressor, "coef_"):
        if model_type in ["Ridge", "LinearRegression"]:
            print(
                f"Number of non-zero coefficients: {np.count_nonzero(regressor.coef_)}."
            )
        else:
            print(
                f"Number of non-zero coefficients: {np.count_nonzero(regressor.coef_)} "
                f"out of {len(regressor.coef_)}."
            )
        print("First 10 coefficients:\n", regressor.coef_[:10])

    print("=" * 72 + "\n")


def summarize_bagging_model(
    bagger,
    selection_model: Optional[dict] = None,
    regression_model: Optional[dict] = None,
):
    """
    Print a compact summary for a fitted BaggingRegressor whose base estimator
    is the nonlinear CE pipeline.
    """
    print("\n" + "=" * 72)
    print("Model Summary Report — BAGGING ENSEMBLE")
    print("=" * 72)

    print(f"Number of fitted base estimators: {len(bagger.estimators_)}")
    if hasattr(bagger, "estimators_features_"):
        print("Feature subsampling: yes")
    else:
        print("Feature subsampling: no")

    first_est = bagger.estimators_[0]
    print("\nFirst fitted base estimator:")
    print(first_est)

    if isinstance(first_est, Pipeline):
        if selection_model is not None and "selector" in first_est.named_steps:
            sel = first_est.named_steps["selector"].estimator_
            print("\nSelector inside first base estimator:")
            print(sel)

        if "regressor" in first_est.named_steps:
            reg = first_est.named_steps["regressor"]
            print("\nRegressor inside first base estimator:")
            print(reg)

            if hasattr(reg, "coef_"):
                print(
                    f"Non-zero coefficients in first base regressor: "
                    f"{np.count_nonzero(reg.coef_)}"
                )
                print(
                    "First 10 coefficients of first base regressor:\n", reg.coef_[:10]
                )

    print("=" * 72 + "\n")


def _selection_importance(estimator, selection_must_include=None):
    if selection_must_include is None:
        selection_must_include = []

    coef = estimator.coef_
    if coef.ndim == 2:
        base = np.max(np.abs(coef), axis=0)
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
    selection_must_include: Optional[list[int]] = None,
    standardize: bool = False,
):
    """
    Build a regression pipeline with optional feature selection.

    REGRESSOR < SELECTOR(None) < SCALER(None) < TRANSFORMER(None)
    """
    if selection_must_include is None:
        selection_must_include = []

    pipeline_steps = []

    if nonlinear_transformation is not None:
        module = importlib.import_module(nonlinear_transformation["module"])
        ft_class = getattr(module, nonlinear_transformation["class"])
        ft = ft_class(**nonlinear_transformation["args"])
        pipeline_steps.append(("transformer", ft))

    if standardize:
        pipeline_steps.append(("scaler", StandardScaler()))

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

    module = importlib.import_module(regression_model["module"])
    rm_class = getattr(module, regression_model["class"])
    rm = rm_class(**regression_model["args"])
    pipeline_steps.append(("regressor", rm))

    return Pipeline(steps=pipeline_steps)


def _wrap_with_bagging_if_requested(
    base_estimator,
    bagging_model: Optional[dict] = None,
    random_state: Optional[int] = None,
):
    """
    Optionally wrap the base estimator in a BaggingRegressor-like ensemble.
    """
    if bagging_model is None:
        return base_estimator

    module = importlib.import_module(bagging_model["module"])
    bag_class = getattr(module, bagging_model["class"])
    bag_args = dict(bagging_model.get("args", {}))

    # sklearn API changed from base_estimator -> estimator in newer versions
    bag_init_vars = bag_class.__init__.__code__.co_varnames
    if "estimator" in bag_init_vars:
        bag_args["estimator"] = base_estimator
    else:
        bag_args["base_estimator"] = base_estimator

    if "random_state" not in bag_args:
        bag_args["random_state"] = random_state

    return bag_class(**bag_args)
