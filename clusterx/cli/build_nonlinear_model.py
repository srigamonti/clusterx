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
from clusterx.cli.report_utils import MarkdownReport
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
    shuffle: bool = False,
):
    """Build and serialize a nonlinear cluster expansion (CE) model.

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
        :class:`~clusterx.correlations.CorrelationsCalculator` object.
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

    fit = fit_nonlinear_pipeline(
        comat,
        pvals,
        weights=weights,
        regression_model=regression_model,
        selection_model=selection_model,
        selection_threshold=selection_threshold,
        selection_must_include=selection_must_include,
        nonlinear_transformation=nonlinear_transformation,
        standardize=standardize,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        report=MarkdownReport(),
    )

    print(f"Info({get_command_name()}): Building and serializing Model object")

    nlmodel = Model(corrc=ccalc, property_name=property_name, estimator=fit["pipeline"])
    nlmodel.serialize(model_filepath)
    print(f"Info({get_command_name()}): Model serialized to '{model_filepath}'")


def fit_nonlinear_pipeline(
    comat,
    pvals,
    weights=None,
    regression_model: Optional[dict] = None,
    selection_model: Optional[dict] = None,
    selection_threshold: Optional[float] = None,
    selection_must_include: list[int] = [],
    nonlinear_transformation: Optional[dict] = None,
    standardize: bool = False,
    test_size: Optional[float] = None,
    random_state: int = 42,
    shuffle: bool = False,
    report: Optional[MarkdownReport] = None,
):
    """
    Fit and evaluate the pipeline built by `_build_pipeline` on a correlations matrix.

    Applies the same defaults as `build_nonlinear_model`, performs the optional
    train/test split, fits the pipeline, and writes metrics and model summaries
    to ``report``.

    Returns
    -------
    dict
        Keys: ``pipeline`` (fitted), ``used_columns`` (indices of the columns of
        ``comat`` that influence predictions), ``x_train``, ``y_train``,
        ``w_train``, ``x_test``, ``y_test``, ``w_test``, ``metrics``.
    """
    if report is None:
        report = MarkdownReport(echo=False)

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

        report.section("Data split")
        report.bullets(
            [
                f"test_size = {test_size}, random_state = {random_state}, shuffle = {shuffle}",
                f"Total samples: {len(pvals)}",
            ]
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

        report.bullets([f"Train samples: {len(y_train)}", f"Test samples: {len(y_test)}"])
    else:
        x_train = comat
        y_train = pvals
        x_test = None
        y_test = None
        w_train = weights
        w_test = None

        report.section("Data split")
        report.text(f"No train/test split requested. Training samples: {len(y_train)}")

    report.section("Model training")
    report.code(
        f"regression_model         : {regression_model}\n"
        f"selection_model          : {selection_model}\n"
        f"selection_threshold      : {selection_threshold}\n"
        f"selection_must_include   : {selection_must_include}\n"
        f"nonlinear_transformation : {nonlinear_transformation}\n"
        f"standardize              : {standardize}"
    )

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

    metrics = {}
    train_preds = reg.predict(x_train)
    metrics["train"] = _regression_metrics(y_train, train_preds, w_train)

    report.section("Training-set performance")
    report.table(["metric", "value"], _metrics_rows(metrics["train"]))

    # ------------------------------------------------------------------
    # Hold-out test metrics
    # ------------------------------------------------------------------
    if x_test is not None:
        test_preds = reg.predict(x_test)
        metrics["test"] = _regression_metrics(y_test, test_preds, w_test)

        report.section("Hold-out test-set performance")
        report.table(["metric", "value"], _metrics_rows(metrics["test"]))

        abs_errors = np.abs(y_test - test_preds)
        sorted_indices = np.argsort(abs_errors)[::-1]
        topn = min(10, len(sorted_indices))

        report.section(f"Largest absolute errors on hold-out test set (top {topn})", 3)
        report.table(
            ["rank", "test index", "abs err", "pred", "target", "target-pred"],
            [
                [
                    rank,
                    idx,
                    f"{abs_errors[idx]:.8f}",
                    f"{test_preds[idx]:.8f}",
                    f"{y_test[idx]:.8f}",
                    f"{(y_test[idx] - test_preds[idx]):.8f}",
                ]
                for rank, idx in enumerate(sorted_indices[:topn], start=1)
            ],
        )

    # ------------------------------------------------------------------
    # Model summaries on training data only
    # ------------------------------------------------------------------
    input_names = [f"c{i}" for i in range(x_train.shape[1])]
    index_map = None
    if selection_model is not None:
        selector_idx = [name for name, _ in reg.steps].index("selector")
        pre_selector = reg[:selector_idx]
        selector = reg.named_steps["selector"]
        x_sel = _transform(pre_selector, x_train)
        names_sel = _feature_names(pre_selector, input_names)
        summarize_model(
            selector.estimator_,
            x_sel,
            y_train,
            selection_model["class"],
            label="SELECTOR",
            feature_names=names_sel,
            report=report,
        )
        _report_selection(
            report, selector, x_sel.shape[1], names_sel, selection_must_include
        )
        index_map = np.flatnonzero(selector.get_support())

    pre_regressor = reg[:-1]
    summarize_model(
        reg.named_steps["regressor"],
        _transform(pre_regressor, x_train),
        y_train,
        regression_model["class"],
        label="REGRESSOR",
        feature_names=_feature_names(pre_regressor, input_names),
        index_map=index_map,
        report=report,
    )

    used_cols = used_input_columns(reg, x_train.shape[1])
    report.section("Input columns used by the final model")
    report.text(
        f"{len(used_cols)} of {x_train.shape[1]} input columns (clusters) "
        "influence the predictions:"
    )
    report.code(str(used_cols.tolist()))

    return {
        "pipeline": reg,
        "used_columns": used_cols,
        "x_train": x_train,
        "y_train": y_train,
        "w_train": w_train,
        "x_test": x_test,
        "y_test": y_test,
        "w_test": w_test,
        "metrics": metrics,
    }


def _regression_metrics(y, preds, weights=None):
    out = {
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "MAE": float(mean_absolute_error(y, preds)),
        "MaxAE": float(np.max(np.abs(y - preds))),
        "R^2": float(r2_score(y, preds)),
    }
    if weights is not None:
        out["wRMSE"] = float(
            np.sqrt(mean_squared_error(y, preds, sample_weight=weights))
        )
    return out


def _metrics_rows(metrics: dict):
    return [[k, f"{v:.8f}"] for k, v in metrics.items()]


def _transform(pipeline_part, x):
    """Transform x with the leading steps of a pipeline (identity if empty)."""
    return pipeline_part.transform(x) if len(pipeline_part.steps) > 0 else x


def _feature_names(pipeline_part, input_names):
    if len(pipeline_part.steps) == 0:
        return list(input_names)
    try:
        return list(pipeline_part.get_feature_names_out(input_names))
    except Exception:
        return None


def summarize_model(
    regressor,
    x,
    y,
    model_type: str,
    label: str = "MODEL",
    feature_names=None,
    index_map=None,
    report: Optional[MarkdownReport] = None,
    top_n: int = 15,
):
    """Write a Markdown summary of a fitted estimator (selector or regressor).

    Reports the estimator, cross-validation information (LassoCV, RidgeCV, or
    leave-one-out for Lasso), the number of non-zero coefficients, and the
    largest coefficients by absolute value.

    Parameters
    ----------
    regressor : fitted estimator
        Estimator with ``coef_``, e.g. the ``estimator_`` of a SelectFromModel
        or the final regression step.

    x : array-like
        Features the estimator was fitted on, i.e. the input *after* all
        preceding pipeline steps.

    y : array-like
        Ground truth values.

    model_type : str
        Class name of the estimator, e.g. 'LassoCV', 'RidgeCV', 'Lasso', 'Ridge'.

    label : str
        Label of the pipeline stage, e.g. 'SELECTOR' or 'REGRESSOR'.

    feature_names : list of str, optional
        Names of the columns of ``x``, used when listing coefficients.

    index_map : array-like of int, optional
        For each column of ``x``, its index before feature selection. If given,
        it is listed next to the column index.

    report : MarkdownReport, optional
        Where to write the summary. Defaults to a console-echoing report.

    top_n : int
        Number of largest coefficients to list.
    """
    if report is None:
        report = MarkdownReport()

    report.section(f"Model summary: {label} ({model_type})")
    report.code(str(regressor))

    cv_rows = None
    if model_type == "LassoCV":
        if not hasattr(regressor, "mse_path_"):
            report.text("`mse_path_` not available. Did you forget to fit the model?")
            return
        mean_cv_scores = np.mean(regressor.mse_path_, axis=1)  # mean over folds
        alphas = regressor.alphas_
        cv_rows = (alphas, mean_cv_scores)

    elif model_type == "RidgeCV":
        if not hasattr(regressor, "cv_results_"):
            report.text(
                "`cv_results_` not available. Set store_cv_results to True, and cv to None."
            )
            return
        mean_cv_scores = np.mean(regressor.cv_results_, axis=0)  # mean over samples
        alphas = regressor.alphas
        cv_rows = (alphas, mean_cv_scores)

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
        report.section("Leave-one-out cross-validation", 3)
        report.table(
            ["RMSE-CV", "MAE-CV", "MaxAE-CV"],
            [
                [
                    f"{np.sqrt(-np.mean(cvs)):.8f}",
                    f"{np.mean(absolute_errors):.8f}",
                    f"{np.amax(absolute_errors):.8f}",
                ]
            ],
        )

        top = np.argsort(absolute_errors)[::-1][:10]
        report.text("Largest leave-one-out errors:")
        report.table(
            ["structure", "abs err", "prediction (CV)", "target", "target-pred"],
            [
                [
                    i,
                    f"{absolute_errors[i]:.6f}",
                    f"{pred_cv[i]:.6f}",
                    f"{y[i]:.6f}",
                    f"{y[i] - pred_cv[i]:.6f}",
                ]
                for i in top
            ],
        )
    elif model_type in ("Ridge", "LinearRegression"):
        pass
    else:
        report.text(f"No cross-validation summary for model type `{model_type}`.")

    if cv_rows is not None:
        alphas, mean_cv_scores = cv_rows
        best = int(np.argmin(mean_cv_scores))
        report.section("Cross-validation RMSE by alpha", 3)
        report.table(
            ["alpha", "RMSE", ""],
            [
                [f"{a:.5g}", f"{np.sqrt(s):.6f}", "<- optimal" if i == best else ""]
                for i, (a, s) in enumerate(zip(alphas, mean_cv_scores))
            ],
        )
        report.text(f"Optimal alpha: {regressor.alpha_}")

    coef = np.ravel(np.asarray(regressor.coef_))
    n_nonzero = int(np.count_nonzero(coef))
    report.section("Coefficients", 3)
    report.text(
        f"Non-zero coefficients: {n_nonzero} out of {len(coef)} features "
        f"entering this {label.lower()}."
    )

    if n_nonzero > 0:
        order = np.argsort(np.abs(coef))[::-1][: min(top_n, n_nonzero)]
        report.text(f"Largest {len(order)} coefficients by absolute value:")
        headers = ["feature index", "feature", "coefficient"]
        if index_map is not None:
            headers.insert(1, "index before selection")
        rows = []
        for i in order:
            row = [i, feature_names[i] if feature_names is not None else ""]
            if index_map is not None:
                row.insert(1, int(index_map[i]))
            rows.append(row + [f"{coef[i]:.8g}"])
        report.table(headers, rows)


def _report_selection(report, selector, n_input, input_names, selection_must_include):
    """Explain how many features the selector keeps and why."""
    importances = np.asarray(
        selector.importance_getter(selector.estimator_), dtype=float
    )
    support = selector.get_support()
    threshold = selector.threshold_
    must = sorted(set(int(i) for i in selection_must_include))
    n_nonzero = int(np.count_nonzero(np.ravel(selector.estimator_.coef_)))
    must_below = [i for i in must if abs(importances[i]) < threshold]

    report.section("Feature selection (selector -> regressor)", 2)
    report.table(
        ["step", "number of features"],
        [
            ["features entering the selector", n_input],
            [f"non-zero {type(selector.estimator_).__name__} coefficients", n_nonzero],
            [
                f"importance >= threshold ({threshold:.6g}, "
                f"selection_threshold={selector.threshold!r})",
                int(np.count_nonzero(importances >= threshold)),
            ],
            [f"forced by selection_must_include {must}", len(must)],
            ["selected features entering the regressor", int(support.sum())],
        ],
    )
    report.text(
        "The selector keeps a feature if its importance (absolute selector "
        "coefficient) is at least the threshold, or if it is in "
        "`selection_must_include`. Selector coefficients that are non-zero but "
        "below the threshold are discarded, so the regressor receives fewer "
        "features than the selector has non-zero coefficients."
    )
    if must_below:
        report.text(
            f"Features kept only because of `selection_must_include`: {must_below}."
        )

    names = input_names if input_names is not None else [""] * n_input
    kept = np.flatnonzero(support)
    order = kept[np.argsort(importances[kept])[::-1]]
    report.text("Selected features (index before selection, in order of importance):")
    report.table(
        ["position in regressor input", "index before selection", "feature", "importance"],
        [
            [int(np.flatnonzero(kept == i)[0]), int(i), names[i], f"{importances[i]:.6g}"]
            for i in order
        ],
    )


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
    """Build a regression pipeline with optional feature selection.

    REGRESSOR < SELECTOR(None) < SCALER(None) < TRANSFORMER(None)

    Parameters
    ----------
    regression_model : dict
        Final regression model configuration::

            {
                "module": "sklearn.linear_model",
                "class": "RidgeCV",
                "args": {"alphas": [0.1, 1.0, 10.0]}
                "store_cv_results": True
            }

    nonlinear_transformation : dict
        Feature transformation configuration (e.g., PolynomialFeatures)::

            {
                "module": "sklearn.preprocessing",
                "class": "PolynomialFeatures",
                "args": {"degree": 2, "include_bias": False}
            }

    selection_model : dict or None, default None
        Optional selection model configuration for SelectFromModel::

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


def used_input_columns(pipeline, n_input_columns: int) -> np.ndarray:
    """
    Return the indices of the input columns (e.g. of comat) that actually
    influence the predictions of a fitted pipeline built by `_build_pipeline`.

    A column is used if it enters (through the nonlinear transformer) at least
    one feature that survives the selector (selection_threshold, Lasso,
    must-include) and has a non-zero coefficient in the final regressor.
    """
    steps = pipeline.named_steps
    reg_coef = np.asarray(steps["regressor"].coef_)
    alive = np.any(reg_coef != 0, axis=tuple(range(reg_coef.ndim - 1)))
    alive = np.atleast_1d(alive)

    if "selector" in steps:
        # Map regressor features back to the features entering the selector
        n_before = len(steps["selector"].get_support())
        full = np.zeros(n_before, dtype=bool)
        full[steps["selector"].get_support()] = alive
        alive = full

    transformer = steps.get("transformer")
    if transformer is None:
        return np.flatnonzero(alive)

    if hasattr(transformer, "powers_"):
        # Exact: a column is used if its exponent is > 0 in an alive feature
        return np.flatnonzero(np.any(transformer.powers_[alive] > 0, axis=0))

    # Generic fallback: probe which input columns change the alive features
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(8, n_input_columns))
    ref = transformer.transform(x0)[:, alive]
    used = []
    for j in range(n_input_columns):
        x1 = x0.copy()
        x1[:, j] += 1.0
        if not np.allclose(transformer.transform(x1)[:, alive], ref):
            used.append(j)
    return np.array(used, dtype=int)
