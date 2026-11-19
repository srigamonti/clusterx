# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from pathlib import Path

import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.cli.info_utils import load_pickle, report_ccalc, start_report
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model

commands = ["print_info_model"]


def _class_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


def _estimator_steps(estimator, prefix=""):
    if hasattr(estimator, "steps"):
        for name, step in estimator.steps:
            yield from _estimator_steps(step, prefix + name + ".")
    elif estimator is not None and not isinstance(estimator, str):
        yield prefix.rstrip(".") or "estimator", estimator


def _report_estimator(report, estimator):
    steps = list(_estimator_steps(estimator))
    if not steps:
        return
    report.section("Estimator / pipeline")
    rows = []
    for name, step in steps:
        n_input = getattr(step, "n_features_in_", "not stored")
        n_output = getattr(step, "n_output_features_", "-")
        if hasattr(step, "kept_columns") and hasattr(step, "n_columns_full"):
            n_input, n_output = len(step.kept_columns), step.n_columns_full
        if hasattr(step, "get_support"):
            try:
                n_output = int(np.count_nonzero(step.get_support()))
            except (ValueError, AttributeError):
                pass
        rows.append([name, _class_name(step), n_input, n_output])
    report.table(["Step", "Class", "Input features", "Output / selected features"], rows)
    for name, step in steps:
        report.section(f"Parameters: {name}", 3)
        if hasattr(step, "get_params"):
            report.table(["Parameter", "Value"], sorted(step.get_params(deep=False).items()))
        fitted = []
        for attr in ("n_iter_", "n_samples_seen_", "mean_", "scale_", "threshold_", "classes_", "kept_columns"):
            value = getattr(step, attr, None)
            if value is not None:
                fitted.append([attr, value])
        members = getattr(step, "estimators_", None)
        if members is not None:
            fitted.append(["Fitted ensemble members", len(members)])
            if len(members):
                fitted.append(["First ensemble member class", _class_name(members[0])])
        if fitted:
            report.table(["Fitted attribute", "Value"], fitted)


def _report_coefficients(report, model, estimator, n_clusters):
    coefficients = getattr(model, "ecis", None)
    if coefficients is None:
        report.section("Coefficients")
        report.text("No explicit coefficients stored; predictions use the fitted estimator.")
        return
    coefficients = np.asarray(coefficients)
    report.section("Coefficients")
    report.table(["Quantity", "Value"], [["Shape", coefficients.shape], ["Total", coefficients.size],
                 ["Nonzero", int(np.count_nonzero(coefficients))], ["Intercept", getattr(model, "intercept", None)]])
    report.text("ECIs include cluster multiplicities." if estimator is None
                else "These coefficients refer to the estimator's feature basis, including any transformations.")
    if coefficients.size == 0 or coefficients.dtype.kind not in "biufc":
        return
    names = None
    if hasattr(estimator, "steps") and n_clusters is not None:
        try:
            names = estimator[:-1].get_feature_names_out([f"c{i}" for i in range(n_clusters)])
        except (ValueError, AttributeError, TypeError):
            pass
    elif coefficients.ndim and n_clusters == coefficients.shape[-1]:
        names = [f"c{i}" for i in range(n_clusters)]
    report.text("Largest stored coefficients by absolute value (up to 20):")
    rows = []
    for flat_index in np.argsort(np.abs(coefficients).ravel())[::-1][:20]:
        index = tuple(int(i) for i in np.unravel_index(flat_index, coefficients.shape))
        feature = names[index[-1]] if names is not None and len(names) == coefficients.shape[-1] else "-"
        rows.append([index, feature, coefficients[index]])
    report.table(["Coefficient index", "Feature", "Value"], rows)


@plac.pos("model_filepath", help="Path to a pickled CELL Model file.")
def print_info_model(model_filepath: str):
    """Inspect model properties, coefficients, fitted pipelines, ensembles, and correlations."""
    cmd_message("head")
    report = start_report("CE model information", model_filepath)
    model = load_pickle(model_filepath, Model)
    companion = Path(model_filepath).with_name(Path(model_filepath).stem + "_CCALC.pickle")
    ccalc = getattr(model, "corrc", None)
    source = "embedded in model" if ccalc is not None else "not available"
    if companion.is_file():
        ccalc = load_pickle(companion, CorrelationsCalculator)
        source = str(companion)
    n_clusters = len(ccalc.get_cpool()) if ccalc is not None else None
    estimator = getattr(model, "estimator", None)
    report.section("Model")
    report.table(
        ["Quantity", "Value"],
        [["Class", _class_name(model)], ["Property / properties", getattr(model, "property_name", None)],
         ["Estimator", _class_name(estimator) if estimator is not None else "explicit ECIs"],
         ["Correlation features / clusters", n_clusters], ["Calculator source", source],
         ["Model standardize flag", getattr(model, "standardize", None)]],
    )
    _report_estimator(report, estimator)
    _report_coefficients(report, model, estimator, n_clusters)
    if ccalc is not None:
        report_ccalc(report, ccalc)
