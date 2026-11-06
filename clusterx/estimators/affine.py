# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

"""Analytically fold recognized fitted affine estimators into correlation space."""

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from clusterx.estimators.column_expander import ColumnExpander


_REGRESSORS = (LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV)


def affine_parameters(estimator, n_features):
    """Return ``(coefficients, intercept)`` or None for an unsupported estimator.

    Pull fitted coefficients back through explicitly supported affine steps.
    Exact types are required: subclasses can override predict/transform with
    nonlinear behavior. Polynomial degree must be at most one;
    agreement at zero and unit vectors is not evidence of affine behavior.
    """
    if type(estimator) is Pipeline:
        steps = estimator.steps
    else:
        steps = [("regressor", estimator)]
    if not steps or type(steps[-1][1]) not in _REGRESSORS:
        return None
    final = steps[-1][1]
    if not hasattr(final, "coef_") or not hasattr(final, "intercept_"):
        return None
    coefficients = np.asarray(final.coef_, dtype=float)
    scalar_output = coefficients.ndim == 1
    if coefficients.ndim not in (1, 2):
        return None
    weights = np.atleast_2d(coefficients).copy()
    intercept = np.broadcast_to(np.asarray(final.intercept_, dtype=float), (weights.shape[0],)).copy()

    for _, step in reversed(steps[:-1]):
        if step is None or (isinstance(step, str) and step == "passthrough"):
            continue
        if type(step) is ColumnExpander:
            kept = np.asarray(step.kept_columns)
            if (kept.ndim != 1 or not np.issubdtype(kept.dtype, np.integer)
                    or len(np.unique(kept)) != len(kept) or (kept < 0).any()
                    or (kept >= step.n_columns_full).any() or weights.shape[1] != step.n_columns_full):
                return None
            weights = weights[:, kept]
        elif type(step) is PolynomialFeatures:
            if not hasattr(step, "n_features_in_"):
                return None
            degree = step.degree if np.isscalar(step.degree) else step.degree[1]
            if degree not in (0, 1):
                return None
            bias = int(step.include_bias)
            if weights.shape[1] != bias + (step.n_features_in_ if degree == 1 else 0):
                return None
            # Degree <= 1 emits an optional bias, then input columns in order.
            # Avoid powers_, which materializes a quadratic-size dense matrix.
            if bias:
                intercept += weights[:, 0]
            weights = weights[:, bias:] if degree == 1 else np.zeros((weights.shape[0], step.n_features_in_))
        elif type(step) is StandardScaler:
            if getattr(step, "n_features_in_", None) != weights.shape[1]:
                return None
            if step.with_std:
                scale = np.asarray(step.scale_)
                if not np.isfinite(scale).all() or (scale <= 0).any():
                    return None
                weights = weights / scale
            if step.with_mean:
                intercept -= weights @ step.mean_
        elif type(step) is SelectFromModel:
            if not hasattr(step, "n_features_in_"):
                return None
            support = step.get_support(indices=True)
            if len(support) != weights.shape[1]:
                return None
            expanded = np.zeros((weights.shape[0], step.n_features_in_), dtype=float)
            expanded[:, support] = weights
            weights = expanded
        else:
            return None

    if weights.shape[1] != n_features or not np.isfinite(weights).all() or not np.isfinite(intercept).all():
        return None
    if scalar_output:
        return weights[0], intercept[0]
    return weights, intercept
