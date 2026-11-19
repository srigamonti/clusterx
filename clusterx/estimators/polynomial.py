# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

"""Compiled prediction for supported polynomial regression pipelines."""

from itertools import chain, combinations, combinations_with_replacement

import numpy as np
from numba import jit
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from clusterx.estimators.column_expander import ColumnExpander


_REGRESSORS = (LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV)
_MAX_CACHE_BYTES = 16 * 1024 * 1024


@jit(nopython=True)
def _evaluate_sparse_polynomial(x, columns, mean, inverse_scale, offsets, features, powers, coefficients, intercept):
    transformed = np.zeros(len(mean))
    for index in range(len(columns)):
        transformed[columns[index]] = x[index]
    for index in range(len(transformed)):
        transformed[index] = (transformed[index] - mean[index]) * inverse_scale[index]
    result = intercept.copy()
    for term in range(len(offsets) - 1):
        monomial = 1.0
        for entry in range(offsets[term], offsets[term + 1]):
            monomial *= transformed[features[entry]] ** powers[entry]
        for output in range(len(result)):
            result[output] += coefficients[term, output] * monomial
    return result


def _selected_powers(polynomial, selected_indices):
    """Yield only selected exponent rows in sklearn PolynomialFeatures order."""
    n_features = polynomial.n_features_in_
    min_degree = getattr(polynomial, "_min_degree", None)
    max_degree = getattr(polynomial, "_max_degree", None)
    if min_degree is None or max_degree is None:
        degree = polynomial.degree
        if np.isscalar(degree):
            min_degree, max_degree = 0, degree
        else:
            min_degree, max_degree = degree
    combination = combinations if polynomial.interaction_only else combinations_with_replacement
    start_degree = max(1, min_degree)
    rows = chain.from_iterable(
        combination(range(n_features), degree) for degree in range(start_degree, max_degree + 1)
    )
    if polynomial.include_bias:
        rows = chain(combination(range(n_features), 0), rows)
    selected = set(int(index) for index in selected_indices)
    for index, columns in enumerate(rows):
        if index in selected:
            powers = np.zeros(n_features, dtype=np.int32)
            for column in columns:
                powers[column] += 1
            yield index, powers


def polynomial_parameters(estimator, n_features):
    """Return sparse polynomial parameters or None for unsupported pipelines.

    The returned CSR-like monomial representation is small when polynomial
    features are sparse (as they are for low-degree expansions). Only the
    exact sklearn transformations whose fitted effects are understood here
    are accepted. A None result leaves prediction to the fitted pipeline.
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
    raw_coef = np.asarray(final.coef_, dtype=float)
    scalar_output = raw_coef.ndim == 1
    if raw_coef.ndim not in (1, 2):
        return None
    coefficients = np.atleast_2d(raw_coef).T.copy()
    intercept = np.broadcast_to(np.asarray(final.intercept_, dtype=float), (coefficients.shape[1],)).copy()

    polynomial_index = None
    for index, (_, step) in enumerate(steps[:-1]):
        if type(step) is PolynomialFeatures:
            if polynomial_index is not None:
                return None
            polynomial_index = index
    if polynomial_index is None:
        return None

    # Fold post-polynomial affine transforms back into monomial coefficients.
    for _, step in reversed(steps[polynomial_index + 1:-1]):
        if step is None or (isinstance(step, str) and step == "passthrough"):
            continue
        if type(step) is StandardScaler:
            if getattr(step, "n_features_in_", None) != coefficients.shape[0]:
                return None
            if step.with_std:
                scale = np.asarray(step.scale_, dtype=float)
                if scale.shape != (coefficients.shape[0],) or not np.isfinite(scale).all() or (scale <= 0).any():
                    return None
                coefficients /= scale[:, None]
            if step.with_mean:
                mean = np.asarray(step.mean_, dtype=float)
                if mean.shape != (coefficients.shape[0],):
                    return None
                intercept -= mean @ coefficients
        elif type(step) is SelectFromModel:
            if not hasattr(step, "n_features_in_"):
                return None
            support = step.get_support(indices=True)
            if len(support) != coefficients.shape[0]:
                return None
            expanded = np.zeros((step.n_features_in_, coefficients.shape[1]), dtype=float)
            expanded[support] = coefficients
            coefficients = expanded
        else:
            return None

    polynomial = steps[polynomial_index][1]
    if not hasattr(polynomial, "n_features_in_") or not hasattr(polynomial, "n_output_features_"):
        return None
    if polynomial.n_output_features_ != coefficients.shape[0]:
        return None

    # Before PolynomialFeatures, support ColumnExpander followed by an optional
    # StandardScaler. Apply these transforms in the compiled evaluator so
    # centering offsets in omitted columns are preserved exactly.
    input_width = polynomial.n_features_in_
    columns = np.arange(n_features)
    expanded_width = n_features
    mean = np.zeros(n_features)
    inverse_scale = np.ones(n_features)
    saw_expander = False
    saw_scaler = False
    for _, step in steps[:polynomial_index]:
        if step is None or (isinstance(step, str) and step == "passthrough"):
            continue
        if type(step) is not ColumnExpander:
            if type(step) is StandardScaler and not saw_scaler:
                if getattr(step, "n_features_in_", None) != expanded_width:
                    return None
                if step.with_mean:
                    mean = np.asarray(step.mean_, dtype=float).copy()
                if step.with_std:
                    scale = np.asarray(step.scale_, dtype=float)
                    if scale.shape != (expanded_width,) or not np.isfinite(scale).all() or (scale <= 0).any():
                        return None
                    inverse_scale = 1.0 / scale
                saw_scaler = True
                continue
            return None
        if saw_expander or saw_scaler:
            return None
        kept = np.asarray(step.kept_columns)
        if (step.n_columns_full != input_width or kept.ndim != 1
                or not np.issubdtype(kept.dtype, np.integer) or len(kept) != n_features
                or len(np.unique(kept)) != len(kept) or (kept < 0).any() or (kept >= input_width).any()):
            return None
        columns = kept
        expanded_width = input_width
        mean = np.zeros(expanded_width)
        inverse_scale = np.ones(expanded_width)
        saw_expander = True
    if expanded_width != input_width or len(mean) != input_width:
        return None

    # Feature selectors often retain only a small fraction of a large
    # polynomial expansion. Generate those exponent rows directly rather
    # than materializing PolynomialFeatures.powers_ (terms x input columns).
    selected_indices = np.flatnonzero(np.any(coefficients != 0, axis=1))
    offsets = [0]
    feature_indices = []
    exponents = []
    selected_coefficients = []
    cache_bytes = columns.nbytes + mean.nbytes + inverse_scale.nbytes + intercept.nbytes
    found = 0
    for index, row in _selected_powers(polynomial, selected_indices):
        nonzero = np.flatnonzero(row)
        feature_indices.extend(nonzero.tolist())
        exponents.extend(row[nonzero].tolist())
        offsets.append(len(feature_indices))
        selected_coefficients.append(coefficients[index])
        found += 1
        cache_bytes += 8 + 2 * 4 * len(nonzero) + coefficients.shape[1] * np.dtype(float).itemsize
        if cache_bytes > _MAX_CACHE_BYTES:
            return None
    if found != len(selected_indices):
        return None
    cache_bytes += 8  # Final CSR offset.

    offsets = np.asarray(offsets, dtype=np.int64)
    feature_indices = np.asarray(feature_indices, dtype=np.int32)
    exponents = np.asarray(exponents, dtype=np.int32)
    coefficients = np.asarray(selected_coefficients, dtype=float).reshape((-1, intercept.size))
    if offsets.nbytes + feature_indices.nbytes + exponents.nbytes + coefficients.nbytes + columns.nbytes + mean.nbytes \
            + inverse_scale.nbytes + intercept.nbytes > _MAX_CACHE_BYTES:
        return None
    if not all(np.isfinite(array).all() for array in (coefficients, intercept, mean, inverse_scale)):
        return None
    coefficients = np.ascontiguousarray(coefficients)
    parameters = (columns, mean, inverse_scale, offsets, feature_indices, exponents, coefficients, intercept)
    if scalar_output:
        return parameters, True
    return parameters, False


def evaluate_polynomial(parameters, correlations, scalar_output):
    """Evaluate cached polynomial parameters for one correlation vector."""
    columns, mean, inverse_scale, offsets, features, powers, coefficients, intercept = parameters
    result = _evaluate_sparse_polynomial(
        np.asarray(correlations, dtype=float), columns, mean, inverse_scale,
        offsets, features, powers, coefficients, intercept,
    )
    return result[0] if scalar_output else result
