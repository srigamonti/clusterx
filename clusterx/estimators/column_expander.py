# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnExpander(BaseEstimator, TransformerMixin):
    """Scatter a reduced set of columns into a zero-padded full-width matrix.

    Used as first step of a pipeline trained on ``n_columns_full`` input columns
    when only the columns ``kept_columns`` are actually used by the fitted
    pipeline. The reduced input (e.g. correlations of a pruned clusters pool)
    is expanded back to the width expected by the trained steps. The
    discarded columns are filled with zeros; since they do not influence the
    pipeline predictions, the expansion is exact.

    Parameters
    ----------
    n_columns_full : int
        Number of columns of the full (original) input matrix.

    kept_columns : sequence of int
        Indices, in the full input matrix, of the columns present in the
        reduced input. Must be sorted in increasing order, which is the
        order in which they appear in the reduced input.
    """

    def __init__(self, n_columns_full, kept_columns):
        self.n_columns_full = n_columns_full
        self.kept_columns = kept_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        kept = np.asarray(self.kept_columns, dtype=int)
        if X.ndim != 2 or X.shape[1] != len(kept):
            raise ValueError(
                f"ColumnExpander expects {len(kept)} input columns, got shape {X.shape}."
            )
        out = np.zeros((X.shape[0], self.n_columns_full), dtype=float)
        out[:, kept] = X
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"x{i}" for i in range(self.n_columns_full)], dtype=object)
