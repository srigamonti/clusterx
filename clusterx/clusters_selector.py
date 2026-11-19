# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.correlations import CorrelationsCalculator
import numpy as np


class ClustersSelector:
    """Select optimal cluster expansion models from training data and a clusters pool.

    Objects of this class select the optimal set of clusters for the given
    training data and clusters pool. After initializing a ``ClustersSelector``
    object, call ``ClustersSelector.select_clusters()`` to perform the selection.
    Different optimality criteria can be used.

    Parameters
    ----------
    basis : str, default "trigonometric"
        Cluster basis used during the optimization task. For details and allowed
        values, see :class:`clusterx.correlations.CorrelationsCalculator`.
    selector_type : str, default "identity"
        Selection method:

        - ``"identity"``: no selection is performed. The optimal clusters pool is
          the same as the argument ``cpool`` in
          :meth:`~clusterx.clusters_selector.ClustersSelector.select_clusters`.
        - ``"subsets_cv"``: cross-validation optimization of models defined by
          subsets of the clusters pool.
        - ``"lasso_cv"``: cross-validation optimization of the sparsity parameter.
        - ``"lasso_on_residual"``: LASSO selection on the residual between a model
          defined by ``set0`` and the output. The final model contains the union
          of clusters in ``set0`` and those selected by LASSO on the residual.
        - ``"OMP"``: Orthogonal Matching Pursuit.

        Deprecated names are described in Notes.
    **selector_opts : dict
        Keyword options passed directly to the constructor. These include general
        options and options specific to ``selector_type``, described below.

    Other Parameters
    ----------------
    fit_intercept : bool, default True
        General option controlling whether to fit an intercept.
    cv_type : {"loo", "5-fold", "10-fold"}, default "loo"
        General cross-validation option. Use ``"loo"`` for leave-one-out CV,
        ``"5-fold"`` for 5-fold CV, or ``"10-fold"`` for 10-fold CV.
    cv_shuffle : bool, default False
        General option. If ``cv_type`` is n-fold, this specifies whether to shuffle
        the data for CV.
    cv_random_state : int or None
        General option. If ``cv_shuffle`` is ``True``, this controls the ordering
        of the shuffling. If None, the shuffling is different in every call. An
        integer gives the same shuffling for the same input, allowing reproducible
        results.
    standardize : bool, default False
        General option. If ``True``, standardize correlations.
    sparsity_max : float, default 1
        Positive maximal sparsity parameter for ``selector_type="lasso_cv"``.
    sparsity_min : float, default 0.01
        Positive minimal sparsity parameter for ``selector_type="lasso_cv"``.
    sparsity_step : float, optional
        Positive sparsity step for ``selector_type="lasso_cv"``. If set to 0.0,
        a logarithmic grid from ``sparsity_max`` to ``sparsity_min`` is created
        automatically.
    max_iter : int
        Maximum number of iterations for the LASSO algorithm when
        ``selector_type="lasso_cv"``.
    tol : float
        Small positive tolerance of the LASSO solution when
        ``selector_type="lasso_cv"``.
    sparsity_scale : {"log", "piece_log"}
        Sparsity scale for ``selector_type="lasso_cv"``.
    clusters_sets : {"size", "combinations", "size+combinations"}
        Subset construction strategy for ``selector_type="subsets_cv"``:

        - ``"size"``: subpools of increasing size are extracted from the initial
          pool, and cross-validation selects the optimal subpool.
        - ``"combinations"``: all possible combinations of clusters from the pool
          are considered. This can be very computationally demanding.
        - ``"size+combinations"``: a fixed pool of clusters up to the size specified
          by ``set0`` is always kept. Combinations are searched only for subsets of
          ``nclmax`` clusters.

    set0 : array of two numbers
        For ``selector_type="subsets_cv"`` and
        ``clusters_sets="size+combinations"``, specifies the size of the fixed
        pool above which a combinatorial search is performed. The two elements
        ``[int, float]`` give the maximum number of cluster points and the maximum
        radius of the fixed subpool, respectively. This option also defines the
        initial model for ``selector_type="lasso_on_residual"``, as described above.
    nclmax : int
        For ``selector_type="subsets_cv"`` and
        ``clusters_sets="size+combinations"``, the maximum number of clusters in
        the combinatorial subsets searched on top of the fixed subpool.
    alpha : float, default 0
        For subset selection, determines the regularization strength of the
        scikit-learn ridge estimator. If set to 0, the estimator switches to
        ordinary least squares.
    alphas : list or ndarray of float, default []
        For subset selection, determines the list of regularization strengths.
        If ``len(alphas) > 1``, a scikit-learn RidgeCV estimator is used.
    n_nonzero_coefs : int, default 1
        Number of nonzero parameters for ``selector_type="OMP"``.

    Notes
    -----
    The deprecated selector type ``"lasso"`` is identical to ``"lasso_cv"``;
    the deprecated type ``"linreg"`` is identical to ``"subsets_cv"``.
    The deprecated keyword option ``method`` overrides ``selector_type``.

    .. todo::

        * make ``optimal_ecis`` private
        * Documentation of selector_opts when selector_type is lasso_on_residual
    """

    def __init__(
        self, basis="trigonometric", selector_type="identity", **selector_opts
    ):
        self.method = selector_opts.pop(
            "method", selector_type
        )  # selector_type argument replaces old method argument. This ensures backward compatibility.

        # general options
        self.fit_intercept = selector_opts.pop("fit_intercept", True)
        self.cv_type = selector_opts.pop("cv_type", "loo")
        self.cv_shuffle = selector_opts.pop("cv_shuffle", False)
        self.cv_random_state = selector_opts.pop("cv_random_state", False)
        self.standardize = selector_opts.pop("standardize", False)

        # additional arguments for estimator in CV
        self.estimator_for_selector = selector_opts.pop(
            "estimator", "skl_LinearRegression"
        )
        self.estimator_opts_for_selector = selector_opts.pop("estimator_opts", {})

        # additional arguments for lasso_cv
        self.sparsity_max = selector_opts.pop("sparsity_max", 1)
        self.sparsity_min = selector_opts.pop("sparsity_min", 0.01)
        self.sparsity_step = selector_opts.pop("sparsity_step", 0.0)
        self.sparsity_scale = selector_opts.pop("sparsity_scale", "log")
        self.max_iter = selector_opts.pop("max_iter", 10000)
        self.tol = selector_opts.pop("tol", 1e-5)

        # additional arguments for subset_cv
        self.clusters_sets = selector_opts.pop("clusters_sets", "size")
        self.nclmax = selector_opts.pop("nclmax", 0)
        self.set0 = selector_opts.pop("set0", [0, 0])
        self.alpha = selector_opts.pop("alpha", 0)
        self.alphas = selector_opts.pop("alphas", [])

        # additional arguments for omp
        self.n_nonzero_coefs = selector_opts.pop("n_nonzero_coefs", 1)

        self.predictions = []
        self.opt_ecis = []
        self.optimal_clusters = None
        self.optimal_cluster_indices = None
        self.optimal_comat = None
        self.opt_rmse = None
        self.opt_mean_cv = None
        self.opt_sparsity = None
        self.lasso_sparsities = []

        self.rmse = []
        self.cvs = []
        self.set_sizes = []

        self.estimator_cv = None

        self.basis = basis

        # self.update()

    def get_opt_ecis(self):
        return self.opt_ecis

    def get_rmse(self):
        return self.rmse

    def select_clusters(self, sset, cpool, prop, comat=None):
        """Select clusters

        Returns a subpool containing
        the optimal set of clusters.

        Parameters
        ----------
        sset : StructuresSet
            The structures set corresponding to the training data.

        cpool : ClustersPool
            the clusters pool from which the optimal model is selected.

        prop : str
            property label (must be in sset) of property for which the optimal set of clusters is
            to be selected.

        comat : 2D ndarray, default "None"
            if the correlation matrix was precalculated, you can give it here
        """
        self.cpool = cpool
        self.plat = cpool.get_plat()
        self.sset = sset
        self.prop = prop

        corrc = CorrelationsCalculator(self.basis, self.plat, self.cpool)
        if comat is None:
            self.ini_comat = corrc.get_correlation_matrix(self.sset)
        else:
            self.ini_comat = comat

        self.target = self.sset.get_property_values(property_name=self.prop)

        x = self.ini_comat
        p = self.target

        opt = []

        if self.method == "lasso_cv" or self.method == "lasso":  # "lasso" deprecated
            opt = self._select_clusters_lasso_cv(x, p)

        elif self.method == "skl_lasso_cv":
            opt = self._select_clusters_skl_lasso_cv(x, p)

        elif (
            self.method == "lasso_on_residual" or self.method == "lasso-on-residual"
        ):  #  "lasso-on-residual" deprecated, preferred with "_"s
            nb = int(self.set0[0])
            r = float(self.set0[1])

            clset0 = []
            for icl, cl in enumerate(self.cpool):
                if cl.npoints <= nb and cl.get_radius() <= r + 1e-4:
                    clset0.append(icl)

            from sklearn import linear_model

            lre = linear_model.LinearRegression(
                fit_intercept=self.fit_intercept, n_jobs=-1
            )
            rows = np.arange(len(p))
            comat0 = x[np.ix_(rows, clset0)]
            lre.fit(comat0, p)
            pred0 = lre.predict(comat0)

            pres = p - pred0
            clset1 = self._select_clusters_lasso_on_residual_cv(x, pres, clset0)

            opt = np.union1d(clset0, clset1)

        elif (
            self.method == "subsets_cv"
            or self.method == "subset_cv"
            or self.method == "subsets-cv"
            or self.method == "subset-cv"
            or self.method == "linreg"
        ):  # "linreg" is deprecated
            if self.clusters_sets == "size":
                clsets = self.cpool.get_clusters_sets(grouping_strategy="size")
                opt = self._linear_regression_cv(x, p, clsets)
            elif self.clusters_sets == "combinations":
                clsets = self.cpool.get_clusters_sets(
                    grouping_strategy="combinations", nclmax=self.nclmax
                )
                opt = self._linear_regression_cv(x, p, clsets)
            elif self.clusters_sets == "size+combinations":
                clsets = self.cpool.get_clusters_sets(
                    grouping_strategy="size+combinations",
                    nclmax=self.nclmax,
                    set0=self.set0,
                )
                opt = self._linear_regression_cv(x, p, clsets)

        elif self.method == "identity":
            opt = np.arange(len(self.cpool), dtype=int)

        elif self.method == "omp" or self.method == "OMP":
            opt = self._select_clusters_skl_omp(x, p)

        self.optimal_cluster_indices = opt
        self.optimal_comat = self.ini_comat[:, opt]

        self.optimal_clusters = self.cpool.get_subpool(opt)

        return self.optimal_clusters

    def get_optimal_cpool(self):
        """Return optimal ClustersPool object"""
        return self.optimal_clusters

    def get_optimal_cpool_array(self):
        """Return optimal array of clusters"""
        return self.optimal_clusters._cpool

    def _linear_regression_cv(self, x, p, clsets, verbosity=0):
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        from tqdm import tqdm

        disable_tqdm = True

        if verbosity == 0:
            disable_tqdm = True

        if not self.standardize:
            if self.alpha != 0 and len(self.alphas) == 0:
                self.estimator_cv = linear_model.Ridge(
                    alpha=self.alpha, fit_intercept=self.fit_intercept
                )
            elif len(self.alphas) > 1:
                self.estimator_cv = linear_model.RidgeCV(
                    alphas=self.alphas, fit_intercept=self.fit_intercept
                )
            else:
                self.estimator_cv = linear_model.LinearRegression(
                    fit_intercept=self.fit_intercept
                )
        else:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            if self.alpha != 0 and len(self.alphas) == 0:
                self.estimator_cv = make_pipeline(
                    StandardScaler(),
                    linear_model.Ridge(
                        alpha=self.alpha, fit_intercept=self.fit_intercept
                    ),
                )
            elif len(self.alphas) > 1:
                self.estimator_cv = make_pipeline(
                    StandardScaler(),
                    linear_model.RidgeCV(
                        alpha=self.alphas, fit_intercept=self.fit_intercept
                    ),
                )
            else:
                self.estimator_cv = make_pipeline(
                    StandardScaler(),
                    linear_model.LinearRegression(fit_intercept=self.fit_intercept),
                )

        rows = np.arange(len(p))

        opt_cv = -1
        opt_clset = []

        for iset, clset in tqdm(
            enumerate(clsets), total=len(clsets), desc="Subset CV", disable=disable_tqdm
        ):
            _comat = x[np.ix_(rows, clset)]

            _cvs = None
            if self.cv_type == "loo":
                from sklearn.model_selection import LeaveOneOut

                _cvs = cross_val_score(
                    self.estimator_cv,
                    _comat,
                    p,
                    cv=LeaveOneOut(),
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
            else:
                from sklearn.model_selection import KFold

                if self.cv_type == "5-fold":
                    if self.cv_shuffle:
                        kf = KFold(
                            n_splits=5,
                            shuffle=self.cv_shuffle,
                            random_state=self.cv_random_state,
                        )
                    else:
                        kf = KFold(n_splits=5, shuffle=self.cv_shuffle)
                if self.cv_type == "10-fold":
                    if self.cv_shuffle:
                        kf = KFold(
                            n_splits=10,
                            shuffle=self.cv_shuffle,
                            random_state=self.cv_random_state,
                        )
                    else:
                        kf = KFold(n_splits=10, shuffle=self.cv_shuffle)

                _cvs = cross_val_score(
                    self.estimator_cv,
                    _comat,
                    p,
                    cv=kf.split(_comat),
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )

            mean_cv = np.sqrt(-np.mean(_cvs))
            self.cvs.append(mean_cv)
            self.estimator_cv.fit(_comat, p)
            self.rmse.append(
                np.sqrt(mean_squared_error(self.estimator_cv.predict(_comat), p))
            )
            self.set_sizes.append(len(clset))

            if opt_cv <= 0:
                opt_cv = mean_cv
                opt_clset = clset
            else:
                if opt_cv > mean_cv:
                    opt_cv = mean_cv
                    opt_clset = clset

        return opt_clset

    def _select_clusters_lasso_cv(self, x, p):
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error

        sparsity = self.sparsity_max

        if self.sparsity_step == 0.0:
            step = float(sparsity / (1.0 * 10))
        else:
            step = self.sparsity_step

        opt_cv = -1
        opt_clset = []

        idx = 1
        # while sparsity.__ge__(self.sparsity_min):
        while sparsity >= self.sparsity_min:
            if not self.standardize:
                estimator_cv = linear_model.Lasso(
                    alpha=sparsity,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
            else:
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline

                estimator_cv = make_pipeline(
                    StandardScaler(),
                    linear_model.Lasso(
                        alpha=sparsity,
                        fit_intercept=self.fit_intercept,
                        max_iter=self.max_iter,
                        tol=self.tol,
                    ),
                )

            estimator_cv.fit(x, p)

            if not self.standardize:
                ecimult = estimator_cv.coef_
            else:
                ecimult = estimator_cv[-1].coef_
            # reshape to 2d array, if not already
            if len(ecimult.shape) == 1:
                ecimult = np.expand_dims(ecimult, axis=0)

            _cvs = None
            if self.cv_type == "loo":
                from sklearn.model_selection import LeaveOneOut

                _cvs = cross_val_score(
                    estimator_cv,
                    x,
                    p,
                    cv=LeaveOneOut(),
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
            else:
                from sklearn.model_selection import KFold

                if self.cv_type == "5-fold":
                    kf = KFold(
                        n_splits=5,
                        shuffle=self.cv_shuffle,
                        random_state=self.cv_random_state,
                    )
                if self.cv_type == "10-fold":
                    kf = KFold(
                        n_splits=10,
                        shuffle=self.cv_shuffle,
                        random_state=self.cv_random_state,
                    )

                _cvs = cross_val_score(
                    estimator_cv,
                    x,
                    p,
                    cv=kf.split(x),
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )

            mean_cv = np.sqrt(-np.mean(_cvs))

            self.cvs.append(mean_cv)
            self.rmse.append(np.sqrt(mean_squared_error(estimator_cv.predict(x), p)))

            self.set_sizes.append(np.count_nonzero(ecimult))
            self.lasso_sparsities.append(sparsity)

            if opt_cv <= 0:
                opt_cv = mean_cv
                opt_clset = np.flatnonzero(np.any(ecimult > 0, axis=0))
                self.opt_sparsity = sparsity
            else:
                if opt_cv > mean_cv:
                    opt_cv = mean_cv
                    opt_clset = np.flatnonzero(np.any(ecimult > 0, axis=0))
                    self.opt_sparsity = sparsity

            if self.sparsity_step == 0.0:
                if self.sparsity_scale == "log":
                    step = float(sparsity / (1.0 * 10))
                    sparsity = sparsity - 3 * step

                elif self.sparsity_scale == "piece_log":
                    if idx == 10:
                        idx = 2
                        step = float(sparsity / (1.0 * 10))
                        sparsity = sparsity - step
                    else:
                        idx = idx + 1
                        sparsity = sparsity - step

            else:
                sparsity = sparsity - step

        return opt_clset

    def _select_clusters_skl_lasso_cv(self, x, p):
        if p.ndim > 1:
            from sklearn.linear_model import MultiTaskLassoCV as LassoCV
        else:
            from sklearn.linear_model import LassoCV

        opt_clset = []
        alphas = np.geomspace(self.sparsity_min, self.sparsity_max, num=100)

        if not self.standardize:
            est = LassoCV(
                alphas=alphas,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                n_jobs=-1,
                verbose=True,
            )
        else:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            est = make_pipeline(
                StandardScaler(),
                LassoCV(
                    alphas=alphas,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.max_iter,
                    n_jobs=-1,
                    verbose=True,
                ),
            )

        mod = est.fit(x, p)
        ecimult = []
        if not self.standardize:
            for coef in mod.coef_:
                ecimult.append(coef)
            self.cvs = np.mean(est.mse_path_)
            self.rmse = np.mean(est.mse_path_)

            self.set_sizes = np.ones(len(est.alphas_))
            self.lasso_sparsities = est.alphas_

        else:
            for coef in est[-1].coef_:
                ecimult.append(coef)
            self.cvs = np.mean(est[-1].mse_path_)
            self.rmse = np.mean(est[-1].mse_path_)

            self.set_sizes = np.ones(len(est[-1].alphas_))
            self.lasso_sparsities = est[-1].alphas_

        opt_clset = [i for i, e in enumerate(ecimult) if np.count_nonzero(e) > 0]
        if not self.standardize:
            self.opt_sparsity = est.alpha_
        else:
            self.opt_sparsity = est[-1].alpha_

        return opt_clset

    def _select_clusters_skl_omp(self, x, p):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import OrthogonalMatchingPursuit

        opt_clset = []

        if self.standardize:
            est = make_pipeline(
                StandardScaler(),
                OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs),
            )
        else:
            est = make_pipeline(
                OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
            )

        mod = est.fit(x, p)

        opt_clset = [i for i, e in enumerate(mod[-1].coef_) if e != 0]

        return opt_clset

    def _select_clusters_lasso_on_residual_cv(self, x, p, clset0):
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error

        sparsity = self.sparsity_max

        if self.sparsity_step == 0.0:
            step = float(sparsity / (1.0 * 10))
        else:
            step = self.sparsity_step

        opt_cv = -1
        opt_clset = []
        rows = np.arange(len(p))

        idx = 1
        while sparsity.__ge__(self.sparsity_min):
            estimator_cv = linear_model.Lasso(
                alpha=sparsity,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            estimator_cv.fit(x, p)

            estimator_lr = linear_model.LinearRegression(fit_intercept=False)

            cluster_list_lasso = []
            for i, coef in enumerate(estimator_cv.coef_):
                if coef != 0:
                    cluster_list_lasso.append(i)

            if len(cluster_list_lasso) == 0:
                clset = clset0
            else:
                clset = np.union1d(clset0, cluster_list_lasso)
            _comat_model = x[np.ix_(rows, clset)]

            _cvs = None
            if self.cv_type == "loo":
                from sklearn.model_selection import LeaveOneOut

                _cvs = cross_val_score(
                    estimator_lr,
                    _comat_model,
                    p,
                    cv=LeaveOneOut(),
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
            else:
                from sklearn.model_selection import KFold

                if self.cv_type == "5-fold":
                    kf = KFold(
                        n_splits=5,
                        shuffle=self.cv_shuffle,
                        random_state=self.cv_random_state,
                    )
                if self.cv_type == "10-fold":
                    kf = KFold(
                        n_splits=10,
                        shuffle=self.cv_shuffle,
                        random_state=self.cv_random_state,
                    )

                _cvs = cross_val_score(
                    estimator_lr,
                    _comat_model,
                    p,
                    cv=kf.split(x),
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )

            mean_cv = np.sqrt(-np.mean(_cvs))

            self.cvs.append(mean_cv)
            self.rmse.append(np.sqrt(mean_squared_error(estimator_cv.predict(x), p)))

            self.set_sizes.append(np.count_nonzero(clset))
            self.lasso_sparsities.append(sparsity)

            if opt_cv <= 0:
                opt_cv = mean_cv
                opt_clset = clset
                self.opt_sparsity = sparsity
            else:
                if opt_cv > mean_cv:
                    opt_cv = mean_cv
                    opt_clset = clset
                    self.opt_sparsity = sparsity

            if self.sparsity_step == 0.0:
                if self.sparsity_scale == "log":
                    step = float(sparsity / (1.0 * 10))
                    sparsity = sparsity - 3 * step

                elif self.sparsity_scale == "piece_log":
                    if idx == 10:
                        idx = 2
                        step = float(sparsity / (1.0 * 10))
                        sparsity = sparsity - step
                    else:
                        idx = idx + 1
                        sparsity = sparsity - step

            else:
                sparsity = sparsity - step

        return opt_clset

    def display_info(self):
        """Display in screen information about the optimal model"""
        try:
            print(
                "{0:<40s}:{1:>10.4f}".format(
                    "CV score (LOO) for optimal model", self.opt_mean_cv
                )
            )
            print(
                "{0:<40s}:{1:>10.4f}".format(
                    "RMSE of the fit for optimal model", self.opt_rmse
                )
            )
            print(
                "{0:<40s}:{1:>10d}".format(
                    "Size of optimal clusters pool", len(self.optimal_clusters)
                )
            )
        except Exception as e:
            if self.opt_mean_cv is None:
                print("ClustersSelector (Error): Calculation of CV score failed.")
            elif self.opt_rmse is None:
                print("ClustersSelector (Error): Calculation of RMSE failed.")
            else:
                print(type(e), e)
        # print("CV score (LOO) for optimal model: "+str(self.opt_mean_cv))
        # print("RMSE of the fit for optimal model: "+str(self.opt_rmse))
        # print("Size of optimal clusters pool: "+str(len(self.optimal_clusters)))
