# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
import numpy as np
import sys

class ClustersSelector():
    """Clusters selector class

    Objects of this class are used to select optimal cluster expansion models, i.e.
    the optimal set of clusters, for given training data and clusters pool. 

    After initializing a ``ClustersSelector`` object, the actual selection of  clusters 
    is achieved by calling the method ``ClustersSelector.select_clusters()``

    Different optimality criteria can be used.


    **Parameters:**

    ``basis``: *string*, default = ``"trigonometric"``
        Cluster basis used during the optimization task. For details and allowed values, read the documentation in :py:class:`clusterx.correlations.CorrelationsCalculator`

    ``method``: *string*, default = ``"identity"``
        can be "lasso_cv" or "subsets_cv". In both cases a cross-validation optimization
        is performed. In the case of "lasso", the optimal sparsity parameter is
        searched through cross validation, while in "linreg", cross validation
        is directly used as model selector.
        Deprecated options: ``"lasso"``, ``"linreg"``.
        ``"lasso"`` is identical to ``"lasso_cv"``
        ``"linreg"`` is identical to ``"subsets_cv"``

    ``**kwargs``: keyword arguments
        if ``method`` is set to "lasso", the keyword arguments are:
            ``sparsity_max``: positive real, maximal sparsity parameter
            ``sparsity_min``: positive real, minimal sparsity parameter
            ``sparsity_step``: positive real, optional, if set to 0.0, a logarithmic
            grid from sparsity_max to sparsity_min is automatically created.
            ``max_iter``: integer, maximum number of iterations for LASSO algorithm.
            ``tol``: small positive real, tolerance of LASSO solution.
            ``sparsity_scale``: either "log" or "piece_log".
            ``cv_splits``: None or integer, default 3, number of splits for CV. If None, LeaveOneOut is performed
        if ``method`` is set to "linreg", the keyword arguments are:
            ``clusters_sets``: one of "size", "combinations", and "size+combinations".
            In the first case, clusters sub_pools of increasing size are extracted from
            the initial pool, and cross validation selects the optimal sub-pool.
            In the second case, all possible combinations of clusters from the pool
            are considered, this can be very computanionally demanding.
            In the third case, a fixed pool of clusters up to certain size (see ``set0``
            parameter below) is always kept and the combinations are searched only
            for subsets of ``nclmax`` (see below) clusters.

            ``set0``: array with two elements ``[int,float]``
                if ``clusters_sets`` is set to "size+combinations", this indicates
                the size of the fixed pool of clusters, above which a combinatorial
                search is performed. The first element of the array indicates the
                maximum number of cluster points and the second element the maximum radius,
                for the fixed subpool.

            ``nclmax``: integer
                if ``clusters_sets`` is set to "size+combinations", this indicates
                the maximum number of clusters in the combinatorial subsets of clusters
                to be searched for (on top of the fixed subpool, see above).

    .. todo::

        * make ``optimal_ecis`` private
    """
    def __init__(self, basis="trigonometric", method="identity", **kwargs):
        self.method = method

        # additional arguments for lasso
        self.sparsity_max = kwargs.pop("sparsity_max",1)
        self.sparsity_min = kwargs.pop("sparsity_min",0.01)
        self.sparsity_step = kwargs.pop("sparsity_step",0.0)
        self.sparsity_scale = kwargs.pop("sparsity_scale","log")
        self.cv_splits = kwargs.pop("cv_splits",3)
        self.max_iter = kwargs.pop("max_iter",10000)
        self.tol = kwargs.pop("tol",1e-5)

        # additional arguments for linear regression
        self.clusters_sets = kwargs.pop("clusters_sets","size")
        self.nclmax = kwargs.pop("nclmax", 0)
        self.set0 = kwargs.pop("set0",[0, 0])

        self.fit_intercept=False
        #for c in self.cpool._cpool:
        #    if c.npoints == 0:
        #        self.fit_intercept=True
        #        break

        self.predictions = []
        self.opt_ecis = []
        self.optimal_clusters = None
        self.opt_rmse = None
        self.opt_mean_cv = None
        self.opt_sparsity = None
        self.lasso_sparsities = []

        self.rmse = []
        self.cvs = []
        self.set_sizes = []


        self.fitter_cv = None

        self.basis = basis

        #self.update()

    def get_opt_ecis(self):
        return self.opt_ecis

    def get_rmse(self):
        return self.rmse

    def select_clusters(self, sset, cpool, prop, comat = None):
        """Select clusters

        Returns a subpool containing
        the optimal set of clusters.

        **Parameters:**

        ``sset``: StructuresSet object
            The structures set corresponding to the training data.

        ``cpool``: ClustersPool object
            the clusters pool from which the optimal model is selected.

        ``prop``: string
            property label (must be in sset) of property for which the optimal set of clusters is to be selected.
        """
        from sklearn.model_selection import LeaveOneOut

        self.cpool = cpool
        self.plat = cpool.get_plat()
        self.sset = sset
        self.prop = prop

        corrc = CorrelationsCalculator(self.basis, self.plat, self.cpool)
        if comat is None:
            self.ini_comat = corrc.get_correlation_matrix(self.sset)
        else:
            self.ini_comat = comat
            
        self.target = self.sset.get_property_values(property_name = self.prop)

        x = self.ini_comat
        p = self.target

        if self.method == "lasso":
            opt = self._select_clusters_lasso_cv(x, p)

            if self.fit_intercept == True:
                if 0 not in opt:
                    self.fit_intercept = False

        elif self.method == "lasso-on-residual":
            nb = int(self.set0[0])
            r = float(self.set0[1])

            clset0 = []
            for icl, cl in enumerate(self.cpool):
                if cl.npoints <= nb and cl.radius <= r + 1e-4:
                    clset0.append(icl)
                    
            from sklearn import linear_model
            from sklearn.metrics import mean_squared_error

            lre = linear_model.LinearRegression(fit_intercept=False, normalize=False, n_jobs = -1)
            rows = np.arange(len(p))
            cols = np.arange(len(cpool))
            comat0 = x[np.ix_(rows,clset0)]
            lre.fit(comat0,p)
            pred0 = lre.predict(comat0)
            rmse0 = np.sqrt(mean_squared_error(pred0, p))

            pres = p - pred0
            #clset1 = self._select_clusters_lasso_cv(x, pres)
            clset1 = self._select_clusters_lasso_on_residual_cv(x, pres, clset0)
            #comat1 = x[np.ix_(rows,np.delete(cols,clset0))]
            #clset1 = self._select_clusters_lasso_cv(comat1, pres)

            opt = np.union1d(clset0, clset1)

        elif self.method == "linreg":
            if self.clusters_sets == "size":
                clsets = self.cpool.get_clusters_sets(grouping_strategy = "size")
                opt = self._linear_regression_cv(x, p , clsets)
            elif self.clusters_sets == "combinations":
                clsets = self.cpool.get_clusters_sets(grouping_strategy = "combinations",  nclmax=self.nclmax)
                opt = self._linear_regression_cv(x, p, clsets)
            elif self.clusters_sets == "size+combinations":
                clsets = self.cpool.get_clusters_sets(grouping_strategy = "size+combinations", nclmax=self.nclmax , set0=self.set0)
                opt = self._linear_regression_cv(x, p, clsets)

        elif self.method == "identity":
            opt = np.arange(len(self.cpool),dtype=int)


        self.optimal_clusters = self.cpool.get_subpool(opt)

        return self.optimal_clusters

        """
        rows = np.arange(len(p))
        comat_opt =  x[np.ix_(rows,opt)]

        self.optimal_ecis(comat_opt,p)
        """

    def get_optimal_cpool(self):
        """Return optimal ClustersPool object
        """
        return self.optimal_clusters

    def get_optimal_cpool_array(self):
        """Return optimal array of clusters
        """
        return self.optimal_clusters._cpool


    def _linear_regression_cv(self,x,p,clsets):
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error

        #if self.method == "linreg":
        self.fitter_cv = linear_model.LinearRegression(fit_intercept=self.fit_intercept, normalize=False, n_jobs = -1)

        rows = np.arange(len(p))
        ecis = []
        ranks = []
        sizes = []

        opt_cv=-1
        opt_clset=[]

        el=True

        for iset, clset in enumerate(clsets):
            _comat = x[np.ix_(rows,clset)]

            #_cvs = cross_val_score(self.fitter_cv, _comat, p, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error', n_jobs = -1)
            _cvs = cross_val_score(self.fitter_cv, _comat, p, cv=10, scoring = 'neg_mean_squared_error', n_jobs = -1)
            mean_cv=np.sqrt(-np.mean(_cvs))
            self.cvs.append(mean_cv)
            self.fitter_cv.fit(_comat,p)
            self.rmse.append(np.sqrt(mean_squared_error(self.fitter_cv.predict(_comat),p)))
            self.set_sizes.append(len(clset))

            if opt_cv <= 0:
                opt_cv = mean_cv
                opt_clset = clset
            else:
                if opt_cv > mean_cv:
                    opt_cv = mean_cv
                    opt_clset = clset

        return opt_clset

    def optimal_ecis(self, x, p):
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error

        #if self.method == "linreg":
        self.fitter_cv = linear_model.LinearRegression(fit_intercept=self.fit_intercept, normalize=False)

        if self.fit_intercept:
            if int(x.shape[1]) > 1:
                _comat = np.delete(x, (0), axis=1)
            else:
                self.fitter_cv = linear_model.LinearRegression(fit_intercept=False, normalize=False)
                _comat = x
        else:
            _comat = x

        self.fitter_cv.fit(_comat,p)

        ecimult = []
        if self.fit_intercept:
            ecimult.append(self.fitter_cv.intercept_)

        for coef in self.fitter_cv.coef_:
            ecimult.append(coef)

        self.opt_ecis = ecimult

        self.predictions = [el for el in self.fitter_cv.predict(_comat)]

        self.opt_rmse=np.sqrt(mean_squared_error(self.fitter_cv.predict(_comat),p))

        _cvs = cross_val_score(self.fitter_cv, _comat, p, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
        self.opt_mean_cv=np.sqrt(-np.mean(_cvs))


    def _select_clusters_lasso_cv(self,x,p):
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error

        sparsity = self.sparsity_max

        if self.sparsity_step == 0.0:
            step = float(sparsity/(1.0*10))
            idpot = 1
        else:
            step = sparsity_step

        opt_cv = -1
        opt_clset = []
        rows = np.arange(len(p))

        idx = 1
        while sparsity.__ge__(self.sparsity_min):

            if self.fit_intercept:
                _comat = np.delete(x, (0), axis=1)
            else:
                _comat = x
                
            fitter_cv = linear_model.Lasso(alpha=sparsity, fit_intercept=self.fit_intercept, normalize = False, max_iter = self.max_iter, tol = self.tol)
            fitter_cv.fit(_comat,p)

            ecimult = []
            if self.fit_intercept:
                ecimult.append(fitter_cv.intercept_)

            for coef in fitter_cv.coef_:
                ecimult.append(coef)

            if self.cv_splits is None:
                _cvs = cross_val_score(fitter_cv, _comat, p, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error', n_jobs = -1)
            else:
                _cvs = cross_val_score(fitter_cv, _comat, p, cv=self.cv_splits, scoring = 'neg_mean_squared_error', n_jobs = -1)
                
            mean_cv = np.sqrt(-np.mean(_cvs))

            self.cvs.append(mean_cv)
            self.rmse.append(np.sqrt(mean_squared_error(fitter_cv.predict(_comat),p)))

            self.set_sizes.append(np.count_nonzero(ecimult))
            self.lasso_sparsities.append(sparsity)

            if opt_cv <= 0:
                opt_cv = mean_cv
                opt_clset = [i for i, e in enumerate(ecimult) if e != 0]
                self.opt_sparsity = sparsity
            else:
                if opt_cv > mean_cv:
                    opt_cv = mean_cv
                    opt_clset=[i for i, e in enumerate(ecimult) if e != 0]
                    self.opt_sparsity=sparsity

            if self.sparsity_step == 0.0:
                if self.sparsity_scale == "log":
                    step = float(sparsity/(1.0*10))
                    sparsity = sparsity - 3*step

                elif self.sparsity_scale == "piece_log":
                    if idx==10:
                        idx = 2
                        step = float(sparsity/(1.0*10))
                        sparsity = sparsity - step
                    else:
                        idx=idx+1
                        sparsity = sparsity - step

            else:
                sparsity = sparsity - step

        return opt_clset

    def _select_clusters_lasso_on_residual_cv(self,x,p,clset0):
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error

        sparsity = self.sparsity_max

        if self.sparsity_step == 0.0:
            step = float(sparsity/(1.0*10))
            idpot = 1
        else:
            step = sparsity_step

        opt_cv = -1
        opt_clset = []
        rows = np.arange(len(p))

        idx = 1
        while sparsity.__ge__(self.sparsity_min):

            if self.fit_intercept:
                _comat = np.delete(x, (0), axis=1)
            else:
                _comat = x
                
            fitter_cv = linear_model.Lasso(alpha=sparsity, fit_intercept=self.fit_intercept, normalize = False, max_iter = self.max_iter, tol = self.tol)
            fitter_cv.fit(_comat,p)
            
            fitter_lr = linear_model.LinearRegression(fit_intercept=False, normalize=False)
            
            cluster_list_lasso = []
            for i,coef in enumerate(fitter_cv.coef_):
                if coef != 0:
                    cluster_list_lasso.append(i)

            clset = np.union1d(clset0,cluster_list_lasso)
            _comat_model = x[np.ix_(rows,clset)]

            _cvs = cross_val_score(fitter_lr, _comat_model, p, cv=10, scoring = 'neg_mean_squared_error', n_jobs = -1)
                
            mean_cv = np.sqrt(-np.mean(_cvs))

            self.cvs.append(mean_cv)
            self.rmse.append(np.sqrt(mean_squared_error(fitter_cv.predict(_comat),p)))

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
                    self.opt_sparsity=sparsity

            if self.sparsity_step == 0.0:
                if self.sparsity_scale == "log":
                    step = float(sparsity/(1.0*10))
                    sparsity = sparsity - 3*step

                elif self.sparsity_scale == "piece_log":
                    if idx==10:
                        idx = 2
                        step = float(sparsity/(1.0*10))
                        sparsity = sparsity - step
                    else:
                        idx=idx+1
                        sparsity = sparsity - step

            else:
                sparsity = sparsity - step

        return opt_clset

    def display_info(self):
        """Display in screen information about the optimal model
        """
        try:
            print("{0:<40s}:{1:>10.4f}".format("CV score (LOO) for optimal model", self.opt_mean_cv))
            print("{0:<40s}:{1:>10.4f}".format("RMSE of the fit for optimal model", self.opt_rmse))
            print("{0:<40s}:{1:>10d}".format("Size of optimal clusters pool", len(self.optimal_clusters)))
        except Exception as e:
            if self.opt_mean_cv is None:
                print("ClustersSelector (Error): Calculation of CV score failed.")
            elif self.opt_rmse is None:
                print("ClustersSelector (Error): Calculation of RMSE failed.")
            else:
                print(type(e), e)
        #print("CV score (LOO) for optimal model: "+str(self.opt_mean_cv))
        #print("RMSE of the fit for optimal model: "+str(self.opt_rmse))
        #print("Size of optimal clusters pool: "+str(len(self.optimal_clusters)))
