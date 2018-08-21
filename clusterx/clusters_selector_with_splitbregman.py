from clusterx.clusters.clusters_pool import ClustersPool
import numpy as np
import sys

class ClustersSelector():
    """Clusters selector class

    Objects of this class are used to select optimal cluster expansion models, i.e.
    the optimal set of clusters, for given training data and clusters pool.

    **Parameters:**

    ``method``: string
        can be "lasso" or "linreg". In both cases a cross-validation optimization
        is performed. In the case of "lasso", the optimal sparsity parameter is
        searched through cross validation, while in "linreg", cross validation
        is directly used as model selector.
    ``clusters_pool``:ClustersPool object
        the clusters pool from which the optimal model is selected.
    ``**kwargs``: keyword arguments
        if ``method`` is set to "lasso", the keyword arguments are:
            ``sparsity_max``: positive real, maximal sparsity parameter
            ``sparsity_min``: positive real, minimal sparsity parameter
            ``sparsity_step``: positive real, optional, if set to 0.0, a logarithmic
            grid from sparsity_max to sparsity_min is automatically created.
        if ``method`` is set to "linreg", the keyword arguments are:
            ``clusters_sets``: one of "size", "combinations", and "size+combinations".
            In the first case, clusters sub_pools of increasing size are extracted from
            the initial pool, and cross validation selects the optimal sub-pool.
            In the second case, all possible combinations of clusters from the pool
            are considered, this can be very computanionally demanding.
            In the third case, a fixed pool of clusters up to certain size (see ``set0``
            parameter below) is always kept and the combinations are searched only
            for subsets of ``nclmax`` (see below) clusters.

            ``set0``: array with to elements ``[int,float]``
                if ``clusters_sets`` is set to "size+combinations", this indicates
                the size of the fixed pool of clusters, above which a combinatorial
                search is performed. The first element of the array indicates the
                maximum number of cluster points and the second element the maximum radius,
                for the fixed subpool.

            ``nclmax``: integer
                if ``clusters_sets`` is set to "size+combinations", this indicates
                the maximum number of clusters in the combinatorial subsets of clusters
                to be searched for (on top of the fixed subpool, see above).

    """
    def __init__(self, method, clusters_pool, **kwargs):
        self.method = method

        # additional arguments for lasso
        self.sparsity_max = kwargs.pop("sparsity_max",1)
        self.sparsity_min = kwargs.pop("sparsity_min",0.01)
        self.sparsity_step = kwargs.pop("sparsity_step",0.0)

        # additional arguments for linear regression
        self.clusters_sets = kwargs.pop("clusters_sets","size")
        self.nclmax = kwargs.pop("nclmax", 0)
        self.set0 = kwargs.pop("set0",[0, 0])


        # additional arguments for split bregmann
        self.l = kwargs.pop("l", 0.9)  # lambda
        self.loo_idx = kwargs.pop("LOO_idx", None)  # Leave-one-out index

        self.cpool = clusters_pool
        self.fit_intercept=False
        #for c in self.cpool._cpool:
        #    if c.npoints == 0:
        #        self.fit_intercept=True
        #        break

        self.predictions = []
        self.ecis = []
        self.optimal_clusters = None
        self.opt_rmse = None
        self.opt_mean_cv = None
        self.opt_sparsity = None
        self.lasso_sparsities = []

        self.rmse = []
        self.cvs = []
        self.set_sizes = []


        self.fitter_cv = None

        #self.update()

    def get_ecis(self):
        return self.ecis

    def get_rmse(self):
        return self.rmse

    def select_clusters(self, x, p, mult = None):
        """Select clusters

        Selects best model for the cluster expansion. The input parameters
        :math:`x` and :math:`p` relate to each other as in:

        .. math::

            xJ = p^T

        where J are the effective cluster interactions.

        Parameters:

        x: 2d matrix of cluster correlations
            Rows correspond to structures, columns correspond to clusters.
        p: list of property values
            Property values for the training structures set.
        """
        from sklearn.model_selection import LeaveOneOut

        #if np.shape(p)[0] != np.shape(x)[0]:
        #    print "Error(cross_validation.cv): Number of property values differs from number of rows in correlation matrix."
        #    sys.exit(0)
        print(self.method)
        if self.method == "lasso":
            opt = self._select_clusters_lasso(x, p)
            if self.fit_intercept == True:
                if 0 not in opt:
                    self.fit_intercept = False
        
        elif self.method == "split_bregman":
            #try:
            #    assert (mult is None)
            #except AssertionError:
            #    print("Error: Missing argument: array mult for split Bregman")
            #else:
            if (mult is None):    
                print("Error: Missing argument: array mult for split Bregman")
            else:      
                opt = self._select_clusters_split_bregman(x, p, mult)
        else:
            if self.clusters_sets == "size":
                clsets = self.cpool.get_clusters_sets(grouping_strategy = "size")
            if self.clusters_sets == "combinations":
                clsets = self.cpool.get_clusters_sets(grouping_strategy = "combinations",  nclmax=self.nclmax)
            if self.clusters_sets == "size+combinations":
                clsets = self.cpool.get_clusters_sets(grouping_strategy = "size+combinations", nclmax=self.nclmax , set0=self.set0)

            opt = self._linear_regression(x, p, clsets)

        self.optimal_clusters = self.cpool.get_subpool(opt)

        rows = np.arange(len(p))
        comat_opt =  x[np.ix_(rows,opt)]

        self.optimal_ecis(comat_opt,p)

    def get_optimal_cpool(self):
        """Return optimal ClustersPoll object
        """
        return self.optimal_clusters

    def get_optimal_cpool_array(self):
        """Return optimal array of clusters
        """
        return self.optimal_clusters._cpool


    def _linear_regression(self,x,p,clsets):
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error

        if self.method == "linreg":
            self.fitter_cv = linear_model.LinearRegression(fit_intercept=self.fit_intercept, normalize=False)

        rows = np.arange(len(p))
        ecis = []
        ranks = []
        sizes = []

        opt_cv=-1
        opt_clset=[]

        el=True

        for iset, clset in enumerate(clsets):
            _comat = x[np.ix_(rows,clset)]

            #if self.fit_intercept:
            #    if int(_comat.shape[1]) > 1:
            #        _comat = np.delete(_comat, (0), axis=1)
            #    else:
            #        fitter_cv2 = linear_model.LinearRegression(fit_intercept=False, normalize=False)

            #        _cvs = cross_val_score(fitter_cv2, _comat, p, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
            #        mean_cv = np.sqrt(-np.mean(_cvs))
            #        self.cvs.append(mean_cv)
            #        fitter_cv2.fit(_comat,p)
            #        self.rmse.append(np.sqrt(mean_squared_error(fitter_cv2.predict(_comat),p)))
            #        self.set_sizes.append(len(clset))

            #        if opt_cv <= 0:
            #            opt_cv=mean_cv
            #            opt_clset=clset
            #        else:
            #            if opt_cv > mean_cv:
            #                opt_cv = mean_cv
            #                opt_clset=clset

            #        continue


            _cvs = cross_val_score(self.fitter_cv, _comat, p, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
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
        self.predictions = [el for el in self.fitter_cv.predict(_comat)]

        self.opt_rmse=np.sqrt(mean_squared_error(self.fitter_cv.predict(_comat),p))

        _cvs = cross_val_score(self.fitter_cv, _comat, p, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
        self.opt_mean_cv=np.sqrt(-np.mean(_cvs))

        ecimult = []
        if self.fit_intercept:
            ecimult.append(self.fitter_cv.intercept_)

        for coef in self.fitter_cv.coef_:
            ecimult.append(coef)

        self.ecis = ecimult

    def _select_clusters_lasso(self,x,p):
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error

        sparsity = self.sparsity_max

        if self.sparsity_step == 0.0:
            step=float(sparsity/(1.0*10))
            idpot = 1
        else:
            step=sparsity_step

        opt_cv=-1
        opt_clset=[]
        rows = np.arange(len(p))

        idx=1
        while sparsity.__ge__(self.sparsity_min):

            if self.fit_intercept:

                _comat = np.delete(x, (0), axis=1)
            else:

                _comat = x

            fitter_cv = linear_model.Lasso(alpha=sparsity, fit_intercept=self.fit_intercept, normalize=False, max_iter = 1000000000, tol = 1e-12)
            fitter_cv.fit(_comat,p)

            ecimult = []
            if self.fit_intercept:
                ecimult.append(fitter_cv.intercept_)

            for coef in fitter_cv.coef_:
                ecimult.append(coef)

            _cvs = cross_val_score(fitter_cv, _comat, p, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
            mean_cv=np.sqrt(-np.mean(_cvs))

            self.cvs.append(mean_cv)
            self.rmse.append(np.sqrt(mean_squared_error(fitter_cv.predict(_comat),p)))

            self.set_sizes.append(np.count_nonzero(ecimult))
            self.lasso_sparsities.append(sparsity)

            if opt_cv <= 0:
                opt_cv=mean_cv
                opt_clset=[i for i, e in enumerate(ecimult) if e != 0]
                self.opt_sparsity=sparsity
            else:
                if opt_cv > mean_cv:
                    opt_cv = mean_cv
                    opt_clset=[i for i, e in enumerate(ecimult) if e != 0]
                    self.opt_sparsity=sparsity

            if self.sparsity_step == 0.0:
                if idx==10:
                    idx=2
                    step=float(sparsity/(1.0*10))
                    sparsity = sparsity - step
                else:
                    idx=idx+1
                    sparsity = sparsity - step
            else:
                sparsity = sparsity - step

        return opt_clset

    def _select_clusters_split_bregman(self, corr, evals, clmults, LOO_idx = None):
        import collections
        from collections import defaultdict
        from copy import deepcopy
        from operator import itemgetter
        from sklearn.metrics import mean_squared_error
        from math import sqrt

        lamb = self.l
        mu_min = self.sparsity_min
        mu_max = self.sparsity_max
        if self.sparsity_step == 0.0:
            mu_step=float(sparsity/(1.0*10))
        else:
            mu_step=self.sparsity_step
        LOO_idx = self.loo_idx
        eci_dict = defaultdict(list)
        values = reversed(np.arange(mu_min,mu_max,mu_step))
        LOO_corr = corr[LOO_idx]
        LOO_evals= evals[LOO_idx]
        t_corr = np.delete(corr, (LOO_idx), axis=0)
        t_evals = np.delete(evals, (LOO_idx))
        print("\nRunning split bregman iteration for solving for ECIs\n")
        for idx, mu in enumerate(values):
            eci_dict[idx] = defaultdict(list)
            #eci = self._split_bregman_iteration(evals, corr, clmults, mu, lamb, tol=1.0e-10)
            eci = self._split_bregman_iteration(t_evals, t_corr, clmults, mu, lamb, tol=1.0e-10)
            #mse = self._split_bregman_meansqr(eci, evals, clmults, corr)
            pred_e, mse = self._split_bregman_meansqr(eci, t_evals, clmults, t_corr)
            rmse = sqrt(mean_squared_error(pred_e, t_evals))
            LOO_predE = self._split_bregman_eval_energy(eci, clmults, LOO_corr)
            cv = sqrt(mean_squared_error([LOO_predE], [LOO_evals]))
            print("training MSE", mse, "RMSE", rmse, "LOO RMSE",cv)
            eci_dict[idx]["ecis"] = deepcopy(eci)
            eci_dict[idx]["mse"] = mse
            eci_dict[idx]["rmse"] = rmse
            eci_dict[idx]["cv"] =  cv

        tmp_cv_list = [ eci_dict[i]["cv"] for i in eci_dict.keys() ]
        min_idx, min_cv = min(enumerate(tmp_cv_list), key=itemgetter(1))
        print("\nMin. CV", min_cv, "min rmse", eci_dict[min_idx]["rmse"] ) 
        self.ecis = eci_dict[min_cv]
        self.opt_rmse = float(eci_dict[min_idx]["rmse"])
        print(float(eci_dict[min_idx]["rmse"])*1.0e10)
        opt_clset=[i for i, e in enumerate(eci_dict[idx]["ecis"]) if e != 0.0]
        
        return opt_clset

    def _split_bregman_iteration(self, evals, corr, mult, mu, lamb, tol=1.0e-10):
        Ncl = np.shape(mult)[0]
        lastd = np.zeros((Ncl))
        nextd = np.zeros((Ncl))
        lastb = np.ones((Ncl))
        nextb = np.zeros((Ncl))
        last_eci = np.zeros((Ncl))
        diff = 1
        while diff > float(tol):
            next_eci = self._split_bregman_rms_opt(evals, corr, mult, mu, lamb, nextb, lastd, shrink_threshold = 1.0e-3)
            next_eci_mu_prod = np.dot(next_eci, mu)
            nextd = self._split_bregman_shrink(next_eci_mu_prod, nextb, lamb)
            tmp = np.subtract(next_eci_mu_prod, nextd)
            lastb = nextb
            nextb = np.add(lastb, tmp)
            lastd = nextd
            diff = float(sum(np.subtract(last_eci, next_eci)))
            last_eci = next_eci

        return next_eci

    def _split_bregman_rms_opt(self, evals, corr, mult, mu, lamb, b, d, shrink_threshold = None):
        # E or pvals is the ab initio energy
        # corr is the correlation matrix 
        N = np.shape(evals)
        Ncl = np.shape(mult)[0]
        corr_trans = corr.T 
        corr_prod = np.dot(corr_trans, corr)
        lambmu2_matrix = np.zeros((Ncl, Ncl))
        np.fill_diagonal(lambmu2_matrix, float(mu**2*lamb))
        sum_XtX_lambmu2 = np.add(corr_prod, lambmu2_matrix)
        lambmu_matrix = np.zeros((Ncl, Ncl))
        np.fill_diagonal(lambmu_matrix, float(mu*lamb))
        sub_b_d = np.subtract(d,b)
        lambmu_identy_sub_b_d = np.dot(lambmu_matrix,sub_b_d)
        XtE = np.dot(corr_trans, evals)
        first_term = np.linalg.solve(sum_XtX_lambmu2,XtE)
        second_term = np.linalg.solve(sum_XtX_lambmu2,lambmu_identy_sub_b_d)
        ecimult = first_term + second_term
        eci = []
        for i in range(np.shape(ecimult)[0]):
            eci.append(ecimult[i] / (1.0*float(mult[i])))
            if shrink_threshold is not None:
                if abs(eci[i]) < shrink_threshold:
                    eci[i] = 0.0

        return np.array(eci) 

    def _split_bregman_shrink(self, eci, b, lamb):
        
        nextd = np.add(eci, b)
        for i, x in enumerate(nextd):
            if abs(float(x)) <= float(1/float(lamb)):
                nextd[i] = 0.0
            else:
                if float(x) > float(1/float(lamb)):
                    nextd[i] = float(x - 1/float(lamb))
                elif float(x) < float(-1/float(lamb)):
                    nextd[i] = float(x + 1/float(lamb))
                
        return np.array(nextd)

    def _split_bregman_meansqr(self, ecis, propvals, clmult, comat):
        ncl = len(clmult)
        nstr = len(propvals)
        e_list = []
        s2 = 0
        for i in range(nstr):
                ece = 0
                for j in range(ncl):
                        ece = ece + clmult[j] * ecis[j] * comat[i,j]
                s2 = s2 + (ece - propvals[i]) * (ece - propvals[i])
                e_list.append(ece)
        mse = np.sqrt(s2/nstr)

        return e_list, mse

    def _split_bregman_eval_energy(self, ecisE, multE, corr):

        erg = 0
        for j in range(len(ecisE)):
            erg += multE[j] * ecisE[j] * corr[j]
        
        return erg

    def display_info(self):
        """Display in screen information about the optimal model
        """
        print("{0:<40s}:{1:>10.4f}".format("CV score (LOO) for optimal model",self.opt_mean_cv))
        print("{0:<40s}:{1:>10.4f}".format("RMSE of the fit for optimal model",self.opt_rmse))
        print("{0:<40s}:{1:>10d}".format("Size of optimal clusters pool",len(self.optimal_clusters)))
        #print("CV score (LOO) for optimal model: "+str(self.opt_mean_cv))
        #print("RMSE of the fit for optimal model: "+str(self.opt_rmse))
        #print("Size of optimal clusters pool: "+str(len(self.optimal_clusters)))
