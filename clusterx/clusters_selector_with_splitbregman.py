from clusterx.clusters.clusters_pool import ClustersPool
import numpy as np
import sys

class ClustersSelector():
    """Clusters selector class

    Objects of this class are used to select optimal cluster expansion models, i.e.
    the optimal set of clusters, for given training data and clusters pool.

    **Parameters:**

    ``method``: string
        can be "split bregman" or "lasso" or "linreg". 
        The split-bregman routine is implemented as a estimator within scikit learn
        Cross-validation optimization is performed using sklearn.
        In the case of "split-bregman" and "lasso", the optimal sparsity parameter is
        searched through cross validation. 
        For "linreg", cross validation is directly used as model selector.
    ``clusters_pool``:ClustersPool object
        the clusters pool from which the optimal model is selected.
    ``**kwargs``: keyword arguments
        if ``method`` is set to "split bregman", the keyword arguments are:
            ``sparsity_max``: positive real, maximal sparsity parameter
            ``sparsity_min``: positive real, minimal sparsity parameter
            ``sparsity_step``: positive real, optional, if set to 0.0, a logarithmic
            grid from sparsity_max to sparsity_min is automatically created.
            ``lamb``: positive real, optional
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
        self.splitbregman_sparsity = []

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
        if self.method == "split_bregman":
            if (mult is None):    
                print("Error: Missing argument: array mult for split Bregman")
            else:      
                #opt = self._byhand_loo_cv_split_bregman_lasso_compaprison(x, p, mult)
                opt = self._select_clusters_split_bregman(x, p, mult)
            self.optimal_clusters = self.cpool.get_subpool(opt)

        elif  self.method == "lasso":
            opt = self._select_clusters_lasso(x, p)
            if self.fit_intercept == True:
                if 0 not in opt:
                    self.fit_intercept = False
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

    
    def _byhand_loo_cv_split_bregman_lasso_compaprison(self, corr, evals, clmults):
        #todo: implement a more robust comparison 
        import collections
        from collections import defaultdict
        from math import sqrt
        #
        from copy import deepcopy
        from operator import itemgetter
        #
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error
        import clusterx
        from clusterx.split_bregman import SplitBregmanEstimator

        lamb = self.l
        mu_min = self.sparsity_min
        mu_max = self.sparsity_max
        if self.sparsity_step == 0.0:
            mu_step=float(sparsity/(1.0*10))
        else:
            mu_step=self.sparsity_step
                
        eci_dict = defaultdict(list)
        
        print("\nRunning split bregman iteration for solving for ECIs\n")
        
        LOO_corr = np.zeros((1, corr.shape[1]))
        nstruc = corr.shape[0]
        cv_list = []
        lasso_cv_list = []
        values = reversed(np.arange(mu_min,mu_max,mu_step))
        values= [0.01, 0.001, 0.0001]
        for idx, mu in enumerate(values):
            print(mu, lamb)
            self.fitter_cv = SplitBregmanEstimator(mult=clmults, mu=mu, lamb=lamb, tol=1.0e-10)
            lasso_fitter = linear_model.Lasso(alpha=mu, fit_intercept=False, normalize=False, max_iter = 1000000000, tol = 1e-12)
            self.fitter_cv.fit(corr, evals)
            print("INITIAL FIT")
            for i,j in zip(self.fitter_cv.predict(corr), evals):
                print(i,j)
            print("MSE OF IN INITIAL FIT", np.mean(self.fitter_cv.predict(corr)-evals))
            for LOO_idx in range(nstruc):
                print(LOO_idx)
                LOO_corr[0,:] = deepcopy(corr[LOO_idx,:])
                LOO_evals = evals[LOO_idx]
                t_corr = np.delete(corr, (LOO_idx), axis=0)
                t_evals = np.delete(evals, (LOO_idx))
                self.fitter_cv.fit(t_corr, t_evals)
                lasso_fitter.fit(t_corr,t_evals)
                print("FIT")
                for i,j, k in zip(self.fitter_cv.predict(t_corr), lasso_fitter.predict(t_corr), t_evals):
                    print(i,j,k)
                mse=mean_squared_error(self.fitter_cv.predict(t_corr),t_evals)
                lasso_mse=mean_squared_error(lasso_fitter.predict(t_corr),t_evals)
                LOO_predE = self.fitter_cv.predict(LOO_corr)
                lasso_LOO_predE =lasso_fitter.predict(LOO_corr)
                print("SPLIT_BREGMAN", LOO_predE, "LASSO", lasso_LOO_predE, "TRUE", LOO_evals)
                cv = sqrt(mean_squared_error([LOO_predE], [LOO_evals]))
                lasso_cv = sqrt(mean_squared_error([lasso_LOO_predE], [LOO_evals]))
                print("training fit MSE", mse, "RMSE LOO SPLIT_BREGMAN CV",cv, "RMSE LOO LASSO CV", lasso_cv)
                cv_list.append(cv)
                lasso_cv_list.append(lasso_cv)
                print("A COMPARISON OF ECIS FROM SPLIT_BREGMAN AND LASSO")
                for i,j in zip(self.fitter_cv.coef_, lasso_fitter.coef_):
                    print(i,j)
            print("AVERAGE CV SCORE", np.mean(cv_list), np.mean(lasso_cv_list))

        sys.exit()

    def _select_clusters_split_bregman(self, corr, evals, clmults, LOO_idx = None):
        import collections
        from collections import defaultdict
        from math import sqrt
        #
        from copy import deepcopy
        from operator import itemgetter
        #
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score, cross_val_predict
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error
        import clusterx
        from clusterx.split_bregman import SplitBregmanEstimator
        
        lamb = self.l
        mu_min = self.sparsity_min
        mu_max = self.sparsity_max
        if self.sparsity_step == 0.0:
            mu_step=float(sparsity/(1.0*10))
        else:
            mu_step=self.sparsity_step
        eci_dict = defaultdict(list)
        values = reversed(np.arange(mu_min,mu_max,mu_step))
        print("\nRunning split bregman iteration for solving for ECIs\n")
        
        for idx, mu in enumerate(values):
            print(mu, lamb)
            eci_dict[idx] = defaultdict(list)    
            self.fitter_cv = SplitBregmanEstimator(mult=clmults, mu=mu, lamb=lamb, tol=1.0e-10)
            self.fitter_cv.fit(corr, evals)
            mse = mean_squared_error(self.fitter_cv.predict(corr),evals)
            pred_cvs = cross_val_predict(self.fitter_cv, corr, evals, cv=LeaveOneOut())
            # for i, j in zip(evals, pred_cvs):
            #     print(float(i), float(j))
            mean_cv = np.sqrt(np.average(abs(pred_cvs-evals)))
            print("CV MSE score",mean_cv)
            self.cvs.append(mean_cv)
            self.rmse.append(np.sqrt(mse))
            #self.set_sizes.append(np.count_nonzero(ecimult))
            self.splitbregman_sparsity.append(mu)
            eci_dict[idx]["ecis"] = deepcopy(self.fitter_cv.coef_)
            eci_dict[idx]["mse"] = mse
            eci_dict[idx]["rmse"] = np.sqrt(mse)
            eci_dict[idx]["cv"] =  mean_cv

        tmp_cv_list = [ eci_dict[i]["cv"] for i in eci_dict.keys() ]
        min_idx, min_cv = min(enumerate(tmp_cv_list), key=itemgetter(1))
        print("\nMin. CV", min_cv, "min rmse", eci_dict[min_idx]["rmse"] ) 
        self.opt_rmse = float(eci_dict[min_idx]["rmse"])
        opt_clset=[i for i, e in enumerate(eci_dict[idx]["ecis"]) if e != 0.0] 

        return opt_clset

    def display_info(self):
        """Display in screen information about the optimal model
        """
        print("{0:<40s}:{1:>10.4f}".format("CV score (LOO) for optimal model",self.opt_mean_cv))
        print("{0:<40s}:{1:>10.4f}".format("RMSE of the fit for optimal model",self.opt_rmse))
        print("{0:<40s}:{1:>10d}".format("Size of optimal clusters pool",len(self.optimal_clusters)))
        #print("CV score (LOO) for optimal model: "+str(self.opt_mean_cv))
        #print("RMSE of the fit for optimal model: "+str(self.opt_rmse))
        #print("Size of optimal clusters pool: "+str(len(self.optimal_clusters)))
