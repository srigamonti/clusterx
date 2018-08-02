from clusterx.clusters.clusters_pool import ClustersPool
import numpy as np
import sys

class ClustersSelector():
    def __init__(self, method, clusters_pool, **kwargs):
        self.method = method

        # additional arguments for lasso
        self.sparsity_max = kwargs.pop("sparsity_max",1)
        self.sparsity_min = kwargs.pop("sparsity_min",0.01)
        self.sparsity_step = kwargs.pop("sparsity_step",0.0)

        # additional arguments for linear regression
        self.clusters_sets = kwargs.pop("clusters_sets","size")
        self.nclmax = kwargs.pop("nclmax", 0)
        self.set0 = kwargs.pop("size0",[0, 0])

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


    def select_clusters(self, x, p):
        """Select clusters

        Selects best model for the cluster expansion. The input parameters
        :math:`x` and :math:`p` relate to each other as in:

        ..math::
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

        if self.method == "lasso":
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
