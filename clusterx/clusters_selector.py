


class ClustersSelector():
    def __init__(self, method, clusters_pool, **kwargs):
        self.method = method
        self.alpha0 = kwargs.pop("alpha0",0.0)
        self.alpha1 = kwargs.pop("alpha1",10.0)
        self.alpha_step = kwargs.pop("alpha_step",1.0)
        self.fitter_size = kwargs.pop("fitter_size","linreg")

        self.cpool = clusters_pool

        self.ecis = []
        self.rmse = []
        self.cvs = []
        self.set_sizes = []

        self.optimal_clusters = None

        self.lasso_sparsity = []

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

        if self.method == "size":
            oc = self._select_clusters_increasing_size(x, p)
            self.optimal_clusters = self.cpool.get_subpool(oc)
            return self.optimal_clusters

        if self.method == "lasso":
            self._select_clusters_lasso(x, p)


    def _select_clusters_increasing_size(self,x,p):
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error

        clsets = cpool.get_clusters_sets(grouping_strategy = "size")


        if self.fitter_size == "linreg":
            self.fitter_cv = linear_model.LinearRegression(fit_intercept=True, normalize=False)

        #fitter_cv = linear_model.LinearRegression(fit_intercept=False, normalize=False)

        #cv = []
        rmse = []
        rows = np.arange(len(energies))
        ecis = []
        ranks = []
        sizes = []
        for iset, clset in enumerate(clsets):
            _comat = comat[np.ix_(rows,clset)]
            _cvs = cross_val_score(self.fitter_cv, _comat, energies, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
            self.cvs.append(np.sqrt(-np.mean(_cvs)))
            fitter_cv.fit(_comat,energies)
            self.rmse.append(np.sqrt(mean_squared_error(fitter_cv.predict(_comat),energies)))
            ecis.append(fitter_cv.coef_)
            ranks.append(np.linalg.matrix_rank(_comat))
            sizes.append(len(clset))

    def _select_clusters_lasso(self,x,p):
        pass
