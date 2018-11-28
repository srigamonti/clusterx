from clusterx.correlations import CorrelationsCalculator
from clusterx.estimators.estimator_factory import EstimatorFactory
from clusterx.clusters_selector import ClustersSelector
import numpy as np

class Model():
    """Model class

    **Parameters:**

    ``corrc``: CorrelationsCalculator object
        The correlations calculator corresponding to the optimal model.
    ``estimator``: Estimator object
        An estimator object to predict the property values. Alternatively,
        The effective cluster interactions (ECIs) can be set.
    ``ecis``: Array of float
        Effective cluster iteractions (multiplied by the corresponding
        multiplicities). Overrides ``estimator`` object.
    """
    def __init__(self, corrc, property_name, estimator = None, ecis = None):
        self.corrc = corrc
        self.estimator = estimator
        self.ecis = ecis
        self.property = property_name

    def get_ecis(self):
        if self.ecis is not None:
            return self.ecis
        else:
            return self.estimator.coef_

    def predict(self,structure):
        """Predict property with the optimal cluster expansion model.

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property to.
        """
        corrs = self.corrc.get_cluster_correlations(structure)
        if self.estimator is not None:
            return self.estimator.predict(corrs.reshape(1,-1))[0]
        else:
            pv = 0
            for i in range(len(corrs)):
                pv = pv + self.ecis[i]*corrs[i]
            return pv

    def predict_swap_binary_linear(self,structure, ind1 = None, ind2 = None, correlation = False):
        """Predict property difference with the optimal cluster expansion model only for binary-linear)

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property difference to.

        ``ind1``: int
            index of first atom position has been swapped

        ``ind2``: int
            index of second atom position has been swapped

        """
        #corrsx = self.corrc.get_cluster_correlations(structure)
        try:
            cluster_orbits = self.corrc._cluster_orbits_set[0]
        except AttributeError:
            raise AttributeError("Cluster_orbit set is not predefined, look at the documentation.")
        corrs = np.zeros(len(cluster_orbits))

        sigma_ind1=structure.sigmas[ind1]
        sigma_ind2=structure.sigmas[ind2]

        for icl, cluster_orbit in enumerate(cluster_orbits):
            for cluster in cluster_orbit:
                cluster_atomic_idxs = cluster.get_idxs()
                if ind1 in cluster_atomic_idxs:
                    #cluster_alphas = cluster.alphas
                    sigmas = [structure.sigmas[cl_idx] for cl_idx in cluster_atomic_idxs]
                    cf1 = np.prod(np.array(sigmas))
                    sigmas[cluster_atomic_idxs.index(ind1)] = sigma_ind2
                    if ind2 in cluster_atomic_idxs:
                        sigmas[cluster_atomic_idxs.index(ind2)] = sigma_ind1
                    cf0 = np.prod(np.array(sigmas))
                    corrs[icl] += cf1
                    corrs[icl] += (-1)*cf0

                elif ind2 in cluster_atomic_idxs:
                    #cluster_alphas = cluster.alphas
                    sigmas = [structure.sigmas[cl_idx] for cl_idx in cluster_atomic_idxs]
                    cf1 = np.prod(np.array(sigmas))
                    sigmas[cluster_atomic_idxs.index(ind2)] = sigma_ind1
                    cf0 = np.prod(np.array(sigmas))
                    corrs[icl] += cf1
                    corrs[icl] += (-1)*cf0


            corrs[icl] /= len(cluster_orbit)
        corrs = np.around(corrs,decimals=12)
        if correlation:
            return corrs

        if self.estimator is not None:
            return self.estimator.predict(corrs.reshape(1,-1))[0]
        else:
            pv = 0
            for i in range(len(corrs)):
                pv = pv + self.ecis[i]*corrs[i]
            return pv
        
        
    def predict_swap(self, structure, ind1 = None, ind2 = None, correlation = False):
        """Predict property difference with the optimal cluster expansion model.

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property difference to.

        ``ind1``: int
            index of first atom position has been swapped

        ``ind2``: int
            index of second atom position has been swapped

        """
        #corrsx = self.corrc.get_cluster_correlations(structure)
        try:
            cluster_orbits = self.corrc._cluster_orbits_set[0]
        except AttributeError:
            raise AttributeError("Cluster_orbit set is not predefined, look at the documentation.")
        corrs = np.zeros(len(cluster_orbits))

        sigma_ind1=structure.sigmas[ind1]
        sigma_ind2=structure.sigmas[ind2]

        ems_ind1=structure.ems[ind1]
        ems_ind2=structure.ems[ind2]

        for icl, cluster_orbit in enumerate(cluster_orbits):
            for cluster in cluster_orbit:
                cluster_atomic_idxs = cluster.get_idxs()
                contribute = False
                if ind1 in cluster_atomic_idxs:
                    cluster_alphas = cluster.alphas
                    sigmas = [structure.sigmas[cl_idx] for cl_idx in cluster_atomic_idxs]
                    ems = [structure.ems[cl_idx] for cl_idx in cluster_atomic_idxs]

                    cf1 = 1.0
                    for i,alpha in enumerate(cluster_alphas):
                        cf1 *= self.corrc.site_basis_function(alpha, sigmas[i], ems[i])

                    clind =cluster_atomic_idxs.index(ind1)
                    sigmas[clind] = sigma_ind2
                    ems[clind] = ems_ind2
                    if ind2 in cluster_atomic_idxs:
                        clind =cluster_atomic_idxs.index(ind2)
                        sigmas[clind] = sigma_ind1
                        ems[clind] = ems_ind1

                    cf0 = 1.0
                    for i,alpha in enumerate(cluster_alphas):
                        cf0 *= self.corrc.site_basis_function(alpha, sigmas[i], ems[i])

                    corrs[icl] += cf1
                    corrs[icl] += (-1)*cf0

                elif ind2 in cluster_atomic_idxs:
                    cluster_alphas = cluster.alphas
                    sigmas = [structure.sigmas[cl_idx] for cl_idx in cluster_atomic_idxs]
                    ems = [structure.ems[cl_idx] for cl_idx in cluster_atomic_idxs]

                    cf1 = 1.0
                    for i,alpha in enumerate(cluster_alphas):
                        cf1 *= self.corrc.site_basis_function(alpha, sigmas[i], ems[i])

                    clind =cluster_atomic_idxs.index(ind2)
                    sigmas[clind] = sigma_ind1
                    ems[clind] = ems_ind1

                    cf0 = 1.0
                    for i,alpha in enumerate(cluster_alphas):
                        cf0 *= self.corrc.site_basis_function(alpha, sigmas[i], ems[i])

                    corrs[icl] += cf1
                    corrs[icl] += (-1)*cf0

            corrs[icl] /= len(cluster_orbit)
        corrs = np.around(corrs,decimals=12)
        if correlation:
            return corrs

        if self.estimator is not None:
            return self.estimator.predict(corrs.reshape(1,-1))[0]
        else:
            pv = 0
            for i in range(len(corrs)):
                pv = pv + self.ecis[i]*corrs[i]
            return pv

    def report_errors(self, sset):
        """Report fit and CV scores

        **Parameters**:
        ``sset``: StructuresSet object
            the scores are computed for the give structures set
        """
        errfit = self.get_errors(sset)
        errcv = self.get_cv_score(sset)

        print("\n+-----------------------------------------------------------+")
        print("|                Report of Fit and CV scores                |")
        print("+-----------------------------------------------------------+")
        print("|{0:<19s}|{1:^19s}|{2:^19s}|".format("","Fit","CV"))
        print("+-----------------------------------------------------------+")
        print("|{0:^19s}|{1:^19.5f}|{2:^19.5f}|".format("RMSE", errfit["RMSE"], errcv["RMSE-CV"]))
        print("|{0:^19s}|{1:^19.5f}|{2:^19.5f}|".format("MAE", errfit["MAE"], errcv["MAE-CV"]))
        print("|{0:^19s}|{1:^19.5f}|{2:^19.5f}|".format("MaxAE", errfit["MaxAE"], errcv["MaxAE-CV"]))
        print("+-----------------------------------------------------------+\n")

    def get_errors(self,sset):
        """Compute RMSE, MAE and MaxAE for model in structure set.
        """
        calc_vals = sset.get_property_values(property_name = self.property)
        predictions = sset.get_predictions(self)
        rmse = 0
        mae = 0
        maxae = 0
        for e,p in zip(calc_vals,predictions):
            rmse += (e-p)*(e-p)
            mae += np.abs(e-p)
            if np.abs(e-p) > maxae:
                maxae = np.abs(e-p)

        rmse = np.sqrt(rmse/len(calc_vals))
        mae = mae/len(calc_vals)

        return {"RMSE":rmse, "MAE": mae, "MaxAE":maxae}

    def get_cv_score(self,sset,fit_params=None):
        """Get leave-one-out cross-validation score over structures set.

        **Parameters:**

        ``fit_params``: dictionary
            Parameters to pass to the fit method of the estimator.
        """
        from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import mean_squared_error
        from sklearn import linear_model

        X = self.corrc.get_correlation_matrix(sset)
        y = sset.get_property_values(self.property)

        # cross_val_score internally clones the estimator, so the optimal one in Model is not changed.
        cvs = cross_val_score(self.estimator, X, y, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
        pred_cv = cross_val_predict(self.estimator, X, y, fit_params=fit_params, cv=LeaveOneOut())
        #cv_results = cross_validate(self.estimator, X, y, cv=LeaveOneOut(), scoring = 'neg_mean_squared_error')
        #cvs = cv_results['test_score']

        aes = np.sqrt(-cvs)
        cv = np.sqrt(-np.mean(cvs))
        maxae = 0
        for ae in aes:
            if ae > maxae:
                maxae = ae
        mae = 0
        for ae in aes:
            mae += ae
        mae /= len(aes)
        return {"RMSE-CV": cv, "MAE-CV": mae, "MaxAE-CV": maxae, "Predictions-CV":pred_cv}

class ModelBuilder():
    """Model class

    Objects of this class represent a cluster expansion model.

    **Parameters:**

    ``basis``: string
        Basis set used to calculate structure-cluster correlations.

    ``selector_type``: string
        Cluster selector type. For the possible values, look at the
        documentation for ``ClustersSelector`` class.

    ``selector_opts``: dictionary
        Cluster selector options. For the possible values, look at the
        documentation for ``ClustersSelector`` class.

    ``estimator_type``: string
        Estimator type. For the possible values, look at the documentation
        for ``EstimatorFactory`` class.

    ``estimator_opts``: dictionary
        Estimator options. For the possible values, look at the documentation
        for ``EstimatorFactory`` class.
    """
    def __init__(self,
                 basis="trigonometric",
                 selector_type="identity",
                 selector_opts={},
                 estimator_type="skl_LinearRegression",
                 estimator_opts={}):

        self.basis = basis
        self.selector_type = selector_type
        self.selector_opts = selector_opts
        self.estimator_type = estimator_type
        self.estimator_opts = estimator_opts

        self.opt_estimator = None
        self.opt_cpool = None
        self.opt_corrc = None
        self.opt_comat = None
        self.opt_estimator = None
        self.selector = None

    def get_selector(self):
        """
        Return selector used in build.

        When the ``build`` method is called, a ``ClustersSelector`` object
        is created to perform the cluster selection task. This selector
        can be obtained by calling this function.
        """
        return self.selector

    def get_opt_cpool(self):
        """
        Return optimal clusters pool found in build.

        When the ``build`` method is called, a ``ClustersSelector`` object
        is created to perform the cluster selection task. The selected clusters pool
        can be obtained by calling this function.
        """
        return self.opt_cpool

    def get_estimator(self):
        """Return estimator object used to create the optimal model
        """
        return self.opt_estimator

    def build(self, sset, cpool, prop):
        """Build optimal cluster expansion model

        Acts as a Model factory.

        **Parameters:**

        ``sset``: StructuresSet object
            structures set used for model training.

        ``cpool``: ClustersPool object
            clusters pool from which to select the best model using the method
            indicated in ``selector_type``.

        ``prop``: String
            Property name. Must be a valid name as stored in ``sset``. The list of names
            can be obtained using ``sset.get_property_names()``.

        """
        self.sset = sset
        self.cpool = cpool
        self.plat = self.cpool.get_plat()
        self.prop = prop

        corrc = CorrelationsCalculator(self.basis, self.plat, self.cpool)
        self.ini_comat = corrc.get_correlation_matrix(self.sset)
        self.target = self.sset.get_property_values(property_name = self.prop)

        # Select optimal clusters using the clusters_selector module
        self.selector = ClustersSelector(basis=self.basis, method=self.selector_type, **self.selector_opts)
        self.opt_cpool = self.selector.select_clusters(sset, cpool, prop)
        self.opt_corrc = CorrelationsCalculator(self.basis, self.plat, self.opt_cpool)
        self.opt_comat = self.opt_corrc.get_correlation_matrix(self.sset)

        # Find out the ECIs using an estimator
        self.opt_estimator = EstimatorFactory.create(self.estimator_type, **self.estimator_opts)
        self.opt_estimator.fit(self.opt_comat,self.target)

        return Model(self.opt_corrc, prop, estimator = self.opt_estimator)
