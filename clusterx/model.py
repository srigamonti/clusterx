from clusterx.correlations import CorrelationsCalculator
from clusterx.estimators.estimator_factory import EstimatorFactory
from clusterx.clusters_selector import ClustersSelector

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
    def __init__(self, corrc, property=None, estimator = None, ecis = None):
        self.corrc = corrc
        self.estimator = estimator
        self.ecis = ecis
        self.property = property

    def predict(self,structure):
        """Predic property with the optimal cluster expansion model.

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

class ModelBuilder():
    """Model class

    Objects of this class represent a cluster expansion model.

    **Parameters:**

    ``basis``: string
        Basis set used to calculate structure-cluster correlations.

    ``selection_method``: string
        Cluster selection method. For the possible values, look at the
        documentation for ``ClustersSelector`` class.

    ``estimator_type``: string
        Estimator type. For the possible values, look at the documentation
        for ``EstimatorFactory`` class.

    """
    def __init__(self,
                 basis="trigonometric",
                 selection_method="identity",
                 estimator_type="skl_LinearRegression",
                 estimator_opts={}):

        self.basis = basis
        self.selection_method = selection_method
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

    def build(self, sset, cpool, prop):
        """Build optimal cluster expansion model

        Acts as a Model factory.

        **Parameters:**

        ``sset``: StructuresSet object
            structures set used for model training.

        ``cpool``: ClustersPool object
            clusters pool from which to select the best model using the method
            indicated in ``selection_method``.

        ``prop``: String
            Property name. Must be a valid name as stored in ``sset``. The list of names
            can be obtained using ``sset.get_property_names()``.

        """
        self.sset = sset
        self.cpool = cpool
        self.plat = self.cpool.get_plat()
        self.prop = prop

        corrc = CorrelationsCalculator(self.basis, self.plat, self.cpool)
        comat = corrc.get_correlation_matrix(self.sset)
        pvals_tr = self.sset.get_property_values(property_name = self.prop)

        # Select optimal clusters using the clusters_selector module
        self.selector = ClustersSelector(self.selection_method,self.cpool)
        self.opt_cpool = self.selector.select_clusters(comat, pvals_tr)
        self.opt_corrc = CorrelationsCalculator(self.basis, self.plat, self.opt_cpool)
        self.opt_comat = self.opt_corrc.get_correlation_matrix(self.sset)

        # Find out the ECIs using an estimator
        self.opt_estimator = EstimatorFactory.create(self.estimator_type, **self.estimator_opts)
        self.opt_estimator.fit(self.opt_comat,pvals_tr)

        return Model(self.opt_corrc,estimator = self.opt_estimator, property=prop)
