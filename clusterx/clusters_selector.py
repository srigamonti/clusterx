


class ClustersSelector():
    def __init__(self,method,**kwargs):
        self.method = method
        self.alpha = kwargs.pop("alpha",0.0)
        self._fitter = None
        self.update()

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
        
        if self.method == "lr_loocv":
            if np.shape(p)[0] != np.shape(x)[0]:
                print "Error(cross_validation.cv): Number of property values differs from number of rows in correlation matrix."
                sys.exit(0)

                

        X = [1, 2, 3, 4]
        loo = LeaveOneOut()
        for train, test in loo.split(X):
            print("%s %s" % (train, test))
            

