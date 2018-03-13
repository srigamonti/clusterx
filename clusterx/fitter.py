import numpy as np
from sklearn import linear_model


class Fitter():
    """Fitter object.

    The fitter object provides a solver for the linear problem 
    .. math::

        XJ = P
    
    where :math:`X` is the correlation matrix (columns correspond to clusters, rows to structures),
    :math:`J` is a column vector of effective cluster-interactions, and :math:`P` is a column
    vector of calculated physical properties.

    Parameters:

    method: str
        The method to be used. Possible values are: "skl_LinearRegression", "skl_Ridge", "skl_RidgeCV",
        "skl_Lasso". For the options starting with "skl_", see the corresponding documentation in 
        `scikit-learn <http://scikit-learn.org/stable/modules/linear_model.html>`_. In these cases the 
        \*\*kwargs are directly passed to the corresponding scikit-learn objects. 
    \*\*kwargs: dictionary of key value pairs
        See explanation for method 
    """
    def __init__(self, method, **kwargs):
        self.alpha = kwargs.pop("alpha",0.0)
        self.alphas = kwargs.pop("alphas",np.zeros(1))
        self.fit_intercept = kwargs.pop("fit_intercept",True)
        self.reg = None
        self.update()

    def update(self):
        
        if method == "skl_LinearRegression":
            self.reg = linear_model.LinearRegression(**kwargs)
