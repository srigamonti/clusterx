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
        self.method = method
        self.alpha = kwargs.pop("alpha",0.0)
        self.alphas = kwargs.pop("alphas",np.zeros(1))
        self.fit_intercept = kwargs.pop("fit_intercept",True)
        self.normalize = kwargs.pop("normalize",True)
        self.skl_reg = None
        self.update()
        
    def update(self):
        if self.method == "skl_LinearRegression":
            self.skl_reg = linear_model.LinearRegression(fit_intercept=self.fit_intercept, normalize=self.normalize)


    def set_alpha(self,alpha):
        self.alpha = alpha
        if self.method == "skl_Ridge":
            self.update()
            
    def set_alphas(self,alphas):
        self.alphas = alphas

    def fit(self, data, target, sample_weight = None):
        if self.method == "skl_LinearRegression":
            self.skl_reg.fit(data, target, sample_weight)
            

    def predict(self, data):
        if self.method == "skl_LinearRegression":
            return self.skl_reg.predict(data)
            
