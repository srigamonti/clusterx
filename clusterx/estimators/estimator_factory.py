# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
from importlib import import_module

class EstimatorFactory(object):
    """Property-estimator factory
    """

    @staticmethod
    def create(estimator_type, **estimator_opts):
        """Create an estimator object from (or compatible with) the `scikit-learn <https://scikit-learn.org/stable/index.html>`_ library.
            
        **Parameters:**

        ``estimator_type``: string
            If the string starts with ``"skl_"``, and the full string is ``"skl_EstimatorName"``, 
            then an instance of the ``"EstimatorName"`` estimator of the
            `scikit-learn <https://scikit-learn.org/stable/index.html>`_ library is created.
          
        ``estimator_opts``: dictionary
            The estimator ``"EstimatorName"`` is initialized with the parameters given by the 
            dictionary ``estimator_opts``


        **Examples:**
        
        In both examples below, ``X`` is the input matrix, ``y`` is the vector of property values, and ``X0`` is the input
        vector for a sample for which we want to predict the property value.
        
        The precise meaning and the complete list of parameters in the ``estimator_opts`` dictionary is to be taken from the 
        documentation for the input parameters of the corresponding sklearn class (see example links below).
 
        Create a linear regression estimator object from the scikit-learn class 
        `sklearn.linear_model.LinearRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_::
        
            from clusterx.estimator_factory import EstimatorFactory

            linreg = EstimatorFactory.create("skl_LinearRegression", {"fit_intercept": True})

            linreg.fit(X,y)
            
            prediction0 = linreg.predict(X0)
            ...
            
        Create a LASSO estimator object from the scikit-learn class 
        `sklearn.linear_model.Lasso <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_::
        
            from clusterx.estimator_factory import EstimatorFactory

            lasso = EstimatorFactory.create("skl_Lasso", {"fit_intercept": True, "alpha": 0.1})

            lasso.fit(X,y)

            prediction0 = lasso.predict(X0)
            ...
            
        
        """
        
        if estimator_type.startswith("skl_"):
            class_name = estimator_type[4:]
            _module = import_module(".linear_model", package="sklearn")
        else:
            class_name = estimator_type
            if class_name == "SplitBregman":
                raise NotImplementedError

        _class = getattr(_module, class_name)
        estimator = _class(**estimator_opts)

        return estimator
