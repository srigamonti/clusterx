import numpy as np
from importlib import import_module
#import sklearn

class EstimatorFactory(object):
    """Property-estimator factory
    """

    @staticmethod
    def create(estimator_type, **kwargs):
        """Create an estimator object compatible with scikit learn library
        """
        if estimator_type.startswith("skl_"):
            class_name = estimator_type[4:]
            _module = import_module(".linear_model", package="sklearn")
        else:
            class_name = estimator_type
            if class_name == "SplitBregman":
                _module = import_module(".estimators.split_bregman", package="clusterx")

        _class = getattr(_module, class_name)
        estimator = _class(**kwargs)

        return estimator
