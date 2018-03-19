

class Model():
    def __init__(correlations_calulator, ecis):
        self.ecis = ecis
        self.corcal = correlations_calculator

    def predict_prop(self, structure):
        # Calculate correlations X for structure
        # Calculate property X*ecis
        # return the result
        pass
    
class ModelConstructor():
    """
    The model constructor will keep a large pool of clusters and pop out models with a correlations calculator
    containing only the selected clusters. Thus, the model constructor is the heavy weapon, while the model
    is sort of a cheep calcultor for predicting properties.
    

    Ideally one would be able to construct various models with the constructor 
    (for instance by changing any of the fitter, the property, the clusters_selector, etc..). The returned models
    can then be used for several purposes.
    """
    def __init__(prop = "energy", # The property to be modeled
                 method = "", # The method for cluster selection, e.g. L2-CV, LASSO, LASSO+L0, ...
                 method_params = {} # The parameters for cluster selection
                 training_set = None, # The set of training structures, containing the calculated property
                 fitter = None, # Once clusters are selected, the fitter is used to determine the ECI's
                 correlations_calculator = None
    ):
        self.prop = prop
        self.method = method
        self.training_set = training_set
        self.fitter = fitter
        self._model = Model()


    def build_model():

        if method == "L2-CV"
        return self._model
