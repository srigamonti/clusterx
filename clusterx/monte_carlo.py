#from clusterx.model import Model


class MonteCarlo():
    """MonteCarlo class
    
    Description...

    Parameters:
    
    ``energy_model``: Model object
          Model used for acceptance and rejection

    ``scell``: SuperCell object
          ...

    """
    def __init__(energy_model, scell, models=[]):
        self._em = energy_model
        self._scell = scell
        self._models = models

    def metropolis(self, temp, nmc):
        """Perform metropolis simulation
        
        Description:
        
        Parameters:
           
        """
        pass
    
