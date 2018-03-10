from collections import Counter
import numpy as np
import sys

class CorrelationsCalculator():
    """
    Correlations calculator object.
    trigonometric basis CALPHAD 33, 266 (2009)
    """
    def __init__(self, basis, parent_lattice):
        self.basis = basis
        self._plat = parent_lattice
        self._2pi = 2*np.pi
        
    def site_basis_function(self, cluster_atomic_nr, site_atomic_nrs, structure_atomic_nr):
        """
        Calculates the site basis function. 
        """
        alpha = np.argwhere(site_atomic_nrs == cluster_atomic_nr).flatten()
        if len(alpha) == 0:
            sys.exit("Error: cluster atomic number not in allowed species for site.")
        alpha = alpha[0]
        
        sigma = np.argwhere(site_atomic_nrs == structure_atomic_nr).flatten()
        if len(sigma) == 0:
            sys.exit("Error: structure atomic number not in allowed species for site.")
        sigma = sigma[0]
        
        m = len(site_atomic_nrs)
        if self.basis = "trigonometric":
            if alpha == 0:
                return 1

            elif alpha%2 != 0:
                return -np.cos(self._2pi*np.ceil(alpha/2.0)*sigma/m)

            else:
                return -np.sin(self._2pi*np.ceil(alpha/2.0)*sigma/m)
                
    def cluster_function(self, cluster, structure_atomic_nrs):
        
