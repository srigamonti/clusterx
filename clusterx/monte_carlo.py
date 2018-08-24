import random
from clusterx.structures_set import StructuresSet

class MonteCarlo():
    """MonteCarlo class
    
    Description:
        Perform MonteCarlo samplings

    Parameters:
    
    ``energy_model``: Model object
        Model used for acceptance and rejection. Usually, the Model enables to calculate the total energy of a given configuration. 

    ``scell``: SuperCell object
        Super cell in which the sampling is performed.

    ``nsub``: dictionary
        The format of the dictionary is as follows::                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                             
               {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                            
        It indicates how many substitutional atoms of every kind (n#1, n#2, ...)                                                                                                                                                                                           
        may replace the pristine species for sites of site_type#.

        Defines the number of substituted atoms in each sublattice 
        Supercell in which the sampling is performed.

    ``ensemble``: String
        "canonical" allows only for swapping of atoms inside scell
        "grandcanonical"  allows for replacing atoms within the given scell 
            (the number of subsitutents in each sub-lattice is not kept)

    .. todo:
        Samplings in the grand canonical ensemble are not yet possible.

    """
    
    def __init__(self, energy_model, scell, nsubs, models = [], ensemble = "canonical", no_of_swaps = 1):
        self._em = energy_model
        self._scell = scell
        self._nsubs = nsubs
        self._models = models
        self._ensemble = ensemble
        self._no_of_swaps = no_of_swaps

    def metropolis(self, temp, nmc, initial_structure = None):
        """Perform metropolis simulation
        
        Description: Perfom Metropolis sampling for nmc sampling steps at temperature temp.
        
        Parameters:

        ``nmc``: integer
            Number of sampling steps
        
        ``temp``: integer
            Temperature in the Boltzmann distribution exp(-E/(kb*temp)) defining the acceptance probability

        ``initial_structure``: Structure object
            Sampling starts with the structure defined by this Structure object. 
            If initial_structure = None: Sampling starts with a structure randomly generated.
           
        """
        import math
        
        kb=3.16681009610757e-6
        
        if initial_structure == None:
            struc = self._scell.gen_random(self._nsubs)

        numbers = struc.get_atoms().get_atomic_numbers()
        print(numbers)
        e = self._em.predict_prop(struc)
                
        for i in range(nmc):
            ind1_list = []
            ind2_list = []
            for j in range(self._no_of_swaps):
                ind1,ind2 = struc.swap_random_binary(0)
                ind1_list.append(ind1)
                ind2_list.append(ind2)
                
            e1 = self._em.predict_prop(struc)


            if e >= e1:
                accept_swap=True
            else:
                boltzmann_factor = math.exp((e-e1)/(kb*temp))
                if random.uniform(0,1) <= boltzmann_factor:
                    accept_swap=True
                else:
                    accept_swap=False

            if accept_swap:
                e=e1
            else:
                for j in range(self._no_of_swaps):
                    m=int(self._no_of_swaps-j)
                    ind1=ind1_list[j]
                    ind2=ind2_list[j]
                    struc.swap(0,ind1,ind2)
            
            print(i,e)


class MonteCarloTrajectory():
    pass
                                    
    
